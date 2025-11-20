import time
import os
import numpy as np
import torch
import open3d as o3d

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import REGISTER, TRACKER, SEGMENTER
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.modules.segmenter.dummy_segmenter import DummySegmenter
from point2pose.io.outputs.point_cloud_io import save_reg_pcd, save_pcd
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.lie import se3_to_vec, log_SE3, exp_se3, vec_to_se3
from point2pose.data_types.front_end_result import FrontEndResult


class FrontEnd:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        self.num_obj = self.pipeline_cfg.get("max_num_obj", 1)

        # ------------- depth related -------------
        self.min_depth = self.pipeline_cfg.get("min_depth", 0.05)
        self.max_depth = self.pipeline_cfg.get("max_depth", 1.0)
        self.fill_missing_depth = self.pipeline_cfg.get("fill_missing_depth", False)
        self.fill_missing_depth_window_size = self.pipeline_cfg.get(
            "fill_missing_depth_window_size", 3
        )
        self.fill_missing_depth_min_neighbors = self.pipeline_cfg.get(
            "fill_missing_depth_min_neighbors", 1
        )

        # ------------- front end modules -------------
        self.use_segmenter = self.pipeline_cfg.get("use_segmenter", False)

        if self.use_segmenter:
            self.segmenter = build_from_cfg(cfg.segmenter, SEGMENTER)
        else:
            self.segmenter = DummySegmenter(cfg.segmenter)

        self.tracker = build_from_cfg(cfg.tracker, TRACKER)
        self.register = build_from_cfg(cfg.register, REGISTER)

        # -------------- registration params --------------
        self._frame_reg_mode = self.pipeline_cfg.get("frame_reg_mode", "hybrid")
        self.reg_uncer_thres = self.cfg.register.params.get("uncer_thres", 0.3)
        self.reg_residual_thres = self.cfg.register.params.get("residual_thres", 0.07)
        self._reg_remove_outside_mask = self.cfg.register.params.get(
            "remove_outside_mask", False
        )

        # -------------- debug related --------------
        self.debug_level = self.pipeline_cfg.get("debug_level", 0)
        self.debug_dir = self.pipeline_cfg.get("debug_dir", None)

        if self.debug_level > 0 and self.debug_dir is not None:
            self._reg_debug_dir = os.path.join(self.debug_dir, "register")
            os.makedirs(self._reg_debug_dir, exist_ok=True)

        self.cropped_pcd_dir = self.pipeline_cfg.get(
            "cropped_pcd_dir",
            os.path.join(self.debug_dir if self.debug_dir else "./debug", "pcd"),
        )
        self.save_cropped_pcd = self.pipeline_cfg.get("save_cropped_pcd", False)
        if self.save_cropped_pcd:
            os.makedirs(self.cropped_pcd_dir, exist_ok=True)

        self.frame_id = 0

    def initialize(self, frame):
        """Initialize segmentation and tracker with the first frame."""
        self.segmenter.initialize(frame.rgb)

        if self.use_segmenter:
            # get number of objects
            # self.num_obj = np.sum(np.asarray(self.segmenter.input_labels) == 1)
            # get segmentation mask
            obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
            frame.mask = mask_logits

        self.tracker.initialize(frame)
        self.frame_id = 0

    def step(self, frame, track_table: PointTrackTable, objects: list):
        """
        Perform one tracking step.
        Note: if using segmentation, this function will update the frame.mask in place.
        Args:
            frame: Current frame object
            track_table: Current point track table
            objects: List of Object instances
        Returns:
            FrontEndResult
        """
        ##########################################################
        ##                     segmenter                        ##
        ##########################################################
        if self.use_segmenter:
            _, mask_logits = self.segmenter.segment(frame.rgb)
            frame.mask = mask_logits  # mask is a torch tensor on gpu

        ##########################################################
        ##                      tracker                         ##
        ##########################################################
        tracks, uncertainties, visibles = self.tracker.track_once(frame)

        ##########################################################
        ##              2D -> 3D conversion                     ##
        ##########################################################
        track_3d, track_valid = convert_pixel_to_world(
            pixel=tracks,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            fill_missing_depth=self.fill_missing_depth,
            window_size=self.fill_missing_depth_window_size,
            min_neighbors=self.fill_missing_depth_min_neighbors,
        )

        ##########################################################
        ##                      register                        ##
        ##########################################################

        result = FrontEndResult(frame_id=self.frame_id)
        result.tracks = tracks
        result.uncertainties = uncertainties
        result.visibles = visibles
        result.track_3d = track_3d
        result.track_valid = track_valid

        # To support potential external updates to visibles (mask removal)
        current_visibles = visibles.copy()

        for obj_id in range(min(self.num_obj, len(objects))):
            obj = objects[obj_id]

            # Skip if object not initialized properly or empty
            if len(track_table.obj2track_map) <= obj_id:
                print(f"[FrontEnd] Object {obj_id} not initialized properly.")
                continue

            # ----------- Registration Logic -----------
            idx_f2f, prev3d, curr3d_f2f, valid_stats = (
                self._extract_valid_idx_points_for_obj(
                    obj_id, track_table, track_3d, track_valid, current_visibles
                )
            )

            stats_f2f = {}
            T_rel = None
            mean_res_f2f = -1.0

            if prev3d.shape[0] >= 3 and curr3d_f2f.shape[0] >= 3:
                T_rel, stats_f2f = self.register.register(
                    prev3d, curr3d_f2f, sigma_tgt=uncertainties[idx_f2f]
                )

                mean_res_f2f = self._compute_mean_residual(stats_f2f)

            # Predict odom pose
            T_prev = obj.pose.copy()
            if T_rel is not None and 0 < mean_res_f2f < self.reg_residual_thres:
                T_odom = T_rel @ T_prev
            else:
                T_odom = T_prev
                obj.lost = True

            # update results for the object
            self._update_results(
                result,
                obj_id,
                T_odom,
                T_rel,
                idx_f2f,
                prev3d,
                curr3d_f2f,
                mean_res_f2f,
                stats_f2f,
                valid_stats,
            )

            # debug save
            if self.debug_level > 1:
                save_reg_pcd(
                    prev3d,
                    curr3d_f2f,
                    T_rel if T_rel is not None else np.eye(4),
                    self._reg_debug_dir,
                    f"obj_{obj_id}_frame_{self.frame_id}_f2f",
                    stats_f2f,
                )

            print(
                f"[FrontEnd] Frame {self.frame_id} - Object {obj_id} - Odom: \n{T_odom}"
            )

        # Optionally save cropped pcd
        if self.save_cropped_pcd:
            self._save_cropped_pcd(frame)

        self.frame_id += 1

        return result

    def _update_results(
        self,
        result: FrontEndResult,
        obj_id: int,
        pose: np.ndarray,
        rel_pose: np.ndarray,
        idx: np.ndarray,
        key_points: np.ndarray,
        curr3d: np.ndarray,
        mean_res: float,
        stats: dict,
        valid_stats: dict,
    ):
        result.obj_poses[obj_id] = pose
        result.rel_poses[obj_id] = rel_pose
        result.valid_indices[obj_id] = idx
        result.valid_key_points[obj_id] = key_points
        result.valid_curr_3d[obj_id] = curr3d
        result.reg_stats[obj_id] = stats
        result.mean_residuals[obj_id] = mean_res
        result.valid_stats[obj_id] = valid_stats

    def _compute_mean_residual(self, stats: dict):
        residuals = stats.get("residuals", np.array([]))
        inliers = stats.get("inliers", np.array([]))
        if len(residuals) > 0:
            if len(inliers) == len(residuals) and np.any(inliers):
                return float(np.mean(residuals[inliers]))
            else:
                return float(np.mean(residuals))
        return -1.0

    def _save_cropped_pcd(self, frame):
        num_objs = self.num_obj
        for obj_id in range(num_objs):
            try:
                mask = frame.mask[obj_id, 0]
                coords_yx = torch.nonzero(mask > 0, as_tuple=False)
                if coords_yx.numel() == 0:
                    continue
                pxl_xy = coords_yx[:, [1, 0]].cpu().numpy()
                world_pts, valid = convert_pixel_to_world(
                    pixel=pxl_xy,
                    depth_image=frame.depth,
                    cam_intrinsics=frame.intrinsics,
                    depth_factor=frame.depth_factor,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    fill_missing_depth=self.fill_missing_depth,
                    window_size=self.fill_missing_depth_window_size,
                    min_neighbors=self.fill_missing_depth_min_neighbors,
                )
                if world_pts.size == 0 or not np.any(valid):
                    continue
                world_pts = world_pts[valid]
                pxl_xy_valid = pxl_xy[valid]
                rgb_vals = frame.rgb[pxl_xy_valid[:, 1], pxl_xy_valid[:, 0]]
                save_pcd(
                    world_pts,
                    rgb_vals,
                    self.cropped_pcd_dir,
                    f"obj_{obj_id}_frame_{self.frame_id}",
                )
            except Exception:
                pass

    def _extract_valid_idx_points_for_obj(
        self,
        obj_id: int,
        track_table: PointTrackTable,
        curr_pts_3d: np.ndarray,
        curr_valid: np.ndarray,
        curr_visible: np.ndarray,
    ):
        """
        Returns:
            idx:     (M,) int indices where correspondence holds and both frames say valid&visible
            prev3d:  (M,3) 3D points from previous frame
            curr3d:  (M,3) 3D points from current frame
            valid_stats: dict with extra masks for debugging
        """
        obj_idx = track_table.obj2track_map[obj_id]

        n_prev = len(track_table.valid)
        n_curr = len(curr_valid)
        common_idx = obj_idx[(obj_idx < n_prev) & (obj_idx < n_curr)]

        # Ensure boolean masks
        curr_vis_arr = np.asarray(curr_visible, dtype=bool)
        curr_val_arr = np.asarray(curr_valid, dtype=bool)
        track_vis_arr = np.asarray(track_table.visible, dtype=bool)
        track_val_arr = np.asarray(track_table.valid, dtype=bool)

        both_mask = (
            curr_vis_arr[common_idx]
            & curr_val_arr[common_idx]
            & track_vis_arr[common_idx]
            & track_val_arr[common_idx]
        )

        idx = common_idx[both_mask]
        prev3d = track_table.track_3d[idx].copy()
        curr3d = curr_pts_3d[idx].copy()

        valid_stats = {
            "extract_vis_obj_mask": curr_vis_arr[common_idx],
            "extract_val_obj_mask": curr_val_arr[common_idx],
            "extract_valid_kp_mask": track_val_arr[common_idx]
            & track_vis_arr[common_idx],
            "extract_obj_idx": common_idx,
            # Not used in F2F but kept for compatibility
            "extract_uncer_obj_mask": np.empty(0),
            "extract_uncertainty_thres": 0.0,
            "extract_inside_mask": np.empty(0),
            "extract_finite_xy": np.empty(0),
        }

        return idx, prev3d, curr3d, valid_stats

    def _extract_valid_key_points(
        self,
        obj,
        obj_idx,
        cur_pts_3d,
        cur_visible,
        cur_valid,
        cur_uncertainties,
        uncertainty_thres=0.3,
    ):
        obj_idx = np.asarray(obj_idx)
        valid_kp_bool = np.asarray(obj.valid, dtype=bool)

        vis_obj = np.asarray(cur_visible, dtype=bool)[obj_idx]
        val_obj = np.asarray(cur_valid, dtype=bool)[obj_idx]
        uncer_obj = (
            np.asarray(cur_uncertainties, dtype=float)[obj_idx] < uncertainty_thres
        )
        both_mask = vis_obj & val_obj & valid_kp_bool & uncer_obj

        idx = obj_idx[both_mask]
        key_points = obj.key_points[both_mask].copy()
        curr3d = cur_pts_3d[idx].copy()

        valid_stats = {
            "extract_vis_obj_mask": vis_obj,
            "extract_val_obj_mask": val_obj,
            "extract_uncer_obj_mask": uncer_obj,
            "extract_valid_kp_mask": valid_kp_bool,
            "extract_uncertainty_thres": uncertainty_thres,
            "extract_obj_idx": obj_idx,
            "extract_inside_mask": np.empty(0),
            "extract_finite_xy": np.empty(0),
        }

        return idx, key_points, curr3d, valid_stats

    def _extract_valid_key_points_mask_remove(
        self,
        obj,
        obj_idx,
        cur_pts_2d,
        cur_pts_3d,
        cur_visible,
        cur_valid,
        cur_uncertainties,
        frame_mask_gpu,
        uncertainty_thres=0.3,
    ):
        # --- normalize mask to boolean 2D on GPU ---
        if frame_mask_gpu.ndim == 3 and frame_mask_gpu.shape[0] == 1:
            mask2d = frame_mask_gpu[0]
        elif frame_mask_gpu.ndim == 2:
            mask2d = frame_mask_gpu
        else:
            raise ValueError("frame_mask_gpu must be (H,W) or (1,H,W)")

        mask_bool = (mask2d > 0) if mask2d.dtype != torch.bool else mask2d
        H, W = mask_bool.shape

        obj_idx = np.asarray(obj_idx, dtype=np.int64)
        cur_pts_2d = np.asarray(cur_pts_2d, dtype=np.float32)
        cur_pts_3d = np.asarray(cur_pts_3d)
        cur_visible = np.asarray(cur_visible, dtype=bool)
        cur_valid = np.asarray(cur_valid, dtype=bool)
        cur_uncertainties = np.asarray(cur_uncertainties, dtype=np.float32)

        val_obj = cur_valid[obj_idx]
        uncer_obj = cur_uncertainties[obj_idx] < float(uncertainty_thres)
        valid_kp_obj = np.asarray(
            getattr(obj, "valid", np.ones(len(obj_idx), dtype=bool)), dtype=bool
        )

        pts2d_obj = cur_pts_2d[obj_idx]
        finite_xy = np.isfinite(pts2d_obj).all(axis=1)

        x = np.rint(pts2d_obj[:, 0]).astype(np.int64)
        y = np.rint(pts2d_obj[:, 1]).astype(np.int64)
        np.clip(x, 0, W - 1, out=x)
        np.clip(y, 0, H - 1, out=y)

        dev = mask_bool.device
        x_t = torch.from_numpy(x).to(device=dev, dtype=torch.long)
        y_t = torch.from_numpy(y).to(device=dev, dtype=torch.long)
        inside_mask_np = mask_bool[y_t, x_t].detach().cpu().numpy()

        outside_or_bad = (~inside_mask_np) | (~finite_xy)
        if outside_or_bad.any():
            cur_visible[obj_idx[outside_or_bad]] = False

        vis_obj = cur_visible[obj_idx]

        both_mask = (
            vis_obj & val_obj & uncer_obj & valid_kp_obj & finite_xy & inside_mask_np
        )

        idx = obj_idx[both_mask]
        key_points = obj.key_points[both_mask].copy()
        curr3d = cur_pts_3d[idx].copy()

        valid_stats = {
            "extract_vis_obj_mask": vis_obj,
            "extract_val_obj_mask": val_obj,
            "extract_uncer_obj_mask": uncer_obj,
            "extract_valid_kp_mask": valid_kp_obj,
            "extract_uncertainty_thres": uncertainty_thres,
            "extract_obj_idx": obj_idx,
            "extract_inside_mask": inside_mask_np,
            "extract_finite_xy": finite_xy,
        }

        return idx, key_points, curr3d, cur_visible, valid_stats
