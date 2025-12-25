import time
import os
import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional, Dict, Any

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
        print(f"[FrontEnd] Register: {self.register}")
        # -------------- registration params --------------
        self.reg_uncer_thres = self.cfg.register.params.get("uncer_thres", 0.3)
        self.reg_residual_thres = self.cfg.register.params.get("residual_thres", 0.0007)
        self._reg_remove_outside_mask = self.cfg.register.params.get(
            "remove_outside_mask", False
        )

        # dense registration for recovering from bad registration
        self.use_dense_registration = self.pipeline_cfg.get(
            "use_dense_registration", False
        )

        if self.use_dense_registration:
            self.dense_register = build_from_cfg(cfg.dense_register, REGISTER)
        else:
            self.dense_register = None

        # -------------- dense registration recovery params --------------
        self.residual_jump_threshold = self.pipeline_cfg.get(
            "residual_jump_threshold", 0.02
        )  # threshold for sudden residual jump
        self.inlier_drop_ratio = self.pipeline_cfg.get(
            "inlier_drop_ratio", 0.3
        )  # ratio threshold for sudden inlier drop (e.g., 0.5 means 50% drop)

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

        # Track previous frame and registration stats for recovery
        self.prev_frame = None
        self.prev_residuals = {}  # obj_id -> previous mean residual
        self.prev_inlier_counts = {}  # obj_id -> previous inlier count

        self.frame_id = 0

    def initialize(self, frame):
        """Initialize segmentation and tracker with the first frame."""
        self.segmenter.initialize(frame.rgb, mask=frame.mask)

        if self.use_segmenter:
            # get number of objects
            # self.num_obj = np.sum(np.asarray(self.segmenter.input_labels) == 1)
            # get segmentation mask
            obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
            frame.mask = mask_logits

        self.tracker.initialize(frame)
        self.frame_id = 0
        # Store first frame as previous frame for next step
        self.prev_frame = frame

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
        # Timing dictionary for frontend modules
        fe_timings = {}

        ##########################################################
        ##                     segmenter                        ##
        ##########################################################
        t_start = time.time()
        if self.use_segmenter:
            _, mask_logits = self.segmenter.segment(frame.rgb)
            frame.mask = mask_logits  # mask is a torch tensor on gpu
        fe_timings["segmenter"] = time.time() - t_start

        ##########################################################
        ##                      tracker                         ##
        ##########################################################
        t_start = time.time()
        tracks, uncertainties, visibles = self.tracker.track_once(frame)
        fe_timings["tracker"] = time.time() - t_start

        if frame.id == 261:
            print("tracks: ", tracks)

        ##########################################################
        ##              2D -> 3D conversion                     ##
        ##########################################################
        t_start = time.time()
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
        fe_timings["2d_to_3d"] = time.time() - t_start

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

        # Timing for registration per object
        fe_timings["registration"] = 0.0
        fe_timings["dense_recovery"] = 0.0
        fe_timings["extract_valid"] = 0.0

        for obj_id in range(min(self.num_obj, len(objects))):
            obj = objects[obj_id]

            # Skip if object not initialized properly or empty
            if len(track_table.obj2track_map) <= obj_id:
                print(f"[FrontEnd] Object {obj_id} not initialized properly.")
                continue

            # skip frame to frame registration if the object is marked as lost
            # (recovery manager should take care of this by performing frame to map registration)
            if obj.lost:
                print(f"[FrontEnd] Object {obj_id} is lost, skip registration.")
                continue

            # ----------- Registration Logic -----------
            t_start = time.time()
            idx_f2f, prev3d, curr3d, valid_stats = (
                self._extract_valid_idx_points_for_obj(
                    obj_id, track_table, track_3d, track_valid, current_visibles
                )
            )
            fe_timings["extract_valid"] += time.time() - t_start

            stats_f2f = {}
            T_rel = None
            mean_res_f2f = -1.0

            # solve frame to frame registration
            if prev3d.shape[0] >= 3 and curr3d.shape[0] >= 3:
                t_start = time.time()
                T_rel, stats_f2f = self.register.register(
                    prev3d, curr3d, sigma_tgt=uncertainties[idx_f2f]
                )
                fe_timings["registration"] += time.time() - t_start

                mean_res_f2f = self._compute_mean_residual(stats_f2f)

            T_rel_dense = None
            # Store before dense recovery info
            pose_before_dense = None
            rel_before_dense = None
            stats_before_dense = None

            # Initialize dense recovery flags
            result.dense_recovery_triggered[obj_id] = False

            # Check for sudden jump/drop and apply dense registration if needed
            if (
                self.use_dense_registration
                and self.dense_register is not None
                and self.prev_frame is not None
                # and T_rel is not None
                # and mean_res_f2f >= 0
            ):
                should_recover, reason = self._check_registration_quality(
                    obj_id, mean_res_f2f, stats_f2f
                )

                # if frame.id > 940 and frame.id < 1000:
                #     should_recover = True
                #     self.dense_register.debug_level = 2
                #     reason = "test"

                if should_recover:
                    # Capture state before dense recovery
                    pose_before_dense = obj.pose.copy()
                    rel_before_dense = T_rel.copy() if T_rel is not None else np.eye(4)
                    stats_before_dense = stats_f2f.copy() if stats_f2f else {}

                    print(
                        f"[FrontEnd] Frame {self.frame_id} - Object {obj_id} - "
                        f"Dense recovery triggered: {reason}"
                    )

                    # perform dense recovery
                    t_start = time.time()
                    T_rel_dense, stats_dense_after = self._apply_dense_recovery(
                        obj_id, self.prev_frame, frame, T_rel
                    )
                    fe_timings["dense_recovery"] += time.time() - t_start

                    # Store dense recovery info in result
                    result.dense_recovery_triggered[obj_id] = True
                    result.dense_recovery_pose_before[obj_id] = pose_before_dense
                    result.dense_recovery_rel_before[obj_id] = rel_before_dense
                    result.dense_recovery_stats_before[obj_id] = stats_before_dense

                    if T_rel_dense is not None:
                        result.dense_recovery_rel_after[obj_id] = T_rel_dense
                        result.dense_recovery_stats_after[obj_id] = (
                            stats_dense_after if stats_dense_after else {}
                        )
                    else:
                        result.dense_recovery_rel_after[obj_id] = rel_before_dense
                        result.dense_recovery_stats_after[obj_id] = stats_before_dense

            # Predict odom pose
            T_prev = obj.pose.copy()
            if T_rel_dense is not None:
                T_odom = T_rel_dense @ T_prev
                print(
                    f"[FrontEnd] Frame {self.frame_id} - Object {obj_id} - "
                    "Dense recovery successful"
                )
            ## TODO: maybe more than simple residual threshold?
            elif T_rel is not None and 0 < mean_res_f2f < self.reg_residual_thres:
                T_odom = T_rel @ T_prev
            else:
                T_odom = T_prev
                obj.lost = True

            # Store after pose if dense recovery was triggered
            if result.dense_recovery_triggered.get(obj_id, False):
                result.dense_recovery_pose_after[obj_id] = T_odom.copy()

            # update results for the object
            self._update_results(
                result,
                obj_id,
                T_odom,
                T_rel,
                idx_f2f,
                prev3d,
                curr3d,
                mean_res_f2f,
                stats_f2f,
                valid_stats,
            )

            # debug save
            if self.debug_level > 1:
                save_reg_pcd(
                    prev3d,
                    curr3d,
                    T_rel if T_rel is not None else np.eye(4),
                    self._reg_debug_dir,
                    f"obj_{obj_id}_frame_{self.frame_id}_f2f",
                    stats_f2f,
                )

            print(
                f"[FrontEnd] Frame {self.frame_id} - Object {obj_id} - Odom: \n{T_odom}"
            )

            # Update previous stats for next frame
            if mean_res_f2f >= 0:
                self.prev_residuals[obj_id] = mean_res_f2f
                inliers = stats_f2f.get("inliers", np.array([]))
                if len(inliers) > 0:
                    self.prev_inlier_counts[obj_id] = int(np.sum(inliers))
                else:
                    # If no inliers array, use number of points as fallback
                    self.prev_inlier_counts[obj_id] = prev3d.shape[0]

        # Optionally save cropped pcd
        if self.save_cropped_pcd:
            t_start = time.time()
            self._save_cropped_pcd(frame)
            fe_timings["save_cropped_pcd"] = time.time() - t_start

        # Update previous frame for next step
        self.prev_frame = frame

        self.frame_id += 1

        # Print timing summary
        total_fe_time = sum(fe_timings.values())
        timing_str = " | ".join(
            [f"{k}: {v*1000:.2f}ms" for k, v in fe_timings.items() if v > 0]
        )
        print(
            f"[FrontEnd] Frame {self.frame_id-1} timing: {timing_str} | Total: {total_fe_time*1000:.2f}ms"
        )

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

    def _check_registration_quality(self, obj_id, mean_residual, stats):
        """
        Check for sudden jump in residual or sudden drop in inlier count.

        Args:
            obj_id: Object ID
            mean_residual: Current mean residual
            stats: Registration statistics dictionary

        Returns:
            tuple: (should_recover: bool, reason: str)
        """
        # Check if we have previous stats for this object
        if obj_id not in self.prev_residuals:
            return False, "no_previous_stats"

        prev_residual = self.prev_residuals[obj_id]
        prev_inlier_count = self.prev_inlier_counts.get(obj_id, 0)

        # Get current inlier count
        inliers = stats.get("inliers", np.array([]))
        if len(inliers) > 0:
            curr_inlier_count = int(np.sum(inliers))
        else:
            # Fallback: use number of residuals as proxy
            residuals = stats.get("residuals", np.array([]))
            curr_inlier_count = len(residuals) if len(residuals) > 0 else 0

        # Check for sudden residual jump (above threshold)
        # Only check if previous residual was below threshold (good registration)
        if prev_residual < self.reg_residual_thres:
            residual_jump = mean_residual - prev_residual
            if residual_jump > self.residual_jump_threshold:
                return True, f"residual_jump_{residual_jump:.4f}"

        if curr_inlier_count < 5:
            return True, f"insufficient_inliers_{curr_inlier_count}"

        if mean_residual > 0.0007:
            return True, f"high_residual_{mean_residual:.4f}"

        # Check for sudden inlier drop
        # Only check if we had a reasonable number of inliers before
        if prev_inlier_count > 0 and curr_inlier_count > 0:
            inlier_drop_ratio = 1.0 - (curr_inlier_count / prev_inlier_count)
            if inlier_drop_ratio > self.inlier_drop_ratio:
                return True, f"inlier_drop_{inlier_drop_ratio:.2%}"

        return False, "ok"

    def _extract_cropped_point_cloud(self, frame, obj_id: int) -> np.ndarray:
        """
        Extract cropped point cloud from frame using mask for the specified object.

        Args:
            frame: Frame object
            obj_id: Object ID

        Returns:
            point_cloud: (N, 3) numpy array of 3D points, or empty array if extraction fails
        """
        try:
            if frame.mask is None:
                return np.empty((0, 3), dtype=np.float32)

            mask = frame.mask[obj_id, 0]
            coords_yx = torch.nonzero(mask > 0, as_tuple=False)
            if coords_yx.numel() == 0:
                return np.empty((0, 3), dtype=np.float32)

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
                return np.empty((0, 3), dtype=np.float32)

            return world_pts[valid]

        except Exception as e:
            print(
                f"[FrontEnd] Error extracting cropped point cloud for obj {obj_id}: {e}"
            )
            return np.empty((0, 3), dtype=np.float32)

    def _apply_dense_recovery(
        self,
        obj_id: int,
        prev_frame,
        curr_frame,
        init_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Apply dense registration using cropped point clouds from previous and current frames.

        Args:
            obj_id: Object ID
            prev_frame: Previous frame object
            curr_frame: Current frame object
            init_pose: Initial transformation estimate (4x4)

        Returns:
            tuple: (transformation_matrix, stats_dict)
                - transformation_matrix: 4x4 transformation matrix, or None if recovery fails
                - stats_dict: Dictionary with registration statistics
        """
        if self.dense_register is None:
            return None, {}

        if init_pose is None:
            init_pose = np.eye(4)

        # Extract cropped point clouds
        prev_pcd = self._extract_cropped_point_cloud(prev_frame, obj_id)
        curr_pcd = self._extract_cropped_point_cloud(curr_frame, obj_id)

        # Check if we have enough points
        min_points = 100  # Minimum points for dense registration
        if prev_pcd.shape[0] < min_points or curr_pcd.shape[0] < min_points:
            print(
                f"[FrontEnd] Dense recovery skipped: insufficient points "
                f"(prev: {prev_pcd.shape[0]}, curr: {curr_pcd.shape[0]})"
            )
            return None, {}

        # Apply dense registration
        try:
            T_dense, stats_dense = self.dense_register.register(
                source_pcd=prev_pcd,
                target_pcd=curr_pcd,
                init_pose=init_pose,
            )

            if T_dense is not None:
                # Return transformation and stats
                return T_dense, stats_dense if stats_dense else {}

            return None, {}

        except Exception as e:
            print(f"[FrontEnd] Dense recovery failed: {e}")
            return None, {}
