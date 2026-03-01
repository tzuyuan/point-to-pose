import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import REGISTER, TRACKER, SEGMENTER
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.modules.segmenter.dummy_segmenter import DummySegmenter
from point2pose.io.outputs.point_cloud_io import save_reg_pcd, save_pcd
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.lie import se3_to_vec, log_SE3, exp_se3, vec_to_se3, log_SO3
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
        self.reg_mask_border_margin_px = int(
            self.cfg.register.params.get("mask_border_margin_px", 0)
        )
        self._reg_remove_outside_mask = self.cfg.register.params.get(
            "remove_outside_mask", False
        )
        self.max_rel_rotation_deg = float(
            self.pipeline_cfg.get("max_rel_rotation_deg", 20.0)
        )
        self.max_rel_translation = float(self.pipeline_cfg.get("max_rel_translation", 0.05))
        self.pose_jump_guard_enable = bool(
            self.pipeline_cfg.get("pose_jump_guard_enable", True)
        )
        self.pose_jump_guard_trans_thres = float(
            self.pipeline_cfg.get("pose_jump_guard_trans_thres", self.max_rel_translation)
        )
        self.pose_jump_guard_rot_deg_thres = float(
            self.pipeline_cfg.get(
                "pose_jump_guard_rot_deg_thres", self.max_rel_rotation_deg
            )
        )
        reg_min_inliers = int(self.cfg.register.params.get("min_inliers", 5))
        self.pose_jump_guard_min_inliers = int(
            self.pipeline_cfg.get("pose_jump_guard_min_inliers", max(8, reg_min_inliers))
        )
        self.pose_jump_guard_min_inlier_ratio = float(
            self.pipeline_cfg.get("pose_jump_guard_min_inlier_ratio", 0.2)
        )
        self.pose_jump_guard_single_cluster_min_inliers = int(
            self.pipeline_cfg.get(
                "pose_jump_guard_single_cluster_min_inliers",
                max(self.pose_jump_guard_min_inliers, reg_min_inliers + 2),
            )
        )

        # -------------- registration mode --------------
        # "f2f" for frame-to-frame (relative pose), "f2m" for frame-to-map (absolute pose)
        self.frame_reg_mode = self.pipeline_cfg.get("frame_reg_mode", "f2f")
        if self.frame_reg_mode not in ["f2f", "f2m"]:
            raise ValueError(
                f"Invalid frame_reg_mode: {self.frame_reg_mode}. Must be 'f2f' or 'f2m'"
            )
        print(f"[FrontEnd] Registration mode: {self.frame_reg_mode}")

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
        self.dense_recovery_min_points = int(
            self.pipeline_cfg.get("dense_recovery_min_points", 100)
        )
        self.dense_recovery_sdf_percentile = float(
            self.pipeline_cfg.get("dense_recovery_sdf_percentile", 70.0)
        )
        self.dense_recovery_sdf_max_points = int(
            self.pipeline_cfg.get("dense_recovery_sdf_max_points", 2000)
        )
        self.dense_recovery_sdf_rot_bound = float(
            self.pipeline_cfg.get("dense_recovery_sdf_rot_bound", 0.15)
        )
        self.dense_recovery_sdf_trans_bound = float(
            self.pipeline_cfg.get("dense_recovery_sdf_trans_bound", 0.03)
        )
        self.dense_recovery_sdf_max_iter = int(
            self.pipeline_cfg.get("dense_recovery_sdf_max_iter", 25)
        )
        self.dense_recovery_sdf_min_improve = float(
            self.pipeline_cfg.get("dense_recovery_sdf_min_improve", 1e-4)
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

        # Track previous frame and registration stats for recovery
        self.prev_frame = None
        self.prev_residuals = {}  # obj_id -> previous mean residual
        self.prev_inlier_counts = {}  # obj_id -> previous inlier count
        self.pose_histories = {}  # obj_id -> list of frontend poses
        self.vel_hist_len = int(self.pipeline_cfg.get("vel_hist_len", 5))
        self.vel_rot_change_thres_deg = float(
            self.pipeline_cfg.get("vel_rot_change_thres_deg", 15.0)
        )
        self.vel_trans_change_thres = float(
            self.pipeline_cfg.get("vel_trans_change_thres", 0.03)
        )

        # Fallback: when confirmed valid map points are too few, temporarily allow
        # tentative points (obj.valid=False) that pass current frame quality gates.
        fallback_min_cfg = int(
            self.pipeline_cfg.get("tentative_fallback_min_valid_points", 4)
        )
        reg_min_inliers = int(self.cfg.register.params.get("min_inliers", 3))
        reg_sample_size = int(self.cfg.register.params.get("sample_size", 3))
        self.tentative_fallback_min_valid_points = max(
            fallback_min_cfg, reg_min_inliers, reg_sample_size, 3
        )
        if self.tentative_fallback_min_valid_points > fallback_min_cfg:
            print(
                "[FrontEnd] tentative_fallback_min_valid_points raised from "
                f"{fallback_min_cfg} to {self.tentative_fallback_min_valid_points} "
                f"(register min_inliers={reg_min_inliers}, sample_size={reg_sample_size})."
            )

        fallback_max_cfg = int(
            self.pipeline_cfg.get("tentative_fallback_max_points", 20)
        )
        self.tentative_fallback_max_points = max(
            fallback_max_cfg, self.tentative_fallback_min_valid_points
        )
        if self.tentative_fallback_max_points > fallback_max_cfg:
            print(
                "[FrontEnd] tentative_fallback_max_points raised from "
                f"{fallback_max_cfg} to {self.tentative_fallback_max_points} "
                "to satisfy the fallback minimum."
            )
        self.tentative_fallback_recent_frames = int(
            self.pipeline_cfg.get("tentative_fallback_recent_frames", 20)
        )
        self.tentative_fallback_uncer_scale = float(
            self.pipeline_cfg.get("tentative_fallback_uncer_scale", 1.0)
        )
        self.tentative_fallback_allow_stale_when_empty = bool(
            self.pipeline_cfg.get("tentative_fallback_allow_stale_when_empty", True)
        )

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

        print(f"number of tracks: {len(tracks)}")

        # if frame.id == 261:
        #     print("tracks: ", tracks)

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
            compute_depth_uncertainty=False,
            sigma_min=0.002,
            sigma_max=0.05,
            sigma_base_a=0.003,
            sigma_base_b=0.0,
            edge_alpha=5.0,
        )
        fe_timings["2d_to_3d"] = time.time() - t_start

        ##########################################################
        ##                      register                        ##
        ##########################################################

        result = FrontEndResult(frame_id=frame.id)
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

            # skip registration if the object is marked as lost
            # (recovery manager should take care of this by performing frame to map registration)

            # ----------- Registration Logic -----------
            stats_reg = {
                "num_inliers": 0,
            }
            T_rel = None
            T_c2w_est = None
            mean_res = -1.0
            idx = np.array([], dtype=int)
            key_points = np.empty((0, 3))
            prev3d = np.empty((0, 3))
            correspond_curr3d = np.empty((0, 3))
            valid_stats = {}

            if self.frame_reg_mode == "f2f":
                # Frame-to-frame registration (relative pose)
                t_start = time.time()
                if obj.lost:
                    print(f"[FrontEnd] Object {obj_id} is lost, skip f2f registration.")
                    continue

                idx, prev3d, correspond_curr3d, valid_stats = (
                    self._extract_valid_idx_points_for_obj(
                        obj_id, track_table, track_3d, track_valid, current_visibles
                    )
                )
                fe_timings["extract_valid"] += time.time() - t_start

                prev_pose = obj.pose.copy()
                # solve frame to frame registration
                if prev3d.shape[0] >= 3 and correspond_curr3d.shape[0] >= 3:
                    t_start = time.time()
                    T_rel, stats_reg = self.register.register(
                        prev3d,
                        correspond_curr3d,
                        sigma_tgt=uncertainties[idx],
                        prev_T=prev_pose,
                        prev_frame=self.prev_frame,
                        cur_frame=frame,
                        obj_id=obj_id,
                        obj=obj,
                        mode="f2f",
                    )
                    fe_timings["registration"] += time.time() - t_start
                    mean_res = self._compute_mean_residual(stats_reg)

                stats_reg["correspond_curr3d"] = correspond_curr3d
                stats_reg["valid_idx"] = idx
                self._attach_sampling_hint_stats(stats_reg, valid_stats)

            elif self.frame_reg_mode == "f2m":
                # Frame-to-map registration (absolute pose)
                t_start = time.time()
                # idx, key_points, correspond_curr3d, valid_stats = self._extract_valid_key_points(
                #     obj,
                #     track_table.obj2track_map[obj_id],
                #     track_3d,
                #     current_visibles,
                #     track_valid,
                #     uncertainties,
                #     uncertainty_thres=self.reg_uncer_thres,
                # )

                idx, key_points, correspond_curr3d, cur_visible, valid_stats = (
                    self._extract_valid_key_points_mask_remove(
                        obj,
                        track_table.obj2track_map[obj_id],
                        tracks,
                        track_3d,
                        current_visibles,
                        track_valid,
                        uncertainties,
                        frame.mask[obj_id, 0],
                        uncertainty_thres=self.reg_uncer_thres,
                        mask_border_margin_px=self.reg_mask_border_margin_px,
                        frame_id=frame.id,
                    )
                )

                # idx, key_points, correspond_curr3d, cur_visible, valid_stats = (
                #     self._extract_valid_key_points_mask_remove_no_depth_check(
                #         obj,
                #         track_table.obj2track_map[obj_id],
                #         tracks,
                #         track_3d,
                #         current_visibles,
                #         track_valid,
                #         uncertainties,
                #         frame.mask[obj_id, 0],
                #         uncertainty_thres=self.reg_uncer_thres,
                #     )
                # )

                fe_timings["extract_valid"] += time.time() - t_start

                prev_pose = obj.pose.copy()
                # solve frame to map registration
                if key_points.shape[0] >= 3 and correspond_curr3d.shape[0] >= 3:
                    t_start = time.time()
                    if frame.id == 100:
                        print("frame id 100")

                    T_c2w_est, stats_reg = self.register.register(
                        src_pcd=key_points,
                        tgt_pcd=correspond_curr3d,
                        sigma_tgt=uncertainties[idx],
                        init_pose=prev_pose,
                        prev_T=prev_pose,
                        prev_frame=self.prev_frame,
                        cur_frame=frame,
                        obj=obj,
                        obj_id=obj_id,
                        mode="f2m",
                        # img_pts=tracks[idx],
                    )
                    fe_timings["registration"] += time.time() - t_start
                    mean_res = self._compute_mean_residual(stats_reg)

                    T_c2w_est, jump_rejected, jump_info = self._apply_pose_jump_guard(
                        T_candidate=T_c2w_est,
                        T_prev=prev_pose,
                        stats=stats_reg,
                    )
                    stats_reg["pose_jump_guard_info"] = jump_info
                    if jump_rejected:
                        obj.lost = True
                    elif 0 < mean_res < self.reg_residual_thres:
                        obj.lost = False
                else:
                    print(
                        f"[FrontEnd] Frame {frame.id} - Object {obj_id} - Not enough points for registration."
                    )
                    T_c2w_est = prev_pose
                    mean_res = -1.0
                    stats_reg = {"inliers": np.array([]), "residuals": np.array([])}
                    obj.lost = True

                stats_reg["correspond_curr3d"] = correspond_curr3d
                stats_reg["valid_idx"] = idx
                self._attach_sampling_hint_stats(stats_reg, valid_stats)

            T_rel_dense = None
            T_c2w_dense = None
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
                and (
                    (self.frame_reg_mode == "f2f" and self.prev_frame is not None)
                    or self.frame_reg_mode == "f2m"
                )
            ):
                should_recover, reason = self._check_registration_quality(
                    obj_id, mean_res, stats_reg
                )
                # should_recover = True
                # reason = "test"
                # if frame.id > 940 and frame.id < 1000:
                #     should_recover = True
                #     self.dense_register.debug_level = 2
                #     reason = "test"

                if should_recover:
                    # Capture state before dense recovery
                    pose_before_dense = obj.pose.copy()
                    if self.frame_reg_mode == "f2f":
                        rel_before_dense = (
                            T_rel.copy() if T_rel is not None else np.eye(4)
                        )
                    else:
                        rel_before_dense = (
                            T_c2w_est.copy()
                            if T_c2w_est is not None
                            else obj.pose.copy()
                        )
                    stats_before_dense = stats_reg.copy() if stats_reg else {}

                    print(
                        f"[FrontEnd] Frame {frame.id} - Object {obj_id} - "
                        f"Dense recovery triggered: {reason}"
                    )

                    # perform dense recovery
                    t_start = time.time()
                    if self.frame_reg_mode == "f2f":
                        T_rel_dense, stats_dense_after = self._apply_dense_recovery(
                            obj_id, self.prev_frame, frame, T_rel
                        )
                    else:
                        init_pose = T_c2w_est if T_c2w_est is not None else obj.pose
                        T_c2w_dense, stats_dense_after = (
                            self._apply_dense_recovery_f2m_sdf(
                                obj_id=obj_id,
                                curr_frame=frame,
                                obj=obj,
                                init_pose=init_pose,
                            )
                        )
                    fe_timings["dense_recovery"] += time.time() - t_start

                    # Store dense recovery info in result
                    result.dense_recovery_triggered[obj_id] = True
                    result.dense_recovery_pose_before[obj_id] = pose_before_dense
                    result.dense_recovery_rel_before[obj_id] = rel_before_dense
                    result.dense_recovery_stats_before[obj_id] = stats_before_dense

                    dense_after_pose = (
                        T_rel_dense if self.frame_reg_mode == "f2f" else T_c2w_dense
                    )
                    if dense_after_pose is not None:
                        result.dense_recovery_rel_after[obj_id] = dense_after_pose
                        result.dense_recovery_stats_after[obj_id] = (
                            stats_dense_after if stats_dense_after else {}
                        )
                    else:
                        result.dense_recovery_rel_after[obj_id] = rel_before_dense
                        result.dense_recovery_stats_after[obj_id] = stats_before_dense

            # Predict odom pose
            T_prev = obj.pose.copy()
            if self.frame_reg_mode == "f2f" and T_rel_dense is not None:
                # Dense recovery (f2f mode)
                T_odom = T_rel_dense @ T_prev
                print(
                    f"[FrontEnd] Frame {frame.id} - Object {obj_id} - "
                    "Dense recovery successful"
                )
            elif self.frame_reg_mode == "f2m" and T_c2w_dense is not None:
                # Dense recovery (f2m mode)
                T_odom = T_c2w_dense
                print(
                    f"[FrontEnd] Frame {frame.id} - Object {obj_id} - "
                    "Dense recovery successful"
                )
            elif self.frame_reg_mode == "f2f":
                # Frame-to-frame: compose relative pose
                if T_rel is not None and 0 < mean_res < self.reg_residual_thres:
                    T_odom = T_rel @ T_prev
                else:
                    T_odom = T_prev
                    obj.lost = True
            elif self.frame_reg_mode == "f2m":
                # self._seed_pose_history(obj_id, T_prev)
                T_odom = T_c2w_est

                if self.use_dense_registration and self.dense_register is not None:
                    # icp refinement
                    T_icp, stats_dense = self.dense_register.register(
                        source_pcd=key_points,
                        target_pcd=correspond_curr3d,
                        init_pose=T_c2w_est,
                    )

                    T_odom = T_icp
                # T_odom, used_vel_fallback = self._apply_velocity_fallback(
                #     obj_id=obj_id,
                #     T_prev=T_prev,
                #     T_candidate=T_odom,
                # )
                # if used_vel_fallback:
                #     print(
                #         f"[FrontEnd] Frame {frame.id} - Object {obj_id} - "
                #         "Velocity fallback used"
                #     )

                ## test output T_c2w_est by comparing relative pose to previous frame
                # T_rel = T_c2w_est @ inverse_SE3(T_prev)

                # R_rel = T_rel[:3, :3]
                # t_rel = T_rel[:3, 3]
                # theta = np.linalg.norm(log_SO3(R_rel))  # radians
                # d = np.linalg.norm(t_rel)  # meters

            #     dt, ddeg = self._se3_delta(T_c2w_est, T_prev)

            #     if ddeg > 15 or dt > 0.03:
            #         #     # idx, prev3d, correspond_curr3d, _ = (
            #         #     #     self._extract_valid_idx_points_for_obj(
            #         #     #         obj_id, track_table, track_3d, track_valid, current_visibles
            #         #     #     )
            #         #     # )

            #         #     # prev_pose = obj.pose.copy()
            #         #     # # solve frame to frame registration
            #         #     # if prev3d.shape[0] >= 3 and correspond_curr3d.shape[0] >= 3:
            #         #     #     t_start = time.time()
            #         #     #     T_rel, _ = self.register.register(
            #         #     #         prev3d,
            #         #     #         correspond_curr3d,
            #         #     #         sigma_tgt=uncertainties[idx],
            #         #     #         prev_T=prev_pose,
            #         #     #         prev_frame=self.prev_frame,
            #         #     #         cur_frame=frame,
            #         #     #         obj_id=obj_id,
            #         #     #         obj=obj,
            #         #     #         mode="f2f",
            #         #     #     )

            #         #     # stats_reg["correspond_curr3d"] = correspond_curr3d
            #         #     # stats_reg["valid_idx"] = idx

            #         #     # T_odom = T_rel @ T_prev
            #         T_odom = T_prev
            #     else:
            #         T_odom = T_c2w_est

            #     # Frame-to-map: use absolute pose directly
            #     # if T_c2w_est is not None and 0 < mean_res < self.reg_residual_thres:
            #     #     T_odom = T_c2w_est
            #     # else:
            #     #     T_odom = T_prev
            #     #     obj.lost = True

            # else:
            #     T_odom = T_prev
            #     obj.lost = True

            # Store after pose if dense recovery was triggered
            if result.dense_recovery_triggered.get(obj_id, False):
                result.dense_recovery_pose_after[obj_id] = T_odom.copy()

            # update results for the object
            # For f2f: rel_pose is T_rel, key_points is prev3d
            # For f2m: rel_pose is None (absolute pose), key_points is key_points
            rel_pose_for_result = T_rel if self.frame_reg_mode == "f2f" else None
            key_points_for_result = (
                prev3d if self.frame_reg_mode == "f2f" else key_points
            )

            self._update_results(
                result,
                obj_id,
                T_odom,
                rel_pose_for_result,
                idx,
                key_points_for_result,
                correspond_curr3d,
                mean_res,
                stats_reg,
                valid_stats,
            )
            self._update_pose_history(obj_id, T_odom)

            # debug save
            if self.debug_level > 1:
                if self.frame_reg_mode == "f2f":
                    save_reg_pcd(
                        prev3d,
                        correspond_curr3d,
                        T_rel if T_rel is not None else np.eye(4),
                        self._reg_debug_dir,
                        f"obj_{obj_id}_frame_{frame.id}_f2f",
                        stats_reg,
                    )
                elif self.frame_reg_mode == "f2m":
                    save_reg_pcd(
                        key_points,
                        correspond_curr3d,
                        T_c2w_est if T_c2w_est is not None else np.eye(4),
                        self._reg_debug_dir,
                        f"obj_{obj_id}_frame_{frame.id}_f2m",
                        stats_reg,
                    )

            print(f"[FrontEnd] Frame {frame.id} - Object {obj_id} - Odom: \n{T_odom}")

            # Update previous stats for next frame
            if mean_res >= 0:
                self.prev_residuals[obj_id] = mean_res
                inliers = stats_reg.get("inliers", np.array([]))
                if len(inliers) > 0:
                    self.prev_inlier_counts[obj_id] = int(np.sum(inliers))
                else:
                    # If no inliers array, use number of points as fallback
                    if self.frame_reg_mode == "f2f":
                        self.prev_inlier_counts[obj_id] = prev3d.shape[0]
                    elif self.frame_reg_mode == "f2m":
                        self.prev_inlier_counts[obj_id] = key_points.shape[0]

        # Optionally save cropped pcd
        if self.save_cropped_pcd:
            t_start = time.time()
            self._save_cropped_pcd(frame)
            fe_timings["save_cropped_pcd"] = time.time() - t_start

        # Update previous frame for next step
        self.prev_frame = frame

        # Print timing summary
        total_fe_time = sum(fe_timings.values())
        timing_str = " | ".join(
            [f"{k}: {v*1000:.2f}ms" for k, v in fe_timings.items() if v > 0]
        )
        print(
            f"[FrontEnd] Frame {frame.id} timing: {timing_str} | Total: {total_fe_time*1000:.2f}ms"
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
        correspond_curr3d: np.ndarray,
        mean_res: float,
        stats: dict,
        valid_stats: dict,
    ):
        result.obj_poses[obj_id] = pose
        result.rel_poses[obj_id] = rel_pose
        result.valid_indices[obj_id] = idx
        result.valid_key_points[obj_id] = key_points
        result.valid_curr_3d[obj_id] = correspond_curr3d
        result.reg_stats[obj_id] = stats
        result.mean_residuals[obj_id] = mean_res
        result.valid_stats[obj_id] = valid_stats

    def _attach_sampling_hint_stats(self, stats: dict, valid_stats: dict):
        if not isinstance(stats, dict) or not isinstance(valid_stats, dict):
            return

        def _to_int(v):
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        confirmed = _to_int(valid_stats.get("extract_confirmed_count", None))
        tentative_pool = _to_int(valid_stats.get("extract_tentative_pool_count", None))
        tentative_added = _to_int(
            valid_stats.get("extract_tentative_added_count", None)
        )
        tentative_pool_all = _to_int(
            valid_stats.get("extract_tentative_pool_count_all", None)
        )
        tentative_pool_recent = _to_int(
            valid_stats.get("extract_tentative_pool_count_recent", None)
        )

        if confirmed is not None:
            stats["extract_confirmed_count"] = confirmed
        if tentative_pool is not None:
            stats["extract_tentative_pool_count"] = tentative_pool
        if tentative_added is not None:
            stats["extract_tentative_added_count"] = tentative_added
        if tentative_pool_all is not None:
            stats["extract_tentative_pool_count_all"] = tentative_pool_all
        if tentative_pool_recent is not None:
            stats["extract_tentative_pool_count_recent"] = tentative_pool_recent

        corr = stats.get("correspond_curr3d", np.empty((0, 3)))
        try:
            num_used = int(np.asarray(corr).shape[0])
        except Exception:
            num_used = 0

        effective_num_pts = num_used
        if confirmed is not None:
            effective_num_pts = max(effective_num_pts, confirmed)
            if tentative_pool is not None:
                effective_num_pts = max(
                    effective_num_pts, confirmed + max(0, tentative_pool)
                )

        stats["num_pts_used_for_registration"] = int(num_used)
        stats["effective_num_pts_for_sampling"] = int(effective_num_pts)

    def _compute_mean_residual(self, stats: dict):
        residuals = stats.get("residuals", np.array([]))
        inliers = stats.get("inliers", np.array([]))
        if len(residuals) > 0:
            if len(inliers) == len(residuals) and np.any(inliers):
                return float(np.mean(residuals[inliers]))
            else:
                return float(np.mean(residuals))
        return -1.0

    def _count_cluster_candidates(self, stats: dict) -> int:
        clusters = stats.get("clusters", [])
        if clusters is None:
            return 0
        if isinstance(clusters, dict):
            return 1
        if isinstance(clusters, (list, tuple)):
            return len(clusters)
        if isinstance(clusters, np.ndarray):
            if clusters.ndim == 0:
                item = clusters.item()
                if isinstance(item, dict):
                    return 1
                if isinstance(item, (list, tuple)):
                    return len(item)
                return 0
            return int(clusters.shape[0])
        return 0

    def _apply_pose_jump_guard(
        self,
        T_candidate: np.ndarray,
        T_prev: np.ndarray,
        stats: dict,
    ):
        info = {
            "enabled": bool(self.pose_jump_guard_enable),
            "rejected": False,
            "dt": -1.0,
            "ddeg": -1.0,
            "used": 0,
            "ninliers": 0,
            "inlier_ratio": 0.0,
            "num_clusters": 0,
        }
        if (
            (not self.pose_jump_guard_enable)
            or T_candidate is None
            or T_prev is None
            or stats is None
        ):
            return T_candidate, False, info

        dt, ddeg = self._se3_delta(T_candidate, T_prev)
        inliers = np.asarray(stats.get("inliers", np.array([])), dtype=bool)
        used = int(inliers.shape[0])
        ninliers = int(np.sum(inliers)) if used > 0 else 0
        inlier_ratio = float(ninliers / max(1, used))
        num_clusters = self._count_cluster_candidates(stats)

        info.update(
            {
                "dt": float(dt),
                "ddeg": float(ddeg),
                "used": int(used),
                "ninliers": int(ninliers),
                "inlier_ratio": float(inlier_ratio),
                "num_clusters": int(num_clusters),
            }
        )

        big_jump = (dt > self.pose_jump_guard_trans_thres) or (
            ddeg > self.pose_jump_guard_rot_deg_thres
        )
        weak_support = (ninliers < self.pose_jump_guard_min_inliers) or (
            inlier_ratio < self.pose_jump_guard_min_inlier_ratio
        )
        weak_single_hypothesis = (num_clusters <= 1) and (
            ninliers < self.pose_jump_guard_single_cluster_min_inliers
        )

        if big_jump and (weak_support or weak_single_hypothesis):
            info["rejected"] = True
            print(
                f"[FrontEnd] Pose jump guard rejected candidate "
                f"(dT={dt:.3f}m, dR={ddeg:.2f}deg, "
                f"inliers={ninliers}/{used}, clusters={num_clusters})."
            )
            return T_prev, True, info

        return T_candidate, False, info

    def _seed_pose_history(self, obj_id: int, pose: np.ndarray):
        if pose is None:
            return
        if obj_id not in self.pose_histories or len(self.pose_histories[obj_id]) == 0:
            self.pose_histories[obj_id] = [pose.copy()]

    def _update_pose_history(self, obj_id: int, pose: np.ndarray):
        if pose is None:
            return
        hist = self.pose_histories.setdefault(obj_id, [])
        hist.append(pose.copy())
        max_len = max(2, self.vel_hist_len + 1)
        if len(hist) > max_len:
            del hist[: len(hist) - max_len]

    def _pose_velocity(self, T_new: np.ndarray, T_old: np.ndarray) -> np.ndarray:
        dT = T_new @ inverse_SE3(T_old)
        return se3_to_vec(log_SE3(dT))

    def _estimate_history_velocity(self, obj_id: int) -> Optional[np.ndarray]:
        hist = self.pose_histories.get(obj_id, [])
        if len(hist) < 2:
            return None

        start_idx = max(1, len(hist) - self.vel_hist_len)
        velocities = [
            self._pose_velocity(hist[i], hist[i - 1])
            for i in range(start_idx, len(hist))
        ]
        if len(velocities) == 0:
            return None
        return np.mean(np.stack(velocities, axis=0), axis=0)

    def _apply_velocity_fallback(
        self, obj_id: int, T_prev: np.ndarray, T_candidate: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, bool]:
        if T_candidate is None:
            return T_prev, False

        hist_vel = self._estimate_history_velocity(obj_id)
        if hist_vel is None:
            return T_candidate, False

        cur_vel = self._pose_velocity(T_candidate, T_prev)
        rot_change_deg = float(np.degrees(np.linalg.norm(cur_vel[:3] - hist_vel[:3])))
        trans_change = float(np.linalg.norm(cur_vel[3:] - hist_vel[3:]))

        if (
            rot_change_deg > self.vel_rot_change_thres_deg
            or trans_change > self.vel_trans_change_thres
        ):
            T_pred = exp_se3(vec_to_se3(hist_vel)) @ T_prev
            return T_pred, True

        return T_candidate, False

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
                    f"obj_{obj_id}_frame_{frame.id}",
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
            correspond_curr3d:  (M,3) 3D points from current frame
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

        assert (
            curr_pts_3d.shape[0]
            == len(curr_valid)
            == len(curr_visible)
            == len(track_table.valid)
            == len(track_table.visible)
        )

        idx = common_idx[both_mask]
        prev3d = track_table.track_3d[idx].copy()
        correspond_curr3d = curr_pts_3d[idx].copy()

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

        return idx, prev3d, correspond_curr3d, valid_stats

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
        obj_idx = np.asarray(obj_idx, dtype=np.int64)
        obj_rows = self._map_track_ids_to_obj_rows(obj, obj_idx)

        valid_kp_bool = np.zeros(obj_idx.shape[0], dtype=bool)
        row_ok = (obj_rows >= 0) & (obj_rows < len(obj.valid))
        if np.any(row_ok):
            valid_kp_bool[row_ok] = np.asarray(obj.valid, dtype=bool)[obj_rows[row_ok]]

        vis_obj = np.asarray(cur_visible, dtype=bool)[obj_idx]
        val_obj = np.asarray(cur_valid, dtype=bool)[obj_idx]
        uncer_obj = (
            np.asarray(cur_uncertainties, dtype=float)[obj_idx] < uncertainty_thres
        )
        both_mask = vis_obj & val_obj & valid_kp_bool & uncer_obj

        idx = obj_idx[both_mask]
        rows = obj_rows[both_mask]
        rows_ok = (rows >= 0) & (rows < len(obj.key_points))
        if not np.all(rows_ok):
            idx = idx[rows_ok]
            rows = rows[rows_ok]
        key_points = obj.key_points[rows].copy()
        correspond_curr3d = cur_pts_3d[idx].copy()

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

        return idx, key_points, correspond_curr3d, valid_stats

    def _map_track_ids_to_obj_rows(self, obj, track_ids: np.ndarray) -> np.ndarray:
        """
        Robust mapping from global track ids to object keypoint row indices.
        Returns -1 where mapping is unavailable/invalid.
        """
        tids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        rows = np.full(tids.shape[0], -1, dtype=np.int64)

        t2o = getattr(obj, "track_idx_2_obj_idx", None)
        if t2o is None:
            return rows

        t2o = np.asarray(t2o).reshape(-1)
        if t2o.size == 0:
            return rows

        tid_ok = (tids >= 0) & (tids < t2o.shape[0])
        if np.any(tid_ok):
            rows[tid_ok] = np.asarray(t2o[tids[tid_ok]], dtype=np.int64)

        return rows

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
        mask_border_margin_px=0,
        frame_id=None,
    ):

        # idx, key_points, correspond_curr3d, cur_visible, valid_stats = (
        #             self._extract_valid_key_points_mask_remove(
        #                 obj,
        #                 track_table.obj2track_map[obj_id],
        #                 tracks,
        #                 track_3d,
        #                 current_visibles,
        #                 track_valid,
        #                 uncertainties,
        #                 frame.mask[obj_id, 0],
        #                 uncertainty_thres=self.reg_uncer_thres,
        #             )
        #         )
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

        obj_rows = self._map_track_ids_to_obj_rows(obj, obj_idx)

        val_obj = cur_valid[obj_idx]
        uncer_obj = cur_uncertainties[obj_idx] < float(uncertainty_thres)
        valid_kp_obj = np.zeros(obj_idx.shape[0], dtype=bool)
        row_ok = (obj_rows >= 0) & (obj_rows < len(obj.valid))
        if np.any(row_ok):
            valid_kp_obj[row_ok] = np.asarray(obj.valid, dtype=bool)[obj_rows[row_ok]]

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
        border_safe_np = np.ones_like(inside_mask_np, dtype=bool)

        margin = int(mask_border_margin_px)
        if margin > 0:
            # Keep only points that are inside an eroded mask so they are at
            # least `margin` pixels away from the object boundary.
            kernel_size = 2 * margin + 1
            inv_mask = (~mask_bool).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            dilated_inv = F.max_pool2d(
                inv_mask, kernel_size=kernel_size, stride=1, padding=margin
            )
            eroded_mask = (1.0 - dilated_inv).squeeze(0).squeeze(0) > 0.5
            border_safe_np = eroded_mask[y_t, x_t].detach().cpu().numpy()

        outside_or_bad = (~inside_mask_np) | (~finite_xy)
        if outside_or_bad.any():
            cur_visible[obj_idx[outside_or_bad]] = False

        vis_obj = cur_visible[obj_idx]

        quality_mask = (
            vis_obj & val_obj & uncer_obj & finite_xy & inside_mask_np & border_safe_np
        )
        both_mask = quality_mask & valid_kp_obj

        # If confirmed points are too few, temporarily use tentative points
        # (obj.valid=False) so tracking does not drop before promotion catches up.
        n_confirmed = int(np.sum(both_mask))
        n_tentative_added = 0
        n_tentative_pool = 0
        n_tentative_pool_all = 0
        n_tentative_pool_recent = 0
        tentative_pool_used_stale = False
        if (
            self.tentative_fallback_min_valid_points > 0
            and n_confirmed < self.tentative_fallback_min_valid_points
        ):
            tentative_uncer_thres = float(uncertainty_thres) * float(
                self.tentative_fallback_uncer_scale
            )
            tentative_uncer_obj = cur_uncertainties[obj_idx] < tentative_uncer_thres
            tentative_pool_mask_all = (
                vis_obj
                & val_obj
                & tentative_uncer_obj
                & finite_xy
                & inside_mask_np
                & border_safe_np
                & (~valid_kp_obj)
            )
            tentative_pool_mask = tentative_pool_mask_all.copy()
            n_tentative_pool_all = int(np.sum(tentative_pool_mask_all))
            n_tentative_pool_recent = n_tentative_pool_all

            # Keep tentative fallback focused on recently added points.
            recent_limit = int(self.tentative_fallback_recent_frames)
            if recent_limit >= 0 and frame_id is not None:
                kp_frames_all = np.asarray(
                    getattr(obj, "key_point_frames", np.full((0,), -1)),
                    dtype=np.int64,
                )
                kp_frames = np.full(obj_idx.shape[0], -1, dtype=np.int64)
                row_ok = (obj_rows >= 0) & (obj_rows < kp_frames_all.shape[0])
                if np.any(row_ok):
                    kp_frames[row_ok] = kp_frames_all[obj_rows[row_ok]]
                ages = int(frame_id) - kp_frames
                recent_mask = (kp_frames >= 0) & (ages >= 0) & (ages <= recent_limit)
                tentative_pool_mask_recent = tentative_pool_mask_all & recent_mask
                n_tentative_pool_recent = int(np.sum(tentative_pool_mask_recent))
                tentative_pool_mask = tentative_pool_mask_recent

                if (
                    self.tentative_fallback_allow_stale_when_empty
                    and n_tentative_pool_recent == 0
                    and n_tentative_pool_all > 0
                ):
                    tentative_pool_mask = tentative_pool_mask_all
                    tentative_pool_used_stale = True

            n_tentative_pool = int(np.sum(tentative_pool_mask))
            if n_tentative_pool > 0:
                # Once fallback is triggered, fill up to max_points (not just min_valid_points)
                # so registration has more support while promotions catch up.
                fallback_target = max(0, int(self.tentative_fallback_max_points))
                need = fallback_target - n_confirmed
                if need > 0:
                    local_idx = np.flatnonzero(tentative_pool_mask)
                    # Pick lower-uncertainty tentative points first.
                    order = np.argsort(cur_uncertainties[obj_idx][local_idx])
                    take_n = min(local_idx.size, need)
                    if take_n > 0:
                        chosen = local_idx[order[:take_n]]
                        both_mask[chosen] = True
                        n_tentative_added = int(take_n)

        idx = obj_idx[both_mask]
        rows = obj_rows[both_mask]
        rows_ok = (rows >= 0) & (rows < len(obj.key_points))
        if not np.all(rows_ok):
            idx = idx[rows_ok]
            rows = rows[rows_ok]
        key_points = obj.key_points[rows].copy()
        correspond_curr3d = cur_pts_3d[idx].copy()

        valid_stats = {
            "extract_vis_obj_mask": vis_obj,
            "extract_val_obj_mask": val_obj,
            "extract_uncer_obj_mask": uncer_obj,
            "extract_valid_kp_mask": valid_kp_obj,
            "extract_uncertainty_thres": uncertainty_thres,
            "extract_obj_idx": obj_idx,
            "extract_inside_mask": inside_mask_np,
            "extract_border_safe_mask": border_safe_np,
            "extract_mask_border_margin_px": margin,
            "extract_finite_xy": finite_xy,
            "extract_obj_rows": obj_rows,
            "extract_confirmed_count": n_confirmed,
            "extract_tentative_fallback_min_valid_points": int(
                self.tentative_fallback_min_valid_points
            ),
            "extract_tentative_recent_frames_limit": int(
                self.tentative_fallback_recent_frames
            ),
            "extract_tentative_pool_count": n_tentative_pool,
            "extract_tentative_pool_count_all": n_tentative_pool_all,
            "extract_tentative_pool_count_recent": n_tentative_pool_recent,
            "extract_tentative_added_count": n_tentative_added,
            "extract_tentative_fallback_used": bool(n_tentative_added > 0),
            "extract_tentative_pool_used_stale": bool(tentative_pool_used_stale),
        }

        return idx, key_points, correspond_curr3d, cur_visible, valid_stats

    def _extract_valid_key_points_mask_remove_no_depth_check(
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

        # idx, key_points, correspond_curr3d, cur_visible, valid_stats = (
        #             self._extract_valid_key_points_mask_remove(
        #                 obj,
        #                 track_table.obj2track_map[obj_id],
        #                 tracks,
        #                 track_3d,
        #                 current_visibles,
        #                 track_valid,
        #                 uncertainties,
        #                 frame.mask[obj_id, 0],
        #                 uncertainty_thres=self.reg_uncer_thres,
        #             )
        #         )
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

        both_mask = vis_obj & uncer_obj & valid_kp_obj & finite_xy & inside_mask_np

        idx = obj_idx[both_mask]
        key_points = obj.key_points[both_mask].copy()
        correspond_curr3d = cur_pts_3d[idx].copy()

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

        return idx, key_points, correspond_curr3d, cur_visible, valid_stats

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

    def _query_abs_sdf_values(self, obj, pts_obj: np.ndarray) -> np.ndarray:
        """Query absolute SDF values at object-frame points."""
        if obj is None or pts_obj is None or pts_obj.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        sdf_vals = np.empty((0,), dtype=np.float64)

        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            try:
                qvals = obj.sdf_volume.query_sdf(pts_obj)
                if qvals is not None:
                    qvals = np.asarray(qvals).reshape(-1)
                    sdf_vals = np.abs(qvals[np.isfinite(qvals)])
            except Exception:
                sdf_vals = np.empty((0,), dtype=np.float64)

        if (
            sdf_vals.size == 0
            and getattr(obj, "sdf", None) is not None
            and "tsdf" in obj.sdf
        ):
            tsdf = obj.sdf["tsdf"]
            origin = obj.sdf["vol_origin"]
            voxel = float(obj.sdf["voxel_size"])
            vol_dim = np.array(tsdf.shape, dtype=np.int32)
            vox = np.floor((pts_obj - origin[None, :]) / voxel).astype(np.int32)

            inb = np.logical_and(
                np.all(vox >= 0, axis=1), np.all(vox < vol_dim[None, :], axis=1)
            )
            if np.any(inb):
                vox_in = vox[inb]
                sdf_vals = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]])

        return np.asarray(sdf_vals, dtype=np.float64)

    def _sdf_alignment_cost(
        self, pts_cur: np.ndarray, T_cur2obj: np.ndarray, obj, robust_percentile: float
    ) -> Tuple[float, float]:
        """
        Return robust SDF alignment cost and support ratio.
        Smaller cost is better.
        """
        pts_obj = transform_pts(T_cur2obj, pts_cur)
        sdf_vals = self._query_abs_sdf_values(obj, pts_obj)
        if sdf_vals.size == 0:
            return np.inf, 0.0

        q = float(np.clip(robust_percentile, 0.0, 100.0))
        cost = float(np.percentile(sdf_vals, q))
        support = float(sdf_vals.size) / float(max(1, pts_cur.shape[0]))
        return cost, support

    def _apply_dense_recovery_f2m_sdf(
        self,
        obj_id: int,
        curr_frame,
        obj,
        init_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Dense recovery for f2m mode by aligning the current depth crop to object SDF.
        Returns absolute pose T_c2w.
        """
        has_sdf = getattr(obj, "sdf_volume", None) is not None or (
            getattr(obj, "sdf", None) is not None and "tsdf" in obj.sdf
        )
        if not has_sdf:
            return None, {"reason": "no_sdf"}

        pts_cur = self._extract_cropped_point_cloud(curr_frame, obj_id)
        if pts_cur.shape[0] < self.dense_recovery_min_points:
            print(
                f"[FrontEnd] Dense recovery (f2m+sdf) skipped: insufficient points "
                f"({pts_cur.shape[0]} < {self.dense_recovery_min_points})"
            )
            return None, {
                "reason": "insufficient_points",
                "num_points": pts_cur.shape[0],
            }

        if pts_cur.shape[0] > self.dense_recovery_sdf_max_points:
            step = int(
                np.ceil(pts_cur.shape[0] / float(self.dense_recovery_sdf_max_points))
            )
            pts_cur = pts_cur[::step]

        T_seed = init_pose.copy() if init_pose is not None else np.eye(4)
        T_seed_cur2obj = inverse_SE3(T_seed)
        cost_before, support_before = self._sdf_alignment_cost(
            pts_cur,
            T_seed_cur2obj,
            obj,
            robust_percentile=self.dense_recovery_sdf_percentile,
        )
        if not np.isfinite(cost_before):
            return None, {"reason": "invalid_initial_cost"}

        def objective(xi):
            dT = exp_se3(vec_to_se3(xi))
            T_cur = dT @ T_seed
            T_cur2obj = inverse_SE3(T_cur)
            cost, _ = self._sdf_alignment_cost(
                pts_cur,
                T_cur2obj,
                obj,
                robust_percentile=self.dense_recovery_sdf_percentile,
            )
            return 1e6 if not np.isfinite(cost) else float(cost)

        x0 = np.zeros(6, dtype=np.float64)
        bounds = [
            (-self.dense_recovery_sdf_rot_bound, self.dense_recovery_sdf_rot_bound)
        ] * 3 + [
            (-self.dense_recovery_sdf_trans_bound, self.dense_recovery_sdf_trans_bound)
        ] * 3

        try:
            opt = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.dense_recovery_sdf_max_iter},
            )
        except Exception as e:
            print(f"[FrontEnd] Dense recovery (f2m+sdf) optimization failed: {e}")
            return None, {"reason": "optimizer_exception", "error": str(e)}

        if not opt.success:
            return None, {"reason": "optimizer_failed", "message": str(opt.message)}

        T_opt = exp_se3(vec_to_se3(opt.x)) @ T_seed
        cost_after, support_after = self._sdf_alignment_cost(
            pts_cur,
            inverse_SE3(T_opt),
            obj,
            robust_percentile=self.dense_recovery_sdf_percentile,
        )
        improvement = float(cost_before - cost_after)
        if (
            not np.isfinite(cost_after)
            or improvement < self.dense_recovery_sdf_min_improve
        ):
            return None, {
                "reason": "insufficient_improvement",
                "sdf_before": float(cost_before),
                "sdf_after": float(cost_after) if np.isfinite(cost_after) else np.inf,
                "improvement": improvement,
            }

        stats = {
            "mode": "f2m_sdf",
            "num_points": int(pts_cur.shape[0]),
            "sdf_before": float(cost_before),
            "sdf_after": float(cost_after),
            "sdf_improvement": improvement,
            "support_before": float(support_before),
            "support_after": float(support_after),
            "opt_success": bool(opt.success),
            "opt_message": str(opt.message),
            "opt_iterations": int(getattr(opt, "nit", -1)),
        }
        return T_opt, stats

    def _se3_delta(self, T_new, T_old):
        dT = T_new @ inverse_SE3(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        dR = scipy_R.from_matrix(dT[:3, :3]).magnitude()
        ddeg = float(np.degrees(dR))
        return dt, ddeg
