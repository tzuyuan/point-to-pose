import os
from collections import defaultdict
from typing import List, Dict, Tuple

import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import SAMPLER, CRITERION
from point2pose.data_types.criterion_context import CriterionContext
from point2pose.data_types.sampler_context import SamplerContext
from point2pose.data_types.front_end_result import FrontEndResult
from point2pose.data_types.key_frame import KeyFrame
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import transform_pts, inverse_SE3


class KeyFrameManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        self.num_obj = 0
        self.save_key_points = self.pipeline_cfg.get("save_key_points", False)
        self.max_rel_rotation_deg = self.pipeline_cfg.get("max_rel_rotation_deg", 45.0)
        self.fill_missing_depth = self.pipeline_cfg.get("fill_missing_depth", False)
        self.fill_depth_win_size = self.pipeline_cfg.get(
            "fill_missing_depth_window_size", 3
        )
        self.fill_depth_min_neighbors = self.pipeline_cfg.get(
            "fill_missing_depth_min_neighbors", 1
        )
        self.min_depth = self.pipeline_cfg.get("min_depth", 0.05)
        self.max_depth = self.pipeline_cfg.get("max_depth", 1.0)
        self.key_points_save_path = self.pipeline_cfg.get(
            "key_points_save_path", "./key_points"
        )

        if self.save_key_points:
            os.makedirs(self.key_points_save_path, exist_ok=True)

        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)

        self.crit_ctx = CriterionContext()
        self.samp_ctx = SamplerContext(
            frame=None, min_depth=self.min_depth, max_depth=self.max_depth
        )

        # Per-object bookkeeping
        self.is_key_frame: Dict[int, bool] = {}  # obj_id -> bool
        self.keyframes: Dict[int, List[KeyFrame]] = defaultdict(
            list
        )  # obj_id -> List[KeyFrame]

        # -------------------------
        # Sampling anchor gating
        # -------------------------
        self.kf_gating = self.pipeline_cfg.get("kf_gating", False)
        self.anchor_rot_deg = self.pipeline_cfg.get("anchor_rot_deg", 3.0)
        self.anchor_trans = self.pipeline_cfg.get("anchor_trans", 0.02)  # meters

        self.anchor_min_inliers = self.pipeline_cfg.get("anchor_min_inliers", 10)
        self.anchor_min_inlier_ratio = self.pipeline_cfg.get(
            "anchor_min_inlier_ratio", 0.3
        )
        self.anchor_max_median_res = self.pipeline_cfg.get(
            "anchor_max_median_res", 0.01
        )  # meters
        self.anchor_vel_hist_len = self.pipeline_cfg.get("anchor_vel_hist_len", 5)
        self.anchor_vel_rot_deg = self.pipeline_cfg.get("anchor_vel_rot_deg", 5.0)
        self.anchor_vel_trans = self.pipeline_cfg.get(
            "anchor_vel_trans", 0.01
        )  # meters
        self.anchor_vel_vote_ratio = self.pipeline_cfg.get("anchor_vel_vote_ratio", 0.6)
        self.anchor_vel_min_votes = self.pipeline_cfg.get("anchor_vel_min_votes", 2)

        # Per-object "last consistent" snapshot
        self.last_consistent = (
            {}
        )  # obj_id -> {"frame_id": int, "pose": np.ndarray (4,4)}

        # Pending points bookkeeping: obj_id -> list of track_ids
        self.pending_track_ids = defaultdict(list)

        # (obj_id, track_id) -> birth frame_id (or any metadata you want)
        self.pending_birth_frame = {}

        self.pending_meta = {}

        # Pending promotion/rejection thresholds
        self.pending_promote_streak = self.pipeline_cfg.get("pending_promote_streak", 3)
        self.pending_ttl = self.pipeline_cfg.get("pending_ttl", 15)
        self.pending_max_bad = self.pipeline_cfg.get("pending_max_bad", 6)
        self.pending_uncer_thres = self.pipeline_cfg.get("pending_uncer_thres", 0.3)
        self.pending_ttl_on_growth_frames = bool(
            self.pipeline_cfg.get("pending_ttl_on_growth_frames", True)
        )

        # Pose stability gate for promotion (tighter than your max_rel_rotation_deg)
        self.pending_stable_rot_deg = self.pipeline_cfg.get(
            "pending_stable_rot_deg", 2.0
        )
        self.pending_stable_trans = self.pipeline_cfg.get(
            "pending_stable_trans", 0.01
        )  # meters
        self.pending_use_geom_check = self.pipeline_cfg.get(
            "pending_use_geom_check", False
        )
        # Optional: if True, require point's UV to lie inside current mask to count as "good"
        self.pending_require_inside_mask = self.pipeline_cfg.get(
            "pending_require_inside_mask", True
        )
        self.pending_reset_on_bad = self.pipeline_cfg.get("pending_reset_on_bad", False)

        # Optional SDF gate for keyframe/keypoint initialization
        self.sdf_kf_gate = self.pipeline_cfg.get("sdf_kf_gate", False)
        self.sdf_kf_gate_thres = float(self.pipeline_cfg.get("sdf_kf_gate_thres", 0.35))
        self.sdf_kf_gate_percentile = float(
            self.pipeline_cfg.get("sdf_kf_gate_percentile", 70.0)
        )
        self.sdf_kf_gate_min_dense = int(
            self.pipeline_cfg.get("sdf_kf_gate_min_dense", 200)
        )

        # -------------------------
        # Safe map growth / pending promotion robustness
        # -------------------------
        self.map_growth_gate = self.pipeline_cfg.get("map_growth_gate", True)
        self.map_growth_cooldown_after_recovery = int(
            self.pipeline_cfg.get("map_growth_cooldown_after_recovery", 3)
        )
        self.map_growth_max_mean_residual = float(
            self.pipeline_cfg.get("map_growth_max_mean_residual", 0.0007)
        )
        self.map_growth_cooldown_until = {}  # obj_id -> frame_id
        self.pending_update_when_growth_blocked = bool(
            self.pipeline_cfg.get("pending_update_when_growth_blocked", True)
        )

        # Pending point geometric consistency checks (object-frame)
        self.pending_obs_min_count = int(
            self.pipeline_cfg.get("pending_obs_min_count", 4)
        )
        self.pending_obs_max_keep = int(
            self.pipeline_cfg.get("pending_obs_max_keep", 1)
        )
        self.pending_obj_spread_thres = float(
            self.pipeline_cfg.get("pending_obj_spread_thres", 0.008)
        )  # meters
        self.pending_use_view_diversity = self.pipeline_cfg.get(
            "pending_use_view_diversity", True
        )
        self.pending_min_view_angle_deg = float(
            self.pipeline_cfg.get("pending_min_view_angle_deg", 5.0)
        )

        # Optional SDF gate for pending promotion (point-wise)
        self.pending_sdf_gate = self.pipeline_cfg.get("pending_sdf_gate", False)
        self.pending_sdf_gate_thres = float(
            self.pipeline_cfg.get("pending_sdf_gate_thres", 0.02)
        )
        self.pending_sdf_gate_percentile = float(
            self.pipeline_cfg.get("pending_sdf_gate_percentile", 100.0)
        )
        self.pending_sdf_min_support = float(
            self.pipeline_cfg.get("pending_sdf_min_support", 1.0)
        )

        # Newly promoted points: start with lower trust (higher uncertainty), then anneal
        self.promoted_warmup_frames = int(
            self.pipeline_cfg.get("promoted_warmup_frames", 4)
        )
        self.promoted_init_uncer = float(
            self.pipeline_cfg.get("promoted_init_uncer", 0.2)
        )
        self.promoted_min_uncer = float(
            self.pipeline_cfg.get("promoted_min_uncer", 0.02)
        )
        self.promoted_spread_to_uncer = float(
            self.pipeline_cfg.get("promoted_spread_to_uncer", 20.0)
        )
        self.promoted_uncer_decay = float(
            self.pipeline_cfg.get("promoted_uncer_decay", 0.6)
        )
        self.promoted_meta = {}  # (obj_id, track_id) -> {obj_idx, age, target_unc}

        # Pending keyframe promotion (conservative mode):
        # sampled keyframes are staged and only promoted to graph/SDF after checks.
        self.pending_keyframes: Dict[int, List[dict]] = defaultdict(list)
        self.pending_kf_min_promoted_ratio = float(
            self.pipeline_cfg.get("pending_kf_min_promoted_ratio", 0.3)
        )
        self.pending_kf_min_promoted_count = int(
            self.pipeline_cfg.get("pending_kf_min_promoted_count", 8)
        )
        self.pending_kf_ttl = int(self.pipeline_cfg.get("pending_kf_ttl", 30))
        self.pending_kf_ttl_on_growth_frames = bool(
            self.pipeline_cfg.get("pending_kf_ttl_on_growth_frames", True)
        )
        self.pending_kf_require_reg_quality = bool(
            self.pipeline_cfg.get("pending_kf_require_reg_quality", True)
        )
        self.pending_kf_require_pose_stable = bool(
            self.pipeline_cfg.get("pending_kf_require_pose_stable", True)
        )
        self.retire_invalid_tracks = bool(
            self.pipeline_cfg.get("retire_invalid_tracks", True)
        )
        self.retire_protect_keyframe_tracks = bool(
            self.pipeline_cfg.get("retire_protect_keyframe_tracks", True)
        )
        self.pending_kf_sdf_gate = bool(
            self.pipeline_cfg.get("pending_kf_sdf_gate", False)
        )
        self.pending_kf_sdf_gate_thres = float(
            self.pipeline_cfg.get("pending_kf_sdf_gate_thres", 0.02)
        )

    def set_num_obj(self, num_obj: int):
        self.num_obj = max(int(num_obj), 0)

    def _infer_num_obj_from_mask(self, mask) -> int:
        if mask is None or not hasattr(mask, "shape"):
            return 0
        shape = getattr(mask, "shape", None)
        if shape is None or len(shape) == 0:
            return 0
        return int(shape[0])

    # -------------------------------------------------------------------------
    # First frame initialization
    # -------------------------------------------------------------------------
    def initialize(self, frame, track_table, objects, tracker):
        """
        Initial sampling for the first frame.
        This:
        - Samples points for all objects via the sampler
        - Populates track_table and object.key_points
        - Creates an initial KeyFrame for each object
        """
        self.num_obj = len(objects)
        if self.num_obj == 0:
            self.num_obj = self._infer_num_obj_from_mask(getattr(frame, "mask", None))

        self.samp_ctx.frame = frame
        self.samp_ctx.update_sampler_context(frame=frame, track_table=track_table)

        # Sample for all objects (fills track_table.track_2d/3d/valid/visibles and obj2track_map)
        self._sample_for_all_obj(self.samp_ctx, track_table, tracker)

        created_keyframes: List[KeyFrame] = []
        for obj_id in range(self.num_obj):
            if obj_id >= len(objects):
                continue

            obj = objects[obj_id]

            # Use track_table mapping to pull this object's initial points
            track_ids = np.asarray(track_table.obj2track_map[obj_id], dtype=np.int64)
            if track_ids.size == 0:
                self.is_key_frame[obj_id] = False
                continue

            # Assign keypoints to object
            obj.add_key_points(
                new_key_points=track_table.track_3d[track_ids],
                new_uncertainties=0.1 * np.ones(track_ids.shape[0], dtype=float),
                new_valid=track_table.valid[track_ids],
                new_indices=track_ids,
                frame_id=0,
            )
            self.is_key_frame[obj_id] = False

            # Build "new keypoints" struct from these initial points
            kp = self._build_kp_from_track_table(
                track_table=track_table,
                track_ids=track_ids,
                obj_pose=obj.pose,
            )

            # For the first frame, observed points = the same set as kp
            obs = kp

            # Registration stats are empty on first frame
            valid_idx = np.asarray(kp["track_ids"], dtype=int)
            valid_idx = valid_idx[kp["valid"]]
            reg_stats = {
                "correspond_curr3d": obj.key_points,
                "inliers": np.ones((len(valid_idx),), dtype=bool),
                "residuals": np.zeros((len(valid_idx),), dtype=float),
                "valid_idx": valid_idx,
            }

            kf = self._create_keyframe(
                frame=frame,
                obj=obj,
                kp=kp,
                obs=obs,
                reg_stats=reg_stats,
                anchor_pose=obj.pose,
            )
            self.keyframes[obj_id].append(kf)

            created_keyframes.append(kf)

            obj.num_keyframes = 1
            obj.keyframes = self.keyframes[obj_id]

        self.crit_ctx.objects = objects
        self.criterion.initialize(self.crit_ctx)

        return created_keyframes

    # -------------------------------------------------------------------------
    # Per-frame update
    # -------------------------------------------------------------------------
    def update(
        self,
        hist_frames,
        hist_fe_results,
        hist_track_tables,
        track_table,
        objects,
        tracker,
        conservative=True,
    ) -> List[KeyFrame]:
        """
        Check criteria and sample new keyframes if needed.
        Returns a list of KeyFrame created in this update.
        """

        cur_fe_result = hist_fe_results[-1]
        cur_frame = hist_frames[-1]
        cur_track_table_copy = hist_track_tables[-1]
        self.num_obj = len(objects)

        # Update contexts for criterion + sampler
        self.crit_ctx.update_criterion_context(
            cur_iter=cur_fe_result.frame_id,
            uncertainty=cur_fe_result.uncertainties,
            reg_stats=cur_fe_result.reg_stats,
            track_table=track_table,
            frame=cur_frame,
        )
        self.samp_ctx.update_sampler_context(frame=cur_frame, track_table=track_table)

        created_keyframes: List[KeyFrame] = []

        # loop through all objects
        for obj_id in range(min(self.num_obj, len(objects))):
            obj = objects[obj_id]
            self.is_key_frame[obj_id] = False

            # If object is lost, avoid sampling
            if getattr(obj, "lost", False):
                continue

            frame_growth_ok, frame_growth_reason = self._frame_allows_map_growth(
                obj_id=obj_id, frame_id=int(cur_frame.id), fe_result=cur_fe_result
            )

            if conservative:
                self._update_pending_pts_for_obj(
                    obj_id,
                    obj,
                    cur_frame,
                    cur_track_table_copy,
                    cur_fe_result,
                    allow_promotion=frame_growth_ok,
                )
                self._update_pending_keyframes_for_obj(
                    obj_id=obj_id,
                    obj=obj,
                    frame=cur_frame,
                    front_end_result=cur_fe_result,
                    allow_promotion=frame_growth_ok,
                    created_keyframes=created_keyframes,
                )

            if conservative and (not frame_growth_ok):
                # Still update pending bookkeeping above, but do not grow the map on weak frames.
                continue

            # Large rotation jump
            # rel_pose = cur_fe_result.rel_poses.get(obj_id)
            # if rel_pose is not None:
            #     rot_magnitude = scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()
            #     angle_deg = np.degrees(rot_magnitude)

            #     if angle_deg > self.max_rel_rotation_deg:
            #         print(
            #             f"[KeyFrameManager] Frame {cur_frame.id}: Large rotation {angle_deg:.2f} deg "
            #             f"detected for obj {obj_id}. Skipping sampling."
            #         )
            #         continue

            # Skip resampling when the current/anchor pose is already covered by an
            # existing promoted or pending keyframe pose. This uses the configured
            # anchor thresholds, which were previously not applied to revisit rejection.
            # pose_is_novel, novel_info = self._is_pose_novel_for_sampling(
            #     obj_id=obj_id,
            #     pose=np.asarray(obj.pose, dtype=np.float64),
            # )
            # if not pose_is_novel:
            #     if self.pipeline_cfg.get("debug_level", 0) > 0:
            #         print(
            #             "[KeyFrameManager] Frame "
            #             f"{cur_frame.id}: skip sampling for obj {obj_id} "
            #             f"(pose already covered by {novel_info['source']} "
            #             f"{novel_info['index']}, dR={novel_info['rot_deg']:.2f}deg, "
            #             f"dT={novel_info['trans_m']:.3f}m)."
            #         )
            #     continue

            # Decide whether this frame is a keyframe for this object
            if not self.criterion.check_sample_criterion(self.crit_ctx, obj_id):
                continue

            if self.kf_gating:
                (
                    anchor_frame,
                    anchor_pose,
                    anchor_fe_result,
                    anchor_track_table,
                    anchor_is_current,
                ) = self._choose_sampling_anchor(
                    hist_fe_results=hist_fe_results,
                    hist_frames=hist_frames,
                    hist_track_tables=hist_track_tables,
                    cur_fe_result=cur_fe_result,
                    obj_id=obj_id,
                    obj=obj,
                )

                # get registration residual of the anchor pose
                # if (
                #     np.mean(cur_fe_result.reg_stats[obj_id]["residuals"])
                #     > self.anchor_max_median_res
                # ):

                if not anchor_is_current:
                    print(
                        f"[KeyFrameManager] Frame {anchor_frame.id}: Anchor is not current. Skip adding keypoints/keyframe."
                    )
                    continue

            else:
                anchor_frame = cur_frame
                anchor_pose = obj.pose
                anchor_track_table = cur_track_table_copy
                anchor_fe_result = cur_fe_result
                anchor_is_current = True

            # Optional SDF consistency gate: reject this frame if dense depth points
            # disagree too much with the current object SDF map.
            if self.sdf_kf_gate:
                dense_pts_gate, _ = self._extract_dense_pcd(anchor_frame, obj_id)

                inliers = cur_fe_result.reg_stats[obj_id]["inliers"]
                reg_curr3d = cur_fe_result.reg_stats[obj_id]["correspond_curr3d"]
                inlier_curr3d = reg_curr3d[inliers]

                sdf_res = self._sdf_residual_dense(
                    dense_pts=dense_pts_gate, obj_pose=anchor_pose, obj=obj
                )
                if np.isfinite(sdf_res) and sdf_res > self.sdf_kf_gate_thres:
                    print(
                        f"[KeyFrameManager] Frame {anchor_frame.id}: SDF residual too large ({sdf_res:.4f}) for obj {obj_id}. Skip adding keypoints/keyframe."
                    )
                    continue

            # Build a sampler context from the chosen frame
            if anchor_is_current:
                samp_ctx = self.samp_ctx  # already updated with cur_frame above
            else:
                tmp_samp_ctx = SamplerContext(
                    frame=None, min_depth=self.min_depth, max_depth=self.max_depth
                )
                tmp_samp_ctx.update_sampler_context(
                    frame=anchor_frame, track_table=anchor_track_table
                )
                samp_ctx = tmp_samp_ctx

            # 1) Sample new keypoints for this object
            new_sampled_points = self.sampler.sample(samp_ctx, obj_id)
            if len(new_sampled_points) == 0:
                continue
            # Add to tracker and track_table
            # NOTE: TAPIR uses frame.rgb + frame.id to init query features/query_points, so pass anchor_frame when anchoring. :contentReference[oaicite:6]{index=6}
            frame_for_tapir = anchor_frame if not anchor_is_current else cur_frame
            new_indices = tracker.add_query_points(frame_for_tapir, new_sampled_points)
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

            # Lift to 3D using the SAME frame used for sampling
            new_points_3d, valid_new_points_3d = convert_pixel_to_world(
                pixel=new_sampled_points,
                depth_image=anchor_frame.depth,
                cam_intrinsics=anchor_frame.intrinsics,
                depth_factor=anchor_frame.depth_factor,
                fill_missing_depth=self.fill_missing_depth,
                window_size=self.fill_depth_win_size,
                min_neighbors=self.fill_depth_min_neighbors,
                max_depth=self.max_depth,
                min_depth=self.min_depth,
            )

            # Transform to object frame-0 coordinate using ANCHOR pose (not current pose)
            new_points_3d_obj = transform_pts(
                inverse_SE3(anchor_pose),
                new_points_3d,
            )

            if conservative:
                old_m = len(obj.key_points)

                # add but DO NOT use for f2m/map yet
                # Conservative approach: mark all new points as invalid until verified
                tentative_valid = np.zeros(len(new_points_3d_obj), dtype=bool)
                obj.add_key_points(
                    new_points_3d_obj,
                    0.1 * np.ones(len(new_points_3d_obj), dtype=float),
                    tentative_valid,
                    new_indices,
                    cur_frame.id,
                )
                new_obj_indices = np.arange(
                    old_m, old_m + len(new_indices), dtype=np.int64
                )

                # record as pending (promote/reject later)
                for tid, oi in zip(new_indices, new_obj_indices):
                    key = (obj_id, int(tid))
                    if key in self.pending_birth_frame:
                        continue
                    self.pending_track_ids[obj_id].append(int(tid))
                    self.pending_birth_frame[key] = int(cur_frame.id)
                    self.pending_meta[key] = {
                        "obj_idx": int(oi),
                        "good": 0,
                        "bad": 0,
                        "age": 0,
                        "eval_age": 0,
                        "obs_obj": [],  # list[(3,)] robust support in object frame
                        "obs_frame_ids": [],  # list[int]
                        "obs_view_dirs": [],  # list[(3,)] optional view diversity
                        "last_obs_frame": -1,
                    }

                track_table.append_track_table(
                    track_2d=new_sampled_points,
                    track_3d=new_points_3d,
                    valid=valid_new_points_3d,
                    uncertainties=0.5 * np.ones(len(new_sampled_points), dtype=float),
                    visibles=np.ones(len(new_sampled_points), dtype=bool),
                )

                # Build a keyframe candidate, but keep it pending until support checks pass.
                kp = dict(
                    track_ids=np.asarray(new_indices, dtype=np.int64),
                    uv=new_sampled_points,
                    xyz_cam=new_points_3d,
                    xyz_obj=new_points_3d_obj,
                    valid=valid_new_points_3d,
                )
                obs = self._build_obs_for_frame(
                    obj_id=obj_id,
                    frame=anchor_frame,
                    track_table=anchor_track_table,
                    obj=obj,
                    anchor_pose=anchor_pose,
                )
                kf = self._create_keyframe(
                    frame=anchor_frame,
                    obj=obj,
                    kp=kp,
                    obs=obs,
                    reg_stats=anchor_fe_result.reg_stats.get(obj_id, {}),
                    anchor_pose=anchor_pose,
                )
                # Stage conservative keyframes. They are promoted to active keyframes
                # (graph + SDF) only after enough tentative points are verified.
                self._queue_pending_keyframe(
                    obj_id=obj_id,
                    keyframe=kf,
                    track_ids=np.asarray(new_indices, dtype=np.int64)[
                        np.asarray(valid_new_points_3d, dtype=bool)
                    ],
                    frame_id=int(cur_frame.id),
                )
                self.is_key_frame[obj_id] = False
            else:
                # More aggressive approach: trust the new depth measurements (even if noisy) and mark them as valid
                # Update object with these new keypoints
                obj.add_key_points(
                    new_points_3d_obj,
                    0.1
                    * np.ones(
                        len(new_points_3d_obj), dtype=float
                    ),  # initial uncertainty
                    valid_new_points_3d,
                    new_indices,
                    anchor_frame.id,
                )

                track_table.append_track_table(
                    track_2d=new_sampled_points,
                    track_3d=new_points_3d,
                    valid=valid_new_points_3d,
                    uncertainties=0.5 * np.ones(len(new_sampled_points), dtype=float),
                    visibles=np.ones(len(new_sampled_points), dtype=bool),
                )

                # 2) Build keypoint struct for this keyframe
                kp = dict(
                    track_ids=np.asarray(new_indices, dtype=np.int64),
                    uv=new_sampled_points,
                    xyz_cam=new_points_3d,
                    xyz_obj=new_points_3d_obj,
                    valid=valid_new_points_3d,
                )

                # 3) Build observed points for this frame
                obs = self._build_obs_for_frame(
                    obj_id=obj_id,
                    frame=anchor_frame,
                    track_table=anchor_track_table,
                    obj=obj,
                    anchor_pose=anchor_pose,
                )

                # 5) Create and store the keyframe
                # build kp/obs/reg_stats and create keyframe
                kf = self._create_keyframe(
                    frame=anchor_frame,
                    obj=obj,
                    kp=kp,
                    obs=obs,
                    reg_stats=anchor_fe_result.reg_stats.get(obj_id, {}),
                    anchor_pose=anchor_pose,
                )
                self.keyframes[obj_id].append(kf)
                created_keyframes.append(kf)

                obj.last_keyframe_frame_id = kf.frame_id
                obj.last_keyframe = kf
                obj.num_keyframes = len(self.keyframes[obj_id])
                obj.keyframes = self.keyframes[obj_id]

                # This object triggers a keyframe
                self.is_key_frame[obj_id] = True

            # Optionally save debug ply of keypoints
            if self.save_key_points:
                self._save_key_points_for_object(
                    obj_id=obj_id,
                    obj=obj,
                    frame_id=anchor_frame.id,
                )

            # Append these new points to the track_table arrays
            # track_table.append_track_table(
            #     track_2d=new_sampled_points,
            #     track_3d=new_points_3d,
            #     valid=valid_new_points_3d,
            #     uncertainties=0.5 * np.ones(len(new_sampled_points), dtype=float),
            #     visibles=np.ones(len(new_sampled_points), dtype=bool),
            # )

            obj.num_keyframes = len(self.keyframes[obj_id])
            obj.keyframes = self.keyframes[obj_id]

        return created_keyframes

    def _iter_sampling_reference_poses(self, obj_id: int):
        """Yield promoted and pending keyframe poses for sampling-coverage checks."""
        for i, kf in enumerate(self.keyframes.get(obj_id, [])):
            pose = getattr(kf, "pose", None)
            if pose is not None:
                yield "keyframe", i, np.asarray(pose, dtype=np.float64)

        for i, meta in enumerate(self.pending_keyframes.get(obj_id, [])):
            kf = meta.get("keyframe", None)
            pose = getattr(kf, "pose", None) if kf is not None else None
            if pose is not None:
                yield "pending_keyframe", i, np.asarray(pose, dtype=np.float64)

    def _is_pose_novel_for_sampling(self, obj_id: int, pose: np.ndarray):
        """
        Check whether a pose is sufficiently different from existing sampled views.

        Returns:
            (is_novel: bool, info: dict)
        """
        pose = np.asarray(pose, dtype=np.float64)
        best_info = None
        best_score = None

        rot_th = float(self.anchor_rot_deg)
        trans_th = float(self.anchor_trans)

        for source, index, ref_pose in self._iter_sampling_reference_poses(obj_id):
            rot_deg, trans_m = self._compute_abs_motion(ref_pose, pose)
            score = rot_deg + 1000.0 * trans_m

            if best_score is None or score < best_score:
                best_score = score
                best_info = {
                    "source": source,
                    "index": int(index),
                    "rot_deg": float(rot_deg),
                    "trans_m": float(trans_m),
                }

            if (rot_deg <= rot_th) and (trans_m <= trans_th):
                return False, {
                    "source": source,
                    "index": int(index),
                    "rot_deg": float(rot_deg),
                    "trans_m": float(trans_m),
                }

        if best_info is None:
            best_info = {
                "source": "none",
                "index": -1,
                "rot_deg": np.inf,
                "trans_m": np.inf,
            }
        return True, best_info

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _build_kp_from_track_table(self, track_table, track_ids, obj_pose):
        """
        Build the 'kp' dict from existing entries in track_table.
        """
        track_ids = np.asarray(track_ids, dtype=np.int64)
        uv = track_table.track_2d[track_ids]
        xyz_cam = track_table.track_3d[track_ids]
        valid = track_table.valid[track_ids]
        visible = track_table.visible[track_ids]
        uncertainties = track_table.uncertainty[track_ids]

        T_o2c = obj_pose
        xyz_obj = transform_pts(inverse_SE3(T_o2c), xyz_cam)

        return dict(
            track_ids=track_ids,
            uv=uv,
            xyz_cam=xyz_cam,
            xyz_obj=xyz_obj,
            valid=valid,
            visible=visible,
            uncertainties=uncertainties,
        )

    def _build_obs_for_frame(self, obj_id, frame, track_table, obj, anchor_pose):
        """
        Build the 'obs' dict for this object at this frame.
        For now, we take all points belonging to this object in the track_table.
        """
        # obj_track_ids list the indices of the points belonging to the object in the track_table
        obj_track_ids = np.asarray(track_table.obj2track_map[obj_id], dtype=np.int64)

        # if no points belonging to the object, return empty dict
        if obj_track_ids.size == 0:
            return dict(
                track_ids=np.zeros((0,), dtype=np.int64),
                uv=np.zeros((0, 2), dtype=float),
                xyz_cam=np.zeros((0, 3), dtype=float),
                xyz_obj=np.zeros((0, 3), dtype=float),
                valid=np.zeros((0,), dtype=bool),
            )

        # get the 2d, 3d, valid, visible, and uncertainties of the points belonging to the object
        uv = track_table.track_2d[obj_track_ids]
        xyz_cam = track_table.track_3d[obj_track_ids]
        valid = track_table.valid[obj_track_ids]
        visible = track_table.visible[obj_track_ids]
        uncertainties = track_table.uncertainty[obj_track_ids]

        # transform the 3d points to the object frame
        # T_o2c: transformation points from object frame to camera frame
        T_o2c = anchor_pose
        xyz_obj = transform_pts(inverse_SE3(T_o2c), xyz_cam)

        return dict(
            track_ids=obj_track_ids,
            uv=uv,
            xyz_cam=xyz_cam,
            xyz_obj=xyz_obj,
            valid=valid,
            visible=visible,
            uncertainties=uncertainties,
        )

    def _sample_for_all_obj(self, context, track_table, tracker):
        """
        Initial sampling: for each object, sample points using the sampler,
        add them to tracker, update obj2track_map, and update track_table.
        """
        all_sampled_points = np.empty((0, 2))
        frame = context.frame
        num_obj = int(self.num_obj)
        mask_count = self._infer_num_obj_from_mask(getattr(frame, "mask", None))
        if mask_count > 0:
            num_obj = mask_count if num_obj <= 0 else min(num_obj, mask_count)

        for obj_id in range(num_obj):
            new_sampled_points = self.sampler.sample(context, obj_id)
            if len(new_sampled_points) == 0:
                continue

            new_indices = tracker.add_query_points(frame, new_sampled_points)
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

            all_sampled_points = np.vstack((all_sampled_points, new_sampled_points))

        if all_sampled_points.shape[0] == 0:
            # Nothing sampled; leave track_table empty
            return

        track_3d, track_valid = convert_pixel_to_world(
            pixel=all_sampled_points,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            fill_missing_depth=self.fill_missing_depth,
            window_size=self.fill_depth_win_size,
            min_neighbors=self.fill_depth_min_neighbors,
        )

        track_table.update_track_table(
            track_2d=all_sampled_points,
            track_3d=track_3d,
            valid=track_valid,
            uncertainties=0.5 * np.ones(len(all_sampled_points), dtype=float),
            visibles=np.ones(len(all_sampled_points), dtype=bool),
        )

    def _save_key_points_for_object(self, obj_id, obj, frame_id):
        filename = f"obj_{obj_id}_keypoints_frame_{frame_id}.ply"
        save_path = os.path.join(self.key_points_save_path, filename)
        obj.save_key_points_with_colors(save_path, frame_id)

    def _create_keyframe(
        self,
        frame,
        obj,
        kp: dict,
        obs: dict,
        reg_stats: dict,
        anchor_pose: np.ndarray,
    ):
        """
        Build a KeyFrame from the packed kp / obs dicts and object state.
        """
        # get the frame id, object id, and keyframe index
        frame_id = frame.id
        obj_id = obj.id
        kf_idx = len(self.keyframes[obj_id])

        # extract the dense point cloud of the object
        dense_pts, dense_colors = self._extract_dense_pcd(frame, obj_id)

        metadata = {
            "kp_valid_mask": kp["valid"],
        }

        return KeyFrame(
            frame_id=frame_id,
            frame=frame,
            obj_id=obj_id,
            kf_idx=kf_idx,
            timestamp=getattr(frame, "timestamp", None),
            pose=anchor_pose.copy(),
            # new keypoints
            kp_track_indices=np.asarray(kp["track_ids"], dtype=np.int64),
            kp_2d=kp["uv"].copy(),
            kp_3d_camera=kp["xyz_cam"].copy(),
            kp_3d_object=kp["xyz_obj"].copy(),
            kp_valid=kp["valid"].copy(),
            # observed points
            obs_track_indices=np.asarray(obs["track_ids"], dtype=np.int64),
            obs_2d=obs["uv"].copy(),
            obs_3d_camera=obs["xyz_cam"].copy(),
            obs_3d_object=obs["xyz_obj"].copy(),
            obs_valid=obs["valid"].copy(),
            obs_visible=obs["visible"].copy(),
            obs_uncertainties=obs["uncertainties"].copy(),
            # registration stats
            reg_correspond_curr3d=reg_stats["correspond_curr3d"].copy(),
            reg_inliers=reg_stats["inliers"].copy(),
            reg_residuals=reg_stats["residuals"].copy(),
            reg_valid_idx=reg_stats["valid_idx"].copy(),
            # dense pcd
            dense_pts=dense_pts,
            dense_colors=dense_colors,
            metadata=metadata,
        )

    def _extract_dense_pcd(self, frame, obj_id: int):
        """
        Extract a dense point cloud of the object using the frame.mask[obj_id].
        """
        if frame.mask is None or frame.mask.shape[0] <= obj_id:
            return np.empty((0, 3)), None

        mask = frame.mask[obj_id, 0]
        coords_yx = torch.nonzero(mask > 0, as_tuple=False)
        if coords_yx.numel() == 0:
            return np.empty((0, 3)), None

        pxl_xy = coords_yx[:, [1, 0]].cpu().numpy()
        max_depth = self.pipeline_cfg.get("max_depth", 1.0)
        world_pts, valid = convert_pixel_to_world(
            pixel=pxl_xy,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
            max_depth=max_depth,
        )
        if world_pts.size == 0 or not np.any(valid):
            return np.empty((0, 3)), None

        world_pts = world_pts[valid]
        pxl_xy_valid = pxl_xy[valid]
        rgb_vals = frame.rgb[pxl_xy_valid[:, 1], pxl_xy_valid[:, 0]]

        # Optional outlier rejection
        if self.pipeline_cfg.get("stat_outlier_removal", False):
            nb_neighbors = self.pipeline_cfg.get(
                "stat_outlier_removal_nb_neighbors", 20
            )
            std_ratio = self.pipeline_cfg.get("stat_outlier_removal_std_ratio", 2.0)

            if world_pts.shape[0] > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(world_pts)
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, std_ratio=std_ratio
                )
                world_pts = world_pts[ind]
                rgb_vals = rgb_vals[ind]

        return world_pts, rgb_vals

    def get_keyframes(self, obj_id: int | None = None) -> List[KeyFrame]:
        """
        Returns:
            - all keyframes if obj_id is None
            - otherwise all keyframes for the given object
        """
        if obj_id is None:
            return [kf for kfs in self.keyframes.values() for kf in kfs]
        return list(self.keyframes.get(obj_id, []))

    def _pending_kf_age_for_ttl(self, meta: dict) -> int:
        if self.pending_kf_ttl_on_growth_frames:
            return int(meta.get("eval_age", 0))
        return int(meta.get("age", 0))

    def _collect_protected_track_ids_for_obj(self, obj_id: int) -> np.ndarray:
        protected = []

        pending = np.asarray(
            self.pending_track_ids.get(obj_id, []), dtype=np.int64
        ).reshape(-1)
        if pending.size > 0:
            protected.append(pending)

        promoted = np.asarray(
            [tid for (oid, tid) in self.promoted_meta.keys() if oid == obj_id],
            dtype=np.int64,
        ).reshape(-1)
        if promoted.size > 0:
            protected.append(promoted)

        for meta in self.pending_keyframes.get(obj_id, []):
            track_ids = np.asarray(meta.get("track_ids", []), dtype=np.int64).reshape(
                -1
            )
            if track_ids.size > 0:
                protected.append(track_ids)

        if self.retire_protect_keyframe_tracks:
            for kf in self.keyframes.get(obj_id, []):
                for attr in ("kp_track_indices", "obs_track_indices", "reg_valid_idx"):
                    track_ids = np.asarray(
                        getattr(kf, attr, np.zeros((0,), dtype=np.int64)),
                        dtype=np.int64,
                    ).reshape(-1)
                    if track_ids.size > 0:
                        protected.append(track_ids)

        if not protected:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(protected))

    def collect_retired_track_ids(self, objects) -> np.ndarray:
        if not self.retire_invalid_tracks:
            return np.zeros((0,), dtype=np.int64)

        retired = []
        for obj in objects:
            track_ids = np.asarray(obj.kp_track_indices, dtype=np.int64).reshape(-1)
            valid = np.asarray(obj.valid, dtype=bool).reshape(-1)
            if track_ids.size == 0 or valid.size != track_ids.size:
                continue

            invalid_track_ids = track_ids[~valid]
            if invalid_track_ids.size == 0:
                continue

            protected = self._collect_protected_track_ids_for_obj(obj.id)
            if protected.size > 0:
                invalid_track_ids = invalid_track_ids[
                    ~np.isin(invalid_track_ids, protected)
                ]
            if invalid_track_ids.size > 0:
                retired.append(invalid_track_ids)

        if not retired:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(retired))

    def _queue_pending_keyframe(
        self, obj_id: int, keyframe: KeyFrame, track_ids: np.ndarray, frame_id: int
    ):
        # kf_idx is assigned when promoted into the active keyframe list.
        keyframe.kf_idx = -1
        self.pending_keyframes[obj_id].append(
            {
                "keyframe": keyframe,
                "track_ids": np.asarray(track_ids, dtype=np.int64).reshape(-1),
                "birth_frame": int(frame_id),
                "age": 0,
                "eval_age": 0,
            }
        )

    def _pending_kf_support_counts(
        self, obj, track_ids: np.ndarray
    ) -> Tuple[int, int, float]:
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        n_total = int(track_ids.size)
        if n_total == 0:
            return 0, 0, 0.0

        rows = np.full((n_total,), -1, dtype=np.int64)
        in_map = (track_ids >= 0) & (track_ids < obj.track_idx_2_obj_idx.shape[0])
        if np.any(in_map):
            rows[in_map] = obj.track_idx_2_obj_idx[track_ids[in_map]]

        ok = (rows >= 0) & (rows < len(obj.valid))
        promoted = np.zeros((n_total,), dtype=bool)
        if np.any(ok):
            promoted[ok] = np.asarray(obj.valid, dtype=bool)[rows[ok]]

        n_promoted = int(np.count_nonzero(promoted))
        ratio = float(n_promoted / max(1, n_total))
        return n_promoted, n_total, ratio

    def _pending_kf_sdf_ok(self, obj, keyframe: KeyFrame) -> bool:
        if not self.pending_kf_sdf_gate:
            return True

        # If SDF has not been initialized yet, do not block promotion.
        if obj is None or (
            getattr(obj, "sdf", None) is None
            and getattr(obj, "sdf_volume", None) is None
        ):
            return True

        dense_pts = getattr(keyframe, "dense_pts", None)
        if dense_pts is None or dense_pts.shape[0] < int(self.sdf_kf_gate_min_dense):
            return False

        sdf_res = self._sdf_residual_dense(
            dense_pts=dense_pts,
            obj_pose=np.asarray(keyframe.pose, dtype=np.float64),
            obj=obj,
        )
        return bool(np.isfinite(sdf_res) and sdf_res <= self.pending_kf_sdf_gate_thres)

    def _update_pending_keyframes_for_obj(
        self,
        obj_id: int,
        obj,
        frame,
        front_end_result,
        allow_promotion: bool,
        created_keyframes: List[KeyFrame],
    ):
        pend = self.pending_keyframes.get(obj_id, [])
        if not pend:
            return

        reg_ok = (
            self._get_reg_quality(front_end_result, obj_id)
            if self.pending_kf_require_reg_quality
            else True
        )
        pose_ok = (
            self._pending_pose_stable(front_end_result, obj_id)
            if self.pending_kf_require_pose_stable
            else True
        )

        keep = []
        for meta in pend:
            meta["age"] = int(meta.get("age", 0)) + 1
            if allow_promotion:
                meta["eval_age"] = int(meta.get("eval_age", 0)) + 1

            track_ids = np.asarray(meta.get("track_ids", []), dtype=np.int64).reshape(
                -1
            )
            n_prom, n_total, ratio = self._pending_kf_support_counts(obj, track_ids)

            promote_ok = bool(
                allow_promotion
                and reg_ok
                and pose_ok
                and (n_prom >= int(self.pending_kf_min_promoted_count))
                and (ratio >= float(self.pending_kf_min_promoted_ratio))
                and self._pending_kf_sdf_ok(obj=obj, keyframe=meta["keyframe"])
            )

            if promote_ok:
                kf = meta["keyframe"]
                kf.kf_idx = len(self.keyframes[obj_id])
                self.keyframes[obj_id].append(kf)
                created_keyframes.append(kf)

                obj.last_keyframe_frame_id = kf.frame_id
                obj.last_keyframe = kf
                obj.num_keyframes = len(self.keyframes[obj_id])
                obj.keyframes = self.keyframes[obj_id]
                self.is_key_frame[obj_id] = True
                continue

            ttl_age = self._pending_kf_age_for_ttl(meta)
            if ttl_age > int(self.pending_kf_ttl):
                continue

            keep.append(meta)

        self.pending_keyframes[obj_id] = keep

    def _frame_allows_map_growth(
        self, obj_id: int, frame_id: int, fe_result
    ) -> Tuple[bool, str]:
        """
        Frame-level gate for map growth (sampling + pending promotion).

        Blocks growth on dense recovery frames, for a cooldown window after recovery,
        and on weak registration quality.
        """
        if not self.map_growth_gate:
            return True, "disabled"

        dense_triggered = False
        if hasattr(fe_result, "dense_recovery_triggered"):
            dense_triggered = bool(
                fe_result.dense_recovery_triggered.get(obj_id, False)
            )

        if dense_triggered:
            self.map_growth_cooldown_until[obj_id] = max(
                int(self.map_growth_cooldown_until.get(obj_id, -1)),
                int(frame_id) + int(self.map_growth_cooldown_after_recovery),
            )
            return False, "dense_recovery_triggered"

        cooldown_until = int(self.map_growth_cooldown_until.get(obj_id, -1))
        if int(frame_id) <= cooldown_until:
            return False, f"cooldown_until_{cooldown_until}"

        if not self._get_reg_quality(fe_result, obj_id):
            return False, "reg_quality_bad"

        mean_res = None
        if hasattr(fe_result, "mean_residuals"):
            mean_res = fe_result.mean_residuals.get(obj_id, None)
        if mean_res is not None:
            try:
                mean_res = float(mean_res)
            except Exception:
                mean_res = None
        if mean_res is not None and np.isfinite(mean_res):
            if mean_res > self.map_growth_max_mean_residual:
                return False, f"mean_residual_{mean_res:.4e}"

        return True, "ok"

    def _query_abs_sdf_points_obj(self, pts_obj: np.ndarray, obj):
        """
        Query |SDF| values for points already in object frame.
        Returns (abs_sdf_vals, support_ratio). support_ratio is the fraction of query
        points with valid / in-bounds SDF values.
        """
        pts_obj = np.asarray(pts_obj, dtype=np.float32).reshape(-1, 3)
        if pts_obj.shape[0] == 0:
            return np.empty((0,), dtype=np.float32), 0.0

        if obj is None:
            return np.empty((0,), dtype=np.float32), 0.0

        # nvblox / mapper query path
        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            try:
                qvals = obj.sdf_volume.query_sdf(pts_obj)
            except Exception:
                qvals = None
            if qvals is not None:
                qvals = np.asarray(qvals).reshape(-1)
                if qvals.shape[0] == pts_obj.shape[0]:
                    finite = np.isfinite(qvals)
                    support = float(np.mean(finite))
                    return np.abs(qvals[finite]).astype(np.float32), support

        # legacy dense TSDF path
        if getattr(obj, "sdf", None) is not None and ("tsdf" in obj.sdf):
            tsdf = obj.sdf["tsdf"]
            origin = np.asarray(obj.sdf["vol_origin"], dtype=np.float32)
            voxel = float(obj.sdf["voxel_size"])
            vol_dim = np.array(tsdf.shape, dtype=np.int32)
            vox = np.floor((pts_obj - origin[None, :]) / voxel).astype(np.int32)
            inb = np.logical_and(
                np.all(vox >= 0, axis=1), np.all(vox < vol_dim[None, :], axis=1)
            )
            support = float(np.mean(inb))
            if not np.any(inb):
                return np.empty((0,), dtype=np.float32), support
            vox_in = vox[inb]
            vals = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]]).astype(
                np.float32
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.empty((0,), dtype=np.float32), support
            return vals, support

        return np.empty((0,), dtype=np.float32), 0.0

    def _pending_sdf_ok(self, xyz_obj: np.ndarray, obj) -> bool:
        if not self.pending_sdf_gate:
            return True
        # If no SDF yet, do not block promotion.
        if obj is None or (
            getattr(obj, "sdf", None) is None
            and getattr(obj, "sdf_volume", None) is None
        ):
            return True

        vals, support = self._query_abs_sdf_points_obj(
            np.asarray(xyz_obj)[None, :], obj
        )
        if support < self.pending_sdf_min_support:
            return False
        if vals.size == 0:
            return False
        q = float(np.clip(self.pending_sdf_gate_percentile, 0.0, 100.0))
        score = float(np.percentile(vals, q))
        return bool(np.isfinite(score) and score <= self.pending_sdf_gate_thres)

    def _pending_obs_geom_ok(self, meta: dict) -> Tuple[bool, dict]:
        obs = np.asarray(meta.get("obs_obj", []), dtype=np.float32).reshape(-1, 3)
        if obs.shape[0] < self.pending_obs_min_count:
            return False, {"n_obs": int(obs.shape[0]), "spread": np.inf}

        med = np.median(obs, axis=0)
        d = np.linalg.norm(obs - med[None, :], axis=1)
        spread = float(np.median(d)) if d.size else np.inf
        if (not np.isfinite(spread)) or (spread > self.pending_obj_spread_thres):
            return False, {"n_obs": int(obs.shape[0]), "spread": spread}

        if self.pending_use_view_diversity:
            vdirs = np.asarray(meta.get("obs_view_dirs", []), dtype=np.float32).reshape(
                -1, 3
            )
            if vdirs.shape[0] >= 2:
                vnorm = np.linalg.norm(vdirs, axis=1, keepdims=True)
                good = vnorm[:, 0] > 1e-8
                vv = vdirs[good] / np.clip(vnorm[good], 1e-8, None)
                if vv.shape[0] >= 2:
                    c = np.clip(vv @ vv.T, -1.0, 1.0)
                    ang = np.degrees(np.arccos(c))
                    max_ang = float(np.max(ang))
                    if max_ang < self.pending_min_view_angle_deg:
                        return False, {
                            "n_obs": int(obs.shape[0]),
                            "spread": spread,
                            "max_view_angle_deg": max_ang,
                        }

        return True, {"n_obs": int(obs.shape[0]), "spread": spread}

    def _pending_fuse_obs(self, meta: dict) -> Tuple[np.ndarray, float]:
        obs = np.asarray(meta.get("obs_obj", []), dtype=np.float32).reshape(-1, 3)
        if obs.shape[0] == 0:
            return np.zeros(3, dtype=np.float32), np.inf
        xyz = np.median(obs, axis=0).astype(np.float32)
        d = np.linalg.norm(obs - xyz[None, :], axis=1)
        spread = float(np.median(d)) if d.size else 0.0
        return xyz, spread

    def _pending_age_for_ttl(self, meta: dict) -> int:
        if self.pending_ttl_on_growth_frames:
            return int(meta.get("eval_age", 0))
        return int(meta.get("age", 0))

    def _pending_mask_for_obj(self, frame, obj_id: int):
        if (
            self.pending_require_inside_mask
            and getattr(frame, "mask", None) is not None
            and frame.mask.shape[0] > obj_id
        ):
            return frame.mask[obj_id, 0]
        return None

    def _pending_pose_stable(self, front_end_result, obj_id: int) -> bool:
        rel_pose = front_end_result.rel_poses.get(obj_id)
        if rel_pose is None:
            return True
        rot_deg = float(np.degrees(scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()))
        trans = float(np.linalg.norm(rel_pose[:3, 3]))
        return (rot_deg < self.pending_stable_rot_deg) and (
            trans < self.pending_stable_trans
        )

    def _pending_point_inside_mask(self, uv: np.ndarray, mask) -> bool:
        if mask is None:
            return True
        if not np.isfinite(uv).all():
            return False
        H, W = int(mask.shape[0]), int(mask.shape[1])
        x = int(np.clip(np.rint(uv[0]), 0, W - 1))
        y = int(np.clip(np.rint(uv[1]), 0, H - 1))
        return bool(mask[y, x] > 0)

    def _pending_point_gate_good(
        self, track_table, tid: int, pose_stable: bool, mask
    ) -> bool:
        if not pose_stable:
            return False
        if not (bool(track_table.visible[tid]) and bool(track_table.valid[tid])):
            return False
        if float(track_table.uncertainty[tid]) >= self.pending_uncer_thres:
            return False
        return self._pending_point_inside_mask(track_table.track_2d[tid], mask)

    def _pending_mark_good(self, meta: dict):
        meta["good"] = int(meta.get("good", 0)) + 1
        meta["bad"] = 0

    def _pending_mark_bad(self, meta: dict):
        meta["bad"] = int(meta.get("bad", 0)) + 1
        meta["good"] = 0

    def _pending_cleanup(self, key: Tuple[int, int]):
        self.pending_birth_frame.pop(key, None)
        self.pending_meta.pop(key, None)

    def _pending_should_reject(self, meta: dict) -> bool:
        ttl_age = self._pending_age_for_ttl(meta)
        return (ttl_age > self.pending_ttl) or (
            int(meta.get("bad", 0)) > self.pending_max_bad
        )

    def _pending_collect_support_observation(
        self, meta: dict, obj, frame_id: int, xyz_cam: np.ndarray, T_c2o: np.ndarray
    ) -> bool:
        xyz_cam = np.asarray(xyz_cam, dtype=np.float32).reshape(3)
        if not np.isfinite(xyz_cam).all():
            return False

        xyz_obj = transform_pts(T_c2o, xyz_cam[None])[0].astype(np.float32)
        if not self._pending_sdf_ok(xyz_obj, obj):
            return False

        if int(meta.get("last_obs_frame", -1)) != frame_id:
            obs_obj = meta.setdefault("obs_obj", [])
            obs_frames = meta.setdefault("obs_frame_ids", [])
            obs_vdirs = meta.setdefault("obs_view_dirs", [])

            obs_obj.append(xyz_obj.copy())
            obs_frames.append(frame_id)

            cam_center_obj = T_c2o[:3, 3].astype(np.float32)
            v = cam_center_obj - xyz_obj
            n = float(np.linalg.norm(v))
            obs_vdirs.append((v / max(n, 1e-8)).astype(np.float32))

            max_keep = max(1, int(self.pending_obs_max_keep))
            if len(obs_obj) > max_keep:
                del obs_obj[:-max_keep]
                del obs_frames[:-max_keep]
                del obs_vdirs[:-max_keep]

            meta["last_obs_frame"] = frame_id

        return True

    def _pending_try_promote(self, key: Tuple[int, int], meta: dict, obj) -> bool:
        if int(meta.get("good", 0)) < int(self.pending_promote_streak):
            return False
        if self.pending_use_geom_check:
            geom_ok, _ = self._pending_obs_geom_ok(meta)
            if not geom_ok:
                return False

        obj_idx = int(meta.get("obj_idx", -1))
        if 0 <= obj_idx < len(obj.key_points):
            xyz_obj_fused, spread = self._pending_fuse_obs(meta)
            obj.key_points[obj_idx] = xyz_obj_fused
            obj.valid[obj_idx] = True

            if obj_idx < len(obj.uncertainties):
                target_unc = max(
                    float(self.promoted_min_uncer),
                    float(spread) * float(self.promoted_spread_to_uncer),
                )
                init_unc = max(float(self.promoted_init_uncer), target_unc)
                obj.uncertainties[obj_idx] = init_unc
                self.promoted_meta[key] = {
                    "obj_idx": int(obj_idx),
                    "age": 0,
                    "target_unc": float(target_unc),
                }

        self._pending_cleanup(key)
        return True

    def _pending_reject(self, key: Tuple[int, int], meta: dict, obj):
        obj_idx = int(meta.get("obj_idx", -1))
        if 0 <= obj_idx < len(obj.valid):
            obj.valid[obj_idx] = False
        self._pending_cleanup(key)

    def _update_recently_promoted_pts_for_obj(
        self, obj_id, obj, frame, track_table, allow_promotion: bool
    ):
        """
        Newly promoted points are valid but start with larger uncertainty, then anneal
        down over a few good frames. This reduces how much fresh points dominate f2m.
        """
        keys = [k for k in self.promoted_meta.keys() if k[0] == obj_id]
        if not keys:
            return

        mask = None
        if (
            self.pending_require_inside_mask
            and getattr(frame, "mask", None) is not None
        ):
            if frame.mask.shape[0] > obj_id:
                mask = frame.mask[obj_id, 0]

        for key in keys:
            _obj_id, tid = key
            meta = self.promoted_meta.get(key, None)
            if meta is None:
                continue
            obj_idx = int(meta.get("obj_idx", -1))
            if obj_idx < 0 or obj_idx >= len(obj.valid):
                self.promoted_meta.pop(key, None)
                continue
            if not bool(obj.valid[obj_idx]):
                self.promoted_meta.pop(key, None)
                continue

            # Determine whether this frame provides a reliable confirmation of the promoted point
            is_good = bool(allow_promotion)
            if tid >= len(track_table.valid):
                is_good = False
            else:
                is_good = (
                    is_good
                    and bool(track_table.visible[tid])
                    and bool(track_table.valid[tid])
                    and (float(track_table.uncertainty[tid]) < self.pending_uncer_thres)
                )
                if is_good and mask is not None:
                    uv = track_table.track_2d[tid]
                    if np.isfinite(uv).all():
                        H, W = int(mask.shape[0]), int(mask.shape[1])
                        x = int(np.clip(np.rint(uv[0]), 0, W - 1))
                        y = int(np.clip(np.rint(uv[1]), 0, H - 1))
                        is_good = bool(mask[y, x] > 0)
                    else:
                        is_good = False

            if is_good:
                meta["age"] = int(meta.get("age", 0)) + 1
                cur_unc = float(obj.uncertainties[obj_idx])
                target_unc = float(meta.get("target_unc", self.promoted_min_uncer))
                new_unc = max(target_unc, cur_unc * self.promoted_uncer_decay)
                obj.uncertainties[obj_idx] = new_unc
            # If not good, keep uncertainty high and just wait.

            if int(meta.get("age", 0)) >= self.promoted_warmup_frames:
                target_unc = float(meta.get("target_unc", self.promoted_min_uncer))
                obj.uncertainties[obj_idx] = max(
                    target_unc, float(obj.uncertainties[obj_idx])
                )
                self.promoted_meta.pop(key, None)

    def _update_pending_pts_for_obj(
        self,
        obj_id,
        obj,
        frame,
        track_table,
        front_end_result,
        allow_promotion: bool = True,
    ):
        """
        Promote/reject pending points for one object using multi-frame geometric
        consistency in object frame, optional point-wise SDF checks, and robust fusion.

        Pending points remain invalid until promoted; they never affect f2m until then.
        """
        # Update trust annealing for recently promoted points first.
        self._update_recently_promoted_pts_for_obj(
            obj_id=obj_id,
            obj=obj,
            frame=frame,
            track_table=track_table,
            allow_promotion=allow_promotion,
        )

        pend_list = self.pending_track_ids.get(obj_id, [])
        if not pend_list:
            return

        update_pending = bool(
            allow_promotion or self.pending_update_when_growth_blocked
        )
        pose_stable = self._pending_pose_stable(front_end_result, obj_id)
        mask = self._pending_mask_for_obj(frame, obj_id)
        T_c2o = inverse_SE3(obj.pose) if update_pending else None

        keep = []
        for tid in pend_list:
            tid = int(tid)
            key = (obj_id, tid)
            meta = self.pending_meta.get(key, None)
            if meta is None:
                continue

            meta["age"] = int(meta.get("age", 0)) + 1
            if update_pending:
                meta["eval_age"] = int(meta.get("eval_age", 0)) + 1

            if tid >= len(track_table.valid):
                self._pending_mark_bad(meta)
            elif update_pending:
                point_gate_good = self._pending_point_gate_good(
                    track_table=track_table,
                    tid=tid,
                    pose_stable=pose_stable,
                    mask=mask,
                )
                if point_gate_good and self._pending_collect_support_observation(
                    meta=meta,
                    obj=obj,
                    frame_id=int(frame.id),
                    xyz_cam=track_table.track_3d[tid],
                    T_c2o=T_c2o,
                ):
                    self._pending_mark_good(meta)
                else:
                    if self.pending_reset_on_bad == True:
                        self._pending_mark_bad(meta)
            # else: freeze pending updates when explicitly disabled.

            if self._pending_try_promote(key=key, meta=meta, obj=obj):
                continue
            if self._pending_should_reject(meta):
                self._pending_reject(key=key, meta=meta, obj=obj)
                continue

            keep.append(tid)

        self.pending_track_ids[obj_id] = keep

    def _reg_inlier_metrics(self, reg_stats: dict):
        inliers = reg_stats.get("inliers", None)
        residuals = reg_stats.get("residuals", None)
        if inliers is None:
            return 0, 0, 0.0, np.inf
        inliers = np.asarray(inliers).astype(bool)
        n_total = int(inliers.size)
        n_in = int(inliers.sum())
        ratio = float(n_in / max(1, n_total))

        med_res = np.inf
        if residuals is not None and n_in > 0:
            residuals = np.asarray(residuals).reshape(-1)
            med_res = float(np.median(residuals[inliers]))
        return n_in, n_total, ratio, med_res

    def _compute_pose_delta(self, rel_pose: np.ndarray):
        if rel_pose is None:
            return 0.0, 0.0
        rot_deg = float(np.degrees(scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()))
        trans = float(np.linalg.norm(rel_pose[:3, 3]))
        return rot_deg, trans

    def _compute_vel_delta(self, prev_rel_pose: np.ndarray, cur_rel_pose: np.ndarray):
        # compare deltas: inv(prev_delta) * cur_delta
        if prev_rel_pose is None or cur_rel_pose is None:
            return 0.0, 0.0
        delta = inverse_SE3(prev_rel_pose) @ cur_rel_pose
        return self._compute_pose_delta(delta)

    def _get_reg_quality(self, fe_result, obj_id: int):
        """
        Returns (ok: bool).
        Uses reg_stats[obj_id]["inliers"] and optional ["residuals"] if present.
        """
        reg_stats = fe_result.reg_stats.get(obj_id, None)
        if not reg_stats:
            return True  # if stats missing, don't block

        inliers = reg_stats.get("inliers", None)
        if inliers is None:
            return True

        inliers = np.asarray(inliers, dtype=bool).reshape(-1)
        n_tot = int(inliers.size)
        n_in = int(np.count_nonzero(inliers))
        ratio = float(n_in / max(1, n_tot))

        if n_in < self.anchor_min_inliers or ratio < self.anchor_min_inlier_ratio:
            return False

        residuals = reg_stats.get("residuals", None)
        if residuals is not None and n_in > 0:
            residuals = np.asarray(residuals, dtype=float).reshape(-1)
            med = float(np.median(residuals[inliers]))
            if med > self.anchor_max_median_res:
                return False

        return True

    def _compute_abs_motion(self, prev_pose: np.ndarray, cur_pose: np.ndarray):
        rel_pose = inverse_SE3(prev_pose) @ cur_pose
        rot_deg = float(np.degrees(scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()))
        trans = float(np.linalg.norm(rel_pose[:3, 3]))
        return rot_deg, trans

    def _velocity_consistent(self, hist_fe_results, obj_id: int, candidate_idx: int):
        """
        Check whether motion at candidate_idx is consistent with recent velocity history
        using consensus voting over neighboring motion samples.
        """
        if candidate_idx <= 0:
            return True

        # Candidate motion: frame (candidate_idx-1) -> frame candidate_idx
        prev_pose = hist_fe_results[candidate_idx - 1].obj_poses.get(obj_id, None)
        cur_pose = hist_fe_results[candidate_idx].obj_poses.get(obj_id, None)
        if prev_pose is None or cur_pose is None:
            return False

        cand_rot, cand_trans = self._compute_abs_motion(prev_pose, cur_pose)

        # Reference velocity window around candidate motion index
        n = len(hist_fe_results)
        start_motion_idx = max(1, candidate_idx - self.anchor_vel_hist_len)
        end_motion_idx = min(n - 1, candidate_idx + self.anchor_vel_hist_len)
        ref_motions = []
        for i in range(start_motion_idx, end_motion_idx + 1):
            if i == candidate_idx:
                continue
            p0 = hist_fe_results[i - 1].obj_poses.get(obj_id, None)
            p1 = hist_fe_results[i].obj_poses.get(obj_id, None)
            if p0 is None or p1 is None:
                continue
            r_deg, t_m = self._compute_abs_motion(p0, p1)
            ref_motions.append((r_deg, t_m))

        # Not enough history -> don't over-constrain
        n_refs = len(ref_motions)
        if n_refs == 0:
            return True

        votes = 0
        for ref_rot, ref_trans in ref_motions:
            rot_ok = abs(cand_rot - ref_rot) <= self.anchor_vel_rot_deg
            trans_ok = abs(cand_trans - ref_trans) <= self.anchor_vel_trans
            if rot_ok and trans_ok:
                votes += 1

        min_votes = max(
            int(self.anchor_vel_min_votes),
            int(np.ceil(float(self.anchor_vel_vote_ratio) * float(n_refs))),
        )
        return votes >= min_votes

    def _choose_sampling_anchor(
        self,
        hist_fe_results,
        hist_frames,
        hist_track_tables,
        cur_fe_result,
        obj_id: int,
        obj,
    ):
        """
        Returns (anchor_frame, anchor_pose, anchor_is_current).
        anchor_pose is T_o2c at that anchor frame.
        """
        cur_frame = hist_frames[-1]
        n = min(len(hist_fe_results), len(hist_frames))
        if n == 0:
            return cur_frame, obj.pose.copy(), cur_fe_result, True

        # Search from newest to oldest: first valid hit is the closest consistent frame.
        for idx in range(n - 1, -1, -1):
            fe_res = hist_fe_results[idx]
            pose = fe_res.obj_poses.get(obj_id, None)
            if pose is None:
                continue

            reg_ok = self._get_reg_quality(fe_res, obj_id)
            vel_ok = self._velocity_consistent(hist_fe_results, obj_id, idx)
            if not (reg_ok and vel_ok):
                continue

            anchor_frame = hist_frames[idx]
            anchor_pose = pose.copy()
            anchor_fe_result = hist_fe_results[idx]
            anchor_track_table = hist_track_tables[idx]
            is_current = int(anchor_frame.id) == int(cur_frame.id)

            self.last_consistent[obj_id] = {
                "frame_id": int(anchor_frame.id),
                "pose": anchor_pose.copy(),
                "fe_result": anchor_fe_result,
            }
            return (
                anchor_frame,
                anchor_pose,
                anchor_fe_result,
                anchor_track_table,
                is_current,
            )

        # fallback to last remembered consistent frame
        # rec = self.last_consistent.get(obj_id, None)
        # if rec is not None:
        #     anchor_id = int(rec["frame_id"])
        #     for fr in reversed(hist_frames):
        #         if int(fr.id) == anchor_id:
        #             return (
        #                 fr,
        #                 rec["pose"].copy(),
        #                 rec["fe_result"],
        #                 rec["track_table"],
        #                 int(fr.id) == int(cur_frame.id),
        #             )

        # no consistent candidate available
        self.last_consistent[obj_id] = {
            "frame_id": int(cur_frame.id),
            "pose": obj.pose.copy(),
            "fe_result": cur_fe_result,
        }
        return cur_frame, obj.pose.copy(), cur_fe_result, hist_track_tables[-1], True

    def _sdf_residual_dense(self, dense_pts, obj_pose, obj):
        if (
            obj is None
            or getattr(obj, "sdf", None) is None
            or dense_pts is None
            or dense_pts.shape[0] < self.sdf_kf_gate_min_dense
        ):
            return np.inf

        pts_obj = transform_pts(inverse_SE3(obj_pose), dense_pts)
        sdf_vals = None

        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            qvals = obj.sdf_volume.query_sdf(pts_obj)
            if qvals is not None and qvals.shape[0] == pts_obj.shape[0]:
                sdf_vals = np.abs(qvals[np.isfinite(qvals)])

        if (sdf_vals is None or sdf_vals.size == 0) and "tsdf" in obj.sdf:
            tsdf = obj.sdf["tsdf"]
            origin = obj.sdf["vol_origin"]
            voxel = float(obj.sdf["voxel_size"])
            vol_dim = np.array(tsdf.shape, dtype=np.int32)
            vox = np.floor((pts_obj - origin[None, :]) / voxel).astype(np.int32)
            inb = np.logical_and(
                np.all(vox >= 0, axis=1), np.all(vox < vol_dim[None, :], axis=1)
            )
            if not np.any(inb):
                return np.inf
            vox_in = vox[inb]
            sdf_vals = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]])

        if sdf_vals is None or sdf_vals.size == 0:
            return np.inf

        q = float(np.clip(self.sdf_kf_gate_percentile, 0.0, 100.0))
        return float(np.percentile(sdf_vals, q))
