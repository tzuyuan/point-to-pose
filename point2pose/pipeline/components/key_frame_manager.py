import os
from collections import defaultdict
from typing import List, Dict

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

        self.num_obj = self.pipeline_cfg.get("max_num_obj", 1)
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

        # Pose stability gate for promotion (tighter than your max_rel_rotation_deg)
        self.pending_stable_rot_deg = self.pipeline_cfg.get(
            "pending_stable_rot_deg", 2.0
        )
        self.pending_stable_trans = self.pipeline_cfg.get(
            "pending_stable_trans", 0.01
        )  # meters

        # Optional: if True, require point's UV to lie inside current mask to count as "good"
        self.pending_require_inside_mask = self.pipeline_cfg.get(
            "pending_require_inside_mask", True
        )

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

            # self._update_pending_pts_for_obj(
            #     obj_id, obj, frame, track_table, front_end_result
            # )

            # Large rotation jump
            rel_pose = cur_fe_result.rel_poses.get(obj_id)
            if rel_pose is not None:
                rot_magnitude = scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()
                angle_deg = np.degrees(rot_magnitude)

                if angle_deg > self.max_rel_rotation_deg:
                    print(
                        f"[KeyFrameManager] Frame {cur_frame.id}: Large rotation {angle_deg:.2f} deg "
                        f"detected for obj {obj_id}. Skipping sampling."
                    )
                    continue

            # Decide whether this frame is a keyframe for this object
            if not self.criterion.check_sample_criterion(self.crit_ctx, obj_id):
                continue

            if self.kf_gating:
                anchor_frame, anchor_pose, anchor_fe_result, anchor_is_current = (
                    self._choose_sampling_anchor(
                        hist_fe_results=hist_fe_results,
                        hist_frames=hist_frames,
                        cur_fe_result=cur_fe_result,
                        obj_id=obj_id,
                        obj=obj,
                    )
                )
                if not anchor_is_current:
                    continue
            else:
                anchor_frame = cur_frame
                anchor_pose = obj.pose
                anchor_fe_result = cur_fe_result
                anchor_is_current = True

            # Build a sampler context from the chosen frame
            if anchor_is_current:
                samp_ctx = self.samp_ctx  # already updated with cur_frame above
            else:
                tmp_samp_ctx = SamplerContext(
                    frame=None, min_depth=self.min_depth, max_depth=self.max_depth
                )
                tmp_samp_ctx.update_sampler_context(
                    frame=anchor_frame, track_table=track_table
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
                fill_missing_depth=False,
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
                    }

                # this is NOT a committed keyframe
                self.is_key_frame[obj_id] = False

                track_table.append_track_table(
                    track_2d=new_sampled_points,
                    track_3d=new_points_3d,
                    valid=valid_new_points_3d,
                    uncertainties=0.5 * np.ones(len(new_sampled_points), dtype=float),
                    visibles=np.ones(len(new_sampled_points), dtype=bool),
                )
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
                    track_table=track_table,
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

        for obj_id in range(self.num_obj):
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

    def _update_pending_pts_for_obj(
        self, obj_id, obj, frame, track_table, front_end_result
    ):
        """
        Promote/reject pending points for one object.

        Promotion:
        - point is "good" for K consecutive frames
        - then: set obj.valid[obj_idx] = True and overwrite obj.key_points[obj_idx]
                using the CURRENT pose and current xyz_cam.

        Rejection:
        - too old (TTL) or too many consecutive bad frames
        - then: leave obj.valid[obj_idx] = False forever (map ignores it).
        """
        pend_list = self.pending_track_ids.get(obj_id, [])
        if not pend_list:
            return

        # --- pose stability gate
        pose_stable = True
        rel_pose = front_end_result.rel_poses.get(obj_id)
        if rel_pose is not None:
            rot_deg = float(
                np.degrees(scipy_R.from_matrix(rel_pose[:3, :3]).magnitude())
            )
            trans = float(np.linalg.norm(rel_pose[:3, 3]))
            pose_stable = (rot_deg < self.pending_stable_rot_deg) and (
                trans < self.pending_stable_trans
            )

        # --- optional mask for inside check
        mask = None
        if (
            self.pending_require_inside_mask
            and getattr(frame, "mask", None) is not None
        ):
            if frame.mask.shape[0] > obj_id:
                mask = frame.mask[obj_id, 0]  # (H,W) torch/bool-like

        keep = []
        for tid in pend_list:
            key = (obj_id, int(tid))
            meta = self.pending_meta.get(key, None)
            if meta is None:
                continue

            meta["age"] += 1

            # Read track state
            vis = bool(track_table.visible[tid])
            vld = bool(track_table.valid[tid])
            uncer = float(track_table.uncertainty[tid])

            inside = True
            if mask is not None:
                uv = track_table.track_2d[tid]
                if np.isfinite(uv).all():
                    H, W = int(mask.shape[0]), int(mask.shape[1])
                    x = int(np.clip(np.rint(uv[0]), 0, W - 1))
                    y = int(np.clip(np.rint(uv[1]), 0, H - 1))
                    inside = bool(mask[y, x] > 0)
                else:
                    inside = False

            good = (
                pose_stable
                and vis
                and vld
                and inside
                and (uncer < self.pending_uncer_thres)
            )

            if good:
                meta["good"] += 1
                meta["bad"] = 0
            else:
                meta["bad"] += 1
                meta["good"] = 0

            # --- promote
            if meta["good"] >= self.pending_promote_streak:
                obj_idx = meta["obj_idx"]

                xyz_cam = track_table.track_3d[tid]
                # overwrite object-frame coordinate using CURRENT pose (important!)
                xyz_obj = transform_pts(inverse_SE3(obj.pose), xyz_cam[None])[0]

                obj.key_points[obj_idx] = xyz_obj
                obj.valid[obj_idx] = True

                # cleanup
                self.pending_birth_frame.pop(key, None)
                self.pending_meta.pop(key, None)
                continue

            # --- reject
            if meta["age"] > self.pending_ttl or meta["bad"] > self.pending_max_bad:
                obj_idx = meta["obj_idx"]
                obj.valid[obj_idx] = False  # stays inactive for f2m forever

                self.pending_birth_frame.pop(key, None)
                self.pending_meta.pop(key, None)
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
        self, hist_fe_results, hist_frames, cur_fe_result, obj_id: int, obj
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
            is_current = int(anchor_frame.id) == int(cur_frame.id)

            self.last_consistent[obj_id] = {
                "frame_id": int(anchor_frame.id),
                "pose": anchor_pose.copy(),
                "fe_result": anchor_fe_result,
            }
            return anchor_frame, anchor_pose, anchor_fe_result, is_current

        # fallback to last remembered consistent frame
        rec = self.last_consistent.get(obj_id, None)
        if rec is not None:
            anchor_id = int(rec["frame_id"])
            for fr in reversed(hist_frames):
                if int(fr.id) == anchor_id:
                    return (
                        fr,
                        rec["pose"].copy(),
                        rec["fe_result"],
                        int(fr.id) == int(cur_frame.id),
                    )

        # no consistent candidate available
        self.last_consistent[obj_id] = {
            "frame_id": int(cur_frame.id),
            "pose": obj.pose.copy(),
            "fe_result": cur_fe_result,
        }
        return cur_frame, obj.pose.copy(), cur_fe_result, True
