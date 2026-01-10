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
        self.key_points_save_path = self.pipeline_cfg.get(
            "key_points_save_path", "./key_points"
        )

        if self.save_key_points:
            os.makedirs(self.key_points_save_path, exist_ok=True)

        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)

        self.crit_ctx = CriterionContext()
        self.samp_ctx = SamplerContext(frame=None)

        # Per-object bookkeeping
        self.is_key_frame: Dict[int, bool] = {}  # obj_id -> bool
        self.keyframes: Dict[int, List[KeyFrame]] = defaultdict(
            list
        )  # obj_id -> List[KeyFrame]

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
            obj_track_ids = np.asarray(
                track_table.obj2track_map[obj_id], dtype=np.int64
            )
            if obj_track_ids.size == 0:
                self.is_key_frame[obj_id] = False
                continue

            # Assign keypoints to object
            obj.key_points = track_table.track_3d[obj_track_ids]
            obj.key_point_indices = obj_track_ids
            obj.valid = track_table.valid[obj_track_ids]
            obj.key_point_frames = np.full(obj_track_ids.shape[0], 0, dtype=int)

            self.is_key_frame[obj_id] = False

            # Build "new keypoints" struct from these initial points
            kp = self._build_kp_from_track_table(
                track_table=track_table,
                track_ids=obj_track_ids,
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
            )
            self.keyframes[obj_id].append(kf)

            created_keyframes.append(kf)

            obj.last_keyframe_frame_id = kf.frame_id
            obj.num_keyframes = 1

        self.crit_ctx.objects = objects
        self.criterion.initialize(self.crit_ctx)

        return created_keyframes

    # -------------------------------------------------------------------------
    # Per-frame update
    # -------------------------------------------------------------------------
    def update(
        self, frame, front_end_result: FrontEndResult, track_table, objects, tracker
    ) -> List[KeyFrame]:
        """
        Check criteria and sample new keyframes if needed.
        Returns a list of KeyFrame created in this update.
        """
        # Update contexts for criterion + sampler
        self.crit_ctx.update_criterion_context(
            cur_iter=front_end_result.frame_id,
            uncertainty=front_end_result.uncertainties,
            reg_stats=front_end_result.reg_stats,  # TODO: extend to multi-object stats if needed
            track_table=track_table,
            frame=frame,
        )
        self.samp_ctx.update_sampler_context(frame=frame, track_table=track_table)

        created_keyframes: List[KeyFrame] = []

        for obj_id in range(min(self.num_obj, len(objects))):
            obj = objects[obj_id]
            self.is_key_frame[obj_id] = False

            # Check 1: If object is lost, avoid sampling
            if getattr(obj, "lost", False):
                continue

            # Check 2: Large rotation jump
            rel_pose = front_end_result.rel_poses.get(obj_id)
            if rel_pose is not None:
                rot_magnitude = scipy_R.from_matrix(rel_pose[:3, :3]).magnitude()
                angle_deg = np.degrees(rot_magnitude)

                if angle_deg > self.max_rel_rotation_deg:
                    print(
                        f"[KeyFrameManager] Frame {frame.id}: Large rotation {angle_deg:.2f} deg "
                        f"detected for obj {obj_id}. Skipping sampling."
                    )
                    continue

            # Decide whether this frame is a keyframe for this object
            if not self.criterion.check_sample_criterion(self.crit_ctx, obj_id):
                continue

            # This object triggers a keyframe
            self.is_key_frame[obj_id] = True

            # 1) Sample new keypoints for this object
            new_sampled_points = self.sampler.sample(self.samp_ctx, obj_id)
            if len(new_sampled_points) == 0:
                continue

            # Add to tracker and track_table
            new_indices = tracker.add_query_points(frame, new_sampled_points)
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

            # Lift to 3D
            new_points_3d, valid_new_points_3d = convert_pixel_to_world(
                pixel=new_sampled_points,
                depth_image=frame.depth,
                cam_intrinsics=frame.intrinsics,
                depth_factor=frame.depth_factor,
            )

            # Transform to object frame-0 coordinate (using current object pose)
            T_co = obj.pose  # depending on your convention, this may be T_0c or T_wc
            new_points_3d_obj = transform_pts(
                inverse_SE3(T_co),
                new_points_3d,
            )

            # Update object with these new keypoints
            obj.add_key_points(
                new_points_3d_obj,
                0.5
                * np.ones(len(new_points_3d_obj), dtype=float),  # initial uncertainty
                valid_new_points_3d,
                new_indices,
                front_end_result.frame_id,
            )

            # Optionally save debug ply of keypoints
            if self.save_key_points:
                self._save_key_points_for_object(
                    obj_id=obj_id,
                    obj=obj,
                    frame_id=front_end_result.frame_id,
                )

            # Append these new points to the track_table arrays
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

            # 3) Build observed points for this frame (could be richer than just `kp`)
            obs = self._build_obs_for_frame(
                obj_id=obj_id,
                frame=frame,
                track_table=track_table,
                obj=obj,
            )

            # 4) Registration stats for this object in this frame (if any)
            reg_stats = front_end_result.reg_stats.get(obj_id, {})

            # 5) Create and store the keyframe
            kf = self._create_keyframe(
                frame=frame,
                obj=obj,
                kp=kp,
                obs=obs,
                reg_stats=reg_stats,
            )
            self.keyframes[obj_id].append(kf)
            created_keyframes.append(kf)

            obj.last_keyframe_frame_id = kf.frame_id
            obj.num_keyframes = len(self.keyframes[obj_id])

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

    def _build_obs_for_frame(self, obj_id, frame, track_table, obj):
        """
        Build the 'obs' dict for this object at this frame.
        For now, we take all points belonging to this object in the track_table.
        You can refine this to only take visible / valid / recently observed points.
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
        T_o2c = obj.pose
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
            obj_id=obj_id,
            kf_idx=kf_idx,
            timestamp=getattr(frame, "timestamp", None),
            pose=obj.pose.copy(),
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
