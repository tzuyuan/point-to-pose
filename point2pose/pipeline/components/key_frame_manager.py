import os
from collections import defaultdict
from typing import List

import numpy as np
import torch

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
        self.key_points_save_path = self.pipeline_cfg.get(
            "key_points_save_path", "./key_points"
        )

        if self.save_key_points:
            os.makedirs(self.key_points_save_path, exist_ok=True)

        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)

        self.crit_ctx = CriterionContext()
        self.samp_ctx = SamplerContext(frame=None)

        self.is_key_frame = {}  # Dictionary to track keyframe status per object
        self.keyframes = defaultdict(list)  # obj_id -> List[KeyFrame]

    def initialize(self, frame, track_table, objects, tracker):
        """
        Initial sampling for the first frame.
        """
        self.samp_ctx.frame = frame
        self.samp_ctx.update_sampler_context(frame=frame, track_table=track_table)

        # Sample for all objects
        self._sample_for_all_obj(self.samp_ctx, track_table, tracker)

        # Initialize objects keypoints from the sampled tracks
        for obj_id in range(self.num_obj):
            if obj_id < len(objects):
                # assign the key points to the object
                objects[obj_id].key_points = track_table.track_3d[
                    track_table.obj2track_map[obj_id]
                ]
                objects[obj_id].valid = track_table.valid[
                    track_table.obj2track_map[obj_id]
                ]
                # Initialize frame tracking for initial key points
                objects[obj_id].key_point_frames = np.full(
                    len(track_table.obj2track_map[obj_id]), 0, dtype=int
                )
                self.is_key_frame[obj_id] = False

        self.crit_ctx.objects = objects
        self.criterion.initialize(self.crit_ctx)

    def update(
        self, frame, front_end_result: FrontEndResult, track_table, objects, tracker
    ) -> List[KeyFrame]:
        """
        Check criteria and sample new keyframes if needed.
        Returns a list of KeyFrame created in this update.
        """
        # Update contexts
        self.crit_ctx.update_criterion_context(
            cur_iter=front_end_result.frame_id,
            uncertainty=front_end_result.uncertainties,
            reg_stats=front_end_result.reg_stats.get(
                0, {}
            ),  # TODO: handle multi-object stats better in context
            track_table=track_table,
            frame=frame,
        )
        self.samp_ctx.update_sampler_context(frame=frame, track_table=track_table)

        # Reset key frame flag
        created_keyframes: List[KeyFrame] = []

        # Check and sample
        for obj_id in range(min(self.num_obj, len(objects))):
            self.is_key_frame[obj_id] = False

            # Check criterion
            if self.criterion.check_sample_criterion(self.crit_ctx, obj_id):
                self.is_key_frame[obj_id] = True
                # Sample points
                new_sampled_points = self.sampler.sample(self.samp_ctx, obj_id)

                if len(new_sampled_points) > 0:
                    # Add to tracker
                    new_indices = tracker.add_query_points(frame, new_sampled_points)

                    # Update track table maps
                    track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

                    # Get 3D points
                    new_points_3d, valid_new_points_3d = convert_pixel_to_world(
                        pixel=new_sampled_points,
                        depth_image=frame.depth,
                        cam_intrinsics=frame.intrinsics,
                        depth_factor=frame.depth_factor,
                    )

                    # Transform to object frame (frame 0 coord system for key points)
                    # We use the *current* estimated pose from FrontEnd result (or object)
                    # Since objects[obj_id].pose should have been updated by FrontEnd step already
                    current_pose = objects[obj_id].pose

                    new_points_3d_frame_0 = transform_pts(
                        inverse_SE3(current_pose),
                        new_points_3d,
                    )

                    # Add new key points to object
                    objects[obj_id].add_key_points(
                        new_points_3d_frame_0,
                        0.5
                        * np.ones(
                            len(new_points_3d_frame_0), dtype=float
                        ),  # Initial uncertainty
                        valid_new_points_3d,
                        front_end_result.frame_id,
                    )

                    # Save debug ply
                    if self.save_key_points:
                        self._save_key_points_for_object(
                            obj_id, objects[obj_id], front_end_result.frame_id
                        )

                    # Update track table with new points data
                    track_table.append_track_table(
                        track_2d=new_sampled_points,
                        track_3d=new_points_3d,
                        valid=valid_new_points_3d,
                        uncertainties=0.5
                        * np.ones(len(new_sampled_points), dtype=float),
                        visibles=np.ones(len(new_sampled_points), dtype=bool),
                    )

                    keyframe = self._create_keyframe_data(
                        frame=frame,
                        obj=objects[obj_id],
                        obj_id=obj_id,
                        frame_id=front_end_result.frame_id,
                        new_sampled_points=new_sampled_points,
                        new_points_3d=new_points_3d,
                        new_points_3d_obj=new_points_3d_frame_0,
                        valid_mask=valid_new_points_3d,
                        track_indices=new_indices,
                        reg_stats=front_end_result.reg_stats.get(obj_id, {}),
                    )
                    self.keyframes[obj_id].append(keyframe)
                    created_keyframes.append(keyframe)

        return created_keyframes

    def _sample_for_all_obj(self, context, track_table, tracker):
        all_sampled_points = np.empty((0, 2))
        frame = context.frame

        for obj_id in range(self.num_obj):
            new_sampled_points = self.sampler.sample(context, obj_id)
            all_sampled_points = np.vstack((all_sampled_points, new_sampled_points))

            new_indices = tracker.add_query_points(frame, new_sampled_points)
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

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

    def _create_keyframe_data(
        self,
        frame,
        obj,
        obj_id: int,
        frame_id: int,
        new_sampled_points: np.ndarray,
        new_points_3d: np.ndarray,
        new_points_3d_obj: np.ndarray,
        valid_mask: np.ndarray,
        track_indices: np.ndarray,
        reg_stats: dict,
    ) -> KeyFrame:
        dense_pts, dense_colors = self._extract_dense_pcd(frame, obj_id)

        metadata = {
            "reg_stats": reg_stats,
            "keyframe_valid_mask": valid_mask,
        }

        return KeyFrame(
            frame_id=frame_id,
            obj_id=obj_id,
            timestamp=getattr(frame, "timestamp", None),
            pose=obj.pose.copy(),
            keypoint_track_indices=np.asarray(track_indices, dtype=np.int64),
            keypoints_2d=new_sampled_points.copy(),
            keypoints_3d_camera=new_points_3d.copy(),
            keypoints_3d_object=new_points_3d_obj.copy(),
            keypoints_valid_mask=valid_mask.copy(),
            dense_pcd_points=dense_pts,
            dense_pcd_colors=dense_colors,
            metadata=metadata,
        )

    def _extract_dense_pcd(self, frame, obj_id: int):
        if frame.mask is None or frame.mask.shape[0] <= obj_id:
            return np.empty((0, 3)), None

        mask = frame.mask[obj_id, 0]
        coords_yx = torch.nonzero(mask > 0, as_tuple=False)
        if coords_yx.numel() == 0:
            return np.empty((0, 3)), None

        pxl_xy = coords_yx[:, [1, 0]].cpu().numpy()
        world_pts, valid = convert_pixel_to_world(
            pixel=pxl_xy,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
        )
        if world_pts.size == 0 or not np.any(valid):
            return np.empty((0, 3)), None

        world_pts = world_pts[valid]
        pxl_xy_valid = pxl_xy[valid]
        rgb_vals = frame.rgb[pxl_xy_valid[:, 1], pxl_xy_valid[:, 0]]
        return world_pts, rgb_vals

    def get_keyframes(self, obj_id: int | None = None) -> List[KeyFrame]:
        if obj_id is None:
            return [kf for kfs in self.keyframes.values() for kf in kfs]
        return list(self.keyframes.get(obj_id, []))
