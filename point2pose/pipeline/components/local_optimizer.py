import numpy as np
from point2pose.modules.optimizer.isam2_optimizer import ISAM2Optimizer
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.data_types.optimizer_result import OptimizerResult


class LocalOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.optimizer = ISAM2Optimizer(cfg.optimizer)

    def reset(self):
        """
        Resets the underlying optimizer state (e.g. when a new keyframe is added).
        """
        self.optimizer = ISAM2Optimizer(self.cfg.optimizer)

    def optimize(self, object_frame_data: ObjectFrameData) -> OptimizerResult:
        """
        Run the optimization.
        """
        return self.optimizer.optimize(object_frame_data)

    def update_object_state(self, obj, opt_result: OptimizerResult, track_table):
        """
        Apply optimization results to the Object.
        """
        if opt_result is None:
            return

        # Vectorized remapping from track IDs -> object-local keypoint indices
        track_ids_opt = np.asarray(opt_result.key_points_idx_optimized, dtype=int)
        pts_opt = np.asarray(opt_result.key_points_optimized, dtype=float)

        obj_track_ids = np.asarray(track_table.obj2track_map[obj.id], dtype=int)

        # Build vectorized lookup:
        # For each optimized track_id, find its index in obj_track_ids.
        matches = obj_track_ids[:, None] == track_ids_opt[None, :]

        # Convert boolean matrix to indices (rows where match=True)
        local_kp_idx, opt_col_idx = np.where(matches)

        # Select the optimized points that correspond to matched columns
        pts_to_update = pts_opt[opt_col_idx]

        # Update object keypoints in one shot
        if len(local_kp_idx) > 0:
            obj.key_points[local_kp_idx] = pts_to_update

        # Update pose
        obj.pose = opt_result.pose_optimized
