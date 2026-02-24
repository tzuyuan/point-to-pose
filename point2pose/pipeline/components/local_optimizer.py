import numpy as np
from point2pose.core.base_optimizer import Optimizer
from point2pose.modules.optimizer.isam2_optimizer import ISAM2Optimizer
from point2pose.modules.optimizer.lm_optimizer import LMGraphOptimizer
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.data_types.optimizer_result import OptimizerResult

import point2pose.modules as _modules  # trigger registrations
from point2pose.core.module_registry import OPTIMIZER
from point2pose.core.build import build_from_cfg


class LocalOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg
        # One ISAM2 optimizer per object id
        self._optimizers = {}  # obj_id -> ISAM2Optimizer

    def _get_optimizer(self, obj_id: int) -> Optimizer:
        """Lazy-create an optimizer for this object."""
        if obj_id not in self._optimizers:
            self._optimizers[obj_id] = build_from_cfg(
                self.cfg.local_optimizer, OPTIMIZER
            )
        return self._optimizers[obj_id]

    def get_num_frames(self, obj_id: int) -> int:
        """Get the number of frames in the local graph for this object."""
        return self._get_optimizer(obj_id).get_num_poses()

    def reset(self, obj_id: int | None = None):
        """
        Reset optimizer state.

        - If obj_id is None: reset all objects.
        - If obj_id is given: reset only that object's optimizer.
        """
        if obj_id is None:
            self._optimizers = {}
        else:
            self._optimizers[obj_id] = build_from_cfg(
                self.cfg.local_optimizer, OPTIMIZER
            )

    def optimize(self, object_frame_data: ObjectFrameData) -> OptimizerResult | None:
        """
        Run optimization for the corresponding object.
        """
        print("-----------local optimizer------------")
        opt = self._get_optimizer(object_frame_data.obj_id)

        print("------------------------------------")
        return opt.optimize(object_frame_data)

    def update_object_state(self, obj, opt_result: OptimizerResult, track_table):
        """
        Apply optimization results to the Object.
        """
        if opt_result is None:
            return

        # Vectorized remapping from track IDs -> object-local keypoint indices
        # track_ids_opt = np.asarray(opt_result.key_points_idx_optimized, dtype=int)
        # pts_opt = np.asarray(opt_result.key_points_optimized, dtype=float)

        # obj_track_ids = np.asarray(track_table.obj2track_map[obj.id], dtype=int)

        # Build vectorized lookup:
        # For each optimized track_id, find its index in obj_track_ids.
        # matches = obj_track_ids[:, None] == track_ids_opt[None, :]

        # Convert boolean matrix to indices (rows where match=True)
        # local_kp_idx, opt_col_idx = np.where(matches)

        # # Select the optimized points that correspond to matched columns
        # pts_to_update = pts_opt[opt_col_idx]

        # Update object keypoints in one shot
        # if len(local_kp_idx) > 0:
        #     obj.key_points[local_kp_idx] = pts_to_update

        # Update pose
        # before = obj.pose.copy()
        obj.pose = opt_result.pose_optimized.copy()

        lm_global_track_ids = opt_result.key_points_idx_optimized
        lm_idx = obj.track_idx_2_obj_idx[lm_global_track_ids]
        lm_pts = opt_result.key_points_optimized
        obj.key_points[lm_idx] = lm_pts

        # delta = np.linalg.norm(before[:3, 3] - obj.pose[:3, 3])
        # print(f"[LocalOptimizer] obj {obj.id}, frame {opt_result.frame_id}, Δt = {delta:.4f} m")
