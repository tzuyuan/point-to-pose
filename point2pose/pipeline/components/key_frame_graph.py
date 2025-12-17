from typing import Dict, List, Tuple

import numpy as np

from point2pose.core.base_optimizer import Optimizer
from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import OPTIMIZER
from point2pose.data_types.key_frame import KeyFrame
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.utils.transform import inverse_SE3


class KeyFrameGraph:
    """
    Global keyframe graph wrapper.

    This class mirrors LocalOptimizer, but operates on KeyFrame objects instead
    of per-frame ObjectFrameData.

    - One underlying Optimizer (ISAM2Optimizer or LMGraphOptimizer) per object id,
      constructed from cfg.global_optimizer via build_from_cfg.
    - Each keyframe is treated as a "frame" in the optimizer.
    - Uses both:
        * keyframe poses (pose, rel_pose) for pose-graph constraints
        * observed 3D keypoints in that keyframe as measurements (cur_3d, cur_3d_idx)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # One optimizer per object (global graph for each obj_id)
        self._optimizers: Dict[int, Optimizer] = {}

        # Track last keyframe pose & index per object to form relative pose
        self._last_kf_pose: Dict[int, np.ndarray] = {}
        self._last_kf_idx: Dict[int, int] = {}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_optimizer(self, obj_id: int) -> Optimizer:
        """
        Lazy-create an optimizer for this object using cfg.global_optimizer.
        """
        if obj_id not in self._optimizers:
            self._optimizers[obj_id] = build_from_cfg(
                self.cfg.global_optimizer, OPTIMIZER
            )
        return self._optimizers[obj_id]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def reset(self, obj_id: int | None = None):
        """
        Reset global optimizer state.

        - If obj_id is None: reset all objects.
        - If obj_id is given: reset only that object's optimizer and bookkeeping.
        """
        if obj_id is None:
            self._optimizers = {}
            self._last_kf_pose = {}
            self._last_kf_idx = {}
        else:
            self._optimizers[obj_id] = build_from_cfg(
                self.cfg.global_optimizer, OPTIMIZER
            )
            self._last_kf_pose.pop(obj_id, None)
            self._last_kf_idx.pop(obj_id, None)

    def get_num_keyframes(self, obj_id: int) -> int:
        """
        Proxy to underlying optimizer.get_num_poses() for this object.
        Interpreted as the number of keyframes that have been fed to the graph.
        """
        opt = self._get_optimizer(obj_id)
        return opt.get_num_poses()

    def update(
        self, keyframes: List[KeyFrame]
    ) -> Tuple[
        Dict[Tuple[int, int], np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray]]
    ]:
        """
        Add a batch of new keyframes to the global graph and run optimization.

        Args:
            keyframes: list of newly created KeyFrame instances.

        Returns:
            Tuple of:
            1. Dict mapping (obj_id, kf_idx) -> optimized 4x4 pose
            2. Dict mapping obj_id -> (optimized_key_points, optimized_key_point_indices)
        """
        updated_poses: Dict[Tuple[int, int], np.ndarray] = {}
        updated_landmarks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        if not keyframes:
            return updated_poses, updated_landmarks

        for kf in keyframes:
            obj_id = kf.obj_id
            kf_idx = kf.kf_idx
            cur_pose = np.asarray(kf.pose, dtype=float)

            opt = self._get_optimizer(obj_id)

            # -------------------------------------------------------------
            # 1) Relative pose between this KF and previous KF for this obj
            #    rel_pose ≈ T_prev^{-1} * T_cur, identity if it's the first KF.
            # -------------------------------------------------------------
            if obj_id in self._last_kf_pose:
                prev_pose = self._last_kf_pose[obj_id]
                prev_pose_inv = inverse_SE3(prev_pose)
                rel_pose = cur_pose @ prev_pose_inv
            else:
                rel_pose = np.eye(4, dtype=float)

            # -------------------------------------------------------------
            # 2) Build measurement data from this KeyFrame
            #
            #    We'll use the "observed" points:
            #       - obs_3d_camera: 3D points in camera frame
            #       - obs_track_indices: global track ids (used as landmark IDs)
            #       - obs_valid: depth validity mask
            #
            #    These become the cur_3d / cur_3d_idx / inliers / residuals /
            #    uncertainties fields for ObjectFrameData.
            # -------------------------------------------------------------
            if (
                kf.obs_3d_camera is not None
                and kf.obs_3d_camera.size > 0
                and kf.obs_valid is not None
                and kf.obs_visible is not None
            ):
                # (N_o,) boolean mask: visible & valid depth
                valid_mask = np.asarray(kf.obs_valid, dtype=bool) & np.asarray(
                    kf.obs_visible, dtype=bool
                )

                if np.any(valid_mask):
                    cur_3d = np.asarray(kf.obs_3d_camera[valid_mask], dtype=float)
                    cur_3d_idx = np.asarray(kf.obs_track_indices[valid_mask], dtype=int)

                    # Use uncertainties from the KF for those points
                    if kf.obs_uncertainties is not None:
                        uncertainties = np.asarray(
                            kf.obs_uncertainties[valid_mask], dtype=float
                        )
                    else:
                        uncertainties = 0.5 * np.ones((cur_3d.shape[0],), dtype=float)
            else:
                # No observed points in this keyframe
                cur_3d = np.zeros((0, 3), dtype=float)
                cur_3d_idx = np.zeros((0,), dtype=int)

            num_meas = cur_3d.shape[0]

            if num_meas > 0:
                inliers = np.ones((num_meas,), dtype=bool)
                # You can later wire reg_stats to get residual-based weights
                residuals = np.zeros((num_meas,), dtype=float)
                uncertainties = 0.5 * np.ones((num_meas,), dtype=float)
            else:
                inliers = np.zeros((0,), dtype=bool)
                residuals = np.zeros((0,), dtype=float)
                uncertainties = np.zeros((0,), dtype=float)

            # -------------------------------------------------------------
            # 3) Wrap into ObjectFrameData and call the underlying optimizer
            # -------------------------------------------------------------
            object_frame_data = ObjectFrameData(
                obj_id=obj_id,
                frame_id=kf_idx,  # use keyframe index as "frame id"
                pose=cur_pose,  # object/world pose at this keyframe
                rel_pose=rel_pose,  # relative pose to previous keyframe
                cur_3d=cur_3d,
                cur_3d_idx=cur_3d_idx,
                inliers=inliers,
                residuals=residuals,
                uncertainties=uncertainties,
            )

            opt_result = opt.optimize(object_frame_data)

            # If optimization succeeded, update the keyframe pose in-place
            if opt_result is not None and opt_result.pose_optimized is not None:
                kf.pose = opt_result.pose_optimized
                updated_poses[(obj_id, kf_idx)] = opt_result.pose_optimized

                if (
                    opt_result.key_points_optimized is not None
                    and opt_result.key_points_idx_optimized is not None
                ):
                    updated_landmarks[obj_id] = (
                        opt_result.key_points_optimized,
                        np.asarray(opt_result.key_points_idx_optimized, dtype=int),
                    )

            # Update last keyframe bookkeeping for this object
            self._last_kf_pose[obj_id] = np.asarray(kf.pose, dtype=float)
            self._last_kf_idx[obj_id] = kf_idx

        return updated_poses, updated_landmarks
