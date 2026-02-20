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

        # Keep references to all keyframes seen by object and frame_id so we can
        # update every keyframe pose after graph optimization.
        self._keyframes_by_obj_frame: Dict[int, Dict[int, KeyFrame]] = {}

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
            self._keyframes_by_obj_frame = {}
        else:
            self._optimizers[obj_id] = build_from_cfg(
                self.cfg.global_optimizer, OPTIMIZER
            )
            self._last_kf_pose.pop(obj_id, None)
            self._last_kf_idx.pop(obj_id, None)
            self._keyframes_by_obj_frame.pop(obj_id, None)

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
        updated_poses = {}
        updated_landmarks = {}

        if not keyframes:
            return updated_poses, updated_landmarks

        for kf in keyframes:
            obj_id = kf.obj_id
            kf_idx = kf.kf_idx
            cur_pose = np.asarray(kf.pose, dtype=float)

            if obj_id not in self._keyframes_by_obj_frame:
                self._keyframes_by_obj_frame[obj_id] = {}
            self._keyframes_by_obj_frame[obj_id][int(kf.frame_id)] = kf

            opt = self._get_optimizer(obj_id)

            print(opt._prior_noise)

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
            #    These become the reg_cur_3d / reg_cur_3d_idx / reg_inliers /
            #    reg_residuals / reg_uncertainties fields for ObjectFrameData.
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
                # TODO: this is not correct, remove valid_mask indexing
                # if np.any(valid_mask):
                cur_3d = np.asarray(kf.obs_3d_camera, dtype=float)
                cur_3d_idx = np.asarray(kf.obs_track_indices, dtype=int)

                # Use uncertainties from the KF for those points
                if kf.obs_uncertainties is not None:
                    uncertainties = np.asarray(kf.obs_uncertainties, dtype=float)
                else:
                    uncertainties = 0.5 * np.ones((cur_3d.shape[0],), dtype=float)
            else:
                # No observed points in this keyframe
                cur_3d = np.zeros((0, 3), dtype=float)
                cur_3d_idx = np.zeros((0,), dtype=int)
                uncertainties = np.zeros((0,), dtype=float)

            num_meas = cur_3d.shape[0]

            # registration stats
            inliers = kf.reg_inliers
            residuals = kf.reg_residuals
            valid_idx = kf.reg_valid_idx
            reg_cur_3d = kf.reg_correspond_curr3d

            if kf.obs_track_indices is not None and kf.obs_uncertainties is not None:
                obs_ids = np.asarray(kf.obs_track_indices, dtype=np.int64).reshape(-1)
                obs_unc = np.asarray(kf.obs_uncertainties, dtype=float).reshape(-1)
                id2unc = {int(t): float(u) for t, u in zip(obs_ids, obs_unc)}
                reg_uncertainties = np.array(
                    [id2unc.get(int(t), 0.5) for t in valid_idx], dtype=float
                )
            else:
                reg_uncertainties = 0.5 * np.ones((len(valid_idx),), dtype=float)

            visible_pts_2d = kf.obs_2d[kf.obs_visible]
            visible_pts_2d_idx = kf.obs_track_indices[kf.obs_visible]
            visible_uncertainties = kf.obs_uncertainties[kf.obs_visible]

            # -------------------------------------------------------------
            # 3) Wrap into ObjectFrameData and call the underlying optimizer
            # -------------------------------------------------------------
            object_frame_data = ObjectFrameData(
                obj_id=obj_id,
                frame_id=kf.frame_id,  # use keyframe index as "frame id"
                intrinsics=kf.frame.intrinsics,
                pose=cur_pose,  # object/world pose at this keyframe
                rel_pose=rel_pose,  # relative pose to previous keyframe
                visible_pts_2d=visible_pts_2d,
                visible_pts_2d_idx=visible_pts_2d_idx,
                visible_uncertainties=visible_uncertainties,
                reg_cur_3d=kf.reg_correspond_curr3d,
                reg_cur_3d_idx=valid_idx,
                reg_valid_idx=valid_idx,
                reg_inliers=inliers,
                reg_residuals=residuals,
                reg_uncertainties=reg_uncertainties,
            )

            opt_result = opt.optimize(object_frame_data)

            # If optimization succeeded, update all keyframe poses for this object.
            if opt_result is not None and opt_result.pose_optimized is not None:
                poses_all = getattr(opt_result, "poses_optimized", None)
                frame_ids_all = getattr(opt_result, "pose_frame_ids_optimized", None)

                if poses_all is not None and frame_ids_all is not None:
                    for frame_id_opt, pose_opt in zip(frame_ids_all, poses_all):
                        kf_ref = self._keyframes_by_obj_frame.get(obj_id, {}).get(
                            int(frame_id_opt)
                        )
                        if kf_ref is None:
                            continue

                        kf_ref.pose = np.asarray(pose_opt, dtype=float)
                        kf_ref.update_kps()
                        updated_poses[(obj_id, int(kf_ref.kf_idx))] = kf_ref.pose
                else:
                    kf.pose = np.asarray(opt_result.pose_optimized, dtype=float)
                    kf.update_kps()
                    updated_poses[(obj_id, kf_idx)] = kf.pose

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
