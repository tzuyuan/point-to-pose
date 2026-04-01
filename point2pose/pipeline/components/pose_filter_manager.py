from __future__ import annotations

from typing import Any

import numpy as np

from point2pose.utils.se3_cv_pose_filter import SE3ConstantVelocityFilter
from point2pose.utils.transform import inverse_SE3


class PoseFilterManager:
    def __init__(
        self,
        *,
        enabled: bool,
        log_raw_pose: bool,
        min_valid_correspondences: int,
        reset_trans_thres: float,
        reset_rot_deg_thres: float,
        filter_kwargs: dict[str, Any],
        skip_on_jump_reject: bool = True,
    ):
        self.enabled = bool(enabled)
        self.log_raw_pose = bool(log_raw_pose)
        self.min_valid_correspondences = max(int(min_valid_correspondences), 0)
        self.reset_trans_thres = float(reset_trans_thres)
        self.reset_rot_deg_thres = float(reset_rot_deg_thres)
        self.filter_kwargs = dict(filter_kwargs)
        self.skip_on_jump_reject = bool(skip_on_jump_reject)
        self.pose_filters: dict[int, SE3ConstantVelocityFilter] = {}

    def initialize(self, num_obj: int, objects: list, timestamp: float | None) -> None:
        self.pose_filters = {}
        if not self.enabled:
            return

        for obj_id in range(int(num_obj)):
            pose_filter = SE3ConstantVelocityFilter(**self.filter_kwargs)
            pose_filter.initialize(np.asarray(objects[obj_id].pose, dtype=float), timestamp)
            self.pose_filters[obj_id] = pose_filter

    def apply(self, frame, fe_result, objects: list) -> None:
        fe_result.obj_poses_raw = {}
        fe_result.obj_poses_filtered = {}
        fe_result.pose_filter_stats = {}

        if not self.enabled:
            for obj_id in range(len(objects)):
                raw_pose = fe_result.obj_poses.get(obj_id)
                if raw_pose is not None:
                    raw_pose_arr = np.asarray(raw_pose, dtype=float).copy()
                    fe_result.obj_poses_raw[obj_id] = raw_pose_arr
                    fe_result.obj_poses_filtered[obj_id] = raw_pose_arr.copy()
                fe_result.pose_filter_stats[obj_id] = self.default_stats()
            return

        timestamp = getattr(frame, "timestamp", None)
        for obj_id, obj in enumerate(objects):
            raw_pose = fe_result.obj_poses.get(obj_id)
            if raw_pose is not None:
                fe_result.obj_poses_raw[obj_id] = np.asarray(raw_pose, dtype=float).copy()

            pose_filter = self.pose_filters.get(obj_id)
            if pose_filter is None:
                pose_filter = SE3ConstantVelocityFilter(**self.filter_kwargs)
                pose_filter.initialize(np.asarray(obj.pose, dtype=float), timestamp)
                self.pose_filters[obj_id] = pose_filter

            stats = dict(fe_result.reg_stats.get(obj_id, {}) or {})
            stats["mean_residual"] = fe_result.mean_residuals.get(obj_id, -1.0)

            raw_pose_arr = None if raw_pose is None else np.asarray(raw_pose, dtype=float)
            valid_idx = np.asarray(
                fe_result.valid_indices.get(obj_id, np.array([])), dtype=int
            ).reshape(-1)
            jump_info = stats.get("pose_jump_guard_info", {}) or {}
            jump_rejected = bool(jump_info.get("rejected", False))
            measurement_ok = (
                raw_pose_arr is not None
                and raw_pose_arr.shape == (4, 4)
                and valid_idx.size >= self.min_valid_correspondences
                and not bool(getattr(obj, "lost", False))
                and not (jump_rejected and self.skip_on_jump_reject)
            )

            hard_reset = False
            if (
                raw_pose_arr is not None
                and raw_pose_arr.shape == (4, 4)
                and fe_result.dense_recovery_triggered.get(obj_id, False)
            ):
                pose_before = pose_filter.get_pose()
                if pose_before is None:
                    hard_reset = True
                else:
                    dt_reset, ddeg_reset = self._se3_delta(raw_pose_arr, pose_before)
                    hard_reset = (
                        dt_reset > self.reset_trans_thres
                        or ddeg_reset > self.reset_rot_deg_thres
                    )

            filtered_pose, filter_stats = pose_filter.step(
                raw_pose_arr,
                timestamp,
                stats,
                measurement_ok,
                hard_reset=hard_reset,
            )

            if filtered_pose is None:
                filtered_pose = np.asarray(obj.pose, dtype=float).copy()
            else:
                filtered_pose = np.asarray(filtered_pose, dtype=float).copy()

            fe_result.obj_poses[obj_id] = filtered_pose
            fe_result.obj_poses_filtered[obj_id] = filtered_pose.copy()
            fe_result.pose_filter_stats[obj_id] = filter_stats

    def get_logged_raw_pose(self, fe_result, obj_id: int) -> np.ndarray:
        if not self.log_raw_pose:
            return np.full((4, 4), np.nan, dtype=float)
        raw_pose = fe_result.obj_poses_raw.get(obj_id)
        if raw_pose is None:
            return np.full((4, 4), np.nan, dtype=float)
        raw_pose = np.asarray(raw_pose, dtype=float)
        if raw_pose.shape != (4, 4):
            return np.full((4, 4), np.nan, dtype=float)
        return raw_pose.copy()

    @staticmethod
    def get_field(fe_result, obj_id: int, key: str, default):
        stats = fe_result.pose_filter_stats.get(obj_id, {})
        value = stats.get(key, default)
        if isinstance(default, np.ndarray):
            if value is None:
                return default.copy()
            arr = np.asarray(value, dtype=float)
            if arr.shape != default.shape:
                return default.copy()
            return arr.copy()
        return value

    @staticmethod
    def default_stats() -> dict[str, Any]:
        return {
            "dt": 0.0,
            "pred_only": False,
            "measurement_used": False,
            "hard_reset": False,
            "meas_scale": 1.0,
            "innovation": np.zeros(6, dtype=float),
            "twist": np.zeros(6, dtype=float),
            "reason": "disabled",
        }

    @staticmethod
    def update_object_from_frontend(obj_id, obj, fe_result) -> None:
        if obj_id in fe_result.obj_poses:
            obj.pose = fe_result.obj_poses[obj_id]

        obj.curr_frame_points_3d = fe_result.valid_curr_3d.get(obj_id, None)
        obj.curr_frame_indices = fe_result.valid_indices.get(obj_id, None)

        stats = fe_result.reg_stats.get(obj_id, {})
        obj.inliers = stats.get("inliers", None)
        obj.residuals = stats.get("residuals", None)

        if obj.curr_frame_indices is not None and fe_result.uncertainties is not None:
            obj.curr_uncertainties = fe_result.uncertainties[obj.curr_frame_indices]
        else:
            obj.curr_uncertainties = None

        obj.mean_residual = fe_result.mean_residuals.get(
            obj_id, getattr(obj, "mean_residual", -1.0)
        )

    def _se3_delta(self, T_new: np.ndarray, T_old: np.ndarray) -> tuple[float, float]:
        dT = T_new @ inverse_SE3(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        trace = np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        ddeg = float(np.degrees(np.arccos(trace)))
        return dt, ddeg
