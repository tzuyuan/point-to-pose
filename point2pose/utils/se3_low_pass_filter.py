import numpy as np
from collections import deque
from typing import Any, Optional

from point2pose.utils.lie import log_SE3, vec_to_se3, se3_to_vec, exp_se3


class SE3LowPassFilter:
    """
    A low pass filter for SE(3) poses that works in the Lie algebra se(3).

    This filter properly handles the non-Euclidean structure of SE(3) by:
    1. Converting poses to the Lie algebra se(3) using logarithm
    2. Applying low pass filtering in the vector space of se(3)
    3. Converting back to SE(3) using exponential mapping

    Parameters
    ----------
    alpha : float
        Smoothing factor (0 < alpha < 1). Smaller values = more smoothing.
    window_size : int, optional
        Window size for moving average. If None, uses exponential smoothing only.
    """

    def __init__(self, alpha: float = 0.1, window_size: Optional[int] = None):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self.window_size = window_size
        self._initialized = False
        self._filtered_pose: Optional[np.ndarray] = None

        # For moving average if window_size is specified
        if window_size is not None and window_size > 0:
            self._se3_buffer = deque(maxlen=window_size)
            self._use_moving_average = True
        else:
            self._use_moving_average = False

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new SE(3) pose.

        Parameters
        ----------
        pose : np.ndarray
            A 4x4 SE(3) transformation matrix

        Returns
        -------
        np.ndarray
            The filtered 4x4 SE(3) transformation matrix
        """
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix")

        # Ensure the pose is a valid SE(3) matrix
        # if not self._is_valid_se3(pose):
        #     raise ValueError("Input pose is not a valid SE(3) matrix")

        if not self._initialized:
            self._filtered_pose = pose.copy()
            self._initialized = True
            return self._filtered_pose

        if self._use_moving_average:
            return self._update_moving_average(pose)
        else:
            return self._update_exponential_diff(pose)

    def _update_exponential_smoothing(self, pose: np.ndarray) -> np.ndarray:
        """Update using exponential smoothing in se(3)."""
        # Convert current filtered pose to se(3)
        current_se3 = log_SE3(self._filtered_pose)
        current_se3_vec = se3_to_vec(current_se3)

        # Convert new pose to se(3)
        new_se3 = log_SE3(pose)
        new_se3_vec = se3_to_vec(new_se3)

        # Apply exponential smoothing in se(3)
        filtered_se3_vec = (1 - self.alpha) * current_se3_vec + self.alpha * new_se3_vec

        # Convert back to SE(3)
        filtered_se3 = vec_to_se3(filtered_se3_vec)
        self._filtered_pose = exp_se3(filtered_se3)

        return self._filtered_pose

    def _update_exponential_diff(self, pose: np.ndarray) -> np.ndarray:
        """Update using exponential smoothing in se(3)."""
        # Convert current filtered pose to se(3)
        self._filtered_pose = self._filtered_pose @ exp_se3(
            self.alpha * log_SE3(np.linalg.inv(self._filtered_pose) @ pose)
        )
        return self._filtered_pose

    def _update_moving_average(self, pose: np.ndarray) -> np.ndarray:
        """Update using moving average in se(3)."""
        # Convert pose to se(3)
        se3 = log_SE3(pose)
        se3_vec = se3_to_vec(se3)

        # Add to buffer
        self._se3_buffer.append(se3_vec)

        # Compute average in se(3)
        if len(self._se3_buffer) > 0:
            avg_se3_vec = np.mean(self._se3_buffer, axis=0)
            avg_se3 = vec_to_se3(avg_se3_vec)
            self._filtered_pose = exp_se3(avg_se3)

        return self._filtered_pose

    def _is_valid_se3(self, pose: np.ndarray) -> bool:
        """Check if a matrix is a valid SE(3) transformation."""
        # Check dimensions
        if pose.shape != (4, 4):
            return False

        # Check that bottom row is [0, 0, 0, 1]
        if not np.allclose(pose[3, :], [0, 0, 0, 1]):
            return False

        # Check that rotation part is orthogonal
        R = pose[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            return False

        # Check determinant of rotation part is 1
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            return False

        return True

    def reset(self):
        """Reset the filter state."""
        self._initialized = False
        self._filtered_pose = None
        if self._use_moving_average:
            self._se3_buffer.clear()

    def get_filtered_pose(self) -> Optional[np.ndarray]:
        """Get the current filtered pose."""
        return self._filtered_pose.copy() if self._filtered_pose is not None else None


class SE3LowPassFilterAdapter:
    """
    Adapter that wraps :class:`SE3LowPassFilter` to conform to the
    filter-manager interface (``initialize`` / ``step`` / ``get_pose`` /
    ``reset``).

    Adds measurement gating, adaptive trust scaling, outlier rejection,
    and the stats dict expected by the pipeline.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        window_size: Optional[int] = None,
        nominal_dt: float = 1.0 / 30.0,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        residual_ref: float = 0.007,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.2,
        min_valid_correspondences: int = 3,
        max_meas_scale: float = 50.0,
        skip_on_jump_reject: bool = True,
        jump_reject_meas_scale: Optional[float] = None,
        outlier_rejection: bool = False,
        outlier_trans_thres: float = 0.1,
        outlier_rot_deg_thres: float = 30.0,
    ):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._core = SE3LowPassFilter(alpha=self.alpha, window_size=window_size)

        self.nominal_dt = max(float(nominal_dt), 1e-6)
        self.min_dt = max(float(min_dt), 1e-6)
        self.max_dt = max(float(max_dt), self.min_dt)

        self.residual_ref = max(float(residual_ref), 1e-8)
        self.min_inliers = max(int(min_inliers), 0)
        self.min_inlier_ratio = max(float(min_inlier_ratio), 0.0)
        self.min_valid_correspondences = max(int(min_valid_correspondences), 0)
        self.max_meas_scale = max(float(max_meas_scale), 1.0)
        self.skip_on_jump_reject = bool(skip_on_jump_reject)
        self.jump_reject_meas_scale = float(
            jump_reject_meas_scale
            if jump_reject_meas_scale is not None
            else max(10.0, 0.25 * self.max_meas_scale)
        )

        self.outlier_rejection = bool(outlier_rejection)
        self.outlier_trans_thres = float(outlier_trans_thres)
        self.outlier_rot_deg_thres = float(outlier_rot_deg_thres)

        self._last_timestamp: Optional[float] = None

    # -- Public API (filter-manager interface) ---------------------------------

    def initialize(
        self, pose: np.ndarray, timestamp: Optional[float] = None
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        self._core.reset()
        self._core.update(pose)
        self._last_timestamp = self._normalize_timestamp(timestamp)
        return self.get_pose()

    def reset(
        self, pose: Optional[np.ndarray] = None, timestamp: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if pose is None:
            self._core.reset()
            self._last_timestamp = None
            return None
        return self.initialize(pose, timestamp)

    def get_pose(self) -> Optional[np.ndarray]:
        return self._core.get_filtered_pose()

    def step(
        self,
        pose_meas: Optional[np.ndarray],
        timestamp: Optional[float],
        stats: Optional[dict[str, Any]],
        measurement_ok: bool,
        hard_reset: bool = False,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        stats = stats or {}
        pose_meas = self._normalize_pose(pose_meas)
        timestamp = self._normalize_timestamp(timestamp)

        # Hard reset: reinitialize at the given pose.
        if hard_reset and pose_meas is not None:
            self.initialize(pose_meas, timestamp)
            return self.get_pose(), self._make_info(
                measurement_used=True,
                hard_reset=True,
                reason="hard_reset",
            )

        # Bootstrap: first measurement initializes the filter.
        if not self._core._initialized:
            if pose_meas is None:
                return None, self._make_info(reason="uninitialized")
            self.initialize(pose_meas, timestamp)
            return self.get_pose(), self._make_info(
                measurement_used=True,
                reason="bootstrap",
            )

        # Gate: determine whether to accept the measurement.
        reject_reason = self._reject_reason(pose_meas, stats, measurement_ok)
        if reject_reason is not None:
            # Zero-order hold — keep the last filtered pose.
            self._last_timestamp = timestamp
            return self.get_pose(), self._make_info(
                pred_only=True,
                reason=reject_reason,
            )

        # Adaptive alpha: scale down when measurement quality is poor.
        meas_scale = self._measurement_scale(stats)
        effective_alpha = float(np.clip(self.alpha / meas_scale, 0.0, 1.0))

        # Temporarily override core alpha for this update.
        orig_alpha = self._core.alpha
        self._core.alpha = effective_alpha
        self._core.update(pose_meas)
        self._core.alpha = orig_alpha

        self._last_timestamp = timestamp

        return self.get_pose(), self._make_info(
            measurement_used=True,
            meas_scale=meas_scale,
            reason="updated",
        )

    # -- Measurement gating ---------------------------------------------------

    def _reject_reason(
        self,
        pose_meas: Optional[np.ndarray],
        stats: dict[str, Any],
        measurement_ok: bool,
    ) -> Optional[str]:
        if pose_meas is None:
            return "missing_measurement"
        if self._extract_valid_count(stats) < self.min_valid_correspondences:
            return "too_few_correspondences"
        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        if bool(jump_info.get("rejected", False)) and self.skip_on_jump_reject:
            return "jump_rejected"
        if not measurement_ok:
            return "measurement_rejected"
        # Outlier rejection: reject if the measurement jumps too far from
        # the current filtered pose.
        if not self.outlier_rejection:
            return None
        current = self.get_pose()
        if current is not None and pose_meas is not None:
            dt, ddeg = self._se3_delta(pose_meas, current)
            if dt > self.outlier_trans_thres or ddeg > self.outlier_rot_deg_thres:
                return "outlier_rejected"
        return None

    # -- Measurement quality --------------------------------------------------

    def _measurement_scale(self, stats: dict[str, Any]) -> float:
        mean_residual = self._extract_mean_residual(stats)
        ninliers, _used, inlier_ratio = self._extract_inlier_stats(stats)
        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        jump_rejected = bool(jump_info.get("rejected", False))

        scale = 1.0
        if np.isfinite(mean_residual) and mean_residual > 0.0:
            scale = max(scale, mean_residual / self.residual_ref)
        if self.min_inliers > 0:
            if ninliers <= 0:
                scale = self.max_meas_scale
            else:
                scale = max(scale, float(self.min_inliers) / float(ninliers))
        if self.min_inlier_ratio > 0.0:
            if inlier_ratio <= 1e-8:
                scale = self.max_meas_scale
            else:
                scale = max(scale, self.min_inlier_ratio / inlier_ratio)
        if jump_rejected and not self.skip_on_jump_reject:
            scale = max(scale, self.jump_reject_meas_scale)
        return min(scale, self.max_meas_scale)

    # -- Stat extraction ------------------------------------------------------

    def _extract_valid_count(self, stats: dict[str, Any]) -> int:
        if "valid_idx" in stats and stats["valid_idx"] is not None:
            return int(np.asarray(stats["valid_idx"]).reshape(-1).size)
        if "correspond_curr3d" in stats and stats["correspond_curr3d"] is not None:
            return int(np.asarray(stats["correspond_curr3d"]).shape[0])
        if "inliers" in stats and stats["inliers"] is not None:
            return int(np.asarray(stats["inliers"]).reshape(-1).size)
        return 0

    def _extract_inlier_stats(self, stats: dict[str, Any]) -> tuple[int, int, float]:
        inliers = np.asarray(
            stats.get("inliers", np.array([])), dtype=bool
        ).reshape(-1)
        used = int(inliers.size)
        ninliers = int(np.count_nonzero(inliers))
        inlier_ratio = float(ninliers / max(1, used))
        return ninliers, used, inlier_ratio

    def _extract_mean_residual(self, stats: dict[str, Any]) -> float:
        mean_residual = stats.get("mean_residual", -1.0)
        try:
            return float(mean_residual)
        except Exception:
            return -1.0

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _se3_delta(T_new: np.ndarray, T_old: np.ndarray) -> tuple[float, float]:
        dT = T_new @ np.linalg.inv(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        trace = np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        ddeg = float(np.degrees(np.arccos(trace)))
        return dt, ddeg

    def _normalize_pose(self, pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=float)
        if arr.shape != (4, 4):
            return None
        return arr.copy()

    def _normalize_timestamp(self, timestamp: Optional[float]) -> Optional[float]:
        if timestamp is None:
            return None
        try:
            value = float(timestamp)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _make_info(
        self,
        *,
        dt: float = 0.0,
        pred_only: bool = False,
        measurement_used: bool = False,
        hard_reset: bool = False,
        meas_scale: float = 1.0,
        reason: str = "idle",
    ) -> dict[str, Any]:
        return {
            "dt": float(dt),
            "pred_only": bool(pred_only),
            "measurement_used": bool(measurement_used),
            "hard_reset": bool(hard_reset),
            "velocity_prediction_enabled": False,
            "meas_scale": float(meas_scale),
            "innovation": np.zeros(6, dtype=float),
            "twist": np.zeros(6, dtype=float),
            "reason": reason,
            # Compatibility fields for default_stats consumers.
            "twist_obs_used": False,
            "twist_obs": np.zeros(6, dtype=float),
            "twist_obs_sigma": np.zeros(6, dtype=float),
            "twist_obs_method": "none",
            "twist_obs_reason": "disabled",
            "twist_obs_num_poses": 0,
            "twist_obs_num_samples": 0,
            "twist_obs_num_inlier_samples": 0,
            "twist_obs_rot_dispersion": 0.0,
            "twist_obs_trans_dispersion": 0.0,
        }
