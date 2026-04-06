from __future__ import annotations

from typing import Any, Optional

import numpy as np

from point2pose.utils.lie import (
    exp_se3,
    exp_so3,
    log_SE3,
    log_SO3,
    se3_to_vec,
    vec_to_se3,
)


class SE3MaskFusedFilter:
    """
    SE(3) pose filter that uses mask-derived motion as the translation
    prediction and point2pose measurements as the EKF update.

    Instead of constant-velocity extrapolation, the translation prediction
    comes from the frame-to-frame displacement of the mask center in 3D.
    This avoids coordinate-frame mismatch between the mask geometric center
    and the object pose origin — the offset cancels in the delta.

    State is 6D: [delta_theta(3), delta_p(3)] with a 6x6 covariance.

    Rotation is smoothed via exponential smoothing in SO(3) (SLERP-style
    interpolation in the Lie algebra).
    """

    def __init__(
        self,
        *,
        nominal_dt: float = 1.0 / 30.0,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        rot_process_sigma: float = np.deg2rad(5.0),
        trans_process_sigma: float = 0.02,
        rot_meas_sigma: float = np.deg2rad(2.0),
        trans_meas_sigma: float = 0.01,
        residual_ref: float = 0.007,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.2,
        min_valid_correspondences: int = 3,
        max_meas_scale: float = 50.0,
        skip_on_jump_reject: bool = True,
        jump_reject_meas_scale: Optional[float] = None,
        rot_smooth_alpha: float = 0.3,
        mask_trans_trust: float = 0.8,
    ):
        self.nominal_dt = max(float(nominal_dt), 1e-6)
        self.min_dt = max(float(min_dt), 1e-6)
        self.max_dt = max(float(max_dt), self.min_dt)

        self.rot_process_sigma = max(float(rot_process_sigma), 1e-8)
        self.trans_process_sigma = max(float(trans_process_sigma), 1e-8)
        self.rot_meas_sigma = max(float(rot_meas_sigma), 1e-8)
        self.trans_meas_sigma = max(float(trans_meas_sigma), 1e-8)

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

        self.rot_smooth_alpha = float(np.clip(rot_smooth_alpha, 0.0, 1.0))
        self.mask_trans_trust = float(np.clip(mask_trans_trust, 0.0, 1.0))

        self._R: Optional[np.ndarray] = None
        self._p = np.zeros(3, dtype=float)
        self._P = self._make_initial_covariance()
        self._R_smooth: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None
        self._prev_mask_pos: Optional[np.ndarray] = None

    # -- Public API -----------------------------------------------------------

    def initialize(
        self, pose: np.ndarray, timestamp: Optional[float] = None
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        self._R = pose[:3, :3].copy()
        self._p = pose[:3, 3].copy()
        self._P = self._make_initial_covariance()
        self._R_smooth = pose[:3, :3].copy()
        self._last_timestamp = self._normalize_timestamp(timestamp)
        self._prev_mask_pos = None
        return self.get_pose()

    def reset(
        self, pose: Optional[np.ndarray] = None, timestamp: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if pose is None:
            self._R = None
            self._p = np.zeros(3, dtype=float)
            self._P = self._make_initial_covariance()
            self._R_smooth = None
            self._last_timestamp = None
            self._prev_mask_pos = None
            return None
        return self.initialize(pose, timestamp)

    def get_pose(self) -> Optional[np.ndarray]:
        if self._R is None:
            return None
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = self._R_smooth if self._R_smooth is not None else self._R
        pose[:3, 3] = self._p
        return pose

    def step(
        self,
        pose_meas: Optional[np.ndarray],
        timestamp: Optional[float],
        stats: Optional[dict[str, Any]],
        measurement_ok: bool,
        hard_reset: bool = False,
        mask_position_3d: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        stats = stats or {}
        pose_meas = self._normalize_pose(pose_meas)
        timestamp = self._normalize_timestamp(timestamp)

        mask_used = False

        # Hard reset: reinitialize at the given pose.
        if hard_reset and pose_meas is not None:
            self.initialize(pose_meas, timestamp)
            return self.get_pose(), self._make_info(
                measurement_used=True,
                hard_reset=True,
                meas_scale=1.0,
                innovation=np.zeros(6, dtype=float),
                reason="hard_reset",
            )

        # Bootstrap: first measurement initializes the filter.
        if self._R is None:
            if pose_meas is None:
                return None, self._make_info(reason="uninitialized")
            self.initialize(pose_meas, timestamp)
            return self.get_pose(), self._make_info(
                measurement_used=True,
                meas_scale=1.0,
                innovation=np.zeros(6, dtype=float),
                reason="bootstrap",
            )

        # Predict.
        R_pred, p_pred, P_pred, mask_used = self._predict(mask_position_3d)

        # Gate: determine whether to accept the measurement.
        reject_reason = self._reject_reason(pose_meas, stats, measurement_ok)
        if reject_reason is not None:
            self._commit_state(R_pred, p_pred, P_pred, timestamp)
            return self.get_pose(), self._make_info(
                pred_only=True,
                mask_position_used=mask_used,
                reason=reject_reason,
            )

        # Full-pose EKF update.
        return self._pose_update(
            pose_meas, R_pred, p_pred, P_pred, timestamp, stats, mask_used,
        )

    # -- Prediction -----------------------------------------------------------

    def _predict(
        self, mask_position_3d: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        R_pred = self._R.copy()

        # Use frame-to-frame mask displacement as translation prediction.
        # The delta cancels the offset between mask geometric center and
        # object pose origin, so we can safely add it to self._p.
        mask_used = False
        mask_delta = np.zeros(3, dtype=float)
        cur_mask_pos = None

        if mask_position_3d is not None:
            cur_mask_pos = np.asarray(mask_position_3d, dtype=float).reshape(3)
            if not np.all(np.isfinite(cur_mask_pos)):
                cur_mask_pos = None

        if cur_mask_pos is not None and self._prev_mask_pos is not None:
            mask_delta = cur_mask_pos - self._prev_mask_pos
            if np.all(np.isfinite(mask_delta)):
                mask_used = True
            else:
                mask_delta = np.zeros(3, dtype=float)

        # Update stored mask position for next frame.
        if cur_mask_pos is not None:
            self._prev_mask_pos = cur_mask_pos.copy()

        # Blend: p_pred = p_prev + trust * mask_delta
        p_pred = self._p + self.mask_trans_trust * mask_delta

        Q = np.diag(
            np.array(
                [self.rot_process_sigma**2] * 3
                + [self.trans_process_sigma**2] * 3,
                dtype=float,
            )
        )
        P_pred = self._P + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        return R_pred, p_pred, P_pred, mask_used

    # -- EKF update -----------------------------------------------------------

    def _pose_update(
        self,
        pose_meas: np.ndarray,
        R_pred: np.ndarray,
        p_pred: np.ndarray,
        P_pred: np.ndarray,
        timestamp: Optional[float],
        stats: dict[str, Any],
        mask_used: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mean_residual = self._extract_mean_residual(stats)
        ninliers, used, inlier_ratio = self._extract_inlier_stats(stats)
        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        jump_rejected = bool(jump_info.get("rejected", False))

        pred_pose = self._assemble_pose(R_pred, p_pred)
        innovation = self._pose_delta_vec(pose_meas, pred_pose)

        meas_scale = self._measurement_scale(
            mean_residual=mean_residual,
            ninliers=ninliers,
            inlier_ratio=inlier_ratio,
            jump_rejected=jump_rejected,
        )

        H = np.eye(6, dtype=float)
        R_meas = np.diag(
            np.array(
                [self.rot_meas_sigma**2] * 3 + [self.trans_meas_sigma**2] * 3,
                dtype=float,
            )
            * meas_scale
        )

        # Standard EKF update (Joseph form).
        S = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.pinv(S)
        delta = K @ innovation

        pose_corr = exp_se3(vec_to_se3(delta))
        pose_upd = pose_corr @ pred_pose
        R_upd = self._reorthogonalize(pose_upd[:3, :3])
        p_upd = pose_upd[:3, 3].copy()

        I_6 = np.eye(6, dtype=float)
        KH = K @ H
        P_upd = (I_6 - KH) @ P_pred @ (I_6 - KH).T + K @ R_meas @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        # Rotation smoothing in SO(3).
        self._apply_rotation_smoothing(R_upd)

        self._commit_state(R_upd, p_upd, P_upd, timestamp)

        return self.get_pose(), self._make_info(
            measurement_used=True,
            meas_scale=meas_scale,
            innovation=innovation,
            mask_position_used=mask_used,
            reason="updated",
        )

    def _apply_rotation_smoothing(self, R_new: np.ndarray) -> None:
        if self._R_smooth is None:
            self._R_smooth = R_new.copy()
            return
        # SLERP-style interpolation in SO(3) via Lie algebra.
        R_rel = self._R_smooth.T @ R_new
        log_rel = log_SO3(R_rel)
        self._R_smooth = self._reorthogonalize(
            self._R_smooth @ exp_so3(self.rot_smooth_alpha * log_rel)
        )

    # -- State management -----------------------------------------------------

    def _commit_state(
        self,
        R: np.ndarray,
        p: np.ndarray,
        P: np.ndarray,
        timestamp: Optional[float],
    ) -> None:
        self._R = R
        self._p = p
        self._P = P
        self._last_timestamp = timestamp

    def _make_initial_covariance(self) -> np.ndarray:
        diag = np.array(
            [self.rot_meas_sigma**2 * 4.0] * 3
            + [self.trans_meas_sigma**2 * 4.0] * 3,
            dtype=float,
        )
        return np.diag(diag)

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
        return None

    # -- Measurement quality --------------------------------------------------

    def _measurement_scale(
        self,
        *,
        mean_residual: float,
        ninliers: int,
        inlier_ratio: float,
        jump_rejected: bool,
    ) -> float:
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
    def _reorthogonalize(R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R)
        d = np.linalg.det(U @ Vt)
        return U @ np.diag([1.0, 1.0, float(d)]) @ Vt

    @staticmethod
    def _assemble_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = R
        pose[:3, 3] = p
        return pose

    @staticmethod
    def _pose_delta_vec(T_new: np.ndarray, T_old: np.ndarray) -> np.ndarray:
        delta = T_new @ np.linalg.inv(T_old)
        return se3_to_vec(log_SE3(delta))

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
        innovation: Optional[np.ndarray] = None,
        mask_position_used: bool = False,
        reason: str = "idle",
    ) -> dict[str, Any]:
        innovation = (
            np.zeros(6, dtype=float)
            if innovation is None
            else np.asarray(innovation, dtype=float).reshape(6)
        )
        return {
            "dt": float(dt),
            "pred_only": bool(pred_only),
            "measurement_used": bool(measurement_used),
            "hard_reset": bool(hard_reset),
            "velocity_prediction_enabled": False,
            "meas_scale": float(meas_scale),
            "innovation": innovation,
            "twist": np.zeros(6, dtype=float),
            "mask_position_used": bool(mask_position_used),
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
