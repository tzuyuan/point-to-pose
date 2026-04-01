from __future__ import annotations

from typing import Any, Optional

import numpy as np

from point2pose.utils.lie import exp_se3, log_SE3, se3_to_vec, skew, vec_to_se3


class SE3ConstantVelocityFilter:
    """
    Causal SE_2(3)-style pose filter with smooth velocity estimation.

    The nominal state is:
      - R in SO(3)
      - omega in R^3 (angular velocity)
      - p in R^3 (position)
      - v in R^3 (linear velocity)
      - covariance P over [delta_theta, delta_omega, delta_p, delta_v]

    Prediction uses a damped constant-velocity (Ornstein-Uhlenbeck) model:
        T_k|k-1 = Exp([beta * omega, beta * v]) * T_k-1|k-1
        omega_k|k-1 = alpha * omega_k-1|k-1
        v_k|k-1 = alpha * v_k-1|k-1

    where alpha = exp(-lambda * dt), beta = (1 - alpha) / lambda, and
    lambda = ln(2) / velocity_damping_half_life.  When the half-life is
    None (default), lambda = 0 and the model reduces to pure constant
    velocity (alpha = 1, beta = dt).

    This matches the frontend / pipeline convention, where relative motion is
    left-multiplied onto the previous pose:
        T_new = T_rel * T_prev

    The covariance uses a first-order linearization around that same
    left-multiplicative pose update.

    Only pose is measured. Velocities are inferred smoothly through the Kalman
    coupling between orientation/position and their respective rates.
    """

    def __init__(
        self,
        *,
        nominal_dt: float = 1.0 / 30.0,
        min_dt: float = 1e-3,
        max_dt: float = 0.2,
        rot_accel_sigma: float = np.deg2rad(200.0),
        trans_accel_sigma: float = 2.0,
        rot_meas_sigma: float = np.deg2rad(2.0),
        trans_meas_sigma: float = 0.01,
        residual_ref: float = 0.007,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.2,
        min_valid_correspondences: int = 3,
        max_meas_scale: float = 50.0,
        skip_on_jump_reject: bool = True,
        jump_reject_meas_scale: Optional[float] = None,
        init_pose_rot_sigma: Optional[float] = None,
        init_pose_trans_sigma: Optional[float] = None,
        init_twist_rot_sigma: Optional[float] = None,
        init_twist_trans_sigma: Optional[float] = None,
        velocity_damping_half_life: Optional[float] = None,
    ):
        self.nominal_dt = max(float(nominal_dt), 1e-6)
        self.min_dt = max(float(min_dt), 1e-6)
        self.max_dt = max(float(max_dt), self.min_dt)

        self.rot_accel_sigma = max(float(rot_accel_sigma), 1e-8)
        self.trans_accel_sigma = max(float(trans_accel_sigma), 1e-8)
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

        if velocity_damping_half_life is not None and velocity_damping_half_life > 0.0:
            self._lambda = float(np.log(2.0) / velocity_damping_half_life)
        else:
            self._lambda = 0.0

        self.init_pose_rot_sigma = (
            float(init_pose_rot_sigma)
            if init_pose_rot_sigma is not None
            else 2.0 * self.rot_meas_sigma
        )
        self.init_pose_trans_sigma = (
            float(init_pose_trans_sigma)
            if init_pose_trans_sigma is not None
            else 2.0 * self.trans_meas_sigma
        )
        self.init_twist_rot_sigma = (
            float(init_twist_rot_sigma)
            if init_twist_rot_sigma is not None
            else max(self.rot_meas_sigma / self.nominal_dt, 1e-4)
        )
        self.init_twist_trans_sigma = (
            float(init_twist_trans_sigma)
            if init_twist_trans_sigma is not None
            else max(self.trans_meas_sigma / self.nominal_dt, 1e-4)
        )

        self._R: Optional[np.ndarray] = None
        self._omega = np.zeros(3, dtype=float)
        self._p = np.zeros(3, dtype=float)
        self._v = np.zeros(3, dtype=float)
        self._P = self._make_initial_covariance()
        self._last_timestamp: Optional[float] = None

    def initialize(self, pose: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        self._R = pose[:3, :3].copy()
        self._p = pose[:3, 3].copy()
        self._omega = np.zeros(3, dtype=float)
        self._v = np.zeros(3, dtype=float)
        self._P = self._make_initial_covariance()
        self._last_timestamp = self._normalize_timestamp(timestamp)
        return self.get_pose()

    def reset(
        self, pose: Optional[np.ndarray] = None, timestamp: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if pose is None:
            self._R = None
            self._omega = np.zeros(3, dtype=float)
            self._p = np.zeros(3, dtype=float)
            self._v = np.zeros(3, dtype=float)
            self._P = self._make_initial_covariance()
            self._last_timestamp = None
            return None
        return self.initialize(pose, timestamp)

    def get_pose(self) -> Optional[np.ndarray]:
        if self._R is None:
            return None
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = self._R
        pose[:3, 3] = self._p
        return pose

    def get_twist(self) -> np.ndarray:
        return np.hstack((self._omega, self._v))

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

        if hard_reset and pose_meas is not None:
            self.initialize(pose_meas, timestamp)
            info = self._make_info(
                dt=0.0,
                pred_only=False,
                measurement_used=True,
                hard_reset=True,
                meas_scale=1.0,
                innovation=np.zeros(6, dtype=float),
                reason="hard_reset",
            )
            return self.get_pose(), info

        if self._R is None:
            if pose_meas is None:
                return None, self._make_info(reason="uninitialized")
            self.initialize(pose_meas, timestamp)
            info = self._make_info(
                pred_only=False,
                measurement_used=True,
                meas_scale=1.0,
                innovation=np.zeros(6, dtype=float),
                reason="bootstrap",
            )
            return self.get_pose(), info

        dt = self._compute_dt(timestamp)
        R_pred, omega_pred, p_pred, v_pred, P_pred = self._predict(dt)

        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        jump_rejected = bool(jump_info.get("rejected", False))
        valid_count = self._extract_valid_count(stats)
        mean_residual = self._extract_mean_residual(stats)
        ninliers, used, inlier_ratio = self._extract_inlier_stats(stats)

        if pose_meas is None:
            self._commit_prediction(R_pred, omega_pred, p_pred, v_pred, P_pred, timestamp)
            info = self._make_info(
                dt=dt,
                pred_only=True,
                measurement_used=False,
                meas_scale=self.max_meas_scale,
                innovation=np.zeros(6, dtype=float),
                reason="missing_measurement",
            )
            return self.get_pose(), info

        if valid_count < self.min_valid_correspondences:
            self._commit_prediction(R_pred, omega_pred, p_pred, v_pred, P_pred, timestamp)
            info = self._make_info(
                dt=dt,
                pred_only=True,
                measurement_used=False,
                meas_scale=self.max_meas_scale,
                innovation=np.zeros(6, dtype=float),
                reason="too_few_correspondences",
            )
            return self.get_pose(), info

        if jump_rejected and self.skip_on_jump_reject:
            self._commit_prediction(R_pred, omega_pred, p_pred, v_pred, P_pred, timestamp)
            info = self._make_info(
                dt=dt,
                pred_only=True,
                measurement_used=False,
                meas_scale=self.max_meas_scale,
                innovation=np.zeros(6, dtype=float),
                reason="jump_rejected",
            )
            return self.get_pose(), info

        if not measurement_ok:
            self._commit_prediction(R_pred, omega_pred, p_pred, v_pred, P_pred, timestamp)
            info = self._make_info(
                dt=dt,
                pred_only=True,
                measurement_used=False,
                meas_scale=self.max_meas_scale,
                innovation=np.zeros(6, dtype=float),
                reason="measurement_rejected",
            )
            return self.get_pose(), info

        innovation = self._pose_delta_vec(pose_meas, self._assemble_pose(R_pred, p_pred))
        meas_scale = self._measurement_scale(
            mean_residual=mean_residual,
            ninliers=ninliers,
            inlier_ratio=inlier_ratio,
            jump_rejected=jump_rejected,
        )

        H = np.zeros((6, 12), dtype=float)
        H[:3, :3] = np.eye(3, dtype=float)
        H[3:, 6:9] = np.eye(3, dtype=float)
        R = np.diag(
            np.array(
                [self.rot_meas_sigma**2] * 3 + [self.trans_meas_sigma**2] * 3,
                dtype=float,
            )
            * meas_scale
        )

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.pinv(S)
        delta = K @ innovation

        pose_corr = exp_se3(vec_to_se3(np.hstack((delta[:3], delta[6:9]))))
        pose_upd = pose_corr @ self._assemble_pose(R_pred, p_pred)
        R_upd = self._reorthogonalize(pose_upd[:3, :3])
        omega_upd = omega_pred + delta[3:6]
        p_upd = pose_upd[:3, 3]
        v_upd = v_pred + delta[9:12]

        I = np.eye(12, dtype=float)
        KH = K @ H
        P_upd = (I - KH) @ P_pred @ (I - KH).T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        self._R = R_upd
        self._omega = omega_upd
        self._p = p_upd
        self._v = v_upd
        self._P = P_upd
        self._last_timestamp = timestamp

        info = self._make_info(
            dt=dt,
            pred_only=False,
            measurement_used=True,
            meas_scale=meas_scale,
            innovation=innovation,
            reason="updated",
            ninliers=ninliers,
            num_used=used,
            inlier_ratio=inlier_ratio,
            mean_residual=mean_residual,
        )
        return self.get_pose(), info

    def _predict(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alpha, beta, gamma = self._damping_coefficients(dt)

        step_twist = np.hstack((beta * self._omega, beta * self._v))
        delta_pose = exp_se3(vec_to_se3(step_twist))
        pose_pred = delta_pose @ self.get_pose()
        R_pred = self._reorthogonalize(pose_pred[:3, :3])
        omega_pred = alpha * self._omega
        p_pred = pose_pred[:3, 3]
        v_pred = alpha * self._v

        # --- Error-state transition matrix F ---
        F = np.eye(12, dtype=float)
        R_delta = delta_pose[:3, :3]
        t_delta = delta_pose[:3, 3]
        F[0:3, 0:3] = R_delta
        F[0:3, 3:6] = beta * np.eye(3, dtype=float)
        F[3:6, 3:6] = alpha * np.eye(3, dtype=float)
        F[6:9, 0:3] = skew(t_delta) @ R_delta
        F[6:9, 6:9] = R_delta
        F[6:9, 9:12] = beta * np.eye(3, dtype=float)
        F[9:12, 9:12] = alpha * np.eye(3, dtype=float)

        # --- Process noise covariance Q ---
        # Derived from the discrete O-U model with zero-order-hold
        # acceleration input a_k ~ N(0, sigma^2 I):
        #   omega(k+1) = alpha * omega(k) + beta * a_k
        #   theta(k+1) = theta(k) + ... + gamma * a_k  (through Jr)
        # When lambda=0: beta=dt, gamma=dt^2/2, recovering the original Q.
        Q = np.zeros((12, 12), dtype=float)

        G_rot = np.vstack(
            [gamma * np.eye(3, dtype=float), beta * np.eye(3, dtype=float)]
        )
        Q[0:6, 0:6] = (self.rot_accel_sigma**2) * (G_rot @ G_rot.T)

        # Translation block (6x6) — per-axis, no cross-axis coupling.
        trans_q = (self.trans_accel_sigma**2) * np.array(
            [[gamma * gamma, gamma * beta], [gamma * beta, beta * beta]],
            dtype=float,
        )
        for axis in range(3):
            idx = [6 + axis, 9 + axis]
            Q[np.ix_(idx, idx)] = trans_q

        P_pred = F @ self._P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        return R_pred, omega_pred, p_pred, v_pred, P_pred

    def _commit_prediction(
        self,
        R_pred: np.ndarray,
        omega_pred: np.ndarray,
        p_pred: np.ndarray,
        v_pred: np.ndarray,
        P_pred: np.ndarray,
        timestamp: Optional[float],
    ) -> None:
        self._R = R_pred
        self._omega = omega_pred
        self._p = p_pred
        self._v = v_pred
        self._P = P_pred
        self._last_timestamp = timestamp

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

    def _extract_valid_count(self, stats: dict[str, Any]) -> int:
        if "valid_idx" in stats and stats["valid_idx"] is not None:
            return int(np.asarray(stats["valid_idx"]).reshape(-1).size)
        if "correspond_curr3d" in stats and stats["correspond_curr3d"] is not None:
            return int(np.asarray(stats["correspond_curr3d"]).shape[0])
        if "inliers" in stats and stats["inliers"] is not None:
            return int(np.asarray(stats["inliers"]).reshape(-1).size)
        return 0

    def _extract_inlier_stats(self, stats: dict[str, Any]) -> tuple[int, int, float]:
        inliers = np.asarray(stats.get("inliers", np.array([])), dtype=bool).reshape(-1)
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

    def _compute_dt(self, timestamp: Optional[float]) -> float:
        return self._compute_dt_from(self._last_timestamp, timestamp)

    def _compute_dt_from(
        self,
        last_timestamp: Optional[float],
        timestamp: Optional[float],
        *,
        fallback_dt: Optional[float] = None,
    ) -> float:
        if last_timestamp is None or timestamp is None:
            base_dt = self.nominal_dt if fallback_dt is None else fallback_dt
        else:
            base_dt = timestamp - last_timestamp
            if not np.isfinite(base_dt) or base_dt <= 0.0:
                base_dt = self.nominal_dt if fallback_dt is None else fallback_dt
        return float(np.clip(base_dt, self.min_dt, self.max_dt))

    @staticmethod
    def _reorthogonalize(R: np.ndarray) -> np.ndarray:
        """Project a near-rotation matrix back onto SO(3) via SVD."""
        U, _, Vt = np.linalg.svd(R)
        # Ensure det = +1 (proper rotation, not reflection).
        d = np.linalg.det(U @ Vt)
        return U @ np.diag([1.0, 1.0, float(d)]) @ Vt

    def _damping_coefficients(self, dt: float) -> tuple[float, float, float]:
        """Discrete O-U damping coefficients for time step *dt*.

        Returns (alpha, beta, gamma) where:
            alpha = exp(-lambda dt)           velocity decay factor
            beta  = (1 - alpha) / lambda      velocity -> position coupling
            gamma = (dt - beta) / lambda      accel noise -> position coupling

        When lambda = 0 (no damping): alpha=1, beta=dt, gamma=dt^2/2.
        """
        lam = self._lambda
        if lam <= 1e-10:
            return 1.0, dt, 0.5 * dt * dt
        alpha = float(np.exp(-lam * dt))
        beta = (1.0 - alpha) / lam
        gamma = (dt - beta) / lam
        return alpha, float(beta), float(gamma)

    def _make_initial_covariance(self) -> np.ndarray:
        diag = np.array(
            [self.init_pose_rot_sigma**2] * 3
            + [self.init_twist_rot_sigma**2] * 3
            + [self.init_pose_trans_sigma**2] * 3
            + [self.init_twist_trans_sigma**2] * 3,
            dtype=float,
        )
        return np.diag(diag)

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
        reason: str = "idle",
        ninliers: int = 0,
        num_used: int = 0,
        inlier_ratio: float = 0.0,
        mean_residual: float = -1.0,
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
            "meas_scale": float(meas_scale),
            "innovation": innovation,
            "twist": self.get_twist(),
            "reason": reason,
            "ninliers": int(ninliers),
            "num_used": int(num_used),
            "inlier_ratio": float(inlier_ratio),
            "mean_residual": float(mean_residual),
        }
