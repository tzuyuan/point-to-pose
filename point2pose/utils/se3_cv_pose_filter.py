from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from point2pose.utils.lie import exp_se3, log_SE3, se3_to_vec, skew, vec_to_se3


@dataclass(frozen=True)
class _ReliablePoseObservation:
    pose: np.ndarray
    timestamp: Optional[float]


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
        enable_velocity_prediction: bool = True,
        enable_twist_observation: bool = False,
        twist_observation_window_size: int = 5,
        twist_observation_min_poses: int = 3,
        twist_observation_method: str = "median",
        twist_rot_meas_sigma: Optional[float] = None,
        twist_trans_meas_sigma: Optional[float] = None,
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
        self.enable_velocity_prediction = bool(enable_velocity_prediction)

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

        self.enable_twist_observation = bool(enable_twist_observation)
        self.twist_observation_window_size = max(int(twist_observation_window_size), 2)
        self.twist_observation_min_poses = min(
            max(int(twist_observation_min_poses), 2),
            self.twist_observation_window_size,
        )
        self.twist_observation_method = str(twist_observation_method).strip().lower()
        if self.twist_observation_method not in {"mean", "median"}:
            raise ValueError(
                "twist_observation_method must be one of {'mean', 'median'}"
            )
        self.twist_rot_meas_sigma = (
            float(twist_rot_meas_sigma)
            if twist_rot_meas_sigma is not None
            else max(self.init_twist_rot_sigma, self.rot_meas_sigma / self.nominal_dt)
        )
        self.twist_trans_meas_sigma = (
            float(twist_trans_meas_sigma)
            if twist_trans_meas_sigma is not None
            else max(
                self.init_twist_trans_sigma, self.trans_meas_sigma / self.nominal_dt
            )
        )

        self._R: Optional[np.ndarray] = None
        self._omega = np.zeros(3, dtype=float)
        self._p = np.zeros(3, dtype=float)
        self._v = np.zeros(3, dtype=float)
        self._P = self._make_initial_covariance()
        self._last_timestamp: Optional[float] = None
        self._reliable_pose_history = deque(maxlen=self.twist_observation_window_size)

    # -- Public API -----------------------------------------------------------

    def initialize(
        self, pose: np.ndarray, timestamp: Optional[float] = None
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        self._R = pose[:3, :3].copy()
        self._p = pose[:3, 3].copy()
        self._omega = np.zeros(3, dtype=float)
        self._v = np.zeros(3, dtype=float)
        self._P = self._make_initial_covariance()
        self._last_timestamp = self._normalize_timestamp(timestamp)
        self._clear_reliable_pose_history()
        self._record_reliable_pose(pose, self._last_timestamp)
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
            self._clear_reliable_pose_history()
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
        dt = self._compute_dt(timestamp)
        prediction_performed = bool(self.enable_velocity_prediction)
        if prediction_performed:
            R_pred, omega_pred, p_pred, v_pred, P_pred = self._predict(dt)
        else:
            R_pred, omega_pred, p_pred, v_pred, P_pred = self._current_state()

        # Gate: determine whether to accept the measurement.
        reject_reason = self._reject_reason(pose_meas, stats, measurement_ok)
        if reject_reason is not None:
            if prediction_performed:
                self._commit_state(R_pred, omega_pred, p_pred, v_pred, P_pred, timestamp)
            return self.get_pose(), self._make_info(
                dt=dt,
                pred_only=True,
                reason=reject_reason,
            )

        # Full-pose EKF update.
        return self._pose_update(
            pose_meas,
            R_pred,
            omega_pred,
            p_pred,
            v_pred,
            P_pred,
            dt,
            timestamp,
            stats,
        )

    # -- Measurement gating ---------------------------------------------------

    def _reject_reason(
        self,
        pose_meas: Optional[np.ndarray],
        stats: dict[str, Any],
        measurement_ok: bool,
    ) -> Optional[str]:
        """Return a reason string if the measurement should be rejected, else None."""
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

    # -- EKF updates ----------------------------------------------------------

    def _pose_update(
        self,
        pose_meas: np.ndarray,
        R_pred: np.ndarray,
        omega_pred: np.ndarray,
        p_pred: np.ndarray,
        v_pred: np.ndarray,
        P_pred: np.ndarray,
        dt: float,
        timestamp: Optional[float],
        stats: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Full 6-DoF pose EKF update."""
        mean_residual = self._extract_mean_residual(stats)
        ninliers, used, inlier_ratio = self._extract_inlier_stats(stats)
        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        jump_rejected = bool(jump_info.get("rejected", False))

        innovation = self._pose_delta_vec(
            pose_meas, self._assemble_pose(R_pred, p_pred)
        )
        meas_scale = self._measurement_scale(
            mean_residual=mean_residual,
            ninliers=ninliers,
            inlier_ratio=inlier_ratio,
            jump_rejected=jump_rejected,
        )

        H = np.zeros((6, 12), dtype=float)
        H[:3, :3] = np.eye(3)
        H[3:, 6:9] = np.eye(3)
        R_meas = np.diag(
            np.array(
                [self.rot_meas_sigma**2] * 3 + [self.trans_meas_sigma**2] * 3,
                dtype=float,
            )
            * meas_scale
        )

        R_upd, omega_upd, p_upd, v_upd, P_upd = self._ekf_update(
            R_pred,
            omega_pred,
            p_pred,
            v_pred,
            P_pred,
            H,
            R_meas,
            innovation,
        )

        twist_info = self._estimate_twist_observation(
            pose_meas=pose_meas,
            timestamp=timestamp,
            meas_scale=meas_scale,
        )
        if bool(twist_info.get("used", False)):
            R_upd, omega_upd, p_upd, v_upd, P_upd = self._twist_update(
                R_pred=R_upd,
                omega_pred=omega_upd,
                p_pred=p_upd,
                v_pred=v_upd,
                P_pred=P_upd,
                twist_meas=np.asarray(twist_info["twist"], dtype=float).reshape(6),
                twist_sigma=np.asarray(twist_info["sigma"], dtype=float).reshape(6),
            )

        self._commit_state(R_upd, omega_upd, p_upd, v_upd, P_upd, timestamp)
        self._record_reliable_pose(pose_meas, timestamp)

        return self.get_pose(), self._make_info(
            dt=dt,
            measurement_used=True,
            meas_scale=meas_scale,
            innovation=innovation,
            reason="updated",
            ninliers=ninliers,
            num_used=used,
            inlier_ratio=inlier_ratio,
            mean_residual=mean_residual,
            twist_observation_info=twist_info,
        )

    def _ekf_update(
        self,
        R_pred: np.ndarray,
        omega_pred: np.ndarray,
        p_pred: np.ndarray,
        v_pred: np.ndarray,
        P_pred: np.ndarray,
        H: np.ndarray,
        R_meas: np.ndarray,
        innovation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generic EKF update on the 12D error state.

        Works with any measurement dimension: H is (m, 12), R_meas is (m, m),
        and innovation is (m,).
        """
        S = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.pinv(S)
        delta = K @ innovation

        pose_corr = exp_se3(vec_to_se3(np.hstack((delta[:3], delta[6:9]))))
        pose_upd = pose_corr @ self._assemble_pose(R_pred, p_pred)
        R_upd = self._reorthogonalize(pose_upd[:3, :3])
        omega_upd = omega_pred + delta[3:6]
        p_upd = pose_upd[:3, 3]
        v_upd = v_pred + delta[9:12]

        I_12 = np.eye(12, dtype=float)
        KH = K @ H
        P_upd = (I_12 - KH) @ P_pred @ (I_12 - KH).T + K @ R_meas @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        return R_upd, omega_upd, p_upd, v_upd, P_upd

    def _twist_update(
        self,
        R_pred: np.ndarray,
        omega_pred: np.ndarray,
        p_pred: np.ndarray,
        v_pred: np.ndarray,
        P_pred: np.ndarray,
        twist_meas: np.ndarray,
        twist_sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.zeros((6, 12), dtype=float)
        H[:3, 3:6] = np.eye(3, dtype=float)
        H[3:, 9:12] = np.eye(3, dtype=float)
        R_meas = np.diag(np.maximum(twist_sigma, 1e-8) ** 2)
        innovation = twist_meas - np.hstack((omega_pred, v_pred))
        return self._ekf_update(
            R_pred,
            omega_pred,
            p_pred,
            v_pred,
            P_pred,
            H,
            R_meas,
            innovation,
        )

    # -- Prediction -----------------------------------------------------------

    def _predict(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alpha, beta, gamma = self._damping_coefficients(dt)

        step_twist = np.hstack(
            (
                beta * self._omega,
                beta * self._v,
            )
        )
        delta_pose = exp_se3(vec_to_se3(step_twist))
        pose_pred = delta_pose @ self.get_pose()
        R_pred = self._reorthogonalize(pose_pred[:3, :3])
        p_pred = pose_pred[:3, 3]
        R_delta = delta_pose[:3, :3]
        t_delta = delta_pose[:3, 3]

        omega_pred = alpha * self._omega
        v_pred = alpha * self._v

        # Error-state transition matrix F.
        F = np.eye(12, dtype=float)
        F[0:3, 0:3] = R_delta
        F[0:3, 3:6] = beta * np.eye(3, dtype=float)
        F[3:6, 3:6] = alpha * np.eye(3, dtype=float)
        F[6:9, 0:3] = skew(t_delta) @ R_delta
        F[6:9, 6:9] = R_delta
        F[6:9, 9:12] = beta * np.eye(3, dtype=float)
        F[9:12, 9:12] = alpha * np.eye(3, dtype=float)

        # Process noise covariance Q.
        # Derived from the discrete O-U model with zero-order-hold
        # acceleration input a_k ~ N(0, sigma^2 I):
        #   omega(k+1) = alpha * omega(k) + beta * a_k
        #   theta(k+1) = theta(k) + ... + gamma * a_k  (through Jr)
        # When lambda=0: beta=dt, gamma=dt^2/2, recovering the original Q.
        Q = np.zeros((12, 12), dtype=float)

        G_rot = np.vstack(
            [
                gamma * np.eye(3, dtype=float),
                beta * np.eye(3, dtype=float),
            ]
        )
        Q[0:6, 0:6] = (self.rot_accel_sigma**2) * (G_rot @ G_rot.T)

        # Translation block (6x6) -- per-axis, no cross-axis coupling.
        trans_q = (self.trans_accel_sigma**2) * np.array(
            [
                [gamma * gamma, gamma * beta],
                [gamma * beta, beta * beta],
            ],
            dtype=float,
        )
        for axis in range(3):
            idx = [6 + axis, 9 + axis]
            Q[np.ix_(idx, idx)] = trans_q

        P_pred = F @ self._P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        return R_pred, omega_pred, p_pred, v_pred, P_pred

    def _current_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._R.copy(),
            self._omega.copy(),
            self._p.copy(),
            self._v.copy(),
            self._P.copy(),
        )

    def _commit_state(
        self,
        R: np.ndarray,
        omega: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        P: np.ndarray,
        timestamp: Optional[float],
    ) -> None:
        """Write the full state (used after both prediction-only and updates)."""
        self._R = R
        self._omega = omega
        self._p = p
        self._v = v
        self._P = P
        self._last_timestamp = timestamp

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

    # -- Twist observation ----------------------------------------------------

    def _estimate_twist_observation(
        self,
        *,
        pose_meas: Optional[np.ndarray],
        timestamp: Optional[float],
        meas_scale: float,
    ) -> dict[str, Any]:
        info = self._default_twist_observation_info()
        info["enabled"] = bool(self.enable_twist_observation)
        info["method"] = self.twist_observation_method

        if not self.enable_twist_observation:
            info["reason"] = "disabled"
            return info
        if pose_meas is None:
            info["reason"] = "missing_pose"
            return info

        history = list(self._reliable_pose_history)
        history.append(
            _ReliablePoseObservation(
                pose=np.asarray(pose_meas, dtype=float).copy(),
                timestamp=timestamp,
            )
        )
        history = history[-self.twist_observation_window_size :]
        info["num_poses"] = int(len(history))

        if len(history) < self.twist_observation_min_poses:
            info["reason"] = "insufficient_history"
            return info

        samples = self._twist_samples_from_history(history)
        info["num_samples"] = int(samples.shape[0])
        if samples.shape[0] < max(1, self.twist_observation_min_poses - 1):
            info["reason"] = "insufficient_samples"
            return info

        twist_obs, inlier_mask = self._robust_average_twist_samples(samples)
        num_inlier_samples = int(np.count_nonzero(inlier_mask))
        if num_inlier_samples <= 0:
            info["reason"] = "no_inlier_samples"
            return info

        samples_used = samples[inlier_mask]
        rot_dispersion, trans_dispersion = self._twist_sample_dispersion(
            samples_used, twist_obs
        )
        sigma = self._twist_observation_sigma(
            meas_scale=meas_scale,
            rot_dispersion=rot_dispersion,
            trans_dispersion=trans_dispersion,
        )

        info.update(
            {
                "used": True,
                "reason": "updated",
                "twist": twist_obs,
                "sigma": sigma,
                "num_inlier_samples": num_inlier_samples,
                "rot_dispersion": float(rot_dispersion),
                "trans_dispersion": float(trans_dispersion),
            }
        )
        return info

    def _twist_samples_from_history(
        self, history: list[_ReliablePoseObservation]
    ) -> np.ndarray:
        samples = []
        for prev_obs, cur_obs in zip(history[:-1], history[1:]):
            dt = self._compute_twist_sample_dt(prev_obs.timestamp, cur_obs.timestamp)
            if not np.isfinite(dt) or dt <= 0.0:
                continue
            twist = self._pose_delta_vec(cur_obs.pose, prev_obs.pose) / float(dt)
            if np.all(np.isfinite(twist)):
                samples.append(np.asarray(twist, dtype=float).reshape(6))

        if not samples:
            return np.empty((0, 6), dtype=float)
        return np.asarray(samples, dtype=float)

    def _robust_average_twist_samples(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(samples, dtype=float).reshape(-1, 6)
        if samples.shape[0] == 0:
            return np.zeros(6, dtype=float), np.zeros((0,), dtype=bool)

        if self.twist_observation_method == "mean" or samples.shape[0] <= 2:
            return np.mean(samples, axis=0), np.ones((samples.shape[0],), dtype=bool)

        center = np.median(samples, axis=0)
        rot_err = np.linalg.norm(samples[:, :3] - center[:3], axis=1)
        trans_err = np.linalg.norm(samples[:, 3:] - center[3:], axis=1)
        residual = rot_err / max(self.twist_rot_meas_sigma, 1e-8) + trans_err / max(
            self.twist_trans_meas_sigma, 1e-8
        )
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        if not np.isfinite(mad) or mad <= 1e-8:
            keep = residual <= max(med, 1e-6)
        else:
            keep = residual <= (med + 3.0 * mad)
        if not np.any(keep):
            keep = np.ones((samples.shape[0],), dtype=bool)
        return np.mean(samples[keep], axis=0), keep

    @staticmethod
    def _twist_sample_dispersion(
        samples: np.ndarray, twist_obs: np.ndarray
    ) -> tuple[float, float]:
        samples = np.asarray(samples, dtype=float).reshape(-1, 6)
        if samples.shape[0] == 0:
            return 0.0, 0.0
        twist_obs = np.asarray(twist_obs, dtype=float).reshape(6)
        residuals = samples - twist_obs[None, :]
        rot_dispersion = float(np.median(np.linalg.norm(residuals[:, :3], axis=1)))
        trans_dispersion = float(np.median(np.linalg.norm(residuals[:, 3:], axis=1)))
        return rot_dispersion, trans_dispersion

    def _twist_observation_sigma(
        self,
        *,
        meas_scale: float,
        rot_dispersion: float,
        trans_dispersion: float,
    ) -> np.ndarray:
        scale = np.sqrt(max(float(meas_scale), 1.0))
        rot_sigma = np.sqrt(self.twist_rot_meas_sigma**2 + float(rot_dispersion) ** 2)
        trans_sigma = np.sqrt(
            self.twist_trans_meas_sigma**2 + float(trans_dispersion) ** 2
        )
        return scale * np.array(
            [rot_sigma] * 3 + [trans_sigma] * 3,
            dtype=float,
        )

    def _clear_reliable_pose_history(self) -> None:
        self._reliable_pose_history.clear()

    def _record_reliable_pose(
        self, pose: Optional[np.ndarray], timestamp: Optional[float]
    ) -> None:
        pose = self._normalize_pose(pose)
        if pose is None:
            return
        self._reliable_pose_history.append(
            _ReliablePoseObservation(
                pose=pose,
                timestamp=self._normalize_timestamp(timestamp),
            )
        )

    def _default_twist_observation_info(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "used": False,
            "method": self.twist_observation_method,
            "reason": "disabled",
            "twist": np.zeros(6, dtype=float),
            "sigma": np.zeros(6, dtype=float),
            "num_poses": 0,
            "num_samples": 0,
            "num_inlier_samples": 0,
            "rot_dispersion": 0.0,
            "trans_dispersion": 0.0,
        }

    # -- Helpers --------------------------------------------------------------

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

    def _compute_twist_sample_dt(
        self,
        last_timestamp: Optional[float],
        timestamp: Optional[float],
    ) -> float:
        if last_timestamp is None or timestamp is None:
            base_dt = self.nominal_dt
        else:
            base_dt = timestamp - last_timestamp
            if not np.isfinite(base_dt) or base_dt <= 0.0:
                base_dt = self.nominal_dt
        return float(max(base_dt, self.min_dt))

    @staticmethod
    def _reorthogonalize(R: np.ndarray) -> np.ndarray:
        """Project a near-rotation matrix back onto SO(3) via SVD."""
        U, _, Vt = np.linalg.svd(R)
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
        twist_observation_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        innovation = (
            np.zeros(6, dtype=float)
            if innovation is None
            else np.asarray(innovation, dtype=float).reshape(6)
        )
        twist_observation_info = (
            self._default_twist_observation_info()
            if twist_observation_info is None
            else dict(twist_observation_info)
        )
        twist_obs = np.asarray(
            twist_observation_info.get("twist", np.zeros(6, dtype=float)),
            dtype=float,
        ).reshape(6)
        twist_sigma = np.asarray(
            twist_observation_info.get("sigma", np.zeros(6, dtype=float)),
            dtype=float,
        ).reshape(6)
        return {
            "dt": float(dt),
            "pred_only": bool(pred_only),
            "measurement_used": bool(measurement_used),
            "hard_reset": bool(hard_reset),
            "velocity_prediction_enabled": bool(self.enable_velocity_prediction),
            "meas_scale": float(meas_scale),
            "innovation": innovation,
            "twist": self.get_twist(),
            "reason": reason,
            "ninliers": int(ninliers),
            "num_used": int(num_used),
            "inlier_ratio": float(inlier_ratio),
            "mean_residual": float(mean_residual),
            "twist_obs_used": bool(twist_observation_info.get("used", False)),
            "twist_obs": twist_obs,
            "twist_obs_sigma": twist_sigma,
            "twist_obs_method": str(
                twist_observation_info.get("method", self.twist_observation_method)
            ),
            "twist_obs_reason": str(twist_observation_info.get("reason", "disabled")),
            "twist_obs_num_poses": int(twist_observation_info.get("num_poses", 0)),
            "twist_obs_num_samples": int(twist_observation_info.get("num_samples", 0)),
            "twist_obs_num_inlier_samples": int(
                twist_observation_info.get("num_inlier_samples", 0)
            ),
            "twist_obs_rot_dispersion": float(
                twist_observation_info.get("rot_dispersion", 0.0)
            ),
            "twist_obs_trans_dispersion": float(
                twist_observation_info.get("trans_dispersion", 0.0)
            ),
        }
