import numpy as np

from point2pose.utils.lie import exp_se3, vec_to_se3
from point2pose.utils.se3_cv_pose_filter import SE3ConstantVelocityFilter


def _pose_from_twist_step(prev_pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    return exp_se3(vec_to_se3(dt * twist)) @ prev_pose


def _translation_error(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def _rotation_error_rad(T_a: np.ndarray, T_b: np.ndarray) -> float:
    R_delta = T_a[:3, :3] @ T_b[:3, :3].T
    trace = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def _good_stats(*, rejected: bool = False) -> dict:
    return {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": bool(rejected)},
    }


def test_constant_velocity_pose_filter_converges_and_tracks_twist():
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.25, 0.04, -0.01, 0.02], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
    )
    filt.initialize(np.eye(4), 0.0)

    pose = np.eye(4)
    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }

    for step_idx in range(1, 16):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        filtered_pose, info = filt.step(
            pose, step_idx * dt, stats, measurement_ok=True, hard_reset=False
        )
        assert info["measurement_used"] is True
        assert filtered_pose is not None

    assert np.allclose(filt.get_twist(), true_twist, atol=5e-3)
    assert _translation_error(filtered_pose, pose) < 5e-3
    assert _rotation_error_rad(filtered_pose, pose) < 5e-3


def test_constant_velocity_pose_filter_does_not_bootstrap_velocity_from_seed_pose():
    dt = 0.1
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
    )
    filt.initialize(np.eye(4), 0.0)

    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }
    first_pose = np.eye(4)
    first_pose[:3, 3] = np.array([0.02, 0.0, 0.0], dtype=float)
    filt.step(first_pose, dt, stats, measurement_ok=True, hard_reset=False)

    assert 0.0 < filt.get_twist()[3] < 0.2
    assert np.allclose(filt.get_twist()[:3], np.zeros(3), atol=1e-9)

    second_pose = np.eye(4)
    second_pose[:3, 3] = np.array([0.03, 0.0, 0.0], dtype=float)
    filt.step(second_pose, 2.0 * dt, stats, measurement_ok=True, hard_reset=False)

    assert np.allclose(filt.get_twist()[3:], np.array([0.1, 0.0, 0.0]), atol=0.02)


def test_constant_velocity_pose_filter_reduces_noise_vs_raw_pose():
    rng = np.random.default_rng(0)
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.15, 0.03, 0.0, 0.01], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.2,
        trans_accel_sigma=0.05,
        rot_meas_sigma=np.deg2rad(1.5),
        trans_meas_sigma=0.01,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
    )
    filt.initialize(np.eye(4), 0.0)

    pose_true = np.eye(4)
    raw_trans_errors = []
    filt_trans_errors = []
    raw_rot_errors = []
    filt_rot_errors = []
    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 0.004,
        "pose_jump_guard_info": {"rejected": False},
    }

    for step_idx in range(1, 41):
        pose_true = _pose_from_twist_step(pose_true, true_twist, dt)
        noise = np.array(
            [
                rng.normal(scale=np.deg2rad(1.0)),
                rng.normal(scale=np.deg2rad(1.0)),
                rng.normal(scale=np.deg2rad(1.0)),
                rng.normal(scale=0.01),
                rng.normal(scale=0.01),
                rng.normal(scale=0.01),
            ],
            dtype=float,
        )
        pose_meas = exp_se3(vec_to_se3(noise)) @ pose_true
        pose_filt, _ = filt.step(
            pose_meas, step_idx * dt, stats, measurement_ok=True, hard_reset=False
        )

        raw_trans_errors.append(_translation_error(pose_meas, pose_true))
        filt_trans_errors.append(_translation_error(pose_filt, pose_true))
        raw_rot_errors.append(_rotation_error_rad(pose_meas, pose_true))
        filt_rot_errors.append(_rotation_error_rad(pose_filt, pose_true))

    assert float(np.mean(filt_trans_errors)) < float(np.mean(raw_trans_errors))
    assert float(np.mean(filt_rot_errors)) < float(np.mean(raw_rot_errors))


def test_constant_velocity_pose_filter_predict_only_on_rejected_measurement():
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.2, 0.1, 0.0, 0.0], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
    )
    filt.initialize(np.eye(4), 0.0)

    stats_ok = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }
    first_pose = _pose_from_twist_step(np.eye(4), true_twist, dt)
    filt.step(first_pose, dt, stats_ok, measurement_ok=True, hard_reset=False)
    second_pose = _pose_from_twist_step(first_pose, true_twist, dt)
    filt.step(second_pose, 2.0 * dt, stats_ok, measurement_ok=True, hard_reset=False)
    predicted_reference = _pose_from_twist_step(filt.get_pose(), filt.get_twist(), dt)

    bad_pose = np.eye(4)
    bad_pose[:3, 3] = np.array([2.0, 0.0, 0.0], dtype=float)
    rejected_stats = {
        "valid_idx": np.array([0, 1], dtype=int),
        "inliers": np.array([True, False], dtype=bool),
        "mean_residual": 0.2,
        "pose_jump_guard_info": {"rejected": True},
    }
    filtered_pose, info = filt.step(
        bad_pose, 3.0 * dt, rejected_stats, measurement_ok=False, hard_reset=False
    )

    assert info["pred_only"] is True
    assert np.allclose(filtered_pose, predicted_reference, atol=1e-6)
    assert _translation_error(filtered_pose, bad_pose) > 1.0


def test_constant_velocity_pose_filter_hard_reset_and_rebootstrap():
    dt = 0.1
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.1,
        trans_accel_sigma=0.05,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
    )
    filt.initialize(np.eye(4), 0.0)

    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }
    pose_a = np.eye(4)
    pose_a[:3, 3] = np.array([0.05, 0.0, 0.0], dtype=float)
    filt.step(pose_a, dt, stats, measurement_ok=True, hard_reset=False)

    reset_pose = np.eye(4)
    reset_pose[:3, 3] = np.array([1.0, 0.5, 0.0], dtype=float)
    filtered_pose, info = filt.step(
        reset_pose, 2.0 * dt, stats, measurement_ok=True, hard_reset=True
    )
    assert info["hard_reset"] is True
    assert np.allclose(filtered_pose, reset_pose, atol=1e-9)
    assert np.allclose(filt.get_twist(), np.zeros(6), atol=1e-9)

    pose_b = np.eye(4)
    pose_b[:3, 3] = np.array([1.1, 0.5, 0.0], dtype=float)
    filt.step(pose_b, 3.0 * dt, stats, measurement_ok=True, hard_reset=False)
    assert np.allclose(filt.get_twist()[3:], np.array([1.0, 0.0, 0.0]), atol=0.15)


def test_velocity_damping_decays_toward_zero_during_prediction_only():
    """With damping enabled, velocity should decay when no measurements arrive."""
    dt = 0.1
    half_life = 0.5  # velocity halves every 0.5s
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        velocity_damping_half_life=half_life,
    )
    filt.initialize(np.eye(4), 0.0)

    # Feed a few good measurements to build up velocity.
    true_twist = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=float)
    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }
    pose = np.eye(4)
    for i in range(1, 11):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        filt.step(pose, i * dt, stats, measurement_ok=True, hard_reset=False)

    v_before = filt.get_twist()[3:].copy()
    assert np.linalg.norm(v_before) > 0.05  # filter learned some velocity

    # Now run prediction-only steps (no measurement).
    for i in range(11, 31):
        filt.step(None, i * dt, stats, measurement_ok=False, hard_reset=False)

    v_after = filt.get_twist()[3:]
    # After 2 seconds of prediction-only with half_life=0.5s (4 half-lives),
    # velocity should have decayed significantly.
    assert np.linalg.norm(v_after) < 0.15 * np.linalg.norm(v_before)


def test_velocity_damping_disabled_preserves_velocity_during_prediction():
    """Without damping, velocity should not decay during prediction-only steps."""
    dt = 0.1
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.01,
        trans_accel_sigma=0.005,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        # velocity_damping_half_life not set — default no damping
    )
    filt.initialize(np.eye(4), 0.0)

    true_twist = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=float)
    stats = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "mean_residual": 1e-4,
        "pose_jump_guard_info": {"rejected": False},
    }
    pose = np.eye(4)
    for i in range(1, 11):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        filt.step(pose, i * dt, stats, measurement_ok=True, hard_reset=False)

    v_before = filt.get_twist()[3:].copy()

    # Prediction-only steps.
    for i in range(11, 31):
        filt.step(None, i * dt, stats, measurement_ok=False, hard_reset=False)

    v_after = filt.get_twist()[3:]
    # Without damping, velocity should be almost unchanged (process noise
    # doesn't shrink the velocity, only grows the covariance).
    assert np.linalg.norm(v_after - v_before) < 1e-9


def test_velocity_prediction_can_be_disabled_for_prediction_only_steps():
    dt = 0.1
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.01,
        trans_accel_sigma=0.005,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        velocity_damping_half_life=0.5,
        enable_velocity_prediction=False,
    )
    filt.initialize(np.eye(4), 0.0)

    true_twist = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=float)
    pose = np.eye(4)
    for i in range(1, 11):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        filt.step(pose, i * dt, _good_stats(), measurement_ok=True, hard_reset=False)

    pose_before = filt.get_pose().copy()
    twist_before = filt.get_twist().copy()
    for i in range(11, 16):
        filtered_pose, info = filt.step(
            None,
            i * dt,
            _good_stats(rejected=True),
            measurement_ok=False,
            hard_reset=False,
        )
        assert info["pred_only"] is True
        assert info["velocity_prediction_enabled"] is False
        assert np.allclose(filtered_pose, pose_before, atol=1e-9)
        assert np.allclose(filt.get_twist(), twist_before, atol=1e-9)


def test_twist_observation_can_be_disabled():
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        enable_twist_observation=False,
    )
    filt.initialize(np.eye(4), 0.0)

    pose = np.eye(4)
    for i in range(1, 4):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        _, info = filt.step(
            pose, i * dt, _good_stats(), measurement_ok=True, hard_reset=False
        )

    assert info["twist_obs_used"] is False
    assert info["twist_obs_reason"] == "disabled"


def test_twist_observation_uses_only_reliable_raw_pose_history():
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.2, 0.08, -0.01, 0.02], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.05,
        trans_accel_sigma=0.01,
        rot_meas_sigma=1e-3,
        trans_meas_sigma=1e-4,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        enable_twist_observation=True,
        twist_observation_window_size=5,
        twist_observation_min_poses=2,
        twist_observation_method="median",
    )
    filt.initialize(np.eye(4), 0.0)

    pose_1 = _pose_from_twist_step(np.eye(4), true_twist, dt)
    filt.step(pose_1, dt, _good_stats(), measurement_ok=True, hard_reset=False)

    bad_pose = np.eye(4)
    bad_pose[:3, 3] = np.array([2.0, 0.0, 0.0], dtype=float)
    _, bad_info = filt.step(
        bad_pose,
        2.0 * dt,
        _good_stats(rejected=True),
        measurement_ok=False,
        hard_reset=False,
    )
    assert bad_info["pred_only"] is True

    pose_3 = _pose_from_twist_step(pose_1, true_twist, 2.0 * dt)
    _, info = filt.step(
        pose_3,
        3.0 * dt,
        _good_stats(),
        measurement_ok=True,
        hard_reset=False,
    )

    assert info["twist_obs_used"] is True
    assert info["twist_obs_num_poses"] == 3
    assert np.allclose(info["twist_obs"], true_twist, atol=1e-6)


def test_windowed_twist_observation_is_more_robust_than_last_difference():
    dt = 0.1
    true_twist = np.array([0.0, 0.0, 0.15, 0.03, -0.01, 0.02], dtype=float)
    filt = SE3ConstantVelocityFilter(
        nominal_dt=dt,
        min_dt=dt,
        max_dt=dt,
        rot_accel_sigma=0.1,
        trans_accel_sigma=0.03,
        rot_meas_sigma=np.deg2rad(1.0),
        trans_meas_sigma=0.005,
        min_inliers=3,
        min_inlier_ratio=0.1,
        min_valid_correspondences=3,
        enable_twist_observation=True,
        twist_observation_window_size=5,
        twist_observation_min_poses=5,
        twist_observation_method="median",
    )
    filt.initialize(np.eye(4), 0.0)

    pose = np.eye(4)
    prev_clean_pose = np.eye(4)
    for i in range(1, 5):
        pose = _pose_from_twist_step(pose, true_twist, dt)
        prev_clean_pose = pose.copy()
        filt.step(pose, i * dt, _good_stats(), measurement_ok=True, hard_reset=False)

    true_pose = _pose_from_twist_step(prev_clean_pose, true_twist, dt)
    noisy_pose = exp_se3(
        vec_to_se3(
            np.array(
                [0.0, 0.0, 0.08, 0.03, -0.015, 0.01],
                dtype=float,
            )
        )
    ) @ true_pose
    _, info = filt.step(
        noisy_pose,
        5.0 * dt,
        _good_stats(),
        measurement_ok=True,
        hard_reset=False,
    )

    naive_last_twist = SE3ConstantVelocityFilter._pose_delta_vec(
        noisy_pose, prev_clean_pose
    ) / dt
    robust_err = float(np.linalg.norm(info["twist_obs"] - true_twist))
    naive_err = float(np.linalg.norm(naive_last_twist - true_twist))

    assert info["twist_obs_used"] is True
    assert info["twist_obs_num_samples"] == 4
    assert robust_err < naive_err
