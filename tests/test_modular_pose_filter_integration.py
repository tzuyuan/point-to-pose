import numpy as np

from point2pose.data_types.front_end_result import FrontEndResult
from point2pose.pipeline.components.pose_filter_manager import PoseFilterManager
from point2pose.utils.lie import exp_se3, vec_to_se3


class _DummyObject:
    def __init__(self):
        self.pose = np.eye(4, dtype=float)
        self.lost = False
        self.mean_residual = -1.0


def test_modular_pipeline_applies_pose_filter_to_frontend_output():
    manager = PoseFilterManager(
        enabled=True,
        log_raw_pose=True,
        min_valid_correspondences=3,
        reset_trans_thres=0.1,
        reset_rot_deg_thres=15.0,
        skip_on_jump_reject=True,
        filter_kwargs={
            "nominal_dt": 0.1,
            "min_dt": 0.1,
            "max_dt": 0.1,
            "rot_accel_sigma": 0.2,
            "trans_accel_sigma": 0.05,
            "rot_meas_sigma": np.deg2rad(1.0),
            "trans_meas_sigma": 0.01,
            "min_inliers": 3,
            "min_inlier_ratio": 0.1,
            "min_valid_correspondences": 3,
        },
    )
    objects = [_DummyObject()]
    manager.initialize(num_obj=1, objects=objects, timestamp=0.0)

    class _Frame:
        def __init__(self, timestamp):
            self.timestamp = timestamp

    frame = _Frame(0.1)
    first_result = FrontEndResult(frame_id=1)
    first_pose = np.eye(4)
    first_pose[:3, 3] = np.array([0.1, 0.0, 0.0], dtype=float)
    first_result.obj_poses[0] = first_pose
    first_result.valid_indices[0] = np.array([0, 1, 2], dtype=int)
    first_result.reg_stats[0] = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    first_result.mean_residuals[0] = 0.001
    first_result.uncertainties = np.array([0.1, 0.1, 0.1], dtype=float)
    first_result.dense_recovery_triggered[0] = False

    manager.apply(frame, first_result, objects)
    PoseFilterManager.update_object_from_frontend(0, objects[0], first_result)

    assert 0 in first_result.obj_poses_raw
    assert 0 in first_result.obj_poses_filtered
    assert 0 in first_result.pose_filter_stats
    assert np.allclose(
        objects[0].pose, first_result.obj_poses_filtered[0], atol=1e-9
    )

    frame.timestamp = 0.2
    second_result = FrontEndResult(frame_id=2)
    true_pose = np.eye(4)
    true_pose[:3, 3] = np.array([0.2, 0.0, 0.0], dtype=float)
    noisy_pose = exp_se3(
        vec_to_se3(np.array([0.0, 0.0, np.deg2rad(1.0), 0.02, 0.0, 0.0], dtype=float))
    ) @ true_pose
    second_result.obj_poses[0] = noisy_pose
    second_result.valid_indices[0] = np.array([0, 1, 2], dtype=int)
    second_result.reg_stats[0] = {
        "valid_idx": np.array([0, 1, 2], dtype=int),
        "inliers": np.array([True, True, True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    second_result.mean_residuals[0] = 0.002
    second_result.uncertainties = np.array([0.1, 0.1, 0.1], dtype=float)
    second_result.dense_recovery_triggered[0] = False

    manager.apply(frame, second_result, objects)
    PoseFilterManager.update_object_from_frontend(0, objects[0], second_result)

    assert 0 in second_result.obj_poses_raw
    assert 0 in second_result.obj_poses_filtered
    assert 0 in second_result.pose_filter_stats
    assert not np.allclose(
        second_result.obj_poses_raw[0], second_result.obj_poses_filtered[0]
    )
    assert np.allclose(
        objects[0].pose, second_result.obj_poses_filtered[0], atol=1e-9
    )
