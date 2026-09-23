import numpy as np
import torch

from point2pose.data_types.front_end_result import FrontEndResult
from point2pose.pipeline.components.mask_pose_fallback_manager import (
    MaskPoseFallbackManager,
)


class _DummyObject:
    def __init__(self):
        self.pose = np.eye(4, dtype=float)
        self.pose[:3, 3] = np.array([0.0, 0.0, 1.0], dtype=float)
        self.init_pose = np.eye(4, dtype=float)
        self.bbox = {
            "center": np.zeros(3, dtype=float),
            "extent": np.ones(3, dtype=float),
            "frame": "object",
        }
        self.init_bbox = None
        self.lost = False


class _DummyFrame:
    def __init__(self, mask: torch.Tensor, depth: np.ndarray):
        self.mask = mask
        self.depth = depth
        self.depth_factor = 1.0
        self.intrinsics = np.array(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )


def _rect_mask(x0: int, x1: int, y0: int, y1: int, *, h: int = 480, w: int = 640):
    mask = torch.zeros((1, 1, h, w), dtype=torch.bool)
    mask[0, 0, y0:y1, x0:x1] = True
    return mask


def _manager(**kwargs) -> MaskPoseFallbackManager:
    base_kwargs = {
        "enabled": True,
        "only_when_weak": True,
        "weak_min_valid_points": 3,
        "weak_min_inliers": 3,
        "weak_mean_residual": 1e-3,
        "use_on_lost": True,
        "use_on_jump_reject": True,
        "center_mode": "bbox",
        "use_mask_depth": True,
        "min_mask_area": 4,
        "min_depth_samples": 4,
        "max_mask_pixels": 256,
        "gain": 1.0,
        "clear_lost_on_apply": True,
        "min_depth": 0.1,
        "max_depth": 2.0,
    }
    base_kwargs.update(kwargs)
    return MaskPoseFallbackManager(**base_kwargs)


def test_mask_pose_fallback_uses_mask_and_depth_on_weak_frontend_result():
    obj = _DummyObject()
    obj.lost = True
    mask = _rect_mask(325, 336, 235, 246)
    depth = np.ones((480, 640), dtype=np.float32)
    frame = _DummyFrame(mask=mask, depth=depth)

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.rel_poses[0] = np.eye(4, dtype=float)  # f2f mode: FrontEnd sets this
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    manager = _manager()
    manager.apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    assert result.mask_fallback_stats[0]["reason"] == "applied"
    assert np.allclose(
        result.obj_poses[0][:3, 3], np.array([0.1, 0.0, 1.0], dtype=float), atol=1e-6
    )
    assert np.allclose(
        result.rel_poses[0][:3, 3], np.array([0.1, 0.0, 0.0], dtype=float), atol=1e-6
    )
    assert obj.lost is False


def test_mask_pose_fallback_skips_when_frontend_is_strong():
    obj = _DummyObject()
    mask = _rect_mask(325, 336, 235, 246)
    depth = np.ones((480, 640), dtype=np.float32)
    frame = _DummyFrame(mask=mask, depth=depth)

    result = FrontEndResult(frame_id=1)
    pose_before = obj.pose.copy()
    result.obj_poses[0] = pose_before.copy()
    result.valid_indices[0] = np.array([0, 1, 2, 3], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0, 1, 2, 3], dtype=int),
        "inliers": np.array([True, True, True, True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-4

    manager = _manager()
    manager.apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is False
    assert result.mask_fallback_stats[0]["reason"] == "strong_registration"
    assert np.allclose(result.obj_poses[0], pose_before, atol=1e-9)


def test_mask_pose_fallback_accepts_sam2_float_logit_masks():
    """FrontEnd stores `frame.mask` as raw SAM2 logits, not booleans."""
    obj = _DummyObject()
    obj.lost = True
    mask = torch.full((1, 1, 480, 640), -8.0, dtype=torch.float32)
    mask[0, 0, 235:246, 325:336] = 5.0
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    # Only the pixels with positive logits count toward the mask center.
    assert np.allclose(
        result.obj_poses[0][:3, 3], np.array([0.1, 0.0, 1.0], dtype=float), atol=1e-6
    )


def test_mask_pose_fallback_triggers_on_rejected_pose_jump():
    """A rejected jump means the tracks are untrustworthy even when plentiful."""
    obj = _DummyObject()
    mask = _rect_mask(325, 336, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.valid_indices[0] = np.arange(8, dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.arange(8, dtype=int),
        "inliers": np.ones(8, dtype=bool),
        "pose_jump_guard_info": {"rejected": True},
    }
    result.mean_residuals[0] = 1e-4

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    assert result.mask_fallback_stats[0]["weak_reason"] == "jump_rejected"
    assert np.allclose(
        result.obj_poses[0][:3, 3], np.array([0.1, 0.0, 1.0], dtype=float), atol=1e-6
    )


def test_mask_pose_fallback_moves_the_full_distance_in_one_frame():
    obj = _DummyObject()
    obj.lost = True
    # Mask center 200 px right of the principal point => 2.0 m at 1 m depth.
    mask = _rect_mask(515, 526, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    # No per-frame step limit: the whole 2 m lateral correction lands at once.
    assert np.allclose(
        result.obj_poses[0][:3, 3], np.array([2.0, 0.0, 1.0], dtype=float), atol=1e-6
    )


def test_mask_pose_fallback_compute_only_reports_without_moving_the_pose():
    obj = _DummyObject()
    obj.lost = True
    mask = _rect_mask(325, 336, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    pose_before = obj.pose.copy()
    result.obj_poses[0] = pose_before.copy()
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager(compute_only=True).apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is False
    assert result.mask_fallback_stats[0]["reason"] == "compute_only"
    assert np.allclose(
        result.mask_fallback_stats[0]["translation_delta"],
        np.array([0.1, 0.0, 0.0], dtype=float),
        atol=1e-6,
    )
    assert np.allclose(result.obj_poses[0], pose_before, atol=1e-9)
    assert obj.lost is True


def test_mask_pose_fallback_disabled_leaves_pose_untouched():
    obj = _DummyObject()
    obj.lost = True
    mask = _rect_mask(325, 336, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    pose_before = obj.pose.copy()
    result.obj_poses[0] = pose_before.copy()

    _manager(enabled=False).apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is False
    assert result.mask_fallback_stats[0]["reason"] == "disabled"
    assert np.allclose(result.obj_poses[0], pose_before, atol=1e-9)


def test_mask_pose_fallback_keeps_pose_when_mask_is_empty():
    obj = _DummyObject()
    obj.lost = True
    mask = torch.zeros((1, 1, 480, 640), dtype=torch.bool)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    pose_before = obj.pose.copy()
    result.obj_poses[0] = pose_before.copy()
    result.valid_indices[0] = np.array([], dtype=int)
    result.reg_stats[0] = {"pose_jump_guard_info": {"rejected": False}}
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is False
    assert result.mask_fallback_stats[0]["reason"] == "empty_mask"
    assert np.allclose(result.obj_poses[0], pose_before, atol=1e-9)


def test_mask_pose_fallback_never_changes_rotation():
    from scipy.spatial.transform import Rotation

    obj = _DummyObject()
    obj.lost = True
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    obj.pose[:3, :3] = rotation
    mask = _rect_mask(325, 336, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    assert np.allclose(result.obj_poses[0][:3, :3], rotation, atol=1e-12)


def test_mask_pose_fallback_leaves_frame_to_map_rel_pose_unset():
    """In f2m mode FrontEnd leaves rel_pose None, and the optimizers read that as
    'no odometry constraint'. The fallback must not invent one."""
    obj = _DummyObject()
    obj.lost = True
    mask = _rect_mask(325, 336, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.rel_poses[0] = None                     # f2m mode
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    # The pose is still corrected ...
    assert result.mask_fallback_triggered[0] is True
    assert np.allclose(
        result.obj_poses[0][:3, 3], np.array([0.1, 0.0, 1.0], dtype=float), atol=1e-6
    )
    # ... but no relative-pose constraint is fabricated.
    assert result.rel_poses[0] is None


def test_mask_pose_fallback_respects_non_identity_init_pose():
    """The object center, not obj.pose's translation, is what gets corrected.

    With a real init_pose (the bbox pose estimated on the first frame) obj.pose
    is the frame-0 -> frame-i transform, so its translation is nowhere near the
    object center. The correction must be applied to the center and mapped back
    through init_pose, not written into obj.pose's translation directly.
    """
    obj = _DummyObject()
    obj.lost = True
    obj.bbox = None
    obj.init_bbox = None
    # Object sits 1 m in front of the camera; obj.pose is (near) identity and
    # init_pose carries the whole 1 m offset, as the pipeline sets it up.
    obj.pose = np.eye(4, dtype=float)
    obj.init_pose = np.eye(4, dtype=float)
    obj.init_pose[:3, 3] = np.array([0.0, 0.0, 1.0], dtype=float)

    # Mask centered 2 px right of the principal point => a 2 cm correction at
    # 1 m depth with fx = 100, well inside the 10 cm step limit.
    mask = _rect_mask(317, 328, 235, 246)
    frame = _DummyFrame(mask=mask, depth=np.ones((480, 640), dtype=np.float32))

    result = FrontEndResult(frame_id=1)
    result.obj_poses[0] = obj.pose.copy()
    result.valid_indices[0] = np.array([0], dtype=int)
    result.reg_stats[0] = {
        "valid_idx": np.array([0], dtype=int),
        "inliers": np.array([True], dtype=bool),
        "pose_jump_guard_info": {"rejected": False},
    }
    result.mean_residuals[0] = 1e-2

    _manager().apply(frame, result, [obj])

    assert result.mask_fallback_triggered[0] is True
    pose_new = result.obj_poses[0]
    center_new = (pose_new @ obj.init_pose)[:3, 3]
    center_prev = (np.eye(4) @ obj.init_pose)[:3, 3]

    # Depth is preserved (mask depth == previous depth) and only the 2 cm
    # lateral correction is applied to the center.
    assert np.isclose(center_new[2], 1.0, atol=1e-6), f"depth changed: {center_new}"
    assert np.linalg.norm(center_new - center_prev) < 0.03
    # Only the lateral correction moved; rotation is untouched.
    assert np.allclose(pose_new[:3, :3], np.eye(3), atol=1e-9)
