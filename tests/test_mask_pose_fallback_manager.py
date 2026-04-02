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
        "depth_blend": 1.0,
        "min_mask_area": 4,
        "min_depth_samples": 4,
        "max_mask_pixels": 256,
        "gain": 1.0,
        "max_translation_step": 1.0,
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
