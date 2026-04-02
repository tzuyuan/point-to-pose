from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from point2pose.utils.transform import inverse_SE3


class MaskPoseFallbackManager:
    """
    Estimate a conservative pose fallback from the segmentation mask.

    The fallback is translation-only: keep the last object rotation, estimate the
    current box center from the mask center and mask-supported depth, then place
    the object so its box center matches that 3D point.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        only_when_weak: bool,
        weak_min_valid_points: int,
        weak_min_inliers: int,
        weak_mean_residual: float,
        use_on_lost: bool,
        use_on_jump_reject: bool,
        center_mode: str,
        use_mask_depth: bool,
        depth_blend: float,
        min_mask_area: int,
        min_depth_samples: int,
        max_mask_pixels: int,
        gain: float,
        max_translation_step: float,
        clear_lost_on_apply: bool,
        min_depth: float,
        max_depth: float,
        debug: bool = False,
    ):
        self.enabled = bool(enabled)
        self.only_when_weak = bool(only_when_weak)
        self.weak_min_valid_points = max(int(weak_min_valid_points), 0)
        self.weak_min_inliers = max(int(weak_min_inliers), 0)
        self.weak_mean_residual = float(weak_mean_residual)
        self.use_on_lost = bool(use_on_lost)
        self.use_on_jump_reject = bool(use_on_jump_reject)
        self.center_mode = str(center_mode).strip().lower()
        self.use_mask_depth = bool(use_mask_depth)
        self.depth_blend = float(np.clip(depth_blend, 0.0, 1.0))
        self.min_mask_area = max(int(min_mask_area), 1)
        self.min_depth_samples = max(int(min_depth_samples), 1)
        self.max_mask_pixels = max(int(max_mask_pixels), self.min_depth_samples)
        self.gain = float(np.clip(gain, 0.0, 1.0))
        self.max_translation_step = max(float(max_translation_step), 0.0)
        self.clear_lost_on_apply = bool(clear_lost_on_apply)
        self.min_depth = max(float(min_depth), 1e-6)
        self.max_depth = max(float(max_depth), self.min_depth)
        self.debug = bool(debug)

        valid_center_modes = {"bbox", "centroid"}
        if self.center_mode not in valid_center_modes:
            raise ValueError(
                f"Invalid center_mode: {self.center_mode}. "
                f"Expected one of {sorted(valid_center_modes)}."
            )

    def apply(self, frame, fe_result, objects: list) -> None:
        fe_result.mask_fallback_triggered = {}
        fe_result.mask_fallback_pose_before = {}
        fe_result.mask_fallback_pose_after = {}
        fe_result.mask_fallback_stats = {}

        for obj_id in range(len(objects)):
            stats = self.default_stats()
            stats["enabled"] = bool(self.enabled)
            if not self.enabled:
                stats["reason"] = "disabled"
            fe_result.mask_fallback_triggered[obj_id] = False
            fe_result.mask_fallback_stats[obj_id] = stats

        if not self.enabled:
            return

        for obj_id, obj in enumerate(objects):
            pose_before = fe_result.obj_poses.get(obj_id)
            if pose_before is not None:
                fe_result.mask_fallback_pose_before[obj_id] = np.asarray(
                    pose_before, dtype=float
                ).copy()

            pose_after, stats = self._apply_one(
                obj_id=obj_id,
                obj=obj,
                frame=frame,
                fe_result=fe_result,
            )
            fe_result.mask_fallback_stats[obj_id] = stats
            fe_result.mask_fallback_triggered[obj_id] = bool(stats["applied"])

            if pose_after is None:
                continue

            pose_after_arr = np.asarray(pose_after, dtype=float).copy()
            if stats["applied"]:
                fe_result.obj_poses[obj_id] = pose_after_arr
                fe_result.mask_fallback_pose_after[obj_id] = pose_after_arr.copy()
                prev_pose = np.asarray(getattr(obj, "pose", np.eye(4)), dtype=float)
                if prev_pose.shape == (4, 4):
                    fe_result.rel_poses[obj_id] = pose_after_arr @ inverse_SE3(prev_pose)

    @staticmethod
    def default_stats() -> dict[str, Any]:
        return {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "weak_reason": "n/a",
            "center_mode": "bbox",
            "mask_area": 0,
            "mask_center_uv": np.full(2, np.nan, dtype=float),
            "prev_center_cam": np.full(3, np.nan, dtype=float),
            "target_center_cam": np.full(3, np.nan, dtype=float),
            "depth_estimate": np.nan,
            "depth_source": "none",
            "num_depth_samples": 0,
            "translation_delta": np.zeros(3, dtype=float),
        }

    def _apply_one(self, *, obj_id: int, obj, frame, fe_result):
        stats = self.default_stats()
        stats["enabled"] = True
        stats["center_mode"] = self.center_mode

        should_apply, weak_reason = self._should_apply(obj_id, obj, fe_result)
        stats["weak_reason"] = weak_reason
        if not should_apply:
            stats["reason"] = "strong_registration"
            return fe_result.obj_poses.get(obj_id), stats

        base_pose = np.asarray(getattr(obj, "pose", np.eye(4)), dtype=float)
        if base_pose.shape != (4, 4) or not np.all(np.isfinite(base_pose)):
            stats["reason"] = "invalid_base_pose"
            return fe_result.obj_poses.get(obj_id), stats

        frame_mask = self._get_frame_mask(frame, obj_id)
        if frame_mask is None:
            stats["reason"] = "no_mask"
            return fe_result.obj_poses.get(obj_id), stats

        mask_center_uv, mask_area, sampled_coords = self._extract_mask_center(frame_mask)
        stats["mask_area"] = int(mask_area)
        if mask_center_uv is None:
            stats["reason"] = "empty_mask"
            return fe_result.obj_poses.get(obj_id), stats
        if mask_area < self.min_mask_area:
            stats["reason"] = "small_mask"
            return fe_result.obj_poses.get(obj_id), stats
        stats["mask_center_uv"] = mask_center_uv.copy()

        center_pose_prev, center_local, center_mode = self._resolve_center_pose(
            obj, base_pose
        )
        prev_center_cam = np.asarray(center_pose_prev[:3, 3], dtype=float).reshape(3)
        stats["prev_center_cam"] = prev_center_cam.copy()
        prev_depth = float(prev_center_cam[2])
        if not np.isfinite(prev_depth) or prev_depth <= 0.0:
            stats["reason"] = "invalid_prev_depth"
            return fe_result.obj_poses.get(obj_id), stats

        depth_estimate, depth_source, num_depth_samples = self._estimate_mask_depth(
            frame=frame,
            sampled_coords=sampled_coords,
            prev_depth=prev_depth,
        )
        stats["depth_estimate"] = float(depth_estimate)
        stats["depth_source"] = depth_source
        stats["num_depth_samples"] = int(num_depth_samples)
        if not np.isfinite(depth_estimate) or depth_estimate <= 0.0:
            stats["reason"] = "invalid_depth"
            return fe_result.obj_poses.get(obj_id), stats

        target_center_cam = self._pixel_to_camera(
            uv=mask_center_uv,
            depth=depth_estimate,
            intrinsics=getattr(frame, "intrinsics", None),
        )
        if target_center_cam is None:
            stats["reason"] = "invalid_intrinsics"
            return fe_result.obj_poses.get(obj_id), stats

        target_center_cam = prev_center_cam + self.gain * (
            target_center_cam - prev_center_cam
        )
        stats["target_center_cam"] = target_center_cam.copy()

        pose_new = base_pose.copy()
        if center_mode == "local_bbox_center":
            pose_new[:3, 3] = (
                target_center_cam - pose_new[:3, :3] @ center_local.reshape(3)
            )
        else:
            pose_new[:3, 3] = target_center_cam.copy()

        translation_delta = pose_new[:3, 3] - base_pose[:3, 3]
        step_norm = float(np.linalg.norm(translation_delta))
        if self.max_translation_step > 0.0 and step_norm > self.max_translation_step:
            translation_delta *= self.max_translation_step / step_norm
            pose_new[:3, 3] = base_pose[:3, 3] + translation_delta

        if center_mode == "center_pose_via_init":
            init_pose = np.asarray(getattr(obj, "init_pose", np.eye(4)), dtype=float)
            if init_pose.shape != (4, 4):
                stats["reason"] = "invalid_init_pose"
                return fe_result.obj_poses.get(obj_id), stats
            center_pose_new = center_pose_prev.copy()
            center_pose_new[:3, 3] = pose_new[:3, 3].copy()
            pose_new = center_pose_new @ inverse_SE3(init_pose)
            translation_delta = pose_new[:3, 3] - base_pose[:3, 3]

        stats["applied"] = True
        stats["reason"] = "applied"
        stats["translation_delta"] = np.asarray(translation_delta, dtype=float).copy()

        if self.clear_lost_on_apply:
            obj.lost = False

        if self.debug:
            print(
                f"[MaskPoseFallback] Applied obj {obj_id}: "
                f"weak={weak_reason}, depth={depth_estimate:.4f} ({depth_source}), "
                f"dtrans={translation_delta}"
            )

        return pose_new, stats

    def _should_apply(self, obj_id: int, obj, fe_result) -> tuple[bool, str]:
        if not self.only_when_weak:
            return True, "always"

        reasons = []
        stats = dict(fe_result.reg_stats.get(obj_id, {}) or {})

        pose_meas = fe_result.obj_poses.get(obj_id)
        pose_meas_arr = None if pose_meas is None else np.asarray(pose_meas, dtype=float)
        if pose_meas_arr is None or pose_meas_arr.shape != (4, 4):
            reasons.append("missing_pose")

        valid_idx = np.asarray(
            fe_result.valid_indices.get(obj_id, stats.get("valid_idx", np.array([]))),
            dtype=int,
        ).reshape(-1)
        if valid_idx.size < self.weak_min_valid_points:
            reasons.append(f"few_valid_points_{valid_idx.size}")

        inliers = np.asarray(stats.get("inliers", np.array([])), dtype=bool).reshape(-1)
        ninliers = int(np.count_nonzero(inliers))
        if ninliers < self.weak_min_inliers:
            reasons.append(f"few_inliers_{ninliers}")

        mean_residual = float(fe_result.mean_residuals.get(obj_id, -1.0))
        if mean_residual >= 0.0 and mean_residual > self.weak_mean_residual:
            reasons.append(f"high_residual_{mean_residual:.4e}")

        jump_info = stats.get("pose_jump_guard_info", {}) or {}
        if self.use_on_jump_reject and bool(jump_info.get("rejected", False)):
            reasons.append("jump_rejected")
        if self.use_on_lost and bool(getattr(obj, "lost", False)):
            reasons.append("lost")

        if reasons:
            return True, "+".join(reasons)
        return False, "strong"

    def _get_frame_mask(self, frame, obj_id: int):
        mask = getattr(frame, "mask", None)
        if mask is None:
            return None
        shape = getattr(mask, "shape", None)
        if shape is None:
            return None
        if len(shape) == 4:
            if obj_id >= int(shape[0]):
                return None
            return mask[obj_id, 0]
        if len(shape) == 3:
            if obj_id >= int(shape[0]):
                return None
            return mask[obj_id]
        if len(shape) == 2 and obj_id == 0:
            return mask
        return None

    def _extract_mask_center(self, frame_mask) -> tuple[Optional[np.ndarray], int, np.ndarray]:
        if isinstance(frame_mask, torch.Tensor):
            mask_bool = (
                frame_mask > 0 if frame_mask.dtype != torch.bool else frame_mask
            )
            coords = torch.nonzero(mask_bool, as_tuple=False)
            if coords.numel() == 0:
                return None, 0, np.empty((0, 2), dtype=np.int64)
            coords_np = coords.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            mask_arr = np.asarray(frame_mask)
            coords_np = np.argwhere(mask_arr > 0).astype(np.int64, copy=False)
            if coords_np.size == 0:
                return None, 0, np.empty((0, 2), dtype=np.int64)

        area = int(coords_np.shape[0])
        y = coords_np[:, 0].astype(float)
        x = coords_np[:, 1].astype(float)
        if self.center_mode == "bbox":
            center_uv = np.array(
                [0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())], dtype=float
            )
        else:
            center_uv = np.array([float(np.mean(x)), float(np.mean(y))], dtype=float)

        if coords_np.shape[0] > self.max_mask_pixels:
            step = int(np.ceil(coords_np.shape[0] / float(self.max_mask_pixels)))
            coords_np = coords_np[::step]

        return center_uv, area, coords_np

    def _estimate_mask_depth(
        self,
        *,
        frame,
        sampled_coords: np.ndarray,
        prev_depth: float,
    ) -> tuple[float, str, int]:
        if (not self.use_mask_depth) or sampled_coords.shape[0] == 0:
            return float(prev_depth), "prev_center", 0

        depth_image = getattr(frame, "depth", None)
        if depth_image is None:
            return float(prev_depth), "prev_center", 0

        depth_arr = np.asarray(depth_image)
        if depth_arr.ndim != 2:
            return float(prev_depth), "prev_center", 0

        yy = sampled_coords[:, 0]
        xx = sampled_coords[:, 1]
        raw_depth = depth_arr[yy, xx].astype(float, copy=False)
        depth_factor = float(getattr(frame, "depth_factor", 1.0) or 1.0)
        if abs(depth_factor) < 1e-9:
            depth_factor = 1.0
        depth_vals = raw_depth / depth_factor
        valid = (
            np.isfinite(depth_vals)
            & (depth_vals >= self.min_depth)
            & (depth_vals <= self.max_depth)
        )
        depth_vals = depth_vals[valid]
        if depth_vals.size < self.min_depth_samples:
            return float(prev_depth), "prev_center", int(depth_vals.size)

        mask_depth = float(np.median(depth_vals))
        if not np.isfinite(mask_depth) or mask_depth <= 0.0:
            return float(prev_depth), "prev_center", int(depth_vals.size)

        blended_depth = (1.0 - self.depth_blend) * float(prev_depth) + self.depth_blend * mask_depth
        return float(blended_depth), "mask_depth", int(depth_vals.size)

    def _resolve_center_pose(
        self, obj, base_pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, str]:
        bbox_source = getattr(obj, "bbox", None)
        if bbox_source is None:
            bbox_source = getattr(obj, "init_bbox", None)

        center_local = self._extract_bbox_center(bbox_source)
        if isinstance(bbox_source, dict):
            bbox_frame = str(bbox_source.get("frame", "object")).lower()
            if bbox_frame in {"object", "object_local", "mesh"}:
                if center_local is None:
                    center_local = np.zeros(3, dtype=float)
                center_pose = base_pose.copy()
                center_pose[:3, 3] = (
                    base_pose[:3, :3] @ center_local.reshape(3) + base_pose[:3, 3]
                )
                return center_pose, center_local.reshape(3), "local_bbox_center"

        init_pose = np.asarray(getattr(obj, "init_pose", np.eye(4)), dtype=float)
        if init_pose.shape != (4, 4):
            init_pose = np.eye(4, dtype=float)
        center_pose = np.asarray(base_pose, dtype=float) @ init_pose
        return center_pose, np.zeros(3, dtype=float), "center_pose_via_init"

    @staticmethod
    def _extract_bbox_center(bbox_source) -> Optional[np.ndarray]:
        if bbox_source is None:
            return None

        if isinstance(bbox_source, dict):
            center = bbox_source.get("center", None)
            if center is not None:
                try:
                    return np.asarray(center, dtype=float).reshape(3)
                except Exception:
                    return None
            mm = bbox_source.get("bbox", None)
            if mm is not None:
                mm_arr = np.asarray(mm, dtype=float)
                if mm_arr.shape == (2, 3):
                    return 0.5 * (mm_arr[0] + mm_arr[1])
            mn = bbox_source.get("mn", None)
            mx = bbox_source.get("mx", None)
            if mn is not None and mx is not None:
                return 0.5 * (
                    np.asarray(mn, dtype=float).reshape(3)
                    + np.asarray(mx, dtype=float).reshape(3)
                )
            return None

        if hasattr(bbox_source, "center"):
            try:
                return np.asarray(bbox_source.center, dtype=float).reshape(3)
            except Exception:
                return None
        if hasattr(bbox_source, "get_center"):
            try:
                return np.asarray(bbox_source.get_center(), dtype=float).reshape(3)
            except Exception:
                return None
        return None

    @staticmethod
    def _pixel_to_camera(
        *, uv: np.ndarray, depth: float, intrinsics
    ) -> Optional[np.ndarray]:
        if intrinsics is None:
            return None
        K = np.asarray(intrinsics, dtype=float)
        if K.shape != (3, 3):
            return None

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None

        u = float(uv[0])
        v = float(uv[1])
        z = float(depth)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=float)
