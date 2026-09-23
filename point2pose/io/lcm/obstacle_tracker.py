from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(slots=True)
class ObstacleSphereEstimate:
    """One SAM2-tracked obstacle approximated as a sphere in the camera frame."""

    center_cam: Optional[np.ndarray]  # (3,) meters, camera frame; None if never seen
    radius_m: float
    center_uv: Optional[np.ndarray]  # (2,) mask center in pixels
    mask_area_px: int
    depth_samples: int
    visible: bool  # mask non-empty and depth valid on this frame
    lost_frames: int = 0
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # HxW bool


class Sam2ObstacleTracker:
    """
    SAM2-only tracker for obstacles that need no 6D pose.

    Owns its own SAM2 camera predictor (independent of the pipeline's segmenter,
    so the obstacle never enters the point-track / registration machinery). Each
    frame the mask is turned into an approximate sphere:

      * 2D center  = center of the mask's bounding box (or its centroid)
      * depth      = median valid depth inside the mask (the visible front surface)
      * center_cam = back-projected pixel, optionally pushed
                     `center_offset_factor * radius` further along the viewing
                     ray so it lands at the sphere's center rather than on its
                     front surface (the median depth of a sphere's visible disk
                     sits R/sqrt(2) in front of its center)
      * radius     = fixed (from config) or estimated from the mask's pixel size

    The estimate is optionally smoothed with an exponential moving average.
    """

    def __init__(self, obstacle_cfg, segmenter_cfg, pipeline_params=None):
        pipeline_params = pipeline_params or {}
        self.cfg = obstacle_cfg
        self._segmenter_cfg = segmenter_cfg

        self.radius_mode = str(obstacle_cfg.get("radius_mode", "fixed")).lower()
        self.radius_m = float(obstacle_cfg.get("radius_m", 0.05))
        self.radius_min_m = float(obstacle_cfg.get("radius_min_m", 0.01))
        self.radius_max_m = float(obstacle_cfg.get("radius_max_m", 0.5))
        self.center_offset_by_radius = bool(
            obstacle_cfg.get("center_offset_by_radius", True)
        )
        # The median depth over a sphere's visible disk lies R/sqrt(2) in front
        # of the center (median of sqrt(1 - rho^2) over a uniform disk), not R.
        self.center_offset_factor = float(
            obstacle_cfg.get("center_offset_factor", 1.0 / np.sqrt(2.0))
        )
        self.center_mode = str(obstacle_cfg.get("center_mode", "bbox")).lower()
        self.min_mask_area = int(obstacle_cfg.get("min_mask_area", 64))
        self.min_depth_samples = int(obstacle_cfg.get("min_depth_samples", 16))
        self.max_mask_pixels = max(
            int(obstacle_cfg.get("max_mask_pixels", 4096)), self.min_depth_samples
        )
        self.min_depth = float(
            obstacle_cfg.get("min_depth", pipeline_params.get("min_depth", 0.1))
        )
        self.max_depth = float(
            obstacle_cfg.get("max_depth", pipeline_params.get("max_depth", 10.0))
        )
        self.smoothing_alpha = float(obstacle_cfg.get("smoothing_alpha", 1.0))
        self.publish_when_lost = bool(obstacle_cfg.get("publish_when_lost", True))
        self.max_lost_frames = int(obstacle_cfg.get("max_lost_frames", 30))

        self.segmenter = None
        self.num_obstacles = 0
        self.initialized = False
        self._points: list[list[list[int]]] = []
        self._labels: list[list[int]] = []
        self.estimates: list[ObstacleSphereEstimate] = []

    # ------------------------------------------------------------------ setup
    def _build_segmenter(self):
        # Local import: the SAM2 backend pulls in torch/torchvision, which the
        # runner deliberately imports only after the cv2 window exists.
        from point2pose.modules.segmenter.sam2_real_time_segmenter import (
            Sam2RealTimeSegmenter,
        )

        return Sam2RealTimeSegmenter(self._segmenter_cfg)

    def ensure_segmenter(self):
        if self.segmenter is None:
            self.segmenter = self._build_segmenter()
        return self.segmenter

    def preview(self, rgb, objects_points, objects_labels):
        """SAM2 preview masks for the obstacle prompts, without starting tracking."""
        seg = self.ensure_segmenter()
        return seg.preview(rgb, objects_points, objects_labels)

    def initialize(self, rgb, objects_points, objects_labels) -> int:
        """Start SAM2 tracking of the obstacle prompts on `rgb`. Returns #obstacles."""
        groups = [
            (pts, lbls)
            for pts, lbls in zip(objects_points, objects_labels)
            if len(pts) > 0
        ]
        self._points = [pts for pts, _ in groups]
        self._labels = [lbls for _, lbls in groups]
        if not groups:
            self.num_obstacles = 0
            self.initialized = False
            return 0

        seg = self.ensure_segmenter()
        seg.clear_input_objects()
        for pts, lbls in groups:
            seg.add_input_object(pts, lbls)
        seg.initialize(rgb)
        self.num_obstacles = int(seg.num_obj)
        self.initialized = self.num_obstacles > 0
        self.estimates = [
            ObstacleSphereEstimate(
                center_cam=None,
                radius_m=self.radius_m,
                center_uv=None,
                mask_area_px=0,
                depth_samples=0,
                visible=False,
            )
            for _ in range(self.num_obstacles)
        ]
        return self.num_obstacles

    def reset(self):
        """Forget the tracked obstacles; the SAM2 model itself is kept loaded."""
        self.num_obstacles = 0
        self.initialized = False
        self.estimates = []
        self._points = []
        self._labels = []
        if self.segmenter is not None:
            self.segmenter.clear_input_objects()

    # ------------------------------------------------------------------ step
    def step(self, frame) -> list[ObstacleSphereEstimate]:
        """Segment `frame.rgb` and update the sphere estimate of every obstacle."""
        if not self.initialized:
            return self.estimates

        _, mask_logits = self.segmenter.segment(frame.rgb)
        masks = self._logits_to_bool_masks(mask_logits, frame.rgb.shape[:2])

        for idx, est in enumerate(self.estimates):
            mask = masks[idx] if idx < len(masks) else None
            self._update_one(est, mask, frame)
        return self.estimates

    def _update_one(self, est: ObstacleSphereEstimate, mask, frame):
        est.mask = mask
        est.visible = False

        if mask is None:
            est.lost_frames += 1
            return

        coords = np.argwhere(mask)
        area = int(coords.shape[0])
        est.mask_area_px = area
        if area < self.min_mask_area:
            est.lost_frames += 1
            return

        y = coords[:, 0].astype(float)
        x = coords[:, 1].astype(float)
        if self.center_mode == "centroid":
            center_uv = np.array([x.mean(), y.mean()], dtype=float)
        else:
            center_uv = np.array(
                [0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())], dtype=float
            )
        est.center_uv = center_uv

        depth, n_samples = self._median_mask_depth(frame, coords)
        est.depth_samples = n_samples
        if depth is None:
            est.lost_frames += 1
            return

        K = np.asarray(frame.intrinsics, dtype=float).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            est.lost_frames += 1
            return

        # Radius: fixed, or from the mask's pixel diameter at the measured depth.
        if self.radius_mode == "estimate":
            diam_px = 0.5 * ((x.max() - x.min() + 1.0) + (y.max() - y.min() + 1.0))
            radius = 0.5 * diam_px * depth / (0.5 * (fx + fy))
            radius = float(np.clip(radius, self.radius_min_m, self.radius_max_m))
        else:
            radius = self.radius_m

        # Surface point on the viewing ray through the mask center.
        surface = np.array(
            [
                (center_uv[0] - cx) * depth / fx,
                (center_uv[1] - cy) * depth / fy,
                depth,
            ],
            dtype=float,
        )
        if self.center_offset_by_radius:
            ray = surface / max(np.linalg.norm(surface), 1e-9)
            center = surface + ray * (self.center_offset_factor * radius)
        else:
            center = surface

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        if est.center_cam is None or alpha >= 1.0:
            est.center_cam = center
            est.radius_m = radius
        else:
            est.center_cam = alpha * center + (1.0 - alpha) * est.center_cam
            est.radius_m = alpha * radius + (1.0 - alpha) * est.radius_m

        est.visible = True
        est.lost_frames = 0

    def _median_mask_depth(self, frame, coords: np.ndarray):
        depth_image = getattr(frame, "depth", None)
        if depth_image is None:
            return None, 0
        depth_arr = np.asarray(depth_image)
        if depth_arr.ndim != 2:
            return None, 0

        if coords.shape[0] > self.max_mask_pixels:
            step = int(np.ceil(coords.shape[0] / float(self.max_mask_pixels)))
            coords = coords[::step]

        raw = depth_arr[coords[:, 0], coords[:, 1]].astype(float, copy=False)
        depth_factor = float(getattr(frame, "depth_factor", 1.0) or 1.0)
        if abs(depth_factor) < 1e-9:
            depth_factor = 1.0
        vals = raw / depth_factor
        valid = np.isfinite(vals) & (vals >= self.min_depth) & (vals <= self.max_depth)
        vals = vals[valid]
        if vals.size < self.min_depth_samples:
            return None, int(vals.size)
        med = float(np.median(vals))
        if not np.isfinite(med) or med <= 0.0:
            return None, int(vals.size)
        return med, int(vals.size)

    @staticmethod
    def _logits_to_bool_masks(mask_logits, hw) -> list[np.ndarray]:
        if mask_logits is None:
            return []
        arr = mask_logits
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[:, 0]
        elif arr.ndim == 2:
            arr = arr[None]
        masks = []
        for m in arr:
            m = m > 0.0
            if m.shape != tuple(hw):
                import cv2

                m = cv2.resize(
                    m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            masks.append(m)
        return masks

    # ------------------------------------------------------------------ export
    def should_publish(self, est: ObstacleSphereEstimate) -> bool:
        if est.center_cam is None:
            return False
        if est.visible:
            return True
        return self.publish_when_lost and est.lost_frames <= self.max_lost_frames

    @staticmethod
    def build_sphere_vector(
        est: ObstacleSphereEstimate, camera_to_world: np.ndarray | None = None
    ) -> np.ndarray:
        """[x y z r] with the center lifted to the world frame."""
        center = np.asarray(est.center_cam, dtype=float).reshape(3)
        if camera_to_world is not None:
            T = np.asarray(camera_to_world, dtype=float).reshape(4, 4)
            center = T[:3, :3] @ center + T[:3, 3]
        return np.asarray(
            [center[0], center[1], center[2], float(est.radius_m)], dtype=np.float32
        )

    @staticmethod
    def obstacle_name_from_index(idx: int) -> str:
        return f"obstacle_{int(idx)}"
