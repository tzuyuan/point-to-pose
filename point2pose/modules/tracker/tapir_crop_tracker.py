"""
TAPIR on a mask-centred crop instead of the full frame.

Every call re-centres a fixed-size window on the union of the SAM masks in
`frame.mask`, crops `frame.rgb` to it, and runs the stock TAPIR path on the
crop. Query points and returned tracks are converted between full-image and
crop coordinates, so the rest of the pipeline sees ordinary full-image pixels.

What this buys and what it does not
-----------------------------------
TAPIR resizes its input to a fixed `resize_height x resize_width` and its
per-frame cost is dominated by the per-point refinement (grows with the number
of tracked points, in chunks of 64), not by the pixel count. Cropping therefore
does NOT by itself make TAPIR faster. What it does:

  * the object occupies far more of the resized input, so tracks are sharper on
    small / distant objects;
  * that headroom lets you lower resize_height/width (e.g. 128) with no loss of
    object detail, which trims the backbone part (~4 ms of ~20 ms on an RTX
    4090) and memory.

The crop size is held fixed after initialisation (TAPIR's query features are
scale-specific), only its position follows the mask.
"""

import cv2
import numpy as np
import torch

from tapnet.utils import transforms

from point2pose.core.module_registry import TRACKER
from point2pose.modules.tracker.tapir_tracker import TapirTracker


@TRACKER.register_module("tapir_crop")
class TapirCropTracker(TapirTracker):
    """
    Extra config keys (on top of TapirTracker's):

      crop_size:            int, [w, h], or "auto" (default). "auto" sizes the
                            window from the mask bbox on the first frame:
                            max(extent) * crop_pad_factor, square.
      crop_pad_factor:      float, default 2.0 (auto mode only).
      crop_min_size:        int px, default 128. Lower bound for auto.
      crop_max_size:        int px, optional upper bound for auto (defaults to
                            the smaller image side).
      crop_recenter_thres_px: float, default 0. Move the window only when the
                            mask centre drifts further than this from the
                            window centre (hysteresis against jitter).
      crop_center_mode:     "bbox" (default) or "centroid" of the mask union.
      crop_debug:           bool, print the window each frame.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "tapir_crop"
        self._crop_size_cfg = config.get("crop_size", "auto")
        self._crop_pad_factor = float(config.get("crop_pad_factor", 2.0))
        self._crop_min_size = int(config.get("crop_min_size", 128))
        self._crop_max_size = config.get("crop_max_size", None)
        self._crop_recenter_thres = float(config.get("crop_recenter_thres_px", 0.0))
        self._crop_center_mode = str(config.get("crop_center_mode", "bbox")).lower()
        self._crop_debug = bool(config.get("crop_debug", False))

        self._crop_wh = None  # (w, h) in full-image pixels, fixed after init
        self._crop_xy = None  # (x0, y0) top-left, follows the mask
        self._crop_frame_id = None  # frame id the current window was computed for
        self.last_crop_box = None  # (x0, y0, w, h) for visualisation / debug

    # ------------------------------------------------------------ window
    def _mask_center_and_extent(self, frame):
        """Union of all object masks -> (center_xy, extent_wh) or (None, None)."""
        mask = getattr(frame, "mask", None)
        if mask is None:
            return None, None
        if torch.is_tensor(mask):
            m = mask
            if m.dim() == 4:
                m = m[:, 0]
            if m.dim() == 3:
                m = (m > 0).any(dim=0)
            else:
                m = m > 0
            coords = torch.nonzero(m, as_tuple=False)
            if coords.numel() == 0:
                return None, None
            coords = coords.cpu().numpy()
        else:
            m = np.asarray(mask)
            if m.ndim == 4:
                m = m[:, 0]
            if m.ndim == 3:
                m = (m > 0).any(axis=0)
            else:
                m = m > 0
            coords = np.argwhere(m)
            if coords.size == 0:
                return None, None
        ys = coords[:, 0].astype(float)
        xs = coords[:, 1].astype(float)
        extent = np.array([xs.max() - xs.min() + 1.0, ys.max() - ys.min() + 1.0])
        if self._crop_center_mode == "centroid":
            center = np.array([xs.mean(), ys.mean()])
        else:
            center = np.array([0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())])
        return center, extent

    def _resolve_crop_size(self, extent):
        max_side = min(self._img_width, self._img_height)
        if self._crop_max_size is not None:
            max_side = min(max_side, int(self._crop_max_size))
        cfg = self._crop_size_cfg
        if isinstance(cfg, str) and cfg.lower() == "auto":
            if extent is None:
                side = max_side
            else:
                side = int(np.ceil(float(np.max(extent)) * self._crop_pad_factor))
            side = int(np.clip(side, self._crop_min_size, max_side))
            w = h = side
        elif isinstance(cfg, (list, tuple)):
            w, h = int(cfg[0]), int(cfg[1])
        else:
            w = h = int(cfg)
        w = int(np.clip(w, 8, self._img_width))
        h = int(np.clip(h, 8, self._img_height))
        return w, h

    def _place_window(self, center):
        """Top-left for a window of self._crop_wh centred on `center`, clamped."""
        w, h = self._crop_wh
        x0 = int(round(center[0] - 0.5 * w))
        y0 = int(round(center[1] - 0.5 * h))
        x0 = int(np.clip(x0, 0, self._img_width - w))
        y0 = int(np.clip(y0, 0, self._img_height - h))
        return x0, y0

    def _update_window(self, frame):
        """Re-centre the window on this frame's mask (with hysteresis)."""
        if self._crop_wh is None:
            _, extent = self._mask_center_and_extent(frame)
            self._crop_wh = self._resolve_crop_size(extent)
        center, _ = self._mask_center_and_extent(frame)
        if center is not None:
            if self._crop_xy is None:
                self._crop_xy = self._place_window(center)
            else:
                w, h = self._crop_wh
                cur_center = np.array([self._crop_xy[0] + 0.5 * w, self._crop_xy[1] + 0.5 * h])
                if np.linalg.norm(center - cur_center) > self._crop_recenter_thres:
                    self._crop_xy = self._place_window(center)
        elif self._crop_xy is None:
            # No mask yet: centre on the image.
            self._crop_xy = self._place_window(
                np.array([0.5 * self._img_width, 0.5 * self._img_height])
            )
        self._crop_frame_id = getattr(frame, "id", None)
        self.last_crop_box = (*self._crop_xy, *self._crop_wh)
        if self._crop_debug:
            print(f"[TAPIR-crop] frame {self._crop_frame_id}: window {self.last_crop_box}")

    def _window_for_frame(self, frame):
        """Window to use for `frame` without disturbing the tracking state."""
        if self._crop_xy is not None and getattr(frame, "id", None) == self._crop_frame_id:
            return self._crop_xy, self._crop_wh
        if self._crop_wh is None:
            self._update_window(frame)
            return self._crop_xy, self._crop_wh
        center, _ = self._mask_center_and_extent(frame)
        if center is None:
            return self._crop_xy, self._crop_wh
        return self._place_window(center), self._crop_wh

    def _crop_and_resize(self, rgb, xy, wh):
        x0, y0 = xy
        w, h = wh
        crop = rgb[y0 : y0 + h, x0 : x0 + w]
        return cv2.resize(crop, (self._resize_width, self._resize_height))

    # ------------------------------------------------------------ API
    def initialize(self, frame):
        ok = super().initialize(frame)
        self._crop_wh = None
        self._crop_xy = None
        self._crop_frame_id = None
        self._update_window(frame)
        print(
            f"[TAPIR-crop] crop window {self._crop_wh[0]}x{self._crop_wh[1]} px "
            f"-> resized to {self._resize_width}x{self._resize_height}"
        )
        return ok

    def track_once(self, frame):
        self._update_window(frame)
        xy, wh = self._crop_xy, self._crop_wh
        rgb_resize = self._crop_and_resize(frame.rgb, xy, wh)
        rgb_resize_tensor = torch.from_numpy(rgb_resize).pin_memory().to(
            self._device, non_blocking=True
        )
        with torch.no_grad():
            tracks_resized, uncertainty, visibles, self._causal_state = (
                self._online_model_predict(
                    self._model,
                    rgb_resize_tensor[None, None],
                    self.query_features,
                    self._causal_state,
                )
            )
            tracks = transforms.convert_grid_coordinates(
                tracks_resized.cpu(), (self._resize_width, self._resize_height), wh
            ).view(-1, 2)
            tracks = tracks.float().numpy()
            tracks[:, 0] += xy[0]
            tracks[:, 1] += xy[1]
            return (
                tracks,
                uncertainty.cpu().float().numpy().reshape(-1),
                visibles.cpu().numpy().reshape(-1),
            )

    def add_query_points(self, frame, new_points):
        xy, wh = self._window_for_frame(frame)
        pts = np.asarray(np.stack(new_points), dtype=np.float32).reshape(-1, 2)
        # full-image -> crop -> resized grid, [t, y, x]
        q = np.zeros((pts.shape[0], 3), dtype=np.float32)
        q[:, 0] = frame.id
        q[:, 1] = (pts[:, 1] - xy[1]) / wh[1] * self._resize_height
        q[:, 2] = (pts[:, 0] - xy[0]) / wh[0] * self._resize_width
        q[:, 1] = np.clip(q[:, 1], 0, self._resize_height - 1)
        q[:, 2] = np.clip(q[:, 2], 0, self._resize_width - 1)
        new_query_points = torch.tensor(q, dtype=torch.float32, device=self._device)

        rgb_resize = self._crop_and_resize(frame.rgb, xy, wh)
        rgb_resize_tensor = torch.tensor(rgb_resize).to(self._device)

        if self.query_points is None:
            old_len = 0
            self.query_points = new_query_points
            self.query_features = self._online_model_init(
                self._model, rgb_resize_tensor[None, None], self.query_points[None]
            )
            self._causal_state = self._model.construct_initial_causal_state(
                self.query_points.shape[0], len(self.query_features.resolutions) - 1
            )
            with torch.no_grad():
                for i in range(len(self._causal_state)):
                    for k, v in self._causal_state[i].items():
                        self._causal_state[i][k] = v.to(self._device)
        else:
            old_len = self.query_points.shape[0]
            self.query_points = torch.cat((self.query_points, new_query_points), axis=0)
            new_qf = self._online_model_init(
                self._model, rgb_resize_tensor[None, None], new_query_points[None]
            )
            self.query_features = self._concat_query_features(self.query_features, new_qf)
            self._causal_state = self._expand_causal_state(new_query_points.shape[0])

        return np.arange(old_len, self.query_points.shape[0])
