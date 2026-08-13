import os
import sys

import numpy as np

import torch

# lite-tracker uses top-level imports (src.*), so its repo root must be on
# sys.path before importing it.
_LITE_TRACKER_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "third_party", "lite-tracker",
)
_LITE_TRACKER_ROOT = os.path.normpath(_LITE_TRACKER_ROOT)
if _LITE_TRACKER_ROOT not in sys.path:
    sys.path.insert(0, _LITE_TRACKER_ROOT)

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER

# NOTE: the LiteTracker import is deferred to __init__ to keep the tracker
# registry import light (see tapnext_tracker.py cv2/torchvision note).


@TRACKER.register_module("litetracker")
class LiteTrackerTracker(Tracker):
    """
    LiteTracker implementation (MICCAI 2025).

    Training-free causal variant of CoTracker3 online: per-frame tracking via
    a temporal memory buffer with feature reuse and EMA motion-prior track
    initialization. Uses standard CoTracker3 online weights (scaled_online.pth).

    Notes:
    - New points added mid-stream are committed on the next track_once call;
      their appearance template is sampled from that frame at the given
      position. Internal FIFO buffers are padded along the point axis.
    - Anchor-frame (past keyframe) queries are injected by position only;
      correlation search is local, so positions should be approximately valid
      in the current frame.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "litetracker"

        from src.lite_tracker import LiteTracker  # deferred import

        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._device = torch.device(config.get("device", "cuda"))
        self._use_bf16 = bool(config.get("bf16", True))

        checkpoint_path = config.get("checkpoint_path", "scaled_online.pth")

        self._model = LiteTracker(
            window_len=config.get("window_len", 16),
            iters=config.get("iters", 1),
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        self._autocast_dtype = (
            torch.bfloat16
            if self._use_bf16
            and self._device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        # tracking state
        self._queries = None  # [1, N, 3], [t, x, y] in original pixels
        self._pending_points = []  # list of (n_i, 2) float arrays, [x, y]
        self._frame_count = 0

    def initialize(self, frame):
        self._img_height = frame.rgb.shape[0]
        self._img_width = frame.rgb.shape[1]
        self._queries = None
        self._pending_points = []
        self._frame_count = 0
        self._model.init_video_online_processing()
        print(
            f"[LiteTracker] Initialized with image size "
            f"{self._img_height}x{self._img_width}."
        )
        return True

    def add_query_points(self, frame, new_points):
        """
        Add new query points. Committed on the next track_once call with the
        appearance template sampled from that frame.
        Args:
            frame: Frame object (position source only).
            new_points: np.ndarray (num_new_points, 2), [x, y] in original
                        image coordinates.
        Returns:
            indices of the newly added points
        """
        pts = np.stack(new_points).astype(np.float32).reshape(-1, 2)
        old_len = self._num_points()
        self._pending_points.append(pts)
        return np.arange(old_len, old_len + pts.shape[0])

    def track_once(self, frame):
        """
        Track all active points in the frame once.
        Args:
            frame: Frame object. Must contain rgb [H,W,3] np.uint8.

        Returns:
            tracks: np.ndarray (num_points, 2), [x, y] in original image coords
            uncertainties: np.ndarray (num_points,), 1 - confidence
            visibles: np.ndarray (num_points,), bool
        """
        if self._pending_points:
            self._commit_pending()

        if self._queries is None or self._queries.shape[1] == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        frame_t = self._frame_to_tensor(frame.rgb)

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype != torch.float32,
        ):
            coords, viss, confs = self._model(frame_t, self._queries)

        self._frame_count += 1

        # take the newest frame of the window
        tracks = coords[0, -1].float().cpu().numpy().reshape(-1, 2)
        visibles = viss[0, -1].cpu().numpy().reshape(-1).astype(bool)
        uncertainty = (
            (1.0 - confs[0, -1].float()).cpu().numpy().reshape(-1)
        )

        return tracks, uncertainty, visibles

    def _num_points(self):
        n = 0 if self._queries is None else self._queries.shape[1]
        return n + sum(p.shape[0] for p in self._pending_points)

    def _commit_pending(self):
        """Append pending points as queries starting at the next processed
        frame, padding the model's FIFO buffers along the point axis."""
        pts = np.concatenate(self._pending_points, axis=0)
        self._pending_points = []
        n_new = pts.shape[0]

        new_queries = torch.zeros(
            (1, n_new, 3), dtype=torch.float32, device=self._device
        )
        new_queries[0, :, 0] = self._frame_count  # committed on this frame
        new_queries[0, :, 1] = torch.from_numpy(pts[:, 0])  # x
        new_queries[0, :, 2] = torch.from_numpy(pts[:, 1])  # y

        if self._queries is None:
            self._queries = new_queries
            return

        self._queries = torch.cat([self._queries, new_queries], dim=1)

        # Pad model buffers (dim 2 = point axis) for the new tracks. This
        # mirrors what a from-scratch run with late queries would contain:
        # coords_buffer holds the query position (model resolution / stride
        # space) for pre-query frames, vis/conf/flow/corr entries are zero.
        if self._model.online_ind > 0:
            h_ratio = (self._model.model_resolution[0] - 1) / (
                self._img_height - 1
            )
            w_ratio = (self._model.model_resolution[1] - 1) / (
                self._img_width - 1
            )
            coords_pad = torch.stack(
                [
                    torch.from_numpy(pts[:, 0]) * w_ratio,
                    torch.from_numpy(pts[:, 1]) * h_ratio,
                ],
                dim=-1,
            ).to(self._device) / self._model.stride  # (n_new, 2)

            for name in (
                "ema_flow_buffer",
                "corr_embs_buffer",
                "coords_buffer",
                "vis_buffer",
                "conf_buffer",
            ):
                buf = getattr(self._model, name)
                if buf.numel() == 0:
                    continue
                pad_shape = list(buf.shape)
                pad_shape[2] = n_new
                pad = buf.new_zeros(pad_shape)
                if name == "coords_buffer":
                    pad[:] = coords_pad.to(buf.dtype)[None, None]
                setattr(self._model, name, torch.cat([buf, pad], dim=2))
            for i, feat in enumerate(self._model.track_feat_cache):
                if feat.numel() == 0:
                    continue
                pad_shape = list(feat.shape)
                pad_shape[2] = n_new
                self._model.track_feat_cache[i] = torch.cat(
                    [feat, feat.new_zeros(pad_shape)], dim=2
                )

    def _frame_to_tensor(self, rgb):
        """[H, W, 3] uint8 RGB -> (1, 3, H, W) float tensor in [0, 255]."""
        t = torch.from_numpy(np.ascontiguousarray(rgb)).pin_memory()
        t = t.to(self._device, non_blocking=True)
        return t.permute(2, 0, 1).unsqueeze(0).float()
