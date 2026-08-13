import numpy as np

import torch
import torch.nn.functional as F

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER

# NOTE: tapnet.tapnext imports torchvision, and loading torchvision BEFORE the
# first cv2.namedWindow call makes that call spin forever (torchvision 0.19 /
# opencv-python 4.11 GUI conflict). The imports are therefore deferred to
# __init__, which in the realsense demo runs after the window is created.


@TRACKER.register_module("tapnext")
class TapnextTracker(Tracker):
    """
    TAPNext / TAPNext++ tracker implementation.

    Purely causal, per-frame recurrent tracker (ViT + SSM). Tracks 2D points
    in RGB image streams with constant memory and latency that is nearly
    independent of the number of points.

    Notes:
    - Query points are position-only (no appearance features). New points
      added mid-stream are committed on the next track_once call by expanding
      the recurrent state with freshly initialized (zero) caches.
    - Anchor-frame queries (frame.id != current frame) are injected by
      position only; TAPNext cannot rewind its recurrent state to a past
      frame. Positions should therefore be approximately valid in the
      current frame.
    """

    # TAPNext coordinate space is fixed at 256x256 regardless of input
    # resolution; the coordinate head has 256 bins per axis.
    MODEL_SIZE = 256

    def __init__(self, config):
        super().__init__(config)
        self.name = "tapnext"

        # deferred heavy imports (see module docstring note)
        from tapnet.tapnext import tapnext_lru_modules, tapnext_torch_utils
        from tapnet.tapnext.tapnext_torch import TAPNext, TAPNextTrackingState

        self._lru_modules = tapnext_lru_modules
        self._torch_utils = tapnext_torch_utils
        self._tracking_state_cls = TAPNextTrackingState

        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        # 256 for the standard checkpoint; 512 for the 512-finetuned one.
        self._input_resolution = config.get("input_resolution", 256)
        self._visible_threshold = config.get("visible_threshold", 0.5)
        self._certainty_radius = config.get("certainty_radius", 8)
        # "visibility": 1 - sigmoid(visible_logit) — matches the value range the
        #   pipeline's uncertainty gates/sigmas were tuned for on TAPIR.
        # "certainty": 1 - positional probability mass within certainty_radius —
        #   much harsher scale; do not feed it to the pipeline gates untouched.
        self._uncertainty_mode = config.get("uncertainty_mode", "visibility")
        self._use_half = bool(config.get("half_precision", True))

        self._device = torch.device(config.get("device", "cuda"))

        checkpoint_path = config.get("checkpoint_path", "tapnextpp_ckpt.pt")

        self._model = TAPNext(image_size=(self.MODEL_SIZE, self.MODEL_SIZE))
        if checkpoint_path.endswith(".npz"):
            # JAX checkpoint (e.g. bootstapnext_ckpt.npz)
            self._model = tapnext_torch_utils.restore_model_from_jax_checkpoint(
                self._model, checkpoint_path
            )
        else:
            ckpt = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {
                k.removeprefix("tapnext."): v for k, v in state_dict.items()
            }
            self._model.load_state_dict(state_dict)
        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        # tracking state
        self._state = None
        self._pending_points = []  # list of (n_i, 2) float arrays, [x, y]
        self._num_points = 0

    def initialize(self, frame):
        """
        Initialize the tracker with the first frame.
        Args:
            frame: Frame object with rgb [height, width, 3], np.uint8
        """
        self._img_height = frame.rgb.shape[0]
        self._img_width = frame.rgb.shape[1]
        self._state = None
        self._pending_points = []
        self._num_points = 0
        print(
            f"[TAPNext] Initialized with image size "
            f"{self._img_height}x{self._img_width}, "
            f"input resolution {self._input_resolution}."
        )
        return True

    def add_query_points(self, frame, new_points):
        """
        Add new query points to the tracker. Points are committed on the next
        track_once call (TAPNext queries are position-only, so no features are
        extracted here).
        Args:
            frame: Frame object (used for API compatibility; TAPNext injects
                   queries by position on the next processed frame).
            new_points: np.ndarray, shape (num_new_points, 2), [x, y] in
                        original image coordinates.
        Returns:
            indices of the newly added points
        """
        pts = np.stack(new_points).astype(np.float32).reshape(-1, 2)
        self._pending_points.append(pts)
        old_len = self._num_points
        self._num_points += pts.shape[0]
        return np.arange(old_len, self._num_points)

    def track_once(self, frame):
        """
        Track the points in the frame once.
        Args:
            frame: Frame object. Must contain rgb [H,W,3] np.uint8.

        Returns:
            tracks: np.ndarray (num_points, 2), [x, y] in original image coords
            uncertainties: np.ndarray (num_points,), higher = more uncertain
            visibles: np.ndarray (num_points,), bool
        """
        frame_t = self._preprocess_frame(frame.rgb)

        query_points = None
        if self._pending_points:
            new_queries = self._commit_pending()
            if self._state is None:
                query_points = new_queries[None]  # [1, Q, 3]
            else:
                self._state = self._expand_state(self._state, new_queries)

        if self._state is None and query_points is None:
            # no points to track yet
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        autocast_ctx = torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=self._use_half and self._device.type == "cuda",
        )
        with torch.no_grad(), autocast_ctx:
            tracks, track_logits, visible_logits, self._state = self._model(
                video=frame_t,
                query_points=query_points,
                state=self._state,
            )

        # tracks: [1, 1, Q, 2] in [y, x] model space
        tracks_yx = tracks[0, 0].float()
        visible_prob = torch.sigmoid(visible_logits[0, 0, :, 0].float())
        visibles = (visible_prob >= self._visible_threshold).cpu().numpy()

        if self._uncertainty_mode == "certainty":
            certainty = self._torch_utils.tracker_certainty(
                tracks_yx[None, None],
                track_logits.float(),
                self._certainty_radius,
            )
            uncertainty = (1.0 - certainty[0, 0, :, 0]).cpu().numpy()
        else:
            uncertainty = (1.0 - visible_prob).cpu().numpy()

        tracks_xy = tracks_yx.cpu().numpy()[:, ::-1].copy()
        tracks_xy[:, 0] *= self._img_width / self.MODEL_SIZE
        tracks_xy[:, 1] *= self._img_height / self.MODEL_SIZE

        return tracks_xy, uncertainty, visibles

    def _commit_pending(self):
        """Convert pending [x, y] points to a [Q_new, 3] query tensor
        ([t, y, x] in model space) at the current recurrent step."""
        pts = np.concatenate(self._pending_points, axis=0)
        self._pending_points = []

        queries = np.zeros((pts.shape[0], 3), dtype=np.float32)
        queries[:, 0] = 0 if self._state is None else self._state.step
        queries[:, 1] = pts[:, 1] * self.MODEL_SIZE / self._img_height  # y
        queries[:, 2] = pts[:, 0] * self.MODEL_SIZE / self._img_width  # x
        return torch.tensor(queries, dtype=torch.float32, device=self._device)

    def _expand_state(self, state, new_queries):
        """
        Grow the recurrent state to accommodate new query tokens.

        Point tokens sit at the end of the token axis, so appending
        zero-initialized cache entries is equivalent to those queries starting
        fresh at the current frame (rnn_scan treats a None/zero h0 the same).
        Args:
            state: TAPNextTrackingState
            new_queries: [Q_new, 3] tensor, [t, y, x] model space, absolute t
        """
        n_new = new_queries.shape[0]
        query_points = torch.cat(
            [state.query_points, new_queries[None]], dim=1
        )

        new_hidden = []
        for cache in state.hidden_state:
            rg = cache.rg_lru_state  # [(n_tokens), lru_width]
            conv = cache.conv1d_state  # [(n_tokens), taps, width]
            rg_pad = rg.new_zeros((n_new, *rg.shape[1:]))
            conv_pad = conv.new_zeros((n_new, *conv.shape[1:]))
            new_hidden.append(
                self._lru_modules.RecurrentBlockCache(
                    rg_lru_state=torch.cat([rg, rg_pad], dim=0),
                    conv1d_state=torch.cat([conv, conv_pad], dim=0),
                )
            )

        return self._tracking_state_cls(
            step=state.step,
            query_points=query_points,
            hidden_state=new_hidden,
        )

    def _preprocess_frame(self, rgb):
        """[H, W, 3] uint8 RGB -> [1, 1, S, S, 3] float32 in [-1, 1]."""
        t = torch.from_numpy(np.ascontiguousarray(rgb)).pin_memory()
        t = t.to(self._device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        t = F.interpolate(
            t,
            size=(self._input_resolution, self._input_resolution),
            mode="bilinear",
            align_corners=False,
        )
        t = t.div_(127.5).sub_(1.0)
        return t.permute(0, 2, 3, 1).unsqueeze(0)  # [1, 1, S, S, 3]
