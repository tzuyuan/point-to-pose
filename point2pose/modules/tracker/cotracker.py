import numpy as np
import torch
import cv2
from collections import deque
from typing import Tuple, List
from contextlib import nullcontext

from cotracker.predictor import CoTrackerOnlinePredictor

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER


def _no_autocast_ctx(device: str):
    dev = device.lower()
    if "cuda" in dev:
        return torch.autocast(device_type="cuda", enabled=False)
    if dev == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return nullcontext()


def _orig_xy_to_resized_xy(
    xy: np.ndarray, orig_w: int, orig_h: int, resize_w: int, resize_h: int
) -> np.ndarray:
    """
    Map original [x,y] to resized [x,y] pixels using simple scaling.
    (x,y) input -> (x,y) output
    """
    x_scale = resize_w / orig_w
    y_scale = resize_h / orig_h
    x = xy[:, 0] * x_scale
    y = xy[:, 1] * y_scale
    return np.stack([x, y], axis=-1).astype(np.float32)


def _resized_xy_to_orig_xy(
    xy: np.ndarray, orig_w: int, orig_h: int, resize_w: int, resize_h: int
) -> np.ndarray:
    """
    Map resized [x,y] pixels back to original [x,y] using simple scaling.
    (x,y) input -> (x,y) output
    """
    x_scale = orig_w / resize_w
    y_scale = orig_h / resize_h
    x = xy[:, 0] * x_scale
    y = xy[:, 1] * y_scale
    return np.stack([x, y], axis=-1).astype(np.float32)


@TRACKER.register_module("cotracker3_online")
class CoTrackerRealtimeTracker(Tracker):
    """
    Online CoTracker3 tracker using sangminkim-99/co-tracker-realtime.

    Public API (matches TAPIR wrapper):
      - initialize(frame) -> bool
      - add_query_points(frame, new_points[xy]) -> np.ndarray indices
      - track_once(frame) -> (tracks[N,2], uncertainties[N], visibles[N])
        NOTE: tracks are returned in [x, y] format.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "cotracker3_online"

        # Original and model canvas sizes
        self._img_height = int(config.get("img_height", 480))
        self._img_width = int(config.get("img_width", 640))
        self._canvas_h = int(config.get("resize_height", 256))
        self._canvas_w = int(config.get("resize_width", 256))

        # Device
        if "device" in config:
            self._device = str(config["device"])
        else:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        # Sliding window
        self._window_len = int(config.get("window_len", 16))
        assert self._window_len >= 2, "window_len must be >= 2"

        # Model / checkpoint
        self._ckpt = config.get("checkpoint_path", None)
        self._v2 = bool(config.get("v2", False))

        # Build predictor and set FP32
        self._model = CoTrackerOnlinePredictor(
            checkpoint=self._ckpt, v2=self._v2, window_len=self._window_len
        ).to(self._device)
        try:
            if hasattr(self._model, "model") and hasattr(self._model.model, "float"):
                self._model.model.float()
        except Exception:
            pass

        # Online state
        self._frame_ids: deque[int] = deque(maxlen=self._window_len)
        self._frame_buf: deque[np.ndarray] = deque()  # resized frames, uint8

        # Global queries (we store in ORIGINAL image pixels, [x,y])
        self._q_birth: List[int] = []
        self._q_xy_orig: List[Tuple[float, float]] = []  # Stores [x,y]

        # Mapping of currently committed queries to global indices
        self._committed_indices: List[int] = []

        # Predictor interaction flags
        self._need_commit = False
        self._queries_committed = False
        self._primed = False
        self._ever_committed = False

        # Query frame within the window where we inject queries (last frame)
        self._grid_query_frame = self._window_len - 1

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def initialize(self, frame) -> bool:
        return True

    def add_query_points(self, frame, new_points: np.ndarray) -> np.ndarray:
        """
        Add new query points mid-stream.
        new_points: (K,2) original image coords [x, y]
        Returns: global indices of newly added points.
        """
        pts_xy = np.asarray(new_points, dtype=np.float32)
        assert (
            pts_xy.ndim == 2 and pts_xy.shape[1] == 2
        ), "new_points must be (K,2) [x,y]"

        old_len = len(self._q_birth)
        birth = int(frame.id)
        for p in pts_xy:  # Iterate over original [x, y] points
            self._q_birth.append(birth)
            self._q_xy_orig.append((float(p[0]), float(p[1])))
        self._need_commit = True
        self._queries_committed = False
        new_len = len(self._q_birth)
        return np.arange(old_len, new_len, dtype=np.int64)

    def track_once(self, frame):
        """
        Ingest one RGB frame and return predictions at THIS frame for all queries.

        Returns
        -------
        tracks : (N,2) np.float32 in original image coords [x, y]
        uncertainties : (N,) np.float32  ~ (1 - visibility)
        visibles : (N,) np.float32 in [0,1]
        """
        # Simple resize (no letterboxing, like TAPIR)
        resized_img = cv2.resize(
            frame.rgb, (self._canvas_w, self._canvas_h), interpolation=cv2.INTER_LINEAR
        )
        self._frame_buf.append(resized_img)
        self._frame_ids.append(int(frame.id))

        N_global = len(self._q_birth)
        if N_global == 0:
            return (
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
            )

        # Default outputs (global)
        tracks_out = np.zeros((N_global, 2), dtype=np.float32)
        vis_out = np.zeros((N_global,), dtype=np.float32)
        unc_out = np.ones((N_global,), dtype=np.float32)

        # 1. BUFFERS NOT FULL (Frames 0 to 14): Echo spawn positions [x,y]
        if len(self._frame_buf) < self._window_len:
            idx_all = list(range(N_global))
            if idx_all:
                xy_orig = np.array(
                    [self._q_xy_orig[i] for i in idx_all], dtype=np.float32
                )
                tracks_out[idx_all] = xy_orig  # Keep as [x,y]
                vis_out[idx_all] = 1.0
                unc_out[idx_all] = 0.0
            return tracks_out, unc_out, vis_out

        # 2. BUFFERS ARE FULL (Frame 15 onwards): Build video chunk
        print(f"self._frame_buf: {len(self._frame_buf)}")
        vid_np = np.stack(list(self._frame_buf), axis=0)  # (T, Hc, Wc, 3)
        print(f"vid_np shape: {vid_np.shape}")
        vid_np = vid_np[..., -self._window_len :, :, :, :]
        video_chunk = torch.tensor(
            vid_np, device=self._device, dtype=torch.float32
        ).permute(0, 3, 1, 2)[None]

        print(f"video_chunk shape: {video_chunk.shape}")

        # Helper to call predictor with autocast disabled and FP32
        def _predict(**kwargs):
            with _no_autocast_ctx(self._device):
                if "video_chunk" in kwargs and isinstance(
                    kwargs["video_chunk"], torch.Tensor
                ):
                    kwargs["video_chunk"] = kwargs["video_chunk"].to(
                        self._device, dtype=torch.float32
                    )
                if "queries" in kwargs and isinstance(kwargs["queries"], torch.Tensor):
                    kwargs["queries"] = kwargs["queries"].to(
                        self._device, dtype=torch.float32
                    )
                return self._model(**kwargs)

        # 3. FIRST COMMIT (Frame 15, if points were added early):
        # This handles the transition from fixed points (0-14) to tracking (15+)
        if self._need_commit and not self._ever_committed:
            q_txy = self._build_query_tensor_at_query_frame()  # [t, x, y]
            # Clamp to canvas bounds
            q_txy[:, :, 1].clamp_(0, self._canvas_w - 1)  # x
            q_txy[:, :, 2].clamp_(0, self._canvas_h - 1)  # y
            print(f"q_txy: {q_txy}")
            is_first_commit = True

            # Perform initial commit and priming
            try:
                pred_tracks, pred_vis = _predict(
                    video_chunk=video_chunk,
                    queries=q_txy,
                    add_support_grid=False,
                    grid_query_frame=self._grid_query_frame,
                    is_first_step=is_first_commit,
                )
            except TypeError:
                pred_tracks, pred_vis = _predict(
                    video_chunk=video_chunk,
                    queries=q_txy,
                    add_support_grid=False,
                    grid_query_frame=self._grid_query_frame,
                )

            # Update state
            self._committed_indices = list(range(N_global))
            self._queries_committed = True
            self._need_commit = False
            self._ever_committed = True
            self._primed = True

            # Process prediction from commit call and return [x,y]
            if pred_tracks is None or pred_vis is None:
                return tracks_out, unc_out, vis_out

            # Use the prediction from the commit step
            T = pred_tracks.shape[1]
            print(f"pred_tracks: {pred_tracks}")
            xy_resized_last = (
                pred_tracks[0, T - 1].detach().float().cpu().numpy()
            )  # [x,y] resized
            vis_last = pred_vis[0, T - 1].detach().float().cpu().numpy()
            print(f"xy_resized_last: {xy_resized_last}")

            # Map back to original [x,y] using simple scaling
            xy_orig = _resized_xy_to_orig_xy(
                xy_resized_last,
                self._img_width,
                self._img_height,
                self._canvas_w,
                self._canvas_h,
            )  # [x,y]
            print(f"xy_orig: {xy_orig}")

            # Scatter into global outputs using committed mapping
            N_model = xy_orig.shape[0]
            scatter_idx: list[int] = (
                self._committed_indices[:N_model]
                if self._committed_indices
                else list(range(N_model))
            )
            tracks_out[scatter_idx] = xy_orig  # Assign as [x,y]
            vis_out[scatter_idx] = vis_last
            unc_out[scatter_idx] = 1.0 - vis_last
            print(f"tracks_out: {tracks_out}")

            return tracks_out, unc_out, vis_out

        # 4. REGULAR TRACKING STEP (Frame 16+):

        if self._ever_committed:
            # Regular online step (only happens if not a commit step)
            # try:
            #     pred_tracks, pred_vis = _predict(
            #         video_chunk=video_chunk,
            #         queries=None,
            #         add_support_grid=False,
            #         grid_query_frame=self._grid_query_frame,
            #     )
            # except TypeError:
            pred_tracks, pred_vis = _predict(
                video_chunk=video_chunk,
                queries=None,
                add_support_grid=False,
            )

            print(f"pred_tracks: {pred_tracks}")

            if pred_tracks is None or pred_vis is None:
                return tracks_out, unc_out, vis_out

            # Process tracks
            T = pred_tracks.shape[1]
            xy_resized_last = (
                pred_tracks[0, T - 1].detach().float().cpu().numpy()
            )  # [x,y] resized
            vis_last = pred_vis[0, T - 1].detach().float().cpu().numpy()
            print(f"xy_resized_last: {xy_resized_last}")
            # Map back to original [x,y] using simple scaling
            xy_orig = _resized_xy_to_orig_xy(
                xy_resized_last,
                self._img_width,
                self._img_height,
                self._canvas_w,
                self._canvas_h,
            )  # [x,y]
            print(f"xy_orig: {xy_orig}")

            # Scatter into global outputs using committed mapping
            N_model = xy_orig.shape[0]
            scatter_idx = (
                self._committed_indices[:N_model]
                if self._committed_indices
                else list(range(N_model))
            )
            tracks_out[scatter_idx] = xy_orig  # Assign as [x,y]
            vis_out[scatter_idx] = vis_last
            unc_out[scatter_idx] = 1.0 - vis_last

            print(f"tracks_out: {tracks_out}")

        return tracks_out, unc_out, vis_out

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _build_query_tensor_at_query_frame(self) -> torch.Tensor:
        """
        Build queries tensor [1, N, 3] in model format: [t, y, x] in resized px.
        Loads from internal [x,y] storage and converts to model format.
        """
        N = len(self._q_xy_orig)
        q = np.zeros((N, 3), dtype=np.float32)
        if N > 0:
            q[:, 0] = float(self._grid_query_frame)  # constant t

            # Convert original [x,y] (stored) to resized [x,y]
            xy_orig = np.array(self._q_xy_orig, dtype=np.float32)  # [x,y]
            xy_resized = _orig_xy_to_resized_xy(
                xy_orig,
                self._img_width,
                self._img_height,
                self._canvas_w,
                self._canvas_h,
            )  # [x,y]

            # Model expects [t, x, y] - note the swap here
            q[:, 1] = xy_resized[:, 0]  # y from [x,y]
            q[:, 2] = xy_resized[:, 1]  # x from [x,y]

        return torch.from_numpy(q)[None].to(self._device, dtype=torch.float32)
