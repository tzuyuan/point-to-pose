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
    # [x, y] -> [x, y] simple scale (no letterboxing)
    x_scale = resize_w / orig_w
    y_scale = resize_h / orig_h
    x = xy[:, 0] * x_scale
    y = xy[:, 1] * y_scale
    return np.stack([x, y], axis=-1).astype(np.float32)


def _resized_xy_to_orig_xy(
    xy: np.ndarray, orig_w: int, orig_h: int, resize_w: int, resize_h: int
) -> np.ndarray:
    # [x, y] -> [x, y] inverse scale
    x_scale = orig_w / resize_w
    y_scale = orig_h / resize_h
    x = xy[:, 0] * x_scale
    y = xy[:, 1] * y_scale
    return np.stack([x, y], axis=-1).astype(np.float32)


@TRACKER.register_module("cotracker3_online")
class CoTrackerRealtimeTracker(Tracker):
    """
    Online CoTracker3 wrapper that matches TAPIR-like API.

    Public API:
      - initialize(frame) -> bool
      - add_query_points(frame, new_points[xy]) -> np.ndarray indices
      - track_once(frame) -> (tracks[N,2], uncertainties[N], visibles[N])
        tracks are [x, y] in ORIGINAL image pixels.
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
        self._frame_ids: deque[int] = deque(maxlen=self._window_len)  # frame ids
        self._frame_buf: deque[np.ndarray] = deque(
            maxlen=self._window_len
        )  # resized frames (Hc,Wc,3) uint8 or float

        # Global queries (we store in ORIGINAL image pixels, [x,y]) and their birth frame id
        self._q_birth: List[int] = []
        self._q_xy_orig: List[Tuple[float, float]] = []  # [x,y]

        # Indices that have been handed to the predictor (0..len-1)
        self._committed_indices: List[int] = []

        # Flags
        self._need_commit = False
        self._ever_committed = False

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
        for p in pts_xy:
            self._q_birth.append(birth)
            self._q_xy_orig.append((float(p[0]), float(p[1])))
        self._need_commit = True
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
        # Resize (no letterbox). If your input is BGR, convert here to RGB before normalization if desired.
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

        # Default outputs
        tracks_out = np.zeros((N_global, 2), dtype=np.float32)
        vis_out = np.zeros((N_global,), dtype=np.float32)
        unc_out = np.ones((N_global,), dtype=np.float32)

        # Not enough frames yet: just echo spawn positions
        if len(self._frame_buf) < self._window_len:
            idx_all = list(range(N_global))
            if idx_all:
                xy_orig = np.array(
                    [self._q_xy_orig[i] for i in idx_all], dtype=np.float32
                )
                tracks_out[idx_all] = xy_orig
                vis_out[idx_all] = 1.0
                unc_out[idx_all] = 0.0
            return tracks_out, unc_out, vis_out

        # Build the video chunk (B=1, T, 3, Hc, Wc), normalized to [0,1]
        vid_np = np.stack(list(self._frame_buf), axis=0).astype(np.float32) / 255.0
        video_chunk = torch.tensor(
            vid_np, device=self._device, dtype=torch.float32
        ).permute(0, 3, 1, 2)[
            None
        ]  # (1, T, 3, Hc, Wc)

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
                return self._model(**kwargs)

        # -----------------------------------------------------------------
        # Commit new points (first commit OR late commits)
        # -----------------------------------------------------------------
        if self._need_commit:
            new_start = len(self._committed_indices)
            new_idxs = list(range(new_start, len(self._q_birth)))

            if new_idxs:
                q_txy = self._build_query_tensor_for_indices(
                    new_idxs
                )  # [1,K,3] (t,x,y)
                # Clamp x,y to canvas bounds (t is already in-window index)
                q_txy[:, :, 1].clamp_(0, self._canvas_w - 1)  # x
                q_txy[:, :, 2].clamp_(0, self._canvas_h - 1)  # y

                if not self._ever_committed:
                    # First-ever commit: initialize online processing and store queries inside predictor
                    _ = _predict(
                        video_chunk=video_chunk,
                        is_first_step=True,
                        queries=q_txy,
                        add_support_grid=False,
                    )
                    # No outputs this step (predictor returns (None, None) on first step)
                else:
                    # Late commit: append new queries to predictor's internal query buffer
                    with torch.no_grad():
                        if self._model.queries is None:
                            # Edge case: shouldn't happen, but be safe
                            self._model.queries = q_txy
                            self._model.N = q_txy.shape[1]
                        else:
                            self._model.queries = torch.cat(
                                [
                                    self._model.queries,
                                    q_txy.to(self._model.queries.device),
                                ],
                                dim=1,
                            )
                            # Update N = number of *non-support* queries
                            self._model.N = self._model.queries.shape[1]

                # Bookkeeping
                self._committed_indices.extend(new_idxs)
                self._ever_committed = True
                self._need_commit = False

                # On the exact frame of a first commit, we don't have predictions yet; echo current positions
                if (
                    not self._model.v2
                    and self._model.queries is not None
                    and not new_start
                ):
                    # nothing special needed; we'll fall through to regular step below
                    pass

        # -----------------------------------------------------------------
        # Regular tracking step (after we have committed at least once)
        # -----------------------------------------------------------------
        if self._ever_committed and self._model.queries is not None:
            pred_tracks, pred_vis = _predict(
                video_chunk=video_chunk,
                add_support_grid=False,
            )

            if pred_tracks is None or pred_vis is None:
                # Can happen on the very first commit frame; just echo current positions
                idx_all = list(range(N_global))
                if idx_all:
                    xy_orig = np.array(
                        [self._q_xy_orig[i] for i in idx_all], dtype=np.float32
                    )
                    tracks_out[idx_all] = xy_orig
                    vis_out[idx_all] = 1.0
                    unc_out[idx_all] = 0.0
                return tracks_out, unc_out, vis_out

            # Shapes: tracks (B, T, N, 2) in (x,y), vis (B, T, N) bool/float
            T = pred_tracks.shape[1]
            xy_resized_last = (
                pred_tracks[0, T - 1].detach().float().cpu().numpy()
            )  # (N,2) (x,y)
            vis_last = (
                pred_vis[0, T - 1].detach().float().cpu().numpy().astype(np.float32)
            )  # (N,)

            # Back to original resolution
            xy_orig_all = _resized_xy_to_orig_xy(
                xy_resized_last,
                self._img_width,
                self._img_height,
                self._canvas_w,
                self._canvas_h,
            )  # (N,2) [x,y] for all committed points (order = predictor.queries order)

            # Scatter into global outputs using committed mapping
            # Note: predictor.queries holds exactly len(self._committed_indices) points in that order
            scatter_idx = self._committed_indices
            n_scatter = min(len(scatter_idx), xy_orig_all.shape[0])

            if n_scatter > 0:
                tracks_out[scatter_idx[:n_scatter]] = xy_orig_all[:n_scatter]
                vis_out[scatter_idx[:n_scatter]] = vis_last[:n_scatter]
                unc_out[scatter_idx[:n_scatter]] = 1.0 - vis_last[:n_scatter]

        return tracks_out, unc_out, vis_out

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _build_query_tensor_for_indices(self, idxs: List[int]) -> torch.Tensor:
        """
        Build [1, K, 3] queries for selected global point indices `idxs`.
        Format: [t, x, y] in resized pixels.
        For each point, t is the index of its birth frame within the current window.
        """
        assert len(self._frame_ids) > 0
        frame_id_list = list(self._frame_ids)  # length == window_len when full
        K = len(idxs)

        # Original [x,y] -> resized [x,y]
        xy_orig = np.array(
            [self._q_xy_orig[i] for i in idxs], dtype=np.float32
        )  # [K,2]
        xy_resized = _orig_xy_to_resized_xy(
            xy_orig, self._img_width, self._img_height, self._canvas_w, self._canvas_h
        )  # [K,2] = [x,y]

        q = np.zeros((K, 3), dtype=np.float32)
        for j, gi in enumerate(idxs):
            birth_fid = self._q_birth[gi]
            try:
                t = frame_id_list.index(birth_fid)  # 0..T-1
            except ValueError:
                # Birth frame fell out of the window; attach to last frame and fix birth
                t = len(frame_id_list) - 1
                self._q_birth[gi] = frame_id_list[-1]
            q[j, 0] = float(t)  # t
            q[j, 1] = xy_resized[j, 0]  # x
            q[j, 2] = xy_resized[j, 1]  # y

        return torch.from_numpy(q)[None].to(self._device, dtype=torch.float32)
