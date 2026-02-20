import cv2
import numpy as np
import torch

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER
from third_party.cotracker.predictor import CoTrackerPredictor


@TRACKER.register_module("cotracker3_offline")
class CoTrackerOfflineTracker(Tracker):
    """Offline CoTracker wrapper with the same external API as online tracker."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "cotracker3_offline"

        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._resize_height = config.get("resize_height", 480)
        self._resize_width = config.get("resize_width", 640)
        self._device = config.get("device", "cuda")
        self._window_len = config.get("window_len", 60)
        self._v2 = config.get("v2", False)
        self._checkpoint_path = config.get("checkpoint_path", "cotracker3_offline.pth")

        self._model = CoTrackerPredictor(
            checkpoint=self._checkpoint_path,
            offline=True,
            v2=self._v2,
            window_len=self._window_len,
        ).to(self._device)

        self._video_frames = []
        self._frame_id_to_local_idx = {}
        self._query_points = torch.zeros(
            (0, 3), dtype=torch.float32, device=self._device
        )

    def initialize(self, frame):
        self._img_height = frame.rgb.shape[0]
        self._img_width = frame.rgb.shape[1]
        print(
            f"[CoTrackerOffline] Initialized with image size {self._img_height}x{self._img_width}."
        )
        return True

    def add_query_points(self, frame, new_points: np.ndarray) -> np.ndarray:
        """Add new query points in [x, y] image coordinates."""
        if new_points is None or len(new_points) == 0:
            return np.zeros((0,), dtype=np.int64)

        frame_local_idx = self._ensure_frame_in_buffer(frame)
        new_query_points = self._convert_select_points_to_query_points(
            frame_local_idx, new_points
        )
        new_query_points = torch.tensor(
            new_query_points, dtype=torch.float32, device=self._device
        )

        old_len = self._query_points.shape[0]
        self._query_points = torch.cat((self._query_points, new_query_points), dim=0)
        new_len = self._query_points.shape[0]
        return np.arange(old_len, new_len)

    def track_once(self, frame):
        frame_local_idx = self._ensure_frame_in_buffer(frame)

        if self._query_points.shape[0] == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        video_chunk = (
            torch.stack(self._video_frames, dim=0).float().permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        with torch.no_grad():
            tracks, visibles = self._model(
                video_chunk,
                queries=self._query_points[None],
                backward_tracking=False,
            )

        current_tracks = tracks[0, frame_local_idx]
        current_visibles = visibles[0, frame_local_idx]
        current_uncertainties = 1.0 - current_visibles.float()

        return (
            current_tracks.cpu().numpy(),
            current_uncertainties.cpu().numpy(),
            current_visibles.cpu().numpy(),
        )

    def _ensure_frame_in_buffer(self, frame) -> int:
        if frame.id in self._frame_id_to_local_idx:
            return self._frame_id_to_local_idx[frame.id]

        rgb_resize = cv2.resize(frame.rgb, (self._resize_width, self._resize_height))
        rgb_resize_tensor = torch.from_numpy(rgb_resize).to(
            self._device, non_blocking=True
        )
        self._video_frames.append(rgb_resize_tensor)
        local_idx = len(self._video_frames) - 1
        self._frame_id_to_local_idx[frame.id] = local_idx
        return local_idx

    def _convert_select_points_to_query_points(self, frame_id, points):
        points = np.stack(points)
        query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = frame_id
        query_points[:, 1] = points[:, 0] / self._img_width * self._resize_width
        query_points[:, 2] = points[:, 1] / self._img_height * self._resize_height
        return query_points
