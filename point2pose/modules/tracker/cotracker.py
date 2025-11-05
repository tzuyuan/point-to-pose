import numpy as np
import torch
import cv2
from collections import deque

from typing import Tuple, List
from contextlib import nullcontext

from third_party.cotracker.predictor import CoTrackerOnlinePredictor

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER


@TRACKER.register_module("cotracker3_online")
class CoTrackerRealtimeTracker(Tracker):

    def __init__(self, config):
        super().__init__(config)
        self.name = "cotracker3_online"

        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._resize_height = config.get("resize_height", 480)
        self._resize_width = config.get("resize_width", 640)
        self._device = config.get("device", "cuda")
        self._window_len = config.get("window_len", 16)
        self._v2 = config.get("v2", False)
        self._checkpoint_path = config.get("checkpoint_path", "cotracker3_online.pth")

        # Build predictor and set FP32
        self._model = CoTrackerOnlinePredictor(
            checkpoint=self._checkpoint_path, v2=self._v2, window_len=self._window_len
        ).to(self._device)

        self._window_frames = deque(maxlen=self._window_len)
        self._is_first_frame = True
        self._query_points = torch.tensor([], dtype=torch.float32, device=self._device)
        self._new_query_points = torch.tensor(
            [], dtype=torch.float32, device=self._device
        )
        self._need_commit = False

    def initialize(self, frame):
        return True

    def add_query_points(self, frame, new_points: np.ndarray) -> np.ndarray:
        """Add new query points to the tracker.

        Args:
            frame: Frame object.
            new_points: [num_points, 2], [x, y]

        Returns:
            indices: [num_points], indices of the newly added points
        """
        new_query_points = self._convert_select_points_to_query_points(
            frame.id, new_points
        )
        new_query_points = torch.tensor(
            new_query_points, dtype=torch.float32, device=self._device
        )
        self._query_points = torch.cat((self._query_points, new_query_points), axis=0)
        self._new_query_points = new_query_points
        self._need_commit = True

        print("new query points: ", new_query_points[:, 0])
        print("_new query points: ", self._new_query_points[:, 0])

        print(f"Added {new_query_points.shape[0]} query points")
        print(f"Total query points: {self._query_points.shape[0]}")
        print(
            f"return indices: {np.arange(self._query_points.shape[0] - new_query_points.shape[0], self._query_points.shape[0])}"
        )

        return np.arange(
            self._query_points.shape[0] - new_query_points.shape[0],
            self._query_points.shape[0],
        )

    def track_once(self, frame):
        """
        Ingest one RGB frame and return predictions at THIS frame for all queries.

        Returns
        -------
        predictions: (K,2) in resized px [x,y]
        uncertainties: (K,1)
        visibles: (K,1)
        """
        rgb_resize = cv2.resize(frame.rgb, (self._resize_width, self._resize_height))
        rgb_resize_pinned = torch.from_numpy(rgb_resize).pin_memory()
        rgb_resize_tensor = rgb_resize_pinned.to(self._device, non_blocking=True)

        predictions = None
        visibles = None
        print("need commit: ", self._need_commit)
        # we copy the first window_len frames to the window_frames deque
        if self._is_first_frame:
            for i in range(self._model.model.window_len):
                self._window_frames.append(rgb_resize_tensor)

            with torch.no_grad():
                predictions, uncertainties, visibles = self._process_step_queries(
                    self._window_frames,
                    is_first_step=True,
                    queries=self._new_query_points[None],
                    need_commit=self._need_commit,
                )
                self._need_commit = False
                self._is_first_frame = False
        else:
            self._window_frames.append(rgb_resize_tensor)

            with torch.no_grad():
                predictions, uncertainties, visibles = self._process_step_queries(
                    self._window_frames,
                    is_first_step=False,
                    queries=self._new_query_points[None],
                    need_commit=self._need_commit,
                )
                self._need_commit = False
        if predictions is not None and visibles is not None:
            print("predictions: ", predictions.shape)
            print("uncertainties: ", uncertainties.shape)
            print("uncertainties: ", uncertainties)
            print("visibles: ", visibles.shape)
            return (
                predictions.squeeze(0).cpu().numpy()[-1, :, :],
                uncertainties.squeeze(0).cpu().numpy()[-1, :],
                visibles.squeeze(0).cpu().numpy()[-1, :],
            )
        else:
            return (
                self._query_points.cpu().numpy(),
                0.1 * np.ones((self._query_points.shape[0])),
                np.ones((self._query_points.shape[0])),
            )

    def _process_step_queries(self, window_frames, is_first_step, queries, need_commit):
        video_chunk = (
            torch.stack(list(window_frames)).float().permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        if need_commit:
            return self._model(
                video_chunk,
                is_first_step=is_first_step,
                queries=queries,
            )
        else:
            return self._model(
                video_chunk,
                is_first_step=is_first_step,
            )

    def _process_step_grid(
        self, window_frames, is_first_step, grid_size, grid_query_frame
    ):
        video_chunk = (
            torch.stack(list(window_frames)).float().permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self._model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    def _convert_select_points_to_query_points(self, frame_id, points):
        """Convert select points to query points.

        Args:
            points: [num_points, 2], [x, y]

        Returns:
            query_points: [frame_id, num_points, 3], [t, x, y]
        """
        points = np.stack(points)
        query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = frame_id
        query_points[:, 1] = points[:, 0] / self._img_width * self._resize_width
        query_points[:, 2] = points[:, 1] / self._img_height * self._resize_height
        return query_points

    def _convert_track_to_image_points(self, tracks):
        """Convert track to image points.

        Args:
            tracks: [num_points, 2], [ x, y]

        Returns:
            image_points: [num_points, 2], [x, y]
        """
        return (
            tracks[:, 0] * self._img_height / self._resize_height,
            tracks[:, 1] * self._img_width / self._resize_width,
        )
