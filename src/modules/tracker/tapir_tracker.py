import os
import numpy as np
import copy
from typing import Optional

import torch
import torch.nn.functional as F

import jax

import cv2

from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

from point2pose.src.core.base_tracker import Tracker
from point2pose.src.core.registry import TRACKER


@TRACKER.register_module("tapir")
class TapirTracker(Tracker):
    def __init__(self, config):
        super().__init__(config)
        self.name = "tapir"

        # image related variables
        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._resize_height = config.get("resize_height", 256)
        self._resize_width = config.get("resize_width", 256)

        self._device = config.get("device", "cpu")
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = config.get(
            "checkpoint_path", "causal_bootstapir_checkpoint.pt"
        )

        # TAPIR model
        self._model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self._model.load_state_dict(torch.load(checkpoint_path))
        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        # model related variables
        self._query_points = None
        self._query_features = None
        self._causal_state = None
        self._initialized = False

    def initialize(self, frame):
        """
        Initialize the tracker with the first frame and the selected points.
        This function assumes add_query_points has been called to add the selected points.
        Args:
            frame: [height, width, 3], np.uint8
        """

        # if no query points, return False
        if self._query_points is None:
            return False

        # Initialize query features
        self._query_features = self._online_model_init(
            self._model,
            frame.unsqueeze(0).unsqueeze(0),
            self._query_points[None],
        )
        # Initialize causal state
        self._causal_state = self._model.construct_initial_causal_state(
            self._query_points.shape[0],
            len(self._query_features.resolutions) - 1,
        )
        # self._causal_state = tree.map_structure(lambda x: x.to(self._device), self._causal_state)
        with torch.no_grad():
            for i in range(len(self._causal_state)):
                for k, v in self._causal_state[i].items():
                    self._causal_state[i][k] = v.to(self._device)

        print(f"[Tracker] Initialized with {self._query_points.shape[0]} points.")
        return True

    def track_once(self, rgb_image):
        """
        Track the points in the frame once.
        """
        rgb_resize = cv2.resize(rgb_image, (self._resize_width, self._resize_height))
        frame = torch.tensor(rgb_resize).to(self._device)

        if not self._initialized:
            self._initialized = self.initialize(frame)
            return None, None, None
        else:
            with torch.no_grad():
                # Predict trajectories and occlusions
                tracks_resized, uncertainty, visibles, self._causal_state = (
                    self._online_model_predict(
                        self._model,
                        frame[None, None],
                        self._query_features,
                        self._causal_state,
                    )
                )
                tracks = transforms.convert_grid_coordinates(
                    tracks_resized.cpu(),
                    (self._resize_width, self._resize_height),
                    (self._img_width, self._img_height),
                ).view(-1, 2)

                return tracks, uncertainty.cpu().numpy(), visibles.cpu().numpy()

    def add_query_points(self, frame_id, new_points):
        """ """
        new_query_points = self._convert_select_points_to_query_points(
            frame_id, new_points
        )  # [num_new_points, 3], [t, y, x]

        if self._query_points is None:
            self._query_points = torch.tensor(
                new_query_points, dtype=torch.float32, device=self._device
            )
        else:
            self._query_points = torch.cat(
                (self._query_points, new_query_points), axis=0
            )

    def _preprocess_frames(self, frames):
        """Preprocess frames to model inputs.

        Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8

        Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
        """
        frames = frames.float()
        frames = frames / 255 * 2 - 1
        return frames

    def _sample_query_points(self, click_point):
        if not self._use_multi_points:
            return click_point
        else:
            # Sample random points around the clicked point

            points = self._sample_random_points_in_box_around_pt(
                click_point, self._num_sample_points, box_size=self._sample_box_size
            )
            points = np.concatenate((points, click_point), axis=0)
            return points

    def _convert_select_points_to_query_points(self, frame_id, points):
        """Convert select points to query points.

        Args:
            points: [num_points, 2], [y, x]

        Returns:
            query_points: [frame_id, num_points, 3], [t, y, x]
        """
        points = np.stack(points)
        query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = frame_id
        query_points[:, 1] = points[:, 1] / self._img_height * self._resize_height
        query_points[:, 2] = points[:, 0] / self._img_width * self._resize_width
        return query_points

    def _postprocess_occlusions(self, occlusions, expected_dist):
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles

    def _online_model_init(self, model, frames, query_points):
        """Initialize query features for the query points."""
        frames = self._preprocess_frames(frames)
        feature_grids = model.get_feature_grids(frames, is_training=False)

        query_features = model.get_query_features(
            frames,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        return query_features

    def _online_model_predict(self, model, frames, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""

        frames = self._preprocess_frames(frames)

        # obtain feature grids for the frames
        feature_grids = model.get_feature_grids(frames, is_training=False)

        trajectories = model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )

        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]

        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        expected_distance = trajectories["expected_dist"][-1]
        uncertainty = copy.deepcopy(F.sigmoid(expected_distance))
        visibles = self._postprocess_occlusions(occlusions, expected_distance)

        return tracks, uncertainty, visibles, causal_context
