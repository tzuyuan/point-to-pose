import os
import time
import numpy as np
import copy
from typing import Optional

import torch
import torch.nn.functional as F


import cv2

from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

from point2pose.data_types.query_feature import QueryFeatures
from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER

# if torch.cuda.is_available():
#     torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         # turn on tfloat32 for Ampere GPUs
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True


@TRACKER.register_module("tapir")
class TapirTracker(Tracker):
    """
    TAPIR tracker implementation.
    TAPIR performs 2D point tracking in RGB image streams.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "tapir"

        # image related variables
        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._resize_height = config.get("resize_height", 256)
        self._resize_width = config.get("resize_width", 256)
        self._visible_threshold = config.get("visible_threshold", 0.5)
        self._query_chunk_size = config.get("query_chunk_size", 64)
        self._enable_tf32 = config.get("enable_tf32", False)
        self._inactive_uncertainty = float(config.get("inactive_uncertainty", 1.0))

        self._device = config.get("device", "cpu")
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._enable_tf32 and str(self._device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        checkpoint_path = config.get(
            "checkpoint_path", "causal_bootstapir_checkpoint.pt"
        )

        # TAPIR model
        self._model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path)
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        # model related variables
        self.query_points = None
        self.query_features = None
        self._causal_state = None
        self._initialized = False
        self._active_global_ids = np.empty((0,), dtype=np.int64)
        self._num_global_points = 0
        self._last_tracks_full = np.empty((0, 2), dtype=np.float32)

    def initialize(self, frame):
        """
        Initialize the tracker with the first frame and the selected points.
        This function assumes add_query_points has been called to add the selected points.
        Args:
            frame: [height, width, 3], np.uint8
        """
        self._img_height = frame.rgb.shape[0]
        self._img_width = frame.rgb.shape[1]
        print(
            f"[TAPIR] Initialized with image size {self._img_height}x{self._img_width}."
        )
        # if no query points, return False
        if self.query_points is None:
            return False

        print(f"[TAPIR] Initialized with {self.query_points.shape[0]} points.")
        return True

    def track_once(self, frame):
        """
        Track the points in the frame once.
        Args:
            frame: Frame object. Must contain rgb [H,W,3].

        Returns:
            tracks: tracked points in the original image coordinates.
                    np.ndarray, shape (num_points, 2), [y, x]
            uncertainties: np.ndarray, shape (num_points, 1), [uncertainty]
            visibles: np.ndarray, shape (num_points, 1), [visible]
        """
        if self.query_features is None or self._active_global_ids.size == 0:
            return self._build_full_outputs()

        rgb_resize = cv2.resize(frame.rgb, (self._resize_width, self._resize_height))
        rgb_resize_pinned = torch.from_numpy(rgb_resize).pin_memory()
        rgb_resize_tensor = rgb_resize_pinned.to(self._device, non_blocking=True)

        with torch.no_grad():
            # Predict trajectories and occlusions
            tracks_resized, uncertainty, visibles, self._causal_state = (
                self._online_model_predict(
                    self._model,
                    rgb_resize_tensor[None, None],
                    self.query_features,
                    self._causal_state,
                )
            )
            tracks = transforms.convert_grid_coordinates(
                tracks_resized.cpu(),
                (self._resize_width, self._resize_height),
                (self._img_width, self._img_height),
            ).view(-1, 2)

            active_tracks = tracks.float().numpy()
            active_uncertainty = uncertainty.cpu().float().numpy().reshape(-1)
            active_visibles = visibles.cpu().numpy().reshape(-1)
            return self._scatter_active_outputs(
                active_tracks, active_uncertainty, active_visibles
            )

    def add_query_points(self, frame, new_points):
        """
        Add new query points to the tracker.
        Args:
            frame_id: int
            new_points: np.ndarray, shape (num_new_points, 2), [y, x]
        Returns:
            newly added indices
        """
        frame_id = frame.id
        new_query_points = self._convert_select_points_to_query_points(
            frame_id, new_points
        )  # [num_new_points, 3], [t, y, x]
        new_query_points = torch.tensor(
            new_query_points, dtype=torch.float32, device=self._device
        )

        rgb_resize = cv2.resize(frame.rgb, (self._resize_width, self._resize_height))
        rgb_resize_tensor = torch.tensor(rgb_resize).to(self._device)

        global_old_len = self._num_global_points

        if self.query_points is None:
            self.query_points = new_query_points
            # Initialize query features
            self.query_features = self._online_model_init(
                self._model,
                rgb_resize_tensor.unsqueeze(0).unsqueeze(0),
                self.query_points[None],
            )
            # Initialize causal state
            self._causal_state = self._model.construct_initial_causal_state(
                self.query_points.shape[0],
                len(self.query_features.resolutions) - 1,
            )
            # self._causal_state = tree.map_structure(lambda x: x.to(self._device), self._causal_state)
            with torch.no_grad():
                for i in range(len(self._causal_state)):
                    for k, v in self._causal_state[i].items():
                        self._causal_state[i][k] = v.to(self._device)
        else:
            self.query_points = torch.cat((self.query_points, new_query_points), axis=0)

            new_qf = self._online_model_init(
                self._model, rgb_resize_tensor[None, None], new_query_points[None]
            )
            # print(self.query_features.shape, new_qf.shape)
            self.query_features = self._concat_query_features(
                self.query_features, new_qf
            )

            # expand causal state
            self._causal_state = self._expand_causal_state(new_query_points.shape[0])

        # get new length
        n_new = int(new_query_points.shape[0])
        global_new_len = global_old_len + n_new
        new_global_ids = np.arange(global_old_len, global_new_len, dtype=np.int64)
        self._active_global_ids = np.concatenate((self._active_global_ids, new_global_ids))
        self._num_global_points = global_new_len

        new_points_np = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)
        if self._last_tracks_full.shape[0] == 0:
            self._last_tracks_full = new_points_np.copy()
        else:
            self._last_tracks_full = np.concatenate(
                [self._last_tracks_full, new_points_np], axis=0
            )

        return new_global_ids

    def deactivate_query_points(self, global_ids):
        if self.query_features is None or self._active_global_ids.size == 0:
            return np.zeros((0,), dtype=np.int64)

        global_ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        if global_ids.size == 0:
            return np.zeros((0,), dtype=np.int64)

        global_ids = np.unique(global_ids)
        keep_mask = ~np.isin(self._active_global_ids, global_ids)
        if np.all(keep_mask):
            return np.zeros((0,), dtype=np.int64)

        removed = self._active_global_ids[~keep_mask].copy()
        self._active_global_ids = self._active_global_ids[keep_mask]

        keep_mask_t = torch.as_tensor(
            keep_mask, dtype=torch.bool, device=self.query_points.device
        )
        self.query_points = self.query_points[keep_mask_t]
        self.query_features = self._filter_query_features(self.query_features, keep_mask_t)
        self._causal_state = self._filter_causal_state(self._causal_state, keep_mask_t)
        return removed

    def _expand_causal_state(self, n_new: int, point_axis: int = 1):
        """
        Grow every per-point tensor in your causal_state list-of-dicts by n_new along point_axis.
        Leaves non-tensors untouched. Returns a structure with the same nesting.
        """
        if self._causal_state is None or n_new <= 0:
            return self._causal_state

        out = []
        for level_dict in self._causal_state:
            new_level = {}
            for k, v in level_dict.items():
                if torch.is_tensor(v) and v.dim() > point_axis:
                    shape = list(v.shape)
                    shape[point_axis] = n_new
                    pad = torch.zeros(shape, dtype=v.dtype, device=v.device)
                    new_level[k] = torch.cat([v, pad], dim=point_axis)
                else:
                    new_level[k] = v
            out.append(new_level)
        return out

    def _filter_causal_state(self, causal_state, keep_mask, point_axis: int = 1):
        if causal_state is None:
            return None

        out = []
        for level_dict in causal_state:
            new_level = {}
            for k, v in level_dict.items():
                if torch.is_tensor(v) and v.dim() > point_axis:
                    index = [slice(None)] * v.dim()
                    index[point_axis] = keep_mask
                    new_level[k] = v[tuple(index)]
                else:
                    new_level[k] = v
            out.append(new_level)
        return out

    def _concat_query_features(
        self, a: QueryFeatures, b: QueryFeatures, point_axis: int = 1
    ) -> QueryFeatures:
        """
        Concatenate two QueryFeatures along the point axis (default 1: [B, N, C]).
        Assumes same number of pyramid levels and identical resolutions.
        """
        assert (
            len(a.lowres) == len(b.lowres) == len(a.hires) == len(b.hires)
        ), "Pyramid length mismatch."
        if hasattr(a, "resolutions") and hasattr(b, "resolutions"):
            assert a.resolutions == b.resolutions, "Resolutions must match."

        lowres_cat = [
            torch.cat([a.lowres[i], b.lowres[i]], dim=point_axis)
            for i in range(len(a.lowres))
        ]
        hires_cat = [
            torch.cat([a.hires[i], b.hires[i]], dim=point_axis)
            for i in range(len(a.hires))
        ]

        # Rebuild the NamedTuple by name
        return QueryFeatures(
            lowres=tuple(lowres_cat),
            hires=tuple(hires_cat),
            resolutions=a.resolutions,  # keep a's (identical to b's)
        )

    def _filter_query_features(
        self, query_features: QueryFeatures, keep_mask, point_axis: int = 1
    ) -> QueryFeatures:
        lowres = []
        hires = []
        for feat in query_features.lowres:
            index = [slice(None)] * feat.dim()
            index[point_axis] = keep_mask
            lowres.append(feat[tuple(index)])
        for feat in query_features.hires:
            index = [slice(None)] * feat.dim()
            index[point_axis] = keep_mask
            hires.append(feat[tuple(index)])
        return QueryFeatures(
            lowres=tuple(lowres),
            hires=tuple(hires),
            resolutions=query_features.resolutions,
        )

    def _build_full_outputs(self):
        tracks = self._last_tracks_full.copy()
        if tracks.shape[0] != self._num_global_points:
            tracks = np.zeros((self._num_global_points, 2), dtype=np.float32)
            self._last_tracks_full = tracks.copy()
        uncertainty = np.full(
            (self._num_global_points,), self._inactive_uncertainty, dtype=np.float32
        )
        visibles = np.zeros((self._num_global_points,), dtype=bool)
        return tracks, uncertainty, visibles

    def _scatter_active_outputs(self, active_tracks, active_uncertainty, active_visibles):
        tracks, uncertainty, visibles = self._build_full_outputs()
        active_ids = self._active_global_ids
        if active_ids.size > 0:
            tracks[active_ids] = active_tracks
            uncertainty[active_ids] = active_uncertainty
            visibles[active_ids] = active_visibles
        self._last_tracks_full = tracks.copy()
        return tracks, uncertainty, visibles

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
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist))
        visibles = (visibles >= self._visible_threshold).to(torch.bool)
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

        # feature_grid = feature_grids.lowres[0]
        # hires_feats = feature_grids.hires[0]
        # query_feature_hires = query_features.hires[0]
        # print(f"feature_grid lowres: {feature_grid.shape}")
        # print(f"feature_grid hires: {hires_feats.shape}")
        # print(f"feature_grid resolutions: {feature_grids.resolutions}")
        # print(f"query_feature lowres: {query_features.lowres[0].shape}")
        # print(f"query_features hires: {query_feature_hires.shape}")

        # print(f"query_features resolutions: {query_features.resolutions}")

        trajectories = model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=max(1, min(self._query_chunk_size, query_features.lowres[0].shape[1])),
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
