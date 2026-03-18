from typing import List

import cv2
import numpy as np
import torch

from tapnet.tapnext import tapnext_lru_modules
from tapnet.tapnext.tapnext_torch import TAPNext, TAPNextTrackingState
from tapnet.tapnext.tapnext_torch_utils import (
    restore_model_from_jax_checkpoint,
    tracker_certainty,
)

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER


@TRACKER.register_module("tapnext")
class TapNextTracker(Tracker):
    """
    TAPNext tracker backend.

    TAPNext keeps a recurrent tracking state for a fixed query set. This tracker
    maps naturally to the pipeline by maintaining one stateful cohort per batch
    of points that is added at a keyframe.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "tapnext"

        self._img_height = int(config.get("img_height", 480))
        self._img_width = int(config.get("img_width", 640))
        self._resize_height = int(config.get("resize_height", 256))
        self._resize_width = int(config.get("resize_width", 256))
        self._visible_threshold = float(config.get("visible_threshold", 0.5))
        self._certainty_radius = int(config.get("certainty_radius", 8))
        self._use_certainty_for_visibility = bool(
            config.get("use_certainty_for_visibility", True)
        )
        self._inactive_uncertainty = float(config.get("inactive_uncertainty", 1.0))
        self._enable_tf32 = bool(config.get("enable_tf32", False))

        self._device = torch.device(config.get("device", "cpu"))

        patch_size_cfg = config.get("patch_size", (8, 8))
        if isinstance(patch_size_cfg, int):
            self._patch_size = (patch_size_cfg, patch_size_cfg)
        else:
            self._patch_size = tuple(int(x) for x in patch_size_cfg)
        if len(self._patch_size) != 2:
            raise ValueError("tapnext patch_size must contain exactly two integers")

        if (self._resize_height % self._patch_size[0]) != 0 or (
            self._resize_width % self._patch_size[1]
        ) != 0:
            raise ValueError(
                "tapnext resize_height/resize_width must be divisible by patch_size"
            )

        if self._enable_tf32 and self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        checkpoint_path = config.get(
            "checkpoint_path",
            "/home/justin/code/point-to-pose/checkpoints/tapnext/bootstapnext_ckpt.npz",
        )
        self._validate_checkpoint(checkpoint_path)

        self._num_video_tokens = (
            self._resize_height // self._patch_size[0]
        ) * (self._resize_width // self._patch_size[1])

        self._model = TAPNext(
            image_size=(self._resize_height, self._resize_width),
            width=int(config.get("width", 768)),
            patch_size=self._patch_size,
            num_heads=int(config.get("num_heads", 12)),
            lru_width=int(config.get("lru_width", 768)),
            depth=int(config.get("depth", 12)),
            use_checkpointing=bool(config.get("use_checkpointing", False)),
        )
        self._model = restore_model_from_jax_checkpoint(self._model, checkpoint_path)
        self._model = self._model.to(self._device).eval()

        compile_model = bool(config.get("compile_model", False))
        if compile_model and hasattr(torch, "compile"):
            compile_mode = str(config.get("compile_mode", "reduce-overhead"))
            self._model = torch.compile(self._model, mode=compile_mode)

        torch.set_grad_enabled(False)

        self._cohorts: List[dict] = []
        self._active_global_ids = np.empty((0,), dtype=np.int64)
        self._num_global_points = 0
        self._last_tracks_full = np.empty((0, 2), dtype=np.float32)

    def initialize(self, frame):
        self._img_height = int(frame.rgb.shape[0])
        self._img_width = int(frame.rgb.shape[1])
        print(
            f"[TapNext] Initialized with image size {self._img_height}x{self._img_width}."
        )
        if not self._cohorts:
            print("[TapNext] Initialized with 0 points.")
            return False

        n_points = int(sum(len(c["global_ids"]) for c in self._cohorts))
        print(f"[TapNext] Initialized with {n_points} points.")
        return True

    def track_once(self, frame):
        if not self._cohorts:
            return self._build_full_outputs()

        frame_tensor = self._prepare_frame(frame)

        tracks, uncertainty, visibles = self._build_full_outputs()

        with torch.no_grad():
            for cohort in self._cohorts:
                pred_tracks, track_logits, visible_logits, state = self._model(
                    video=frame_tensor,
                    state=cohort["state"],
                )
                cohort["state"] = state

                visibility_score = self._compute_visibility_score(
                    pred_tracks, track_logits, visible_logits
                )[0, 0]
                cohort_tracks = self._convert_tracks_to_image_points(
                    pred_tracks[0, 0]
                )
                cohort_uncertainty = (
                    1.0 - visibility_score
                ).detach().cpu().float().numpy().reshape(-1)
                cohort_visible = (
                    visibility_score >= self._visible_threshold
                ).detach().cpu().numpy().reshape(-1).astype(bool)

                cohort_ids = cohort["global_ids"]
                tracks[cohort_ids] = cohort_tracks
                uncertainty[cohort_ids] = cohort_uncertainty
                visibles[cohort_ids] = cohort_visible

        self._last_tracks_full = tracks.copy()
        return tracks, uncertainty, visibles

    def add_query_points(self, frame, new_points):
        new_points = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)
        if new_points.size == 0:
            return np.zeros((0,), dtype=np.int64)

        frame_tensor = self._prepare_frame(frame)
        query_points = torch.as_tensor(
            self._convert_select_points_to_query_points(new_points),
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            pred_tracks, _, _, state = self._model(
                video=frame_tensor,
                query_points=query_points[None],
            )

        n_new = int(query_points.shape[0])
        global_old_len = self._num_global_points
        global_new_len = global_old_len + n_new
        new_global_ids = np.arange(global_old_len, global_new_len, dtype=np.int64)

        self._cohorts.append({"global_ids": new_global_ids, "state": state})
        self._num_global_points = global_new_len
        self._refresh_active_global_ids()

        current_tracks = self._convert_tracks_to_image_points(pred_tracks[0, 0])
        if self._last_tracks_full.shape[0] == 0:
            self._last_tracks_full = current_tracks.copy()
        else:
            self._last_tracks_full = np.concatenate(
                [self._last_tracks_full, current_tracks], axis=0
            )

        return new_global_ids

    def deactivate_query_points(self, global_ids):
        if not self._cohorts:
            return np.zeros((0,), dtype=np.int64)

        global_ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        if global_ids.size == 0:
            return np.zeros((0,), dtype=np.int64)

        remove_set = set(np.unique(global_ids).tolist())
        removed = []
        kept_cohorts = []

        for cohort in self._cohorts:
            cohort_ids = np.asarray(cohort["global_ids"], dtype=np.int64)
            keep_mask = np.array(
                [int(idx) not in remove_set for idx in cohort_ids], dtype=bool
            )
            if np.all(keep_mask):
                kept_cohorts.append(cohort)
                continue

            removed.extend(cohort_ids[~keep_mask].tolist())
            if not np.any(keep_mask):
                continue

            keep_mask_t = torch.as_tensor(
                keep_mask,
                dtype=torch.bool,
                device=cohort["state"].query_points.device,
            )
            kept_cohorts.append(
                {
                    "global_ids": cohort_ids[keep_mask],
                    "state": self._prune_tracking_state(cohort["state"], keep_mask_t),
                }
            )

        self._cohorts = kept_cohorts
        self._refresh_active_global_ids()
        return np.asarray(removed, dtype=np.int64)

    def _validate_checkpoint(self, checkpoint_path):
        ckpt = np.load(checkpoint_path)
        try:
            pos_emb = ckpt["backbone/pos_embedding"]
            model_width = int(pos_emb.shape[-1])
            expected_tokens = (
                self._resize_height // self._patch_size[0]
            ) * (self._resize_width // self._patch_size[1])
            if int(pos_emb.shape[1]) != expected_tokens:
                raise ValueError(
                    "tapnext checkpoint positional embedding does not match the "
                    f"configured resize {self._resize_height}x{self._resize_width}. "
                    f"Checkpoint expects {int(pos_emb.shape[1])} tokens, config "
                    f"expects {expected_tokens}."
                )
            if model_width != int(ckpt["backbone/unknown_token"].shape[-1]):
                raise ValueError("tapnext checkpoint appears inconsistent")
        finally:
            ckpt.close()

    def _refresh_active_global_ids(self):
        if not self._cohorts:
            self._active_global_ids = np.empty((0,), dtype=np.int64)
            return
        self._active_global_ids = np.concatenate(
            [np.asarray(c["global_ids"], dtype=np.int64) for c in self._cohorts],
            axis=0,
        )

    def _prune_tracking_state(self, state: TAPNextTrackingState, keep_mask_t):
        keep_query_idx = torch.nonzero(keep_mask_t, as_tuple=False).reshape(-1)
        token_keep = torch.cat(
            [
                torch.arange(self._num_video_tokens, device=keep_mask_t.device),
                self._num_video_tokens + keep_query_idx,
            ],
            dim=0,
        )

        hidden_state = []
        for layer_state in state.hidden_state:
            hidden_state.append(
                tapnext_lru_modules.RecurrentBlockCache(
                    rg_lru_state=layer_state.rg_lru_state[token_keep].contiguous(),
                    conv1d_state=layer_state.conv1d_state[token_keep].contiguous(),
                )
            )

        return TAPNextTrackingState(
            step=int(state.step),
            query_points=state.query_points[:, keep_mask_t, :].contiguous(),
            hidden_state=hidden_state,
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

    def _prepare_frame(self, frame):
        rgb_resize = cv2.resize(frame.rgb, (self._resize_width, self._resize_height))
        rgb_resize_pinned = torch.from_numpy(rgb_resize).pin_memory()
        return self._preprocess_frames(
            rgb_resize_pinned.to(self._device, non_blocking=True).unsqueeze(0).unsqueeze(0)
        )

    def _preprocess_frames(self, frames):
        frames = frames.float()
        return frames / 255 * 2 - 1

    def _convert_select_points_to_query_points(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        query_points = np.zeros((points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = 0.0
        query_points[:, 1] = points[:, 1] / self._img_height * self._resize_height
        query_points[:, 2] = points[:, 0] / self._img_width * self._resize_width
        return query_points

    def _convert_tracks_to_image_points(self, tracks_yx):
        tracks_yx = tracks_yx.detach().cpu().float().numpy().reshape(-1, 2)
        tracks_xy = np.empty_like(tracks_yx)
        tracks_xy[:, 0] = tracks_yx[:, 1] * (self._img_width / self._resize_width)
        tracks_xy[:, 1] = tracks_yx[:, 0] * (self._img_height / self._resize_height)
        return tracks_xy

    def _compute_visibility_score(self, pred_tracks, track_logits, visible_logits):
        visible_prob = torch.sigmoid(visible_logits)
        if not self._use_certainty_for_visibility:
            return visible_prob

        certainty = tracker_certainty(
            pred_tracks, track_logits, radius=self._certainty_radius
        )
        return visible_prob * certainty
