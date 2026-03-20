import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER


@TRACKER.register_module("alltracker")
class AllTrackerTracker(Tracker):
    """
    Adapter around the official AllTracker dense anchor-frame tracker.

    Point2Pose adds sparse query points over time, while AllTracker predicts a
    dense correspondence field from one anchor frame to later frames. This
    adapter keeps one anchor-cohort per query batch and advances each cohort
    with a cached sliding-window state, rather than replaying the entire
    anchor-to-current prefix on every frame.
    """

    _DEFAULT_CKPT_URL = (
        "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    )
    _DEFAULT_TINY_CKPT_URL = (
        "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
    )

    def __init__(self, config):
        super().__init__(config)
        self.name = "alltracker"

        self._device = torch.device(config.get("device", "cuda"))
        if self._device.type != "cuda":
            raise ValueError(
                "AllTrackerTracker currently requires CUDA because the upstream "
                "reference implementation uses .cuda() internally."
            )

        self._visible_threshold = float(config.get("visible_threshold", 0.6))
        self._inactive_uncertainty = float(config.get("inactive_uncertainty", 1.0))
        self._inference_iters = int(config.get("inference_iters", 4))
        self._window_len = int(config.get("window_len", 16))
        self._window_stride = config.get("window_stride", None)
        if self._window_stride is not None:
            self._window_stride = int(self._window_stride)
        self._effective_stride = (
            self._window_len // 2
            if self._window_stride is None
            else int(self._window_stride)
        )
        if self._effective_stride <= 0:
            raise ValueError("alltracker window_stride must be positive.")

        self._tiny = bool(config.get("tiny", False))
        self._enable_tf32 = bool(config.get("enable_tf32", False))
        self._use_confidence_product = bool(
            config.get("use_confidence_product", True)
        )

        resize_height = config.get("resize_height", None)
        resize_width = config.get("resize_width", None)
        if (resize_height is None) != (resize_width is None):
            raise ValueError(
                "alltracker resize_height and resize_width must either both be set "
                "or both be omitted."
            )
        self._resize_height = (
            int(resize_height) if resize_height is not None else None
        )
        self._resize_width = int(resize_width) if resize_width is not None else None

        if self._enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        self._repo_path = self._resolve_repo_path(config.get("repo_path", None))
        net_cls, input_padder_cls = self._import_alltracker_modules(self._repo_path)
        self._model = self._build_model(net_cls, config)
        self._input_padder_cls = input_padder_cls

        torch.set_grad_enabled(False)

        self._img_height = 0
        self._img_width = 0
        self._model_height = 0
        self._model_width = 0

        self._grid_xy = None
        self._padder = None
        self._norm_mean = None
        self._norm_std = None
        self._feature_height = 0
        self._feature_width = 0

        self._cohorts = []
        self._history_frame_ids = []
        self._frame_id_to_history_idx = {}
        self._history_fmaps = {}

        self._num_global_points = 0
        self._last_tracks_full = np.empty((0, 2), dtype=np.float32)

    def initialize(self, frame):
        self._img_height = int(frame.rgb.shape[0])
        self._img_width = int(frame.rgb.shape[1])
        self._model_height = (
            int(self._resize_height)
            if self._resize_height is not None
            else self._img_height
        )
        self._model_width = (
            int(self._resize_width)
            if self._resize_width is not None
            else self._img_width
        )

        self._grid_xy = self._build_grid_xy(
            self._model_height, self._model_width
        ).to(self._device)
        self._padder = self._input_padder_cls((1, 3, self._model_height, self._model_width))
        self._norm_mean = torch.as_tensor(
            [0.485, 0.456, 0.406], device=self._device, dtype=torch.float32
        ).reshape(1, 3, 1, 1)
        self._norm_std = torch.as_tensor(
            [0.229, 0.224, 0.225], device=self._device, dtype=torch.float32
        ).reshape(1, 3, 1, 1)

        self._cohorts = []
        self._history_frame_ids = []
        self._frame_id_to_history_idx = {}
        self._history_fmaps = {}
        self._num_global_points = 0
        self._last_tracks_full = np.empty((0, 2), dtype=np.float32)
        self._feature_height = 0
        self._feature_width = 0

        self._append_frame_to_history(frame)

        print(
            f"[AllTracker] Initialized with image size {self._img_height}x{self._img_width}."
        )
        print("[AllTracker] Initialized with 0 points.")
        return False

    def track_once(self, frame):
        if not self._history_frame_ids:
            raise RuntimeError("AllTrackerTracker.initialize(frame) must be called first.")

        current_history_idx = self._append_frame_to_history(frame)
        if not self._cohorts:
            self._prune_history_fmaps(current_history_idx)
            return self._build_full_outputs()

        tracks, uncertainty, visibles = self._build_full_outputs()

        for cohort in self._cohorts:
            cohort_ids = np.asarray(cohort["global_ids"], dtype=np.int64)
            if cohort_ids.size == 0:
                continue

            anchor_history_idx = int(cohort["anchor_history_idx"])
            if current_history_idx < anchor_history_idx:
                raise ValueError(
                    "Current frame appears earlier than an active AllTracker anchor frame."
                )

            if current_history_idx == anchor_history_idx:
                cohort_tracks_model = cohort["query_points_model"].copy()
                visibility_score = np.ones((cohort_ids.shape[0],), dtype=np.float32)
            else:
                last_traj_map, last_visconf_map = self._advance_cohort_window(
                    cohort, current_history_idx
                )
                cohort_tracks_model, visibility_score = self._sample_query_points(
                    last_traj_map,
                    last_visconf_map,
                    cohort["query_points_model"],
                )

            tracks[cohort_ids] = self._convert_model_to_image_points(cohort_tracks_model)
            uncertainty[cohort_ids] = 1.0 - visibility_score
            visibles[cohort_ids] = visibility_score >= self._visible_threshold

        self._last_tracks_full = tracks.copy()
        self._prune_history_fmaps(current_history_idx)
        return tracks, uncertainty, visibles

    def add_query_points(self, frame, new_points):
        if not self._history_frame_ids:
            raise RuntimeError(
                "AllTrackerTracker.initialize(frame) must be called before adding points."
            )

        frame_id = int(frame.id)
        if frame_id not in self._frame_id_to_history_idx:
            raise ValueError(
                f"Frame {frame_id} is not available in AllTracker history. "
                "Anchoring new points requires the anchor frame to still be cached."
            )

        new_points = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)
        if new_points.size == 0:
            return np.zeros((0,), dtype=np.int64)

        anchor_history_idx = self._frame_id_to_history_idx[frame_id]
        query_points_model = self._convert_image_to_model_points(new_points)

        global_old_len = self._num_global_points
        n_new = int(new_points.shape[0])
        global_new_len = global_old_len + n_new
        new_global_ids = np.arange(global_old_len, global_new_len, dtype=np.int64)

        cohort = self._find_cohort(anchor_history_idx)
        if cohort is None:
            anchor_fmap = self._get_history_fmap(anchor_history_idx).unsqueeze(0)
            self._cohorts.append(
                {
                    "anchor_history_idx": anchor_history_idx,
                    "anchor_fmap": anchor_fmap,
                    "global_ids": new_global_ids.copy(),
                    "query_points_model": query_points_model.copy(),
                    "window_start_rel": None,
                    "flows8": None,
                    "visconfs8": None,
                }
            )
        else:
            cohort["global_ids"] = np.concatenate(
                [np.asarray(cohort["global_ids"], dtype=np.int64), new_global_ids],
                axis=0,
            )
            cohort["query_points_model"] = np.concatenate(
                [
                    np.asarray(cohort["query_points_model"], dtype=np.float32),
                    query_points_model,
                ],
                axis=0,
            )

        self._num_global_points = global_new_len
        if self._last_tracks_full.shape[0] == 0:
            self._last_tracks_full = new_points.copy()
        else:
            self._last_tracks_full = np.concatenate(
                [self._last_tracks_full, new_points], axis=0
            )

        return new_global_ids

    def deactivate_query_points(self, global_ids):
        if not self._cohorts:
            return np.zeros((0,), dtype=np.int64)

        global_ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        if global_ids.size == 0:
            return np.zeros((0,), dtype=np.int64)

        global_ids = np.unique(global_ids)
        removed = []
        kept_cohorts = []

        for cohort in self._cohorts:
            cohort_ids = np.asarray(cohort["global_ids"], dtype=np.int64)
            keep_mask = ~np.isin(cohort_ids, global_ids)
            if np.all(keep_mask):
                kept_cohorts.append(cohort)
                continue

            removed.extend(cohort_ids[~keep_mask].tolist())
            if not np.any(keep_mask):
                continue

            kept_cohorts.append(
                {
                    "anchor_history_idx": int(cohort["anchor_history_idx"]),
                    "anchor_fmap": cohort["anchor_fmap"],
                    "global_ids": cohort_ids[keep_mask].copy(),
                    "query_points_model": np.asarray(
                        cohort["query_points_model"], dtype=np.float32
                    )[keep_mask].copy(),
                    "window_start_rel": cohort.get("window_start_rel", None),
                    "flows8": cohort.get("flows8", None),
                    "visconfs8": cohort.get("visconfs8", None),
                }
            )

        self._cohorts = kept_cohorts
        latest_idx = len(self._history_frame_ids) - 1
        if latest_idx >= 0:
            self._prune_history_fmaps(latest_idx)
        return np.asarray(removed, dtype=np.int64)

    def _resolve_repo_path(self, configured_path: Optional[str]) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        candidates = []

        if configured_path:
            candidates.append(Path(configured_path).expanduser())

        env_path = os.environ.get("ALLTRACKER_REPO_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        candidates.append(repo_root / "third_party" / "alltracker")

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            candidate = candidate.resolve()
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            if (candidate / "nets" / "alltracker.py").is_file():
                return candidate

        searched = ", ".join(str(candidate) for candidate in unique_candidates)
        raise FileNotFoundError(
            "Could not find the official AllTracker repo. Set tracker.params.repo_path "
            f"or ALLTRACKER_REPO_PATH. Searched: {searched}"
        )

    def _import_alltracker_modules(self, repo_path: Path):
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        try:
            from nets.alltracker import Net
            from nets.blocks import InputPadder
        except Exception as exc:
            raise ImportError(
                "Failed to import the official AllTracker code from "
                f"{repo_path}. Install its inference dependencies from the upstream "
                "repo and verify tracker.params.repo_path."
            ) from exc

        return Net, InputPadder

    def _build_model(self, net_cls, config):
        if self._tiny:
            model = net_cls(self._window_len, use_basicencoder=True, no_split=True)
        else:
            model = net_cls(self._window_len)

        checkpoint_path = config.get("checkpoint_path", None)
        checkpoint_url = config.get(
            "checkpoint_url",
            self._DEFAULT_TINY_CKPT_URL if self._tiny else self._DEFAULT_CKPT_URL,
        )
        checkpoint_cache_dir = config.get("checkpoint_cache_dir", None)

        if checkpoint_path:
            checkpoint_path = str(Path(checkpoint_path).expanduser())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            load_kwargs = {"map_location": "cpu"}
            if checkpoint_cache_dir:
                load_kwargs["model_dir"] = str(
                    Path(checkpoint_cache_dir).expanduser()
                )
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint_url, **load_kwargs
            )

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device).eval()

        if bool(config.get("compile_model", False)) and hasattr(torch, "compile"):
            compile_mode = str(config.get("compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=compile_mode)

        return model

    def _append_frame_to_history(self, frame):
        frame_id = int(frame.id)
        if frame_id in self._frame_id_to_history_idx:
            return self._frame_id_to_history_idx[frame_id]

        if (
            frame.rgb.shape[0] != self._img_height
            or frame.rgb.shape[1] != self._img_width
        ):
            raise ValueError(
                "AllTrackerTracker assumes a fixed frame size within a sequence."
            )

        rgb = self._resize_rgb(frame.rgb)
        fmap = self._encode_frame(rgb)

        history_idx = len(self._history_frame_ids)
        self._history_frame_ids.append(frame_id)
        self._frame_id_to_history_idx[frame_id] = history_idx
        self._history_fmaps[history_idx] = fmap
        return history_idx

    def _encode_frame(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        rgb_tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self._device, non_blocking=True)
        )
        rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = (rgb_tensor - self._norm_mean) / self._norm_std
        rgb_tensor = self._padder.pad(rgb_tensor)[0].contiguous()

        with torch.no_grad():
            fmap = self._model.get_fmaps(rgb_tensor, B=1, T=1, sw=None, is_training=False)

        fmap = fmap.reshape(
            1,
            fmap.shape[1],
            fmap.shape[2],
            fmap.shape[3],
        ).detach()
        if self._feature_height == 0 or self._feature_width == 0:
            self._feature_height = int(fmap.shape[-2])
            self._feature_width = int(fmap.shape[-1])
        return fmap[0]

    def _resize_rgb(self, rgb):
        if rgb.shape[0] == self._model_height and rgb.shape[1] == self._model_width:
            return rgb

        return cv2.resize(
            rgb,
            (self._model_width, self._model_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _advance_cohort_window(self, cohort, current_history_idx: int):
        anchor_history_idx = int(cohort["anchor_history_idx"])
        relative_idx = current_history_idx - anchor_history_idx
        if relative_idx <= 0:
            raise ValueError("AllTracker cohort advance requires a non-anchor current frame.")

        window_start_rel = self._compute_window_start_rel(relative_idx + 1)
        current_slot = relative_idx - window_start_rel
        window_fmaps = self._assemble_window_fmaps(
            anchor_history_idx, window_start_rel, current_history_idx
        )
        flows8, visconfs8 = self._prepare_sliding_state(
            cohort, window_start_rel, window_fmaps
        )

        with torch.no_grad():
            flow_predictions, visconf_predictions, flows8_out, visconfs8_out, _ = (
                self._model.forward_window(
                    cohort["anchor_fmap"],
                    window_fmaps,
                    visconfs8,
                    iters=self._inference_iters,
                    flowfeat=None,
                    flows8=flows8,
                    is_training=False,
                )
            )

        final_flow = flow_predictions[-1].reshape(
            1, self._window_len, 2, *flow_predictions[-1].shape[-2:]
        )[:, current_slot]
        final_visconf = visconf_predictions[-1].reshape(
            1, self._window_len, 2, *visconf_predictions[-1].shape[-2:]
        )[:, current_slot]

        final_flow = self._padder.unpad(final_flow)[0]
        final_visconf = torch.sigmoid(self._padder.unpad(final_visconf))[0]

        cohort["window_start_rel"] = int(window_start_rel)
        cohort["flows8"] = flows8_out.reshape(
            1, self._window_len, 2, self._feature_height, self._feature_width
        ).detach()
        cohort["visconfs8"] = visconfs8_out.reshape(
            1, self._window_len, 2, self._feature_height, self._feature_width
        ).detach()

        last_traj_map = final_flow + self._grid_xy.to(dtype=final_flow.dtype)
        return last_traj_map, final_visconf

    def _compute_window_start_rel(self, num_frames: int) -> int:
        if num_frames <= self._window_len:
            return 0

        overflow = num_frames - self._window_len
        return (
            (overflow + self._effective_stride - 1) // self._effective_stride
        ) * self._effective_stride

    def _assemble_window_fmaps(
        self, anchor_history_idx: int, window_start_rel: int, current_history_idx: int
    ):
        relative_idx = current_history_idx - anchor_history_idx
        fmaps = []
        for slot in range(self._window_len):
            rel_idx = window_start_rel + slot
            if rel_idx > relative_idx:
                history_idx = current_history_idx
            else:
                history_idx = anchor_history_idx + rel_idx
            fmaps.append(self._get_history_fmap(history_idx))

        return torch.stack(fmaps, dim=0).unsqueeze(0)

    def _prepare_sliding_state(self, cohort, window_start_rel: int, window_fmaps):
        dtype = window_fmaps.dtype
        device = window_fmaps.device
        shape = (
            1,
            self._window_len,
            2,
            self._feature_height,
            self._feature_width,
        )

        prev_start = cohort.get("window_start_rel", None)
        prev_flows8 = cohort.get("flows8", None)
        prev_visconfs8 = cohort.get("visconfs8", None)
        if prev_start is None or prev_flows8 is None or prev_visconfs8 is None:
            flows8 = torch.zeros(shape, dtype=dtype, device=device)
            visconfs8 = torch.zeros(shape, dtype=dtype, device=device)
            return (
                flows8.reshape(-1, 2, self._feature_height, self._feature_width),
                visconfs8.reshape(-1, 2, self._feature_height, self._feature_width),
            )

        prev_flows8 = prev_flows8.to(device=device, dtype=dtype)
        prev_visconfs8 = prev_visconfs8.to(device=device, dtype=dtype)

        if int(prev_start) == int(window_start_rel):
            return (
                prev_flows8.reshape(-1, 2, self._feature_height, self._feature_width),
                prev_visconfs8.reshape(
                    -1, 2, self._feature_height, self._feature_width
                ),
            )

        if (
            self._effective_stride == (self._window_len // 2)
            and int(window_start_rel) == int(prev_start) + self._effective_stride
        ):
            half = self._window_len // 2
            stride = self._effective_stride
            shifted_flows8 = torch.cat(
                [
                    prev_flows8[:, stride : stride + half],
                    prev_flows8[:, stride + half - 1 : stride + half].repeat(
                        1, half, 1, 1, 1
                    ),
                ],
                dim=1,
            )
            shifted_visconfs8 = torch.cat(
                [
                    prev_visconfs8[:, stride : stride + half],
                    prev_visconfs8[:, stride + half - 1 : stride + half].repeat(
                        1, half, 1, 1, 1
                    ),
                ],
                dim=1,
            )
            return (
                shifted_flows8.reshape(
                    -1, 2, self._feature_height, self._feature_width
                ),
                shifted_visconfs8.reshape(
                    -1, 2, self._feature_height, self._feature_width
                ),
            )

        flows8 = torch.zeros(shape, dtype=dtype, device=device)
        visconfs8 = torch.zeros(shape, dtype=dtype, device=device)
        return (
            flows8.reshape(-1, 2, self._feature_height, self._feature_width),
            visconfs8.reshape(-1, 2, self._feature_height, self._feature_width),
        )

    def _get_history_fmap(self, history_idx: int):
        fmap = self._history_fmaps.get(int(history_idx), None)
        if fmap is None:
            raise RuntimeError(
                f"AllTracker feature cache no longer contains history index {history_idx}. "
                "This frame was pruned before all active cohorts finished using it."
            )
        return fmap

    def _prune_history_fmaps(self, current_history_idx: int):
        if not self._history_fmaps:
            return

        keep_from = int(current_history_idx)
        for cohort in self._cohorts:
            anchor_history_idx = int(cohort["anchor_history_idx"])
            start_rel = cohort.get("window_start_rel", None)
            if start_rel is None:
                cohort_keep_from = anchor_history_idx
            else:
                cohort_keep_from = anchor_history_idx + int(start_rel)
            keep_from = min(keep_from, cohort_keep_from)

        stale_keys = [idx for idx in self._history_fmaps.keys() if idx < keep_from]
        for idx in stale_keys:
            del self._history_fmaps[idx]

    def _sample_query_points(self, last_traj_map, last_visconf_map, query_points_model):
        query_points_model = np.asarray(query_points_model, dtype=np.float32).reshape(
            -1, 2
        )
        if query_points_model.size == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        x = np.clip(
            np.rint(query_points_model[:, 0]).astype(np.int64),
            0,
            self._model_width - 1,
        )
        y = np.clip(
            np.rint(query_points_model[:, 1]).astype(np.int64),
            0,
            self._model_height - 1,
        )

        traj = (
            last_traj_map[:, y, x]
            .permute(1, 0)
            .float()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        visconf = (
            last_visconf_map[:, y, x]
            .permute(1, 0)
            .float()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        visibility_score = self._compute_visibility_score(visconf)
        return traj, visibility_score

    def _compute_visibility_score(self, visconf):
        visconf = np.asarray(visconf, dtype=np.float32)
        if visconf.ndim != 2 or visconf.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        primary = np.clip(visconf[:, 0], 0.0, 1.0)
        if visconf.shape[1] < 2 or not self._use_confidence_product:
            return primary

        return np.clip(primary * visconf[:, 1], 0.0, 1.0)

    def _convert_image_to_model_points(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        points[:, 0] *= float(self._model_width) / float(self._img_width)
        points[:, 1] *= float(self._model_height) / float(self._img_height)
        return points

    def _convert_model_to_image_points(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        points[:, 0] *= float(self._img_width) / float(self._model_width)
        points[:, 1] *= float(self._img_height) / float(self._model_height)
        return points

    def _find_cohort(self, anchor_history_idx: int):
        for cohort in self._cohorts:
            if int(cohort["anchor_history_idx"]) == int(anchor_history_idx):
                return cohort
        return None

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

    def _build_grid_xy(self, height: int, width: int):
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack([xs, ys], dim=0)
