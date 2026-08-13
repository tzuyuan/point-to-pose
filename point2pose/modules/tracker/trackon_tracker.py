import os
import sys
from argparse import Namespace

import numpy as np

import torch
import torch.nn.functional as F

# track_on uses top-level imports (model.*, utils.*), so its repo root must be
# on sys.path before importing it.
_TRACK_ON_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "third_party", "track_on",
)
_TRACK_ON_ROOT = os.path.normpath(_TRACK_ON_ROOT)
if _TRACK_ON_ROOT not in sys.path:
    sys.path.insert(0, _TRACK_ON_ROOT)

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER

# NOTE: the Track_On2 import (mmcv, transformers, timm chain) is deferred to
# __init__ — loading torchvision-adjacent libs before the first cv2.namedWindow
# call hangs it (see tapnext_tracker.py note).


@TRACKER.register_module("trackon")
class TrackOnTracker(Tracker):
    """
    Track-On2 / Track-On-R tracker implementation.

    Strictly online, per-frame transformer tracker with a FIFO per-point
    memory. Localization is done by coarse patch classification over the whole
    frame plus offset regression, which enables re-detection after occlusion.

    Notes:
    - Query features are sampled from the frame passed to add_query_points,
      so anchor-frame (past keyframe) queries keep TAPIR-like semantics: the
      appearance template comes from the anchor image and the point is
      re-localized globally in subsequent frames.
    - vit_backbone "dinov2_s" works out of the box (ungated weights,
      trackon2_dinov2_checkpoint.pt). "dinov3_s_plus" requires accepting the
      DINOv3 license on Hugging Face and a newer transformers version
      (see README), and matches trackon2_dinov3_checkpoint.pt / track_on_r.pt.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "trackon"

        from model.trackon import Track_On2  # deferred heavy import

        self._img_height = config.get("img_height", 480)
        self._img_width = config.get("img_width", 640)
        self._device = torch.device(config.get("device", "cuda"))
        self._delta_v = config.get("visible_threshold", 0.8)

        checkpoint_path = config.get(
            "checkpoint_path", "trackon2_dinov2_checkpoint.pt"
        )

        model_args = Namespace(
            input_size=list(config.get("input_size", [384, 512])),
            M=config.get("memory_size", 24),
            D=256,
            K=16,
            decoder_layer_num=3,
            predicton_head_layer_num=3,
            rerank_layer_num=3,
            vit_backbone=config.get("vit_backbone", "dinov2_s"),
            vit_upsample_factor=config.get("vit_upsample_factor", 1.143),
            grad_checkpoint=False,
            M_i=config.get("inference_memory_size", 72),
            delta_v=self._delta_v,
        )

        self._model = Track_On2(model_args)
        self._load_checkpoint(checkpoint_path)

        # Inference-time memory extension (interpolates temporal embeddings)
        if model_args.M_i != model_args.M:
            self._model.memory_extension(model_args.M_i)
        self._memory_size = self._model.M

        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        # tracking state
        self._q_init = None  # (N, D) query features
        self._point_memory = None  # (N, M, D)
        self._temporal_mask = None  # (N, M), True = masked
        self._pending_queries = []  # list of (n_i, D) feature tensors

    def _load_checkpoint(self, checkpoint_path):
        """Load Track-On weights. Backbone weights are intentionally absent
        from the released checkpoints (they are pulled from Hugging Face), so
        only backbone keys may be missing."""
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
        state_dict = {
            k.removeprefix("module."): v for k, v in state_dict.items()
        }
        result = self._model.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {result.unexpected_keys[:5]}"
            )
        not_backbone = [
            k for k in result.missing_keys if not k.startswith("backbone.")
        ]
        if not_backbone:
            raise RuntimeError(
                f"Checkpoint is missing non-backbone keys: {not_backbone[:5]}"
            )
        print(f"[TrackOn] Loaded checkpoint from {checkpoint_path}")

    def initialize(self, frame):
        self._img_height = frame.rgb.shape[0]
        self._img_width = frame.rgb.shape[1]
        self._q_init = None
        self._point_memory = None
        self._temporal_mask = None
        self._pending_queries = []
        print(
            f"[TrackOn] Initialized with image size "
            f"{self._img_height}x{self._img_width}."
        )
        return True

    def add_query_points(self, frame, new_points):
        """
        Add new query points. Appearance features are sampled from the given
        frame (which may be a past anchor keyframe) immediately; the points
        join the per-frame tracking loop on the next track_once call.
        Args:
            frame: Frame object with rgb [H, W, 3] np.uint8.
            new_points: np.ndarray (num_new_points, 2), [x, y] in original
                        image coordinates.
        Returns:
            indices of the newly added points
        """
        pts = np.stack(new_points).astype(np.float32).reshape(-1, 2)
        frame_t = self._frame_to_tensor(frame.rgb)
        old_len = self._num_points()

        # mmcv's deformable attention kernel does not support bf16/fp16, and
        # the sam2 segmenter enters a global bf16 autocast at import time, so
        # explicitly run Track-On in fp32.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            _, _, _, _, f_fused = self._model.extract_frame_features(frame_t)
            q_new = self._sample_query_features(
                f_fused, pts, frame.rgb.shape[0], frame.rgb.shape[1]
            )
        self._pending_queries.append(q_new)

        return np.arange(old_len, old_len + pts.shape[0])

    def track_once(self, frame):
        """
        Track all active points in the frame once.
        Args:
            frame: Frame object. Must contain rgb [H,W,3] np.uint8.

        Returns:
            tracks: np.ndarray (num_points, 2), [x, y] in original image coords
            uncertainties: np.ndarray (num_points,), 1 - visibility prob
            visibles: np.ndarray (num_points,), bool
        """
        frame_t = self._frame_to_tensor(frame.rgb)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            features = self._model.extract_frame_features(frame_t)

            if self._pending_queries:
                self._commit_pending(features[-1].device)

            if self._q_init is None or self._q_init.shape[0] == 0:
                return (
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=bool),
                )

            p, v_logit, q_new = self._model.track_frame(
                self._q_init,
                self._temporal_mask,
                self._point_memory,
                features,
                self._img_height,
                self._img_width,
            )

        # FIFO memory update
        self._point_memory = torch.roll(self._point_memory, shifts=-1, dims=1)
        self._point_memory[:, -1] = q_new
        self._temporal_mask = torch.roll(self._temporal_mask, shifts=-1, dims=1)
        self._temporal_mask[:, -1] = False

        v_prob = torch.sigmoid(v_logit.float())
        visibles = (v_prob >= self._delta_v).cpu().numpy().reshape(-1)
        uncertainty = (1.0 - v_prob).cpu().numpy().reshape(-1)
        tracks = p.float().cpu().numpy().reshape(-1, 2)

        return tracks, uncertainty, visibles

    def _num_points(self):
        n = 0 if self._q_init is None else self._q_init.shape[0]
        return n + sum(q.shape[0] for q in self._pending_queries)

    def _commit_pending(self, device):
        """Append pending query features with fresh (empty) memory rows."""
        q_new = torch.cat(self._pending_queries, dim=0).to(device)
        self._pending_queries = []
        n_new = q_new.shape[0]

        d = self._model.D
        m = self._memory_size
        new_memory = torch.zeros(n_new, m, d, device=device)
        new_mask = torch.ones(n_new, m, device=device, dtype=torch.bool)

        if self._q_init is None:
            self._q_init = q_new
            self._point_memory = new_memory
            self._temporal_mask = new_mask
        else:
            self._q_init = torch.cat([self._q_init, q_new], dim=0)
            self._point_memory = torch.cat(
                [self._point_memory, new_memory], dim=0
            )
            self._temporal_mask = torch.cat(
                [self._temporal_mask, new_mask], dim=0
            )

    def _sample_query_features(self, f_fused, pts_xy, img_h, img_w):
        """Bilinearly sample fused feature map at [x, y] pixel locations.
        Mirrors track_on's Predictor.init_queries feature sampling."""
        coords = torch.from_numpy(pts_xy).to(f_fused.device)
        coords_norm = coords.clone()
        coords_norm[:, 0] = coords_norm[:, 0] / img_w
        coords_norm[:, 1] = coords_norm[:, 1] / img_h
        coords_norm = coords_norm * 2 - 1

        f_map = f_fused.view(
            1, self._model.Hf, self._model.Wf, self._model.D
        ).permute(0, 3, 1, 2)  # (1, D, Hf, Wf)
        grid = coords_norm.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        q = F.grid_sample(
            f_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (1, D, N, 1)
        return q.squeeze(-1).squeeze(0).permute(1, 0)  # (N, D)

    def _frame_to_tensor(self, rgb):
        """[H, W, 3] uint8 RGB -> (1, 3, H, W) float tensor in [0, 255]."""
        t = torch.from_numpy(np.ascontiguousarray(rgb)).pin_memory()
        t = t.to(self._device, non_blocking=True)
        return t.permute(2, 0, 1).unsqueeze(0).float()
