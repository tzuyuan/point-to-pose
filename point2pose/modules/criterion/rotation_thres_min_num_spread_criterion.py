import numpy as np
import torch

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("rotation_threshold_and_min_num_spread")
class RotationThresholdAndMinNumSpreadCriterion(SampleCriterion):
    def __init__(self, config):
        super().__init__()
        self._max_angle_deg = config.get("max_angle_deg", 60)
        self._min_num_pts = config.get("min_num_pts", 10)
        self._min_mask_area = config.get("min_mask_area", 100)

        # --- NEW: spread params (cheap + robust) ---
        self._spread_grid = int(config.get("spread_grid", 4))  # e.g. 4 -> 4x4
        self._min_occupied_cells = int(
            config.get("min_occupied_cells", 4)
        )  # e.g. >=4 cells
        self._min_bbox_area_frac = float(
            config.get("min_bbox_area_frac", 0.05)
        )  # 5% of mask bbox

        self._num_obj = 0
        self._v_list = []

    def initialize(self, crit_ctx: CriterionContext):
        self._num_obj = len(crit_ctx.objects)
        self._v_list = []
        for obj_id in range(self._num_obj):
            self._v_list.append(np.array([[0, 0, 1]]))

    # --- NEW: helper ---
    def _points_spread_ok(
        self, context: CriterionContext, obj_id: int, uv_np: np.ndarray
    ) -> bool:
        """
        uv_np: (N,2) in pixel coords (u=x, v=y)
        Returns True if points cover enough of the current mask region.
        """
        if uv_np is None or uv_np.size == 0:
            return False

        # mask: (H,W) bool
        mask = context.frame.mask[obj_id, 0] > 0
        # quick bbox of mask
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return False

        x0 = int(xs.min().item())
        x1 = int(xs.max().item())
        y0 = int(ys.min().item())
        y1 = int(ys.max().item())
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        mask_bbox_area = float(w * h)

        uv = uv_np.astype(np.float32)
        u = uv[:, 0]
        v = uv[:, 1]

        # keep only points that land inside the mask bbox (avoids weird projections/outliers)
        in_bbox = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        if not np.any(in_bbox):
            return False
        u = u[in_bbox]
        v = v[in_bbox]

        # (1) bbox coverage
        pu0 = float(np.min(u))
        pu1 = float(np.max(u))
        pv0 = float(np.min(v))
        pv1 = float(np.max(v))
        pw = max(1.0, pu1 - pu0 + 1.0)
        ph = max(1.0, pv1 - pv0 + 1.0)
        pts_bbox_area = float(pw * ph)
        if (pts_bbox_area / mask_bbox_area) < self._min_bbox_area_frac:
            return False

        # (2) grid occupancy within the mask bbox
        G = self._spread_grid
        # normalize to [0,1] inside mask bbox, then to [0..G-1]
        gx = np.clip(((u - x0) / w * G).astype(np.int32), 0, G - 1)
        gy = np.clip(((v - y0) / h * G).astype(np.int32), 0, G - 1)
        cells = gy * G + gx
        n_occ = int(np.unique(cells).size)

        return n_occ >= self._min_occupied_cells

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        obj = context.objects[obj_id]
        reg_stats = context.reg_stats[obj_id]

        num_pts = reg_stats["correspond_curr3d"].shape[0]
        print(f"[Criterion] num visible points: {num_pts}")

        mask_area = int(torch.sum(context.frame.mask[obj_id, 0] > 0).item())

        # Your existing trigger
        if (num_pts < self._min_num_pts) and (mask_area > self._min_mask_area):
            return True

        # --- NEW: spread trigger ---
        # Prefer real 2D correspondences if you have them in reg_stats.
        uv = reg_stats.get("correspond_curr2d", None)  # <-- adjust key if yours differs
        if uv is None:
            # Fallback: if you *don't* store 2D, you can still approximate spread by projecting
            # correspond_curr3d using intrinsics if available.
            K = getattr(context.frame, "K", None)
            if K is not None:
                pts = reg_stats["correspond_curr3d"].astype(
                    np.float32
                )  # (N,3) in camera
                z = np.maximum(1e-6, pts[:, 2])
                x = pts[:, 0] / z
                y = pts[:, 1] / z
                uv = np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)

        if uv is not None:
            if isinstance(uv, torch.Tensor):
                uv = uv.detach().cpu().numpy()
            uv = np.asarray(uv)
            if uv.ndim == 2 and uv.shape[1] >= 2:
                spread_ok = self._points_spread_ok(context, obj_id, uv[:, :2])
                if (not spread_ok) and (mask_area > self._min_mask_area):
                    print("[Criterion] points not spread enough in mask -> resample")
                    return True

        # existing rotation logic
        R = obj.pose[:3, :3]
        u = R @ self._v_list[obj_id][0, :]

        inner_product = u @ self._v_list[obj_id].T
        angle = np.arccos(np.clip(inner_product, -1.0, 1.0))
        angle_deg = np.rad2deg(angle)
        print(f"[Criterion] angle_deg: {angle_deg}")

        if np.any(angle_deg < self._max_angle_deg):
            return False
        else:
            self._v_list[obj_id] = np.concatenate(
                [self._v_list[obj_id], u.reshape(1, -1)], axis=0
            )
            return True
