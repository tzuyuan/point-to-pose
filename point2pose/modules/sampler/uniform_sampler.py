import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull

from point2pose.data_types.frame import Frame
from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER
from point2pose.data_types.sampler_context import SamplerContext


@SAMPLER.register_module("uniform_fps")
class UniformFPSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        # Legacy fixed count (used only if density_per_kpx <= 0)
        self.num_points = int(config.get("num_points", 10))

        # New: density in "points per 1,000 pixels" of *safe* area
        self.density_per_kpx = float(config.get("density_per_kpx", 1.0))

        # Optional clamps for auto k
        self.min_points = int(config.get("min_points", 5))
        self.max_points = int(config.get("max_points", 50))

        # Rounding mode: "round", "ceil", "floor"
        self.round_mode = str(config.get("round_mode", "round")).lower()

        # Distance (in pixels) from the mask boundary to exclude
        self.edge_margin_px = int(config.get("edge_margin_px", 5))
        self.edge_margin_shrink_ratio = float(
            config.get("edge_margin_shrink_ratio", 0.8)
        )
        self.min_safe_points = int(config.get("min_safe_points", 5))

        self._remove_convex_hull = bool(config.get("remove_convex_hull", True))

        # NEW: subtract inflated points
        self._use_inflate_points = bool(config.get("inflate_points", False))
        self._inflate_radius_px = int(config.get("inflate_radius_px", 5))

        self._i = 0
        self._initialized_obj_ids = set()

    def sample(self, context: SamplerContext, obj_id: int) -> np.ndarray:
        """
        Uniform-like sampling via FPS inside the (possibly modified) mask.
        """
        frame = context.frame

        # ---- 1) Get mask -> CPU uint8 (for CPU ops) but keep torch ref for device
        mask_t = frame.mask[obj_id, 0]  # [H,W], bool/uint8, on CUDA/CPU
        if isinstance(mask_t, torch.Tensor):
            mask_np = (mask_t.detach().cpu().numpy() > 0).astype(np.uint8)
        else:
            mask_np = (mask_t > 0).astype(np.uint8)

        # ---- 1.1) Optional convex-hull subtraction on subsequent frames
        if obj_id in self._initialized_obj_ids:
            if self._remove_convex_hull and frame.convex_hull_xy is not None:
                mask_np = self.subtract_convex_hull(mask_np, frame.convex_hull_xy)
            elif self._remove_convex_hull and frame.convex_hull_xy is None:
                frame.convex_hull_xy = self._fit_convex_hull(context, obj_id)
                if frame.convex_hull_xy is not None:
                    mask_np = self.subtract_convex_hull(mask_np, frame.convex_hull_xy)

            # ---- 1.2) NEW: subtract inflated points (fast GPU or CPU)
            if self._use_inflate_points:
                pts = self._collect_points_in_mask(context, obj_id)
                if isinstance(mask_t, torch.Tensor) and mask_t.is_cuda:
                    # GPU fast path
                    mask_bool = mask_t > 0
                    mask_bool = self._subtract_inflated_points_from_mask_torch(
                        mask_bool, pts, self._inflate_radius_px
                    )
                    # keep a numpy copy for downstream distance transform
                    mask_np = mask_bool.detach().cpu().numpy().astype(np.uint8)
                else:
                    # CPU path
                    mask_np = self._subtract_inflated_points_from_mask_np(
                        mask_np, pts, self._inflate_radius_px
                    )
        else:
            self._initialized_obj_ids.add(obj_id)

        # Early exit on empty mask
        if mask_np.sum() == 0:
            if getattr(self, "debug_level", 0) >= 1:
                print(f"[UniformFPSampler] Empty mask for object {obj_id}; no points.")
            return np.zeros((0, 2), dtype=np.int32)

        # ---- 2) Exclude pixels within d of boundary using distance transform
        d = max(int(self.edge_margin_px), 0)
        if d > 0:
            dist = cv.distanceTransform(mask_np, distanceType=cv.DIST_L2, maskSize=3)
            safe = dist >= d
        else:
            dist = None
            safe = mask_np.astype(bool)

        # Relax margin if too few remain
        if np.count_nonzero(safe) < self.min_safe_points:
            ratio = self.edge_margin_shrink_ratio
            if not (0.0 < ratio < 1.0):
                ratio = 0.8
            if d > 0 and dist is None:
                dist = cv.distanceTransform(mask_np, cv.DIST_L2, 3)
            d_float = float(d)
            attempts = 0
            while d_float > 0 and np.count_nonzero(safe) < self.min_safe_points:
                d_float *= ratio
                d_new = int(max(0, np.floor(d_float)))
                if d_new == d:
                    if d_new == 0:
                        break
                    d_new = d - 1
                d = d_new
                safe = dist >= d if d > 0 else mask_np.astype(bool)
                attempts += 1
            if getattr(self, "debug_level", 0) >= 1:
                print(
                    f"[UniformFPSampler] Relaxed edge_margin_px to {d} after {attempts} step(s); "
                    f"safe_count={np.count_nonzero(safe)} (min={self.min_safe_points}) for object {obj_id}."
                )

        # ---- 3) Candidate coords (x,y) from safe region
        ys, xs = np.where(safe)
        coords_xy_np = np.stack([xs, ys], axis=1)  # (N,2) int32
        N = coords_xy_np.shape[0]
        if N == 0:
            return np.zeros((0, 2), dtype=np.int32)

        # ---- 3.5) Compute k from density and safe area
        safe_area_px = int(N)
        if self.density_per_kpx > 0:
            raw = self.density_per_kpx * (safe_area_px / 1000.0)
            if self.round_mode == "ceil":
                k_auto = int(np.ceil(raw))
            elif self.round_mode == "floor":
                k_auto = int(np.floor(raw))
            else:
                k_auto = int(np.round(raw))
            k = max(self.min_points, min(self.max_points, k_auto))
        else:
            k = int(self.num_points)
        k = max(0, min(k, N))

        # ---- 4) FPS on device
        device = (
            mask_t.device if isinstance(mask_t, torch.Tensor) else torch.device("cpu")
        )
        coords_xy = torch.from_numpy(coords_xy_np).to(
            device=device, dtype=torch.float32
        )
        picked_xy = self._fps_2d(coords_xy, k)  # (k,2) float32
        sampled_np = picked_xy.detach().cpu().numpy().round().astype(np.int32)

        # ---- 5) Debug viz (optional)
        if getattr(self, "debug_level", 0) >= 1:
            mask_img = cv.cvtColor((mask_np * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)

            rgb_src = frame.rgb
            if isinstance(rgb_src, torch.Tensor):
                rgb_src = rgb_src.detach().cpu().numpy()
            if rgb_src.dtype != np.uint8:
                rgb_src = np.clip(rgb_src, 0, 255).astype(np.uint8)
            rgb_img = cv.cvtColor(rgb_src, cv.COLOR_RGB2BGR).copy()

            for x, y in sampled_np:
                cv.circle(mask_img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv.circle(rgb_img, (int(x), int(y)), 4, (0, 0, 255), -1)

            out_dir = getattr(self, "debug_dir", ".")
            frame_id = getattr(frame, "id", "unk")
            cv.imwrite(f"{out_dir}/{frame_id}_mask_{obj_id}.png", mask_img)
            cv.imwrite(f"{out_dir}/{frame_id}_rgb_{obj_id}.png", rgb_img)
            print(
                f"[UniformFPSampler] Safe area: {safe_area_px}px, "
                f"density={self.density_per_kpx}/kpx -> k={k} (N={N}). "
                f"Saved debug images to {out_dir}"
            )

        if sampled_np.shape[0] < k:
            print(
                f"[UniformFPSampler] Returned {sampled_np.shape[0]} points (limited by safe region)."
            )
        else:
            print(
                f"[UniformFPSampler] Sampled {sampled_np.shape[0]} points for object {obj_id}."
            )
        return sampled_np

    # ---------------- NEW helpers ----------------

    def _collect_points_in_mask(
        self, context: SamplerContext, obj_id: int
    ) -> np.ndarray:
        """
        Collect current visible 2D points that lie inside the object's mask.
        Returns (M,2) float array in (x,y), may be empty.
        """
        mask = context.frame.mask[obj_id, 0] > 0
        obj_idx = context.track_table.obj2track_map[obj_id]
        vis_obj = np.asarray(context.track_table.visible, dtype=bool)[obj_idx]
        idx = obj_idx[vis_obj]
        points = context.track_table.track_2d[idx]

        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        mask_np = (
            mask.detach().cpu().numpy()
            if isinstance(mask, torch.Tensor)
            else np.asarray(mask)
        )
        H, W = mask_np.shape
        px = points[:, 0].astype(np.int32)
        py = points[:, 1].astype(np.int32)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (mask_np[py, px] > 0)
        points = points[valid]
        return points.astype(np.float32)

    def _subtract_inflated_points_from_mask_np(
        self, mask_np: np.ndarray, points_xy: np.ndarray, radius_px: int
    ) -> np.ndarray:
        """
        CPU path (OpenCV). mask_np: HxW {0,1}
        points_xy: (N,2) in (x,y)
        """
        mask_np = (mask_np > 0).astype(np.uint8)
        H, W = mask_np.shape

        if points_xy.size == 0:
            return mask_np

        pts_img = np.zeros((H, W), dtype=np.uint8)
        ix = np.rint(points_xy[:, 0]).astype(np.int32)
        iy = np.rint(points_xy[:, 1]).astype(np.int32)
        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy = ix[valid], iy[valid]
        pts_img[iy, ix] = 1

        k = radius_px * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        inflated = cv.dilate(pts_img, kernel, iterations=1)  # {0,1}

        out = (mask_np & (inflated == 0)).astype(np.uint8)
        return out

    def _subtract_inflated_points_from_mask_torch(
        self, mask_bool: torch.Tensor, points_xy, radius_px: int
    ) -> torch.Tensor:
        """
        GPU/CPU torch path. mask_bool: (H,W) torch.bool/uint8/float; returns torch.bool
        points_xy: (N,2) (x,y) tensor or np
        """
        mask_bool = mask_bool > 0
        H, W = mask_bool.shape
        device = mask_bool.device

        if not torch.is_tensor(points_xy):
            points_xy = torch.as_tensor(points_xy, device=device, dtype=torch.float32)
        else:
            points_xy = points_xy.to(device=device, dtype=torch.float32)

        if points_xy.numel() == 0:
            return mask_bool

        ix = torch.round(points_xy[:, 0]).to(torch.int64)
        iy = torch.round(points_xy[:, 1]).to(torch.int64)
        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy = ix[valid], iy[valid]
        if ix.numel() == 0:
            return mask_bool

        pts_img = torch.zeros((H, W), dtype=torch.float32, device=device)
        pts_img[iy, ix] = 1.0

        k = radius_px * 2 + 1
        yy, xx = torch.meshgrid(
            torch.arange(k, device=device),
            torch.arange(k, device=device),
            indexing="ij",
        )
        rr = (yy - radius_px) ** 2 + (xx - radius_px) ** 2
        ker = (rr <= radius_px**2).to(torch.float32)  # disk kernel

        inp = pts_img.unsqueeze(0).unsqueeze(0)  # 1x1xH xW
        ker = ker.unsqueeze(0).unsqueeze(0)  # 1x1xk xk
        inflated = F.conv2d(inp, ker, padding=radius_px)  # 1x1xH xW
        inflated_bool = inflated.squeeze(0).squeeze(0) > 0

        return mask_bool & (~inflated_bool)

    # --------------- Existing methods ---------------

    def _fps_2d(self, coords_xy: torch.Tensor, k: int) -> torch.Tensor:
        N = coords_xy.shape[0]
        if k <= 0 or N == 0:
            return coords_xy.new_zeros((0, 2))

        k = min(k, N)
        pts = coords_xy  # float32

        selected = torch.empty(k, dtype=torch.long, device=pts.device)
        min_dist2 = torch.full((N,), float("inf"), device=pts.device)

        centroid = pts.mean(dim=0, keepdim=True)  # (1,2)
        init_idx = torch.argmin(((pts - centroid) ** 2).sum(dim=1))
        selected[0] = init_idx
        last = pts[init_idx : init_idx + 1]  # (1,2)

        for i in range(1, k):
            diff = pts - last
            dist2 = (diff * diff).sum(dim=1)
            min_dist2 = torch.minimum(min_dist2, dist2)
            far_idx = torch.argmax(min_dist2)
            selected[i] = far_idx
            last = pts[far_idx : far_idx + 1]

        return pts[selected]

    def _fit_convex_hull(self, context: SamplerContext, obj_id: int):
        mask = context.frame.mask[obj_id, 0] > 0

        obj_idx = context.track_table.obj2track_map[obj_id]
        vis_obj = np.asarray(context.track_table.visible, dtype=bool)[obj_idx]

        idx = obj_idx[vis_obj]
        points = context.track_table.track_2d[idx]

        original_point_count = points.shape[0]
        if points.shape[0] > 0:
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            h, w = mask_np.shape
            valid_points_mask = (
                (points[:, 0] >= 0)
                & (points[:, 0] < w)
                & (points[:, 1] >= 0)
                & (points[:, 1] < h)
                & mask_np[points[:, 1].astype(int), points[:, 0].astype(int)]
            )
            points = points[valid_points_mask]
            idx = idx[valid_points_mask]

            filtered_count = original_point_count - points.shape[0]
            if filtered_count > 0:
                print(
                    f"Filtered out {filtered_count} points outside mask "
                    f"(kept {points.shape[0]}/{original_point_count})"
                )

            if points.shape[0] >= 3:
                hull = ConvexHull(points)
                convex_hull_xy = hull.points[hull.vertices]
                return convex_hull_xy
            else:
                return None
        return None

    def subtract_convex_hull(self, mask: np.ndarray, hull_xy: np.ndarray):
        if hull_xy is None:
            return mask.astype(bool)

        assert mask.ndim in (2, 3), "mask must be HxW or 1xHxW"
        H, W = mask.shape[-2], mask.shape[-1]

        poly = np.round(hull_xy).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        hull_mask_np = np.zeros((H, W), dtype=np.uint8)
        if len(poly) >= 3:
            cv.fillConvexPoly(hull_mask_np, poly.reshape(-1, 1, 2), 1)

        hull_mask = hull_mask_np.astype(bool)
        base = mask.astype(bool)
        out = base & (~hull_mask)
        return out.astype(np.uint8)
