import numpy as np
import cv2 as cv
import torch

from scipy.spatial import ConvexHull

from point2pose.data_types.frame import Frame
from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER
from point2pose.data_types.sampler_context import SamplerContext


@SAMPLER.register_module("uniform_fps")
class UniformFPSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        self.num_points = config.get("num_points", 10)
        # Distance (in pixels) from the mask boundary to exclude
        self.edge_margin_px = config.get("edge_margin_px", 5)
        # If erosion removes too many pixels, shrink margin by this ratio (<1)
        self.edge_margin_shrink_ratio = float(
            config.get("edge_margin_shrink_ratio", 0.8)
        )
        # Minimum number of safe pixels required before accepting the safe region
        self.min_safe_points = int(config.get("min_safe_points", 5))
        self._remove_convex_hull = config.get("remove_convex_hull", True)
        self._i = 0
        self._initialized_obj_ids = set()

    def sample(self, context: SamplerContext, obj_id: int) -> np.ndarray:
        """
        Uniform-like sampling (blue-noise) via Farthest Point Sampling within the mask,
        while staying at least `edge_margin_px` away from the mask boundary.
        Returns integer (x,y) coordinates (np.int32).
        """

        frame = context.frame

        # ---- 1) Get mask -> CPU uint8
        mask_t = frame.mask[obj_id, 0]  # [H,W], bool/uint8, on CUDA/CPU
        if isinstance(mask_t, torch.Tensor):
            mask_np = (mask_t.detach().cpu().numpy() > 0).astype(np.uint8)
        else:
            mask_np = (mask_t > 0).astype(np.uint8)

        if obj_id in self._initialized_obj_ids:
            if self._remove_convex_hull and frame.convex_hull_xy is not None:
                mask_np = self.subtract_convex_hull(mask_np, frame.convex_hull_xy)
            elif self._remove_convex_hull and frame.convex_hull_xy is None:
                frame.convex_hull_xy = self._fit_convex_hull(context, obj_id)
                mask_np = self.subtract_convex_hull(mask_np, frame.convex_hull_xy)
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
            # Euclidean distance to nearest zero (boundary/outside); only computed inside mask
            # Note: cv.distanceTransform expects non-zero = foreground
            dist = cv.distanceTransform(mask_np, distanceType=cv.DIST_L2, maskSize=3)
            safe = dist >= d  # keep pixels at least d px from boundary
        else:
            dist = None
            safe = mask_np.astype(bool)

        # If erosion removes too many pixels, gradually relax d by a ratio until enough remain
        if np.count_nonzero(safe) < self.min_safe_points:
            ratio = self.edge_margin_shrink_ratio
            # Ensure a sane ratio
            if not (0.0 < ratio < 1.0):
                ratio = 0.8
            if d > 0 and dist is None:
                dist = cv.distanceTransform(
                    mask_np, distanceType=cv.DIST_L2, maskSize=3
                )
            d_float = float(d)
            attempts = 0
            while d_float > 0 and np.count_nonzero(safe) < self.min_safe_points:
                d_float *= ratio
                d_new = int(max(0, np.floor(d_float)))
                # Ensure progress
                if d_new == d:
                    if d_new == 0:
                        break
                    d_new = d - 1
                d = d_new
                if d > 0:
                    safe = dist >= d
                else:
                    safe = mask_np.astype(bool)
                attempts += 1
            if getattr(self, "debug_level", 0) >= 1:
                print(
                    f"[UniformFPSampler] Relaxed edge_margin_px to {d} after {attempts} step(s); "
                    f"safe_count={np.count_nonzero(safe)} (min={self.min_safe_points}) for object {obj_id}."
                )

        # ---- 3) Build candidate coords (x,y) from safe region
        ys, xs = np.where(safe)
        coords_xy_np = np.stack([xs, ys], axis=1)  # (N,2) int32
        N = coords_xy_np.shape[0]

        if N == 0:
            return np.zeros((0, 2), dtype=np.int32)

        # Move to torch (use same device as mask_t if it’s a tensor; else CPU)
        device = (
            mask_t.device if isinstance(mask_t, torch.Tensor) else torch.device("cpu")
        )
        coords_xy = torch.from_numpy(coords_xy_np).to(
            device=device, dtype=torch.float32
        )

        # ---- 4) FPS for uniform coverage
        k = min(self.num_points, coords_xy.shape[0])
        picked_xy = self._fps_2d(coords_xy, k)  # (k,2) float32 on device
        sampled_np = picked_xy.detach().cpu().numpy().round().astype(np.int32)

        # ---- 5) Debug viz (optional)
        if getattr(self, "debug_level", 0) >= 1:
            # Mask image (BGR)
            mask_img = cv.cvtColor((mask_np * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)

            # RGB image (assume HxWx3 RGB uint8, else convert)
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
            print(f"[UniformFPSampler] Saved debug images to {out_dir}")

        if sampled_np.shape[0] < self.num_points:
            print(
                f"[UniformFPSampler] Returned {sampled_np.shape[0]} points (limited by safe region)."
            )
        else:
            print(
                f"[UniformFPSampler] Sampled {sampled_np.shape[0]} points for object {obj_id}."
            )

        return sampled_np

    def _fps_2d(self, coords_xy: torch.Tensor, k: int) -> torch.Tensor:
        """
        Farthest Point Sampling over 2D coords (x,y). Returns (k,2).
        Deterministic: starts from the coordinate nearest the centroid.
        """
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
        """
        Fit the convex hull of the object in the mask.
        ## TODO: optimize this
        """
        mask = context.frame.mask[obj_id, 0] > 0

        obj_idx = context.track_table.obj2track_map[obj_id]
        vis_obj = np.asarray(context.track_table.visible, dtype=bool)[obj_idx]

        idx = obj_idx[vis_obj]
        points = context.track_table.track_2d[idx]

        # remove points outside of the mask
        original_point_count = points.shape[0]
        if points.shape[0] > 0:
            # Convert mask to numpy for indexing
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

            # Get image dimensions
            h, w = mask_np.shape

            # Filter points that are within image bounds and inside the mask
            valid_points_mask = (
                (points[:, 0] >= 0)
                & (points[:, 0] < w)  # x within bounds
                & (points[:, 1] >= 0)
                & (points[:, 1] < h)  # y within bounds
                & mask_np[
                    points[:, 1].astype(int), points[:, 0].astype(int)
                ]  # inside mask
            )

            # Keep only valid points
            points = points[valid_points_mask]
            idx = idx[valid_points_mask]

            # Debug info
            filtered_count = original_point_count - points.shape[0]
            if filtered_count > 0:
                print(
                    f"Filtered out {filtered_count} points outside mask (kept {points.shape[0]}/{original_point_count})"
                )

            # --- compute point region area ---

            hull = ConvexHull(points)

            convex_hull_xy = hull.points[hull.vertices]

            return convex_hull_xy

    def subtract_convex_hull(self, mask: np.ndarray, hull_xy: np.ndarray):
        """
        mask: np.ndarray or {0,1} HxW (or 1xHxW)
        hull_xy: (K, 2) numpy array of (x, y) pixel coords for the convex hull boundary
        """
        assert mask.ndim in (2, 3), "mask must be HxW or 1xHxW"
        H, W = mask.shape[-2], mask.shape[-1]

        # Build a CPU uint8 canvas and fill the convex polygon
        poly = np.round(hull_xy).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        hull_mask_np = np.zeros((H, W), dtype=np.uint8)
        if len(poly) >= 3:  # only fill if valid polygon
            cv.fillConvexPoly(hull_mask_np, poly.reshape(-1, 1, 2), 1)

        hull_mask = hull_mask_np.astype(bool)

        out = mask & (~hull_mask)  # mask - convex_hull
        return out
