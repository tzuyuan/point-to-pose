import numpy as np
import cv2 as cv
import torch
from scipy.spatial import ConvexHull, QhullError

from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER
from point2pose.data_types.sampler_context import SamplerContext


@SAMPLER.register_module("orb")
class ORBSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        self.num_points = int(config.get("num_points", 200))
        self.boundary_margin = int(config.get("edge_margin_px", 5))  # square (L∞) px
        self.cell_size = int(config.get("cell_size", 8))  # one-per-cell
        self.remove_convex_hull = bool(config.get("remove_convex_hull", True))
        # self.orb_detector = cv.ORB_create(nfeatures=int(config.get("nfeatures", 1000)))
        self.orb_detector = cv.ORB_create(
            nfeatures=2000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=5,
            patchSize=31,
            fastThreshold=2,
        )
        # if True, compute hull only once per object (you can change this policy)
        self.compute_hull_once = bool(config.get("compute_hull_once", False))
        self._initialized_obj_ids = set()

    def sample(self, context: SamplerContext, obj_id: int):
        frame = context.frame
        rgb = frame.rgb  # HxWx3, uint8 RGB
        H, W = rgb.shape[:2]

        # --- 1) Base mask (uint8 0/255 for OpenCV)
        mask_t = frame.mask[obj_id, 0]  # torch [H,W], bool/uint8 on CUDA
        mask_u8 = (mask_t > 0).to(torch.uint8).mul_(255).cpu().numpy()

        # --- 2) Boundary margin via RECT erosion (Chebyshev distance)
        if self.boundary_margin > 0:
            ksz = 2 * self.boundary_margin + 1
            kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (ksz, ksz))
            mask_inner = cv.erode(mask_u8, kernel_rect, iterations=1)  # uint8 0/255
        else:
            mask_inner = mask_u8

        # --- 3) Optional: subtract convex hull of existing points (in-mask)
        hull_xy = getattr(frame, "convex_hull_xy", None)
        need_fit = (
            (hull_xy is None)
            or (not self.compute_hull_once)
            or (obj_id not in self._initialized_obj_ids)
        )
        if self.remove_convex_hull and need_fit:
            hull_xy = self._fit_convex_hull(
                context, obj_id, mask_inner
            )  # may return None
            frame.convex_hull_xy = hull_xy
            self._initialized_obj_ids.add(obj_id)

        detect_mask = mask_inner
        if self.remove_convex_hull and hull_xy is not None and len(hull_xy) >= 3:
            detect_mask = self.subtract_convex_hull(mask_inner, hull_xy)  # uint8 0/255

        # Early out if nothing left to detect
        if detect_mask.max() == 0:
            if self.debug_level >= 1:
                print(
                    f"[ORB Sampler] Detect mask empty after boundary/hull (obj {obj_id})"
                )
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32), None

        # --- 4) ORB detect+compute with the correct detect_mask
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        kps, des = self.orb_detector.detectAndCompute(gray, detect_mask)

        if not kps or des is None or len(kps) == 0:
            if self.debug_level >= 1:
                print(f"[ORB Sampler] No ORB keypoints for object {obj_id}")
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32), None

        # --- 5) To arrays + sanity filter still inside detect_mask (cheap safety)
        pts_xy = np.array([kp.pt for kp in kps], dtype=np.float32)
        pts_xy_int = np.round(pts_xy).astype(np.int32)
        responses = np.array([kp.response for kp in kps], dtype=np.float32)

        x = np.clip(pts_xy_int[:, 0], 0, W - 1)
        y = np.clip(pts_xy_int[:, 1], 0, H - 1)
        inside = detect_mask[y, x] > 0

        pts_xy_int = pts_xy_int[inside]
        responses = responses[inside]
        des = des[inside] if des is not None else None

        if pts_xy_int.shape[0] == 0:
            if self.debug_level >= 1:
                print(
                    f"[ORB Sampler] All keypoints filtered out by detect_mask (obj {obj_id})"
                )
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32), None

        # --- 6) One-per-square grid rejection (new-new spacing)
        order = np.argsort(-responses)  # strongest first
        pts_xy_int = pts_xy_int[order]
        des = des[order] if des is not None else None

        occupied = set()
        keep_idx = []
        for j, (xj, yj) in enumerate(pts_xy_int):
            cx, cy = int(xj) // self.cell_size, int(yj) // self.cell_size
            if (cx, cy) in occupied:
                continue
            occupied.add((cx, cy))
            keep_idx.append(j)
            if len(keep_idx) >= self.num_points:
                break

        if len(keep_idx) == 0:
            if self.debug_level >= 1:
                print(
                    f"[ORB Sampler] No keypoints survived grid rejection (obj {obj_id})"
                )
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32), None

        keep_idx = np.asarray(keep_idx, dtype=np.int32)
        sel_pts = pts_xy_int[keep_idx]
        sel_des = des[keep_idx] if des is not None else None

        # --- 7) Debug viz
        if self.debug_level >= 1:
            self._viz_hull(
                rgb,
                mask_inner,
                hull_xy,
                detect_mask,
                frame.id,
                obj_id,
                kept_points=sel_pts,
            )

        print(
            f"[ORB Sampler] ORB kept {len(sel_pts)} (from {len(kps)}) for object {obj_id}"
        )
        return sel_pts

    # ---------- helpers ----------

    def _fit_convex_hull(
        self, context: SamplerContext, obj_id: int, mask_inner_u8: np.ndarray
    ):
        """
        Build convex hull from existing tracked 2D points that lie inside the (eroded) mask.
        Returns hull vertices as (K,2) float32 in image (x,y), or None if not enough points.
        """
        # Try a couple of likely locations for the 2D points table; adapt to your codebase
        pts = None
        tbl = getattr(context, "point_track_table", None) or getattr(
            context, "track_table", None
        )
        if tbl is not None:
            # Common: tbl.track_2d (N,2) float, tbl.visible (N,), tbl.obj2track_map[obj_id] gives indices
            track_2d = getattr(tbl, "track_2d", None)
            visible = getattr(tbl, "visible", None)
            obj2track = getattr(tbl, "obj2track_map", None)
            if track_2d is not None and visible is not None and obj2track is not None:
                obj_idx = obj2track[obj_id]  # indices for this object
                vis_mask = np.asarray(visible, dtype=bool)[obj_idx]
                pts = track_2d[obj_idx][vis_mask]  # (M,2) (x,y)

        if pts is None or len(pts) < 3:
            return None

        # Keep only points inside mask_inner
        H, W = mask_inner_u8.shape
        x = np.clip(pts[:, 0].astype(np.int32), 0, W - 1)
        y = np.clip(pts[:, 1].astype(np.int32), 0, H - 1)
        inside = mask_inner_u8[y, x] > 0
        pts_in = pts[inside]
        if pts_in.shape[0] < 3:
            return None

        try:
            hull = ConvexHull(pts_in)
        except QhullError:
            return None

        hull_xy = hull.points[hull.vertices].astype(np.float32)  # (K,2), (x,y)
        return hull_xy

    def subtract_convex_hull(self, mask_u8: np.ndarray, hull_xy: np.ndarray):
        """
        mask_u8: uint8 [H,W] with {0,255}
        hull_xy: (K,2) float/int (x,y)
        returns uint8 mask with hull area removed: mask_u8 & ~hull_mask
        """
        H, W = mask_u8.shape
        poly = np.round(hull_xy).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        hull_mask = np.zeros((H, W), dtype=np.uint8)
        if len(poly) >= 3:
            cv.fillConvexPoly(hull_mask, poly.reshape(-1, 1, 2), 255)  # {0,255}

        # mask_out = mask_u8 AND NOT hull_mask
        inv_hull = cv.bitwise_not(hull_mask)
        out = cv.bitwise_and(mask_u8, inv_hull)  # uint8 {0,255}
        return out

    def _viz_hull(
        self, rgb, mask_inner, hull_xy, detect_mask, frame_id, obj_id, kept_points=None
    ):
        """
        Save:
          - *_mask_removed.png  : the detect_mask (mask with hull removed)
          - *_rgb_overlay.png   : RGB with hull polygon + semi-transparent removed area + kept points
        """
        H, W = mask_inner.shape
        rgb_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR).copy()

        # 1) draw hull polygon + translucent fill (removed area)
        if hull_xy is not None and len(hull_xy) >= 3:
            poly = np.round(hull_xy).astype(np.int32).reshape(-1, 1, 2)
            # translucent fill
            overlay = rgb_bgr.copy()
            cv.fillConvexPoly(overlay, poly, (0, 0, 255))  # red fill (removed region)
            cv.addWeighted(overlay, 0.3, rgb_bgr, 0.7, 0, rgb_bgr)
            # border
            cv.polylines(rgb_bgr, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

        # 2) draw kept points
        if kept_points is not None:
            for xk, yk in kept_points:
                cv.circle(rgb_bgr, (int(xk), int(yk)), 3, (0, 255, 0), -1)

        # 3) dump images
        mask_removed_bgr = cv.cvtColor(detect_mask, cv.COLOR_GRAY2BGR)
        cv.imwrite(
            f"{self.debug_dir}/{frame_id}_mask_removed_{obj_id}.png", mask_removed_bgr
        )
        cv.imwrite(f"{self.debug_dir}/{frame_id}_rgb_overlay_{obj_id}.png", rgb_bgr)
