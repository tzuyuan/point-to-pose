import os
import numpy as np
import cv2 as cv

import torch
from scipy.spatial import ConvexHull, QhullError

from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER
from point2pose.data_types.sampler_context import SamplerContext


@SAMPLER.register_module("super_point")
class SuperPointSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        self.num_points = int(config.get("num_points", 200))
        self.boundary_margin = int(config.get("edge_margin_px", 5))  # square (L∞) px
        self.cell_size = int(config.get("cell_size", 8))  # one-per-cell
        self.remove_convex_hull = bool(config.get("remove_convex_hull", True))
        self.device = config.get("device", "cuda")

        # Cropping controls
        self.crop_to_mask = bool(config.get("crop_to_mask", False))
        self.crop_pad_px = int(config.get("crop_pad_px", 0))

        self.super_point_max_num_keypoints = int(
            config.get("super_point_max_num_keypoints", 512)
        )
        # Debug save controls
        self.debug_save_superpoint = bool(config.get("debug_save_superpoint", False))
        self.debug_save_limit = int(config.get("debug_save_limit", 5000))
        self.debug_save_score_heatmap = bool(
            config.get("debug_save_score_heatmap", True)
        )

        from lightglue import SuperPoint

        self.super_point_extractor = (
            SuperPoint(max_num_keypoints=self.super_point_max_num_keypoints)
            .eval()
            .to(self.device)
        )

        # if True, compute hull only once per object
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

        # --- 3) subtract convex hull of existing points (in-mask)
        hull_xy = getattr(frame, "convex_hull_xy", None)
        need_fit = (
            (hull_xy is None)
            or (not self.compute_hull_once)
            or (obj_id not in self._initialized_obj_ids)
        )
        if self.remove_convex_hull and need_fit:
            hull_xy = self._fit_convex_hull(context, obj_id, mask_inner)  # may be None
            frame.convex_hull_xy = hull_xy
            self._initialized_obj_ids.add(obj_id)

        detect_mask = mask_inner
        if self.remove_convex_hull and hull_xy is not None and len(hull_xy) >= 3:
            detect_mask = self.subtract_convex_hull(mask_inner, hull_xy)  # uint8 0/255

        # --- 4) SuperPoint (with optional cropping)
        if self.crop_to_mask:
            (y0, y1, x0, x1), used_crop = self._compute_crop_from_mask(
                detect_mask, H, W, pad=self.crop_pad_px
            )
            rgb_used = rgb[y0:y1, x0:x1] if used_crop else rgb
            detect_mask_used = detect_mask[y0:y1, x0:x1] if used_crop else detect_mask
            x_offset, y_offset = (x0 if used_crop else 0), (y0 if used_crop else 0)
        else:
            rgb_used = rgb
            detect_mask_used = detect_mask
            x_offset, y_offset = 0, 0
            used_crop = False
            y0, y1, x0, x1 = 0, H, 0, W  # full frame box for metadata

        # Prepare tensor for SuperPoint
        rgb_float = rgb_used.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).to(self.device)

        feats = self.super_point_extractor.extract(rgb_tensor)  # dict
        # RAW (pre-filter) tensors -> numpy
        kps_xy_raw = feats["keypoints"][0]  # (N,2) (x,y) LOCAL (crop) coords if cropped
        kp_sc_raw = feats["keypoint_scores"][0]  # (N,)
        desc_raw = feats.get("descriptors", None)[0]  # (N,D) or None

        if torch.is_tensor(kps_xy_raw):
            kps_xy_raw = kps_xy_raw.detach().cpu().numpy()
        if torch.is_tensor(kp_sc_raw):
            kp_sc_raw = kp_sc_raw.detach().cpu().numpy()
        if desc_raw is not None and torch.is_tensor(desc_raw):
            desc_raw = desc_raw.detach().cpu().numpy()

        # --- Debug SAVE of RAW SuperPoint outputs (before any filtering)
        if self.debug_save_superpoint:
            self._save_sp_debug_raw(
                rgb_full=rgb,
                rgb_used=rgb_used,
                kps_local=kps_xy_raw,
                scores=kp_sc_raw,
                desc=desc_raw,
                frame_id=frame.id,
                obj_id=obj_id,
                used_crop=used_crop,
                x_offset=x_offset,
                y_offset=y_offset,
                crop_box=(y0, y1, x0, x1),
            )
            if (
                self.debug_save_score_heatmap
                and kps_xy_raw is not None
                and len(kps_xy_raw) > 0
            ):
                # RAW score visualizations
                self._save_score_visuals(
                    rgb_full=rgb,
                    rgb_used=rgb_used,
                    used_crop=used_crop,
                    kps_local=kps_xy_raw,
                    scores=kp_sc_raw,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    frame_id=frame.id,
                    obj_id=obj_id,
                    tag="raw",
                )

        if kps_xy_raw is None or kp_sc_raw is None or kps_xy_raw.shape[0] == 0:
            if self.debug_level >= 1:
                print(f"[SuperPoint] No keypoints for object {obj_id}")
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32)

        # --- 5) Mask filtering (keep only points inside detect_mask_used)
        h_used, w_used = detect_mask_used.shape
        pts_xy_int_local = np.round(kps_xy_raw).astype(np.int32)
        x_local = np.clip(pts_xy_int_local[:, 0], 0, w_used - 1)
        y_local = np.clip(pts_xy_int_local[:, 1], 0, h_used - 1)
        inside = detect_mask_used[y_local, x_local] > 0

        kps_xy = kps_xy_raw[inside]
        kp_sc = kp_sc_raw[inside]
        pts_xy_int_local = pts_xy_int_local[inside]
        desc = None if desc_raw is None else desc_raw[inside]

        if pts_xy_int_local.shape[0] == 0:
            if self.debug_level >= 1:
                print(
                    f"[SuperPoint] All keypoints filtered out by detect_mask (obj {obj_id})"
                )
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32)

        # --- 6) Top-N by score (descending)
        n_keep = min(self.num_points, kp_sc.shape[0])
        top_idx = np.argpartition(-kp_sc, n_keep - 1)[:n_keep]
        top_idx = top_idx[np.argsort(-kp_sc[top_idx])]
        top_idx_inside = top_idx.copy()  # keep mapping within 'inside' subset

        pts_xy_int_local = pts_xy_int_local[top_idx]
        if desc is not None:
            desc = desc[top_idx]

        # OPTIONAL: one-per-cell AFTER top-N (still in local coords)
        grid_used = False
        if self.cell_size > 1:
            occupied = set()
            keep_idx = []
            for j, (xj, yj) in enumerate(pts_xy_int_local):
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
                        f"[SuperPoint] No keypoints survived grid rejection (obj {obj_id})"
                    )
                    self._viz_hull(
                        rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id
                    )
                return np.empty((0, 2), dtype=np.int32)
            keep_idx = np.asarray(keep_idx, dtype=np.int32)
            pts_xy_int_local = pts_xy_int_local[keep_idx]
            if desc is not None:
                desc = desc[keep_idx]
            grid_used = True

        # --- NEW: selection vs rejection indices in RAW space
        if self.debug_save_superpoint:
            raw_count = len(kps_xy_raw)
            raw_indices = np.arange(raw_count)
            inside_idx = np.where(inside)[0]  # raw indices that were inside mask

            if grid_used:
                selected_inside_idx = inside_idx[top_idx_inside[keep_idx]]
            else:
                selected_inside_idx = inside_idx[top_idx_inside]

            selected_raw_idx = selected_inside_idx
            rejected_raw_mask = np.ones(raw_count, dtype=bool)
            rejected_raw_mask[selected_raw_idx] = False
            rejected_raw_idx = raw_indices[rejected_raw_mask]

            # global coords for raw keypoints
            if used_crop:
                kps_global_raw = kps_xy_raw + np.array(
                    [x_offset, y_offset], dtype=kps_xy_raw.dtype
                )
            else:
                kps_global_raw = kps_xy_raw

            try:
                self._save_sp_selection_overlay(
                    rgb_full=rgb,
                    rgb_used=rgb_used,
                    used_crop=used_crop,
                    kps_local_all=kps_xy_raw,
                    kps_global_all=kps_global_raw,
                    selected_raw_idx=selected_raw_idx,
                    rejected_raw_idx=rejected_raw_idx,
                    frame_id=frame.id,
                    obj_id=obj_id,
                )
            except Exception as e:
                print(f"[SuperPoint Debug] Failed to save selection overlay: {e}")

        # --- Score visualizations for SELECTED points only (optional)
        if (
            self.debug_save_superpoint
            and self.debug_save_score_heatmap
            and len(kps_xy) > 0
        ):
            self._save_score_visuals(
                rgb_full=rgb,
                rgb_used=rgb_used,
                used_crop=used_crop,
                kps_local=kps_xy,  # selected subset in local coords
                scores=kp_sc,
                x_offset=x_offset,
                y_offset=y_offset,
                frame_id=frame.id,
                obj_id=obj_id,
                tag="selected",
            )

        # --- 6.5) Map local (crop) coords back to original image coords (FINAL selection)
        if used_crop:
            pts_xy_int_global = pts_xy_int_local.copy()
            pts_xy_int_global[:, 0] += x_offset
            pts_xy_int_global[:, 1] += y_offset
        else:
            pts_xy_int_global = pts_xy_int_local

        sel_pts = pts_xy_int_global  # (M,2) int in ORIGINAL IMAGE FRAME
        sel_des = desc  # (M,D) or None (kept for extension)

        # --- 7) Debug viz (always in original frame coords)
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
            f"[SuperPoint Sampler] SuperPoint kept {len(sel_pts)} for object {obj_id}"
        )
        return sel_pts

    # ---------- debug save helpers ----------

    def _ensure_dir(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"[SuperPoint Debug] Failed to create directory {path}: {e}")

    def _normalize_scores(self, scores: np.ndarray):
        if scores is None or len(scores) == 0:
            return np.array([], dtype=np.float32)
        s = scores.astype(np.float32)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if s_max <= s_min + 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    def _draw_score_scatter(
        self,
        base_img_bgr: np.ndarray,
        kps_xy: np.ndarray,
        scores: np.ndarray,
        radius: int = 2,
    ):
        """Draw score-colored scatter on a BGR image (modifies copy and returns it)."""
        out = base_img_bgr.copy()
        if kps_xy is None or len(kps_xy) == 0:
            return out
        s_norm = self._normalize_scores(scores)
        # map to 0..255 for colormap
        s_uint8 = np.clip(np.round(s_norm * 255.0), 0, 255).astype(np.uint8)
        for i, (x, y) in enumerate(kps_xy.astype(np.int32)):
            c = cv.applyColorMap(
                np.array([[s_uint8[i]]], dtype=np.uint8), cv.COLORMAP_JET
            )[0, 0]
            cv.circle(
                out,
                (int(x), int(y)),
                radius,
                color=tuple(int(v) for v in c),
                thickness=-1,
            )
        return out

    def _make_score_heatmap(
        self, shape_hw, kps_xy: np.ndarray, scores: np.ndarray, sigma: float = 3.0
    ):
        """Build a heatmap by splatting scores then blurring."""
        h, w = shape_hw
        heat = np.zeros((h, w), dtype=np.float32)
        if kps_xy is None or len(kps_xy) == 0:
            return (heat * 255).astype(np.uint8)
        s_norm = self._normalize_scores(scores)
        # splat points (cap drawing to limit for speed)
        draw_idx = np.arange(len(kps_xy))
        if len(draw_idx) > self.debug_save_limit:
            draw_idx = draw_idx[: self.debug_save_limit]
        for i in draw_idx:
            x, y = int(round(kps_xy[i, 0])), int(round(kps_xy[i, 1]))
            if 0 <= x < w and 0 <= y < h:
                heat[y, x] = max(heat[y, x], s_norm[i])
        # apply Gaussian blur to spread activation
        ksize = int(max(3, round(6 * sigma) | 1))  # odd kernel
        heat_blur = cv.GaussianBlur(
            heat,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv.BORDER_REPLICATE,
        )
        # normalize to 0..255 uint8
        if heat_blur.max() > 0:
            heat_blur = heat_blur / heat_blur.max()
        return (heat_blur * 255.0).astype(np.uint8)

    def _save_score_visuals(
        self,
        rgb_full: np.ndarray,
        rgb_used: np.ndarray,
        used_crop: bool,
        kps_local: np.ndarray,
        scores: np.ndarray,
        x_offset: int,
        y_offset: int,
        frame_id,
        obj_id: int,
        tag: str = "raw",
    ):
        """
        Save: *_score_heatmap_*.png and *_score_scatter_*.png for both full and crop views.
        tag: 'raw' or 'selected'
        """
        self._ensure_dir(self.debug_dir)

        # Full-frame: map local coords to global if cropped
        if used_crop:
            kps_global = kps_local + np.array(
                [x_offset, y_offset], dtype=kps_local.dtype
            )
        else:
            kps_global = kps_local

        # --- Heatmaps
        # Full
        heat_full_u8 = self._make_score_heatmap(
            rgb_full.shape[:2], kps_global, scores, sigma=3.0
        )
        heat_full_bgr = cv.applyColorMap(heat_full_u8, cv.COLORMAP_JET)
        cv.imwrite(
            f"{self.debug_dir}/{frame_id}_sp_{tag}_score_heatmap_full_obj{obj_id}.png",
            heat_full_bgr,
        )

        # Crop (local)
        if used_crop:
            heat_crop_u8 = self._make_score_heatmap(
                rgb_used.shape[:2], kps_local, scores, sigma=3.0
            )
            heat_crop_bgr = cv.applyColorMap(heat_crop_u8, cv.COLORMAP_JET)
            cv.imwrite(
                f"{self.debug_dir}/{frame_id}_sp_{tag}_score_heatmap_crop_obj{obj_id}.png",
                heat_crop_bgr,
            )

        # --- Scatter overlays (score-colored circles)
        full_bgr = cv.cvtColor(rgb_full, cv.COLOR_RGB2BGR)
        full_scatter = self._draw_score_scatter(full_bgr, kps_global, scores, radius=2)
        cv.imwrite(
            f"{self.debug_dir}/{frame_id}_sp_{tag}_score_scatter_full_obj{obj_id}.png",
            full_scatter,
        )

        if used_crop:
            crop_bgr = cv.cvtColor(rgb_used, cv.COLOR_RGB2BGR)
            crop_scatter = self._draw_score_scatter(
                crop_bgr, kps_local, scores, radius=2
            )
            cv.imwrite(
                f"{self.debug_dir}/{frame_id}_sp_{tag}_score_scatter_crop_obj{obj_id}.png",
                crop_scatter,
            )

    def _save_sp_debug_raw(
        self,
        rgb_full: np.ndarray,
        rgb_used: np.ndarray,
        kps_local: np.ndarray,
        scores: np.ndarray,
        desc: np.ndarray,
        frame_id,
        obj_id: int,
        used_crop: bool,
        x_offset: int,
        y_offset: int,
        crop_box,  # (y0,y1,x0,x1)
    ):
        """
        Save raw (pre-filter) SP outputs and images to disk.
        """
        self._ensure_dir(self.debug_dir)
        y0, y1, x0, x1 = crop_box

        # Compute global coordinates for raw points
        if used_crop:
            kps_global = kps_local + np.array(
                [x_offset, y_offset], dtype=kps_local.dtype
            )
        else:
            kps_global = kps_local.copy()

        # Limit how many points we draw to keep images readable
        draw_idx = np.arange(len(kps_local))
        if len(draw_idx) > self.debug_save_limit:
            draw_idx = draw_idx[: self.debug_save_limit]

        # Save npz (raw arrays)
        # npz_path = f"{self.debug_dir}/{frame_id}_sp_raw_obj{obj_id}.npz"
        # try:
        #     np.savez_compressed(
        #         npz_path,
        #         kps_local=kps_local,
        #         kps_global=kps_global,
        #         scores=scores,
        #         desc=desc if desc is not None else np.array([]),
        #         used_crop=np.array([used_crop], dtype=bool),
        #         x_offset=np.array([x_offset], dtype=np.int32),
        #         y_offset=np.array([y_offset], dtype=np.int32),
        #         crop_box=np.array([y0, y1, x0, x1], dtype=np.int32),
        #     )
        # except Exception as e:
        #     print(f"[SuperPoint Debug] Failed to save npz: {npz_path} ({e})")

        # Save full RGB and crop RGB
        full_img_bgr = cv.cvtColor(rgb_full, cv.COLOR_RGB2BGR)
        full_path = f"{self.debug_dir}/{frame_id}_rgb_full_obj{obj_id}.png"
        try:
            cv.imwrite(full_path, full_img_bgr)
        except Exception as e:
            print(f"[SuperPoint Debug] Failed to save full rgb: {full_path} ({e})")

        if used_crop:
            crop_img_bgr = cv.cvtColor(rgb_used, cv.COLOR_RGB2BGR)
            crop_path = f"{self.debug_dir}/{frame_id}_rgb_crop_obj{obj_id}.png"
            try:
                cv.imwrite(crop_path, crop_img_bgr)
            except Exception as e:
                print(f"[SuperPoint Debug] Failed to save crop rgb: {crop_path} ({e})")

        # Overlays of raw detections (simple green dots)
        try:
            overlay_full = full_img_bgr.copy()
            for i in draw_idx:
                x, y = int(round(kps_global[i, 0])), int(round(kps_global[i, 1]))
                cv.circle(overlay_full, (x, y), 2, (0, 255, 0), -1)
            cv.imwrite(
                f"{self.debug_dir}/{frame_id}_sp_overlay_full_obj{obj_id}.png",
                overlay_full,
            )
        except Exception as e:
            print(f"[SuperPoint Debug] Failed to save full overlay: {e}")

        if used_crop:
            try:
                overlay_crop = crop_img_bgr.copy()
                for i in draw_idx:
                    x, y = int(round(kps_local[i, 0])), int(round(kps_local[i, 1]))
                    cv.circle(overlay_crop, (x, y), 2, (0, 255, 0), -1)
                cv.imwrite(
                    f"{self.debug_dir}/{frame_id}_sp_overlay_crop_obj{obj_id}.png",
                    overlay_crop,
                )
            except Exception as e:
                print(f"[SuperPoint Debug] Failed to save crop overlay: {e}")

    def _save_sp_selection_overlay(
        self,
        rgb_full: np.ndarray,
        rgb_used: np.ndarray,
        used_crop: bool,
        kps_local_all: np.ndarray,
        kps_global_all: np.ndarray,
        selected_raw_idx: np.ndarray,
        rejected_raw_idx: np.ndarray,
        frame_id,
        obj_id: int,
    ):
        """
        Save an overlay with selected points (GREEN) and not-selected points (RED).
        Creates full-frame and (if applicable) crop overlays.
        """
        self._ensure_dir(self.debug_dir)

        # Optional draw limit
        if len(rejected_raw_idx) > self.debug_save_limit:
            rejected_raw_idx = rejected_raw_idx[: self.debug_save_limit]
        if len(selected_raw_idx) > self.debug_save_limit:
            selected_raw_idx = selected_raw_idx[: self.debug_save_limit]

        # Full-frame overlay (global coords)
        full_bgr = cv.cvtColor(rgb_full, cv.COLOR_RGB2BGR).copy()
        for i in rejected_raw_idx:
            x, y = int(round(kps_global_all[i, 0])), int(round(kps_global_all[i, 1]))
            cv.circle(full_bgr, (x, y), 2, (0, 0, 255), -1)  # RED
        for i in selected_raw_idx:
            x, y = int(round(kps_global_all[i, 0])), int(round(kps_global_all[i, 1]))
            cv.circle(full_bgr, (x, y), 2, (0, 255, 0), -1)  # GREEN
        cv.imwrite(
            f"{self.debug_dir}/{frame_id}_sp_sel_vs_rej_full_obj{obj_id}.png", full_bgr
        )

        # Crop overlay (local coords), only if a crop was used
        if used_crop:
            crop_bgr = cv.cvtColor(rgb_used, cv.COLOR_RGB2BGR).copy()
            for i in rejected_raw_idx:
                x, y = int(round(kps_local_all[i, 0])), int(round(kps_local_all[i, 1]))
                cv.circle(crop_bgr, (x, y), 2, (0, 0, 255), -1)  # RED
            for i in selected_raw_idx:
                x, y = int(round(kps_local_all[i, 0])), int(round(kps_local_all[i, 1]))
                cv.circle(crop_bgr, (x, y), 2, (0, 255, 0), -1)  # GREEN
            cv.imwrite(
                f"{self.debug_dir}/{frame_id}_sp_sel_vs_rej_crop_obj{obj_id}.png",
                crop_bgr,
            )

    # ---------- existing helpers ----------

    def _compute_crop_from_mask(
        self, mask_u8: np.ndarray, H: int, W: int, pad: int = 0
    ):
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0 or xs.size == 0:
            return (0, H, 0, W), False

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        if pad > 0:
            y0 = max(0, y0 - pad)
            x0 = max(0, x0 - pad)
            y1 = min(H, y1 + pad)
            x1 = min(W, x1 + pad)

        if (y1 - y0) < 2 or (x1 - x0) < 2:
            return (0, H, 0, W), False

        return (y0, y1, x0, x1), True

    def _fit_convex_hull(
        self, context: SamplerContext, obj_id: int, mask_inner_u8: np.ndarray
    ):
        pts = None
        tbl = getattr(context, "point_track_table", None) or getattr(
            context, "track_table", None
        )
        if tbl is not None:
            track_2d = getattr(tbl, "track_2d", None)
            visible = getattr(tbl, "visible", None)
            obj2track = getattr(tbl, "obj2track_map", None)
            if (
                track_2d is not None
                and visible is not None
                and obj2track is not None
                and obj_id in obj2track
            ):
                obj_idx = obj2track[obj_id]
                vis_mask = np.asarray(visible, dtype=bool)[obj_idx]
                pts = track_2d[obj_idx][vis_mask]  # (M,2) (x,y)

        if pts is None or len(pts) < 3:
            return None

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
        H, W = mask_u8.shape
        poly = np.round(hull_xy).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        hull_mask = np.zeros((H, W), dtype=np.uint8)
        if len(poly) >= 3:
            cv.fillConvexPoly(hull_mask, poly.reshape(-1, 1, 2), 255)
        inv_hull = cv.bitwise_not(hull_mask)
        out = cv.bitwise_and(mask_u8, inv_hull)
        return out

    def _viz_hull(
        self, rgb, mask_inner, hull_xy, detect_mask, frame_id, obj_id, kept_points=None
    ):
        H, W = mask_inner.shape
        rgb_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR).copy()

        if hull_xy is not None and len(hull_xy) >= 3:
            poly = np.round(hull_xy).astype(np.int32).reshape(-1, 1, 2)
            overlay = rgb_bgr.copy()
            cv.fillConvexPoly(overlay, poly, (0, 0, 255))  # red fill
            cv.addWeighted(overlay, 0.3, rgb_bgr, 0.7, 0, rgb_bgr)
            cv.polylines(rgb_bgr, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

        if kept_points is not None:
            for xk, yk in kept_points:
                cv.circle(rgb_bgr, (int(xk), int(yk)), 3, (0, 255, 0), -1)

        mask_removed_bgr = cv.cvtColor(detect_mask, cv.COLOR_GRAY2BGR)
        cv.imwrite(
            f"{self.debug_dir}/{frame_id}_mask_removed_{obj_id}.png", mask_removed_bgr
        )
        cv.imwrite(f"{self.debug_dir}/{frame_id}_rgb_overlay_{obj_id}.png", rgb_bgr)
