import numpy as np
import cv2 as cv
import torch

from point2pose.core.module_registry import SAMPLER
from point2pose.data_types.sampler_context import SamplerContext
from point2pose.modules.sampler.super_point_fps_sampler import SuperPointFPSSampler


@SAMPLER.register_module("super_point_balanced")
class SuperPointBalancedSampler(SuperPointFPSSampler):
    """
    SuperPoint sampler with adaptive quality+coverage balancing.

    Compared to plain top-score + FPS, this sampler:
      1) optionally enforces depth validity,
      2) optionally suppresses near-duplicate candidates (NMS),
      3) selects points with a greedy objective that balances score and novelty.
    """

    def __init__(self, config):
        super().__init__(config)

        # Optional adaptive target count by safe area (points per 1000 pixels).
        self.density_per_kpx = float(config.get("density_per_kpx", -1.0))
        self.min_points = int(config.get("min_points", 5))
        self.max_points = int(config.get("max_points", max(self.num_points, 5)))
        self.round_mode = str(config.get("round_mode", "round")).lower()

        # Candidate preprocessing.
        self.enable_depth_filter = bool(config.get("enable_depth_filter", True))
        self.depth_fallback_to_unfiltered = bool(
            config.get("depth_fallback_to_unfiltered", True)
        )
        self.candidate_score_percentile = float(
            config.get("candidate_score_percentile", 0.0)
        )
        self.nms_radius_px = float(config.get("nms_radius_px", 0.0))
        self.candidate_max_num = int(config.get("candidate_max_num", 0))

        # Selection objective weights.
        self.score_weight = float(np.clip(config.get("score_weight", 0.65), 0.0, 1.0))
        self.novelty_radius_scale = float(config.get("novelty_radius_scale", 1.0))
        self.min_separation_px = float(config.get("min_separation_px", 3.0))
        self.separation_penalty_weight = float(
            config.get("separation_penalty_weight", 0.35)
        )
        self.avoid_existing_points = bool(config.get("avoid_existing_points", True))

    def sample(self, context: SamplerContext, obj_id: int):
        frame = context.frame
        rgb = frame.rgb
        H, W = rgb.shape[:2]

        # --- 1) Base object mask
        mask_t = frame.mask[obj_id, 0]
        mask_u8 = (mask_t > 0).to(torch.uint8).mul_(255).cpu().numpy()

        # --- 2) Boundary margin erosion
        if self.boundary_margin > 0:
            ksz = 2 * self.boundary_margin + 1
            kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (ksz, ksz))
            mask_inner = cv.erode(mask_u8, kernel_rect, iterations=1)
        else:
            mask_inner = mask_u8

        # --- 3) Optional convex hull subtraction
        hull_xy = getattr(frame, "convex_hull_xy", None)
        need_fit = (
            (hull_xy is None)
            or (not self.compute_hull_once)
            or (obj_id not in self._initialized_obj_ids)
        )
        if self.remove_convex_hull and need_fit:
            hull_xy = self._fit_convex_hull(context, obj_id, mask_inner)
            frame.convex_hull_xy = hull_xy
            self._initialized_obj_ids.add(obj_id)

        detect_mask = mask_inner
        if self.remove_convex_hull and hull_xy is not None and len(hull_xy) >= 3:
            detect_mask = self.subtract_convex_hull(mask_inner, hull_xy)

        # --- 4) Optional crop to object mask
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
            y0, y1, x0, x1 = 0, H, 0, W

        # --- 5) SuperPoint extraction (local crop coordinates)
        rgb_float = rgb_used.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).to(self.device)
        feats = self.super_point_extractor.extract(rgb_tensor)

        kps_xy_raw = feats["keypoints"][0]
        kp_sc_raw = feats["keypoint_scores"][0]
        desc_raw = feats.get("descriptors", None)
        if desc_raw is not None:
            desc_raw = desc_raw[0]

        if torch.is_tensor(kps_xy_raw):
            kps_xy_raw = kps_xy_raw.detach().cpu().numpy()
        if torch.is_tensor(kp_sc_raw):
            kp_sc_raw = kp_sc_raw.detach().cpu().numpy()
        if desc_raw is not None and torch.is_tensor(desc_raw):
            desc_raw = desc_raw.detach().cpu().numpy()

        # Optional raw debug dump
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
            if getattr(self, "debug_level", 0) >= 1:
                print(f"[SuperPoint Balanced] No keypoints for object {obj_id}")
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32)

        # --- 6) Keep only keypoints inside detection mask
        h_used, w_used = detect_mask_used.shape
        pts_xy_int_local_all = np.round(kps_xy_raw).astype(np.int32)
        x_local_all = np.clip(pts_xy_int_local_all[:, 0], 0, w_used - 1)
        y_local_all = np.clip(pts_xy_int_local_all[:, 1], 0, h_used - 1)
        inside = detect_mask_used[y_local_all, x_local_all] > 0

        kps_xy = kps_xy_raw[inside]
        kp_sc = kp_sc_raw[inside]
        desc = None if desc_raw is None else desc_raw[inside]
        raw_idx = np.where(inside)[0]

        if kps_xy.shape[0] == 0:
            if getattr(self, "debug_level", 0) >= 1:
                print(
                    f"[SuperPoint Balanced] All keypoints filtered by mask (obj {obj_id})"
                )
                self._viz_hull(rgb, mask_inner, hull_xy, detect_mask, frame.id, obj_id)
            return np.empty((0, 2), dtype=np.int32)

        # --- 7) Optional depth filtering
        pts_xy_int_local = np.round(kps_xy).astype(np.int32)
        if self.enable_depth_filter:
            valid_depth = self._valid_depth_mask(
                frame=frame,
                context=context,
                pts_xy_int_local=pts_xy_int_local,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            if valid_depth is not None:
                if np.any(valid_depth):
                    kps_xy = kps_xy[valid_depth]
                    kp_sc = kp_sc[valid_depth]
                    pts_xy_int_local = pts_xy_int_local[valid_depth]
                    raw_idx = raw_idx[valid_depth]
                    if desc is not None:
                        desc = desc[valid_depth]
                elif not self.depth_fallback_to_unfiltered:
                    if getattr(self, "debug_level", 0) >= 1:
                        print(
                            f"[SuperPoint Balanced] All keypoints invalid depth (obj {obj_id})"
                        )
                    return np.empty((0, 2), dtype=np.int32)

        if pts_xy_int_local.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)

        # --- 8) Sort by score descending (single source of priority)
        sort_idx = np.argsort(-kp_sc)
        kps_xy = kps_xy[sort_idx]
        kp_sc = kp_sc[sort_idx]
        pts_xy_int_local = pts_xy_int_local[sort_idx]
        raw_idx = raw_idx[sort_idx]
        if desc is not None:
            desc = desc[sort_idx]

        # --- 9) Optional score percentile trimming
        if 0.0 < self.candidate_score_percentile < 100.0 and kp_sc.shape[0] > 1:
            score_th = np.percentile(kp_sc, self.candidate_score_percentile)
            keep = kp_sc >= score_th
            if np.any(keep):
                kps_xy = kps_xy[keep]
                kp_sc = kp_sc[keep]
                pts_xy_int_local = pts_xy_int_local[keep]
                raw_idx = raw_idx[keep]
                if desc is not None:
                    desc = desc[keep]

        # --- 10) Optional local NMS
        if self.nms_radius_px > 0.0 and pts_xy_int_local.shape[0] > 1:
            keep_idx = self._greedy_nms_indices(pts_xy_int_local, self.nms_radius_px)
            kps_xy = kps_xy[keep_idx]
            kp_sc = kp_sc[keep_idx]
            pts_xy_int_local = pts_xy_int_local[keep_idx]
            raw_idx = raw_idx[keep_idx]
            if desc is not None:
                desc = desc[keep_idx]

        # --- 11) Optional coarse one-per-cell rejection
        if self.cell_size > 1:
            occupied = set()
            keep = []
            for i, (x_i, y_i) in enumerate(pts_xy_int_local):
                cell = (int(x_i) // self.cell_size, int(y_i) // self.cell_size)
                if cell in occupied:
                    continue
                occupied.add(cell)
                keep.append(i)
            if len(keep) == 0:
                return np.empty((0, 2), dtype=np.int32)
            keep = np.asarray(keep, dtype=np.int32)
            kps_xy = kps_xy[keep]
            kp_sc = kp_sc[keep]
            pts_xy_int_local = pts_xy_int_local[keep]
            raw_idx = raw_idx[keep]
            if desc is not None:
                desc = desc[keep]

        if pts_xy_int_local.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)

        # --- 12) Determine target count
        target_k = self._target_num_points(detect_mask_used)
        target_k = min(target_k, pts_xy_int_local.shape[0])
        if target_k <= 0:
            return np.empty((0, 2), dtype=np.int32)

        # --- 13) Candidate subset cap (for speed)
        max_candidates = self._max_candidates(target_k, pts_xy_int_local.shape[0])
        cand_pts = pts_xy_int_local[:max_candidates]
        cand_scores = kp_sc[:max_candidates]
        cand_kps_xy = kps_xy[:max_candidates]
        cand_raw_idx = raw_idx[:max_candidates]

        # Optional existing-point suppression (novelty initialized from old tracks)
        existing_pts_local = None
        if self.avoid_existing_points:
            existing_pts_local = self._collect_existing_points_local(
                context=context,
                obj_id=obj_id,
                detect_mask_used=detect_mask_used,
                used_crop=used_crop,
                x_offset=x_offset,
                y_offset=y_offset,
            )

        selected_local_idx = self._balanced_greedy_select(
            cand_pts_xy=cand_pts,
            cand_scores=cand_scores,
            k=target_k,
            safe_area_px=int(np.count_nonzero(detect_mask_used)),
            existing_pts_xy=existing_pts_local,
        )
        sel_pts_local = cand_pts[selected_local_idx]
        sel_scores = cand_scores[selected_local_idx]
        sel_kps_local = cand_kps_xy[selected_local_idx]
        selected_raw_idx = cand_raw_idx[selected_local_idx]

        # --- 14) Map local coordinates back to global frame coordinates
        if used_crop:
            sel_pts = sel_pts_local.copy()
            sel_pts[:, 0] += x_offset
            sel_pts[:, 1] += y_offset
        else:
            sel_pts = sel_pts_local

        # Optional selection/rejection debug overlay
        if self.debug_save_superpoint:
            if used_crop:
                kps_global_raw = kps_xy_raw + np.array(
                    [x_offset, y_offset], dtype=kps_xy_raw.dtype
                )
            else:
                kps_global_raw = kps_xy_raw
            all_raw_idx = np.arange(kps_xy_raw.shape[0], dtype=np.int32)
            rejected_raw_idx = all_raw_idx[
                ~np.isin(all_raw_idx, selected_raw_idx, assume_unique=False)
            ]
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
                print(f"[SuperPoint Balanced Debug] Failed selection overlay: {e}")

            if self.debug_save_score_heatmap and len(sel_kps_local) > 0:
                self._save_score_visuals(
                    rgb_full=rgb,
                    rgb_used=rgb_used,
                    used_crop=used_crop,
                    kps_local=sel_kps_local,
                    scores=sel_scores,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    frame_id=frame.id,
                    obj_id=obj_id,
                    tag="selected",
                )

        # Optional hull visualization
        if getattr(self, "debug_level", 0) >= 1:
            self._viz_hull(
                rgb,
                mask_inner,
                hull_xy,
                detect_mask,
                frame.id,
                obj_id,
                kept_points=sel_pts,
            )

        print(f"[SuperPoint Balanced] Kept {len(sel_pts)} points for object {obj_id}")
        return sel_pts

    def _target_num_points(self, detect_mask_u8: np.ndarray) -> int:
        if self.density_per_kpx <= 0:
            return int(self.num_points)

        safe_area_px = int(np.count_nonzero(detect_mask_u8))
        raw = self.density_per_kpx * (safe_area_px / 1000.0)
        if self.round_mode == "ceil":
            k_auto = int(np.ceil(raw))
        elif self.round_mode == "floor":
            k_auto = int(np.floor(raw))
        else:
            k_auto = int(np.round(raw))

        low = max(int(self.min_points), 0)
        high = max(int(self.max_points), low)
        return int(np.clip(k_auto, low, high))

    def _max_candidates(self, target_k: int, total_candidates: int) -> int:
        if self.candidate_max_num > 0:
            max_candidates = self.candidate_max_num
        else:
            max_candidates = max(
                target_k, target_k * max(int(self.fps_oversample_factor), 1)
            )
        max_candidates = max(1, max_candidates)
        return int(min(max_candidates, total_candidates))

    def _valid_depth_mask(
        self,
        frame,
        context: SamplerContext,
        pts_xy_int_local: np.ndarray,
        x_offset: int,
        y_offset: int,
    ):
        if pts_xy_int_local.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        depth = getattr(frame, "depth", None)
        if depth is None:
            return None

        if torch.is_tensor(depth):
            depth_np = depth.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth)

        if depth_np.ndim != 2:
            return None

        h_depth, w_depth = depth_np.shape[:2]
        depth_factor = frame.depth_factor if frame.depth_factor is not None else 1.0
        pts_xy_int_global = pts_xy_int_local.copy()
        pts_xy_int_global[:, 0] += int(x_offset)
        pts_xy_int_global[:, 1] += int(y_offset)

        x = np.clip(pts_xy_int_global[:, 0], 0, w_depth - 1)
        y = np.clip(pts_xy_int_global[:, 1], 0, h_depth - 1)
        depths = depth_np[y, x].astype(np.float32) * float(depth_factor)

        valid = np.isfinite(depths)
        if context.min_depth is not None:
            valid &= depths > float(context.min_depth)
        if context.max_depth is not None:
            valid &= depths < float(context.max_depth)
        return valid

    def _greedy_nms_indices(
        self, pts_xy_int: np.ndarray, radius_px: float
    ) -> np.ndarray:
        if pts_xy_int.shape[0] <= 1 or radius_px <= 0:
            return np.arange(pts_xy_int.shape[0], dtype=np.int32)

        r2 = float(radius_px * radius_px)
        keep = []
        pts = pts_xy_int.astype(np.float32)
        for i in range(pts.shape[0]):
            if len(keep) == 0:
                keep.append(i)
                continue
            kept_pts = pts[np.asarray(keep, dtype=np.int32)]
            diff = kept_pts - pts[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            if np.min(d2) >= r2:
                keep.append(i)
        return np.asarray(keep, dtype=np.int32)

    def _collect_existing_points_local(
        self,
        context: SamplerContext,
        obj_id: int,
        detect_mask_used: np.ndarray,
        used_crop: bool,
        x_offset: int,
        y_offset: int,
    ) -> np.ndarray:
        tbl = getattr(context, "track_table", None)
        if tbl is None:
            return np.empty((0, 2), dtype=np.float32)

        track_2d = getattr(tbl, "track_2d", None)
        visible = getattr(tbl, "visible", None)
        obj2track = getattr(tbl, "obj2track_map", None)
        if track_2d is None or visible is None or obj2track is None:
            return np.empty((0, 2), dtype=np.float32)
        if obj_id not in obj2track:
            return np.empty((0, 2), dtype=np.float32)

        obj_idx = obj2track[obj_id]
        if len(obj_idx) == 0:
            return np.empty((0, 2), dtype=np.float32)

        vis_mask = np.asarray(visible, dtype=bool)[obj_idx]
        if not np.any(vis_mask):
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(track_2d[obj_idx][vis_mask], dtype=np.float32)
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        if used_crop:
            pts = pts.copy()
            pts[:, 0] -= float(x_offset)
            pts[:, 1] -= float(y_offset)

        h_used, w_used = detect_mask_used.shape
        xi = np.round(pts[:, 0]).astype(np.int32)
        yi = np.round(pts[:, 1]).astype(np.int32)
        in_bounds = (xi >= 0) & (xi < w_used) & (yi >= 0) & (yi < h_used)
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.float32)

        xi = xi[in_bounds]
        yi = yi[in_bounds]
        pts = pts[in_bounds]
        in_mask = detect_mask_used[yi, xi] > 0
        return pts[in_mask]

    def _balanced_greedy_select(
        self,
        cand_pts_xy: np.ndarray,
        cand_scores: np.ndarray,
        k: int,
        safe_area_px: int,
        existing_pts_xy: np.ndarray = None,
    ) -> np.ndarray:
        n = cand_pts_xy.shape[0]
        if n == 0 or k <= 0:
            return np.empty((0,), dtype=np.int32)
        if n <= k:
            return np.arange(n, dtype=np.int32)

        pts = cand_pts_xy.astype(np.float32)
        scores = cand_scores.astype(np.float32)
        score_norm = self._normalize_scores(scores)

        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        min_dist2 = np.full(n, np.inf, dtype=np.float32)

        if existing_pts_xy is not None and existing_pts_xy.shape[0] > 0:
            existing = existing_pts_xy.astype(np.float32)
            min_dist2 = np.minimum(min_dist2, self._nearest_dist2_to_refs(pts, existing))

        # Rough "ideal spacing" from area and point count.
        novelty_radius = self.novelty_radius_scale * np.sqrt(
            max(float(safe_area_px), 1.0) / max(float(k), 1.0)
        )
        novelty_radius = max(float(novelty_radius), 1.0)
        min_sep2 = float(max(self.min_separation_px, 0.0) ** 2)
        score_weight = float(self.score_weight)
        novelty_weight = 1.0 - score_weight

        for _ in range(k):
            novelty = np.sqrt(np.clip(min_dist2, 0.0, None))
            novelty_norm = np.clip(novelty / novelty_radius, 0.0, 1.0)
            combined = score_weight * score_norm + novelty_weight * novelty_norm

            if min_sep2 > 0 and self.separation_penalty_weight > 0:
                penalty = np.clip((min_sep2 - min_dist2) / max(min_sep2, 1e-6), 0, 1)
                combined -= self.separation_penalty_weight * penalty

            combined[selected_mask] = -np.inf
            pick = int(np.argmax(combined))
            if not np.isfinite(combined[pick]):
                break

            selected.append(pick)
            selected_mask[pick] = True

            diff = pts - pts[pick]
            d2 = np.einsum("ij,ij->i", diff, diff)
            min_dist2 = np.minimum(min_dist2, d2)
            min_dist2[selected_mask] = 0.0

        if len(selected) < k:
            remain = np.where(~selected_mask)[0]
            if remain.size > 0:
                remain = remain[np.argsort(-score_norm[remain])]
                need = k - len(selected)
                selected.extend(remain[:need].tolist())

        return np.asarray(selected[:k], dtype=np.int32)

    def _nearest_dist2_to_refs(
        self, query_xy: np.ndarray, ref_xy: np.ndarray, chunk_size: int = 1024
    ) -> np.ndarray:
        if ref_xy.shape[0] == 0:
            return np.full(query_xy.shape[0], np.inf, dtype=np.float32)

        out = np.full(query_xy.shape[0], np.inf, dtype=np.float32)
        for i in range(0, ref_xy.shape[0], chunk_size):
            refs = ref_xy[i : i + chunk_size]
            diff = query_xy[:, None, :] - refs[None, :, :]
            d2 = np.einsum("nij,nij->ni", diff, diff)
            out = np.minimum(out, d2.min(axis=1))
        return out
