"""
Lean front-end for model-based tracking.

Unlike the full ``FrontEnd`` (which does online map growth, SDF-based dense recovery, and
supports f2f/f2m modes), this front-end does the minimal frame-to-map loop against a
**pre-built model map**:

    (optional) segment  ->  TAPIR track_once  ->  select valid map points  ->  PnP f2m

The object's ``key_points`` (3D, object frame) are the fixed model map, index-aligned with the
TAPIR query points that were pre-seeded from rendered views. Every ``track_once`` call returns
one 2D track per map point in the live frame; we feed the (map-3D, live-2D) pairs of the
currently-visible, low-uncertainty points into RANSAC PnP to recover ``T_obj2cam``.

No SDF, no graph optimization, no keyframe growth.
"""

from time import perf_counter

import numpy as np

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import REGISTER, SEGMENTER
from point2pose.data_types.front_end_result import FrontEndResult
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.camera import convert_pixel_to_world


class ModelTrackingFrontEnd:
    def __init__(self, cfg, tracker):
        """
        Args:
            cfg: full config. Uses ``cfg.pipeline.params`` for flags, ``cfg.register`` for the
                 PnP solver, and ``cfg.segmenter`` when segmentation is enabled.
            tracker: a TAPIR tracker already seeded with the model-map query points (built by
                     ``MapBuilder``). This front-end does not create the tracker.
        """
        self.cfg = cfg
        p = cfg.pipeline.params

        self.tracker = tracker
        self.use_segmenter = bool(p.get("use_segmenter", False))
        self.segmenter = (
            build_from_cfg(cfg.segmenter, SEGMENTER) if self.use_segmenter else None
        )

        # Pose solver. Two correspondence modes depending on the register:
        #   * PnP-style (consumes img_pts): 2D-3D  -> (map 3D, live 2D pixels).
        #   * everything else (SVD/ICP):    3D-3D  -> (map 3D, live 3D from depth backprojection).
        # We pick the mode from the register type so both work with the same front-end.
        reg_type = str(cfg.register.get("type", ""))
        self.is_pnp = "pnp" in reg_type.lower()
        print(f"[ModelTrackingFE] register='{reg_type}' -> "
              f"{'2D-3D PnP' if self.is_pnp else '3D-3D (live depth backprojection)'} mode")
        self.register = build_from_cfg(cfg.register, REGISTER)

        self.num_obj = int(p.get("max_num_obj", 1))
        self.min_depth = float(p.get("min_depth", 0.05))
        self.max_depth = float(p.get("max_depth", 2.0))
        self.reg_uncer_thres = float(p.get("reg_uncertainty_thres", 0.6))
        self.min_corr = int(p.get("min_correspondences", 6))

        # If set (and a mask exists), black out everything outside the SAM mask BEFORE TAPIR, so
        # the tracker can't drift onto background texture. mask_bg_dilate_px keeps a few pixels
        # of margin around the object so boundary features survive.
        self.mask_background = bool(p.get("mask_background", False))
        self.mask_bg_dilate_px = int(p.get("mask_bg_dilate_px", 4))

        # Pose jump guard (hold last good pose on implausible jumps).
        self.jump_guard_enable = bool(p.get("pose_jump_guard_enable", True))
        self.jump_trans_thres = float(p.get("pose_jump_guard_trans_thres", 0.15))
        self.jump_rot_deg_thres = float(p.get("pose_jump_guard_rot_deg_thres", 30.0))

        self.prev_frame = None
        # Per-object flag: has an absolute pose been locked on yet?
        self._pose_initialized = {}
        # SAM2 must be initialized (load_first_frame + prompts) before segment()/track() works.
        self._segmenter_ready = False

    # ------------------------------------------------------------------ #
    def add_user_points(self, points, labels):
        """Forward click points to the segmenter (only meaningful when use_segmenter)."""
        if self.segmenter is not None:
            self.segmenter.add_input_points(points, labels)

    def initialize_segmenter(self, frame):
        """Init SAM2 from the collected click prompts on this frame; enables per-frame masks."""
        if self.segmenter is not None:
            self.segmenter.initialize(frame.rgb)
            self._segmenter_ready = bool(getattr(self.segmenter, "tracking_started", True))
            print(f"[ModelTrackingFE] segmenter ready: {self._segmenter_ready}")
        return self._segmenter_ready

    @property
    def segmenter_ready(self):
        return self._segmenter_ready

    # ------------------------------------------------------------------ #
    def step(self, frame, track_table, objects):
        """
        One model-tracking step. Returns a FrontEndResult with per-object pose in ``obj_poses``.
        Sets ``obj.pose`` (== T_obj2cam) and ``obj.lost``.
        """
        result = FrontEndResult(frame_id=frame.id)

        # 1) (optional) segmentation -> frame.mask, used only to filter tracked points.
        #    Only run once SAM2 has been initialized from user clicks; before that, mask stays
        #    None and every tracked point is considered (PnP RANSAC handles outliers).
        t0 = perf_counter()
        if self.use_segmenter and self.segmenter is not None and self._segmenter_ready:
            _, mask_logits = self.segmenter.segment(frame.rgb)
            frame.mask = mask_logits
            # Optionally black out the background so TAPIR can't lock onto it.
            if self.mask_background:
                self._black_out_background(frame)
        else:
            frame.mask = None
        t_sam = perf_counter()

        # 2) TAPIR tracks all pre-seeded map points into this frame.
        tracks, uncertainties, visibles = self.tracker.track_once(frame)
        t_tapir = perf_counter()

        print(
            f"[FE timing] sam2={1000 * (t_sam - t0):.1f}ms "
            f"tapir={1000 * (t_tapir - t_sam):.1f}ms"
        )

        result.tracks = tracks
        result.uncertainties = uncertainties
        result.visibles = visibles

        for obj_id in range(min(self.num_obj, len(objects))):
            obj = objects[obj_id]
            if obj_id not in track_table.obj2track_map:
                continue

            g_idx = np.asarray(track_table.obj2track_map[obj_id], dtype=np.int64)
            # Map each global track index -> row in obj.key_points.
            obj_rows = obj.track_idx_2_obj_idx[g_idx]

            # 3) Select valid correspondences: visible, low uncertainty, (in-mask if segmenting),
            #    landing on a pixel with a real depth reading (depth==0 == no LiDAR/stereo
            #    return, e.g. reflective/out-of-range surface -- can't backproject those, and a
            #    stale/garbage depth there would silently corrupt the 3D-3D register).
            m_row = obj_rows >= 0
            m_vis = np.asarray(visibles, dtype=bool)[g_idx]
            m_unc = np.asarray(uncertainties, dtype=float)[g_idx] < self.reg_uncer_thres
            keep = m_row & m_vis & m_unc
            if self.use_segmenter and frame.mask is not None:
                m_mask = self._inside_mask(tracks[g_idx], frame.mask, obj_id)
                keep &= m_mask
            else:
                m_mask = np.ones_like(keep)

            m_depth = self._depth_ok(tracks[g_idx], frame.depth)
            keep &= m_depth

            # DEBUG: per-filter surviving counts (comment out when not needed)
            print(
                f"[FE obj{obj_id}] total={g_idx.size} "
                f"| row_ok={int(m_row.sum())} "
                f"| +visible={int((m_row & m_vis).sum())} "
                f"| +uncer<{self.reg_uncer_thres}={int((m_row & m_vis & m_unc).sum())} "
                f"| +in_mask={int((m_row & m_vis & m_unc & m_mask).sum())} "
                f"| +depth_ok={int(keep.sum())} -> PnP"
            )

            sel_global = g_idx[keep]
            sel_rows = obj_rows[keep]
            img_pts = tracks[sel_global].astype(np.float64)  # (M,2) (x,y)
            map_pts = obj.key_points[sel_rows].astype(np.float64)  # (M,3) object frame

            # For a 3D-3D register, backproject the live 2D tracks with the frame's depth to get
            # live 3D (camera frame). Drop pairs with no valid depth so both point sets align.
            live_3d = None
            if not self.is_pnp and img_pts.shape[0] > 0:
                p3d, p_valid = convert_pixel_to_world(
                    pixel=img_pts, depth_image=frame.depth,
                    cam_intrinsics=frame.intrinsics, depth_factor=frame.depth_factor,
                    min_depth=self.min_depth, max_depth=self.max_depth,
                )
                p_valid = np.asarray(p_valid, dtype=bool).reshape(-1)
                img_pts = img_pts[p_valid]
                map_pts = map_pts[p_valid]
                sel_global = sel_global[p_valid]
                sel_rows = sel_rows[p_valid]
                live_3d = p3d[p_valid]

            prev_pose = obj.pose.copy()
            initialized = self._pose_initialized.get(obj_id, False)
            # Before the pose is initialized there is no meaningful prior for cluster selection.
            init_pose = prev_pose if initialized else None

            if map_pts.shape[0] >= self.min_corr:
                if self.is_pnp:
                    # 2D-3D PnP: img_pts (live 2D) <-> map_pts (object 3D). map_pts is passed as
                    # both src/tgt (RANSAC uses src, residuals use tgt). prev_frame=None avoids
                    # the register's dense projection-consistency path (needs frame masks).
                    T_obj2cam, stats_reg = self.register.register(
                        src_pcd=map_pts, tgt_pcd=map_pts,
                        init_pose=init_pose, prev_T=init_pose, prev_frame=None,
                        cur_frame=frame, obj_id=obj_id, mode="f2m", img_pts=img_pts,
                    )
                else:
                    # 3D-3D: align map 3D (object frame) to live 3D (camera frame). The SVD/ICP
                    # register returns the src->tgt transform == T_obj2cam.
                    T_obj2cam, stats_reg = self.register.register(
                        src_pcd=map_pts, tgt_pcd=live_3d,
                        init_pose=init_pose, prev_T=init_pose, prev_frame=None,
                        cur_frame=frame, obj_id=obj_id, mode="f2m",
                    )
                n_inl = int(np.sum(stats_reg.get("inliers", np.array([]))))
                # DEBUG: inliers vs correspondences fed in (comment out when not needed)
                _re = stats_reg.get("reproj_errors", stats_reg.get("residuals", np.array([])))
                _inl_mask = np.asarray(stats_reg.get("inliers", []), dtype=bool)
                _mre = float(np.mean(_re[_inl_mask])) if _inl_mask.any() and _re.size else -1.0
                _tag = "PnP" if self.is_pnp else "SVD"
                print(f"[FE obj{obj_id}] {_tag}: {n_inl}/{map_pts.shape[0]} inliers "
                      f"err={_mre:.4f}")

                # Store the inlier correspondences for debugging / viser overlay:
                #   valid_key_points = inlier 3D in object frame,
                #   valid_curr_3d    = inlier live 2D pixels,
                #   valid_stats.inlier_rows = object-keypoint ROW of each inlier, so a debugger
                #     can trace it back to (seed view, keypoint-in-view).
                if _inl_mask.size == map_pts.shape[0]:
                    result.valid_key_points[obj_id] = map_pts[_inl_mask]
                    result.valid_curr_3d[obj_id] = img_pts[_inl_mask]
                    result.valid_stats[obj_id] = {"inlier_rows": sel_rows[_inl_mask]}

                if T_obj2cam is None or n_inl < self.min_corr:
                    # PnP failed to find a consensus.
                    T_obj2cam = prev_pose
                    obj.lost = True
                elif not initialized:
                    # First lock-on: accept unconditionally (no prior to guard against).
                    self._pose_initialized[obj_id] = True
                    obj.lost = False
                else:
                    T_obj2cam, jump_rejected = self._pose_jump_guard(T_obj2cam, prev_pose)
                    obj.lost = jump_rejected
                result.reg_stats[obj_id] = stats_reg
            else:
                print(
                    f"[ModelTrackingFE] Frame {frame.id} obj {obj_id}: "
                    f"only {map_pts.shape[0]} correspondences (<{self.min_corr}); holding pose."
                )
                T_obj2cam = prev_pose
                obj.lost = True
                result.reg_stats[obj_id] = {"inliers": np.array([]), "residuals": np.array([])}

            obj.pose = T_obj2cam
            result.obj_poses[obj_id] = T_obj2cam
            result.valid_indices[obj_id] = sel_global

        self.prev_frame = frame
        return result

    # ------------------------------------------------------------------ #
    def _black_out_background(self, frame):
        """Zero out frame.rgb outside the (dilated) union of all object masks."""
        import cv2

        m = frame.mask  # [N,1,H,W]
        if m is None:
            return
        union = None
        for i in range(len(m)):
            mi = m[i, 0]
            if hasattr(mi, "detach"):
                mi = mi.detach().cpu().numpy()
            mi = np.asarray(mi) > 0
            union = mi if union is None else (union | mi)
        if union is None:
            return
        if self.mask_bg_dilate_px > 0:
            k = 2 * self.mask_bg_dilate_px + 1
            union = cv2.dilate(union.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        frame.rgb = np.where(union[..., None], frame.rgb, 0).astype(frame.rgb.dtype)

    def _depth_ok(self, pts_xy, depth):
        """Return a bool array: which (x,y) pixels land on a nonzero (real) depth reading.
        depth==0 means no sensor return (e.g. reflective surface, out of range) -- there's no
        real distance to backproject there, so those correspondences are dropped regardless of
        register mode (PnP doesn't need depth, but a bad TAPIR track there usually means the
        point drifted off the object edge anyway)."""
        if depth is None:
            return np.ones(len(pts_xy), dtype=bool)
        H, W = depth.shape[:2]
        x = np.clip(np.round(pts_xy[:, 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(pts_xy[:, 1]).astype(int), 0, H - 1)
        return np.asarray(depth)[y, x] > 0

    def _inside_mask(self, pts_xy, mask, obj_id):
        """Return a bool array: which (x,y) pixels fall inside the object's mask."""
        m = mask[obj_id, 0]
        # mask may be a torch tensor on GPU; move to numpy bool.
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        m = np.asarray(m) > 0
        H, W = m.shape
        x = np.clip(np.round(pts_xy[:, 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(pts_xy[:, 1]).astype(int), 0, H - 1)
        return m[y, x]

    def _pose_jump_guard(self, T_candidate, T_prev):
        """Reject implausibly large pose jumps relative to the previous pose."""
        if T_candidate is None:
            return T_prev, True
        if not self.jump_guard_enable:
            return T_candidate, False
        T_rel = inverse_SE3(T_prev) @ T_candidate
        trans = float(np.linalg.norm(T_rel[:3, 3]))
        cos = np.clip(0.5 * (np.trace(T_rel[:3, :3]) - 1.0), -1.0, 1.0)
        rot_deg = float(np.rad2deg(np.arccos(cos)))
        if trans > self.jump_trans_thres or rot_deg > self.jump_rot_deg_thres:
            print(
                f"[ModelTrackingFE] pose jump rejected: trans={trans:.3f}m "
                f"rot={rot_deg:.1f}deg"
            )
            return T_prev, True
        return T_candidate, False
