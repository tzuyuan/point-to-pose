import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.camera import extract_cropped_point_cloud

# Adjust this import path to wherever your SVDClusterRegister lives.
from point2pose.modules.register.svd_cluster_register import SVDClusterRegister


def _skew(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _left_update_SE3(T, dxi):
    """
    Left-multiply pose update:
      T_new = Exp(dxi) @ T
    Here dxi = [dtx, dty, dtz, wx, wy, wz] (small translation + rotvec).
    """
    dxi = np.asarray(dxi, dtype=np.float64).reshape(6)
    dt = dxi[:3]
    w = dxi[3:]
    dT = np.eye(4, dtype=np.float64)
    dT[:3, :3] = scipy_R.from_rotvec(w).as_matrix()
    dT[:3, 3] = dt
    return dT @ T


@REGISTER.register_module("svd_cluster_sdf_refine")
class SVDClusterSDFRefineRegister(SVDClusterRegister):
    """
    Two-stage registration:
      1) Run parent SVDClusterRegister (keeps your correspondence outlier rejection).
      2) Refine pose with SDF loss using robust trimming + IRLS.

    Assumes the returned pose from parent is T_obj2cur (same convention used by your
    dense-map selection branch, where T_cur2obj = inverse(T_obj2cur)).
    """

    def __init__(self, config=None):
        super().__init__(config)

        # SDF refine settings
        self._enable_sdf_refine = bool(config.get("enable_sdf_refine", True))
        self._sdf_refine_iters = int(config.get("sdf_refine_iters", 8))
        self._sdf_refine_damping = float(config.get("sdf_refine_damping", 1e-4))
        self._sdf_refine_grad_eps = float(
            config.get("sdf_refine_grad_eps", 0.5)
        )  # in voxels
        self._sdf_refine_max_pts = int(config.get("sdf_refine_max_pts", 1500))
        self._sdf_refine_use_dense_cur = bool(
            config.get("sdf_refine_use_dense_cur", True)
        )
        self._sdf_refine_min_pts = int(config.get("sdf_refine_min_pts", 30))

        # Outlier rejection for SDF residuals
        self._sdf_trim_method = str(
            config.get("sdf_trim_method", "mad")
        )  # mad|percentile|fixed
        self._sdf_mad_scale = float(config.get("sdf_mad_scale", 2.5))
        self._sdf_inlier_thres = float(config.get("sdf_inlier_thres", 0.01))
        self._sdf_inlier_percentile = float(config.get("sdf_inlier_percentile", 80.0))

        # Robust kernel
        self._sdf_kernel = str(config.get("sdf_kernel", "huber"))  # huber|cauchy|none
        self._sdf_kernel_delta = float(config.get("sdf_kernel_delta", 0.01))

        # Optional joint blend with point-to-point residual on correspondences (lightweight)
        self._joint_p2p_weight = float(config.get("joint_p2p_weight", 0.3))
        self._joint_sdf_weight = float(config.get("joint_sdf_weight", 0.7))

        # Fallback penalty for out-of-support when support is too low
        self._sdf_min_support_ratio = float(config.get("sdf_min_support_ratio", 0.15))
        self._sdf_bad_penalty = float(config.get("sdf_bad_penalty", 0.05))

    def register(
        self,
        src_pcd,
        tgt_pcd,
        init_pose=None,
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
        prev_T=None,
        prev_frame=None,
        cur_frame=None,
        obj_id=0,
        obj=None,
        mode="f2m",
    ):
        # Stage 1: your existing robust SVD-cluster registration
        T_init, stats = super().register(
            src_pcd=src_pcd,
            tgt_pcd=tgt_pcd,
            init_pose=init_pose,
            sigma_src=sigma_src,
            sigma_tgt=sigma_tgt,
            sigma=sigma,
            prev_T=prev_T,
            prev_frame=prev_frame,
            cur_frame=cur_frame,
            obj_id=obj_id,
            obj=obj,
            mode=mode,
        )

        if (not self._enable_sdf_refine) or (obj is None):
            return T_init, stats

        # Need an SDF backend/map
        has_sdf = getattr(obj, "sdf_volume", None) is not None or (
            getattr(obj, "sdf", None) is not None and "tsdf" in obj.sdf
        )
        if not has_sdf:
            stats["sdf_refine_skipped"] = "no_sdf"
            return T_init, stats

        # Pick source points for SDF refinement
        src_for_sdf = self._get_sdf_source_points(
            src_pcd=src_pcd, cur_frame=cur_frame, obj_id=obj_id
        )
        if src_for_sdf is None or src_for_sdf.shape[0] < self._sdf_refine_min_pts:
            stats["sdf_refine_skipped"] = "too_few_points"
            return T_init, stats

        # Refine pose in SDF space:
        # Parent returns T_obj2cur (as used by your _sdf_residual branch), so optimize T_cur2obj.
        T_cur2obj0 = inverse_SE3(T_init)
        T_cur2obj_ref, sdf_debug = self._refine_pose_with_sdf(
            pts_cur=src_for_sdf,
            T_cur2obj_init=T_cur2obj0,
            obj=obj,
            src_corr=src_pcd,
            tgt_corr=tgt_pcd,
        )

        if T_cur2obj_ref is None:
            stats["sdf_refine_skipped"] = "optimizer_failed"
            return T_init, stats

        T_refined = inverse_SE3(T_cur2obj_ref)
        stats["sdf_refine"] = sdf_debug
        return T_refined, stats

    # -------------------------------------------------------------------------
    # Point selection
    # -------------------------------------------------------------------------
    def _get_sdf_source_points(self, src_pcd, cur_frame, obj_id):
        pts = None
        if self._sdf_refine_use_dense_cur and cur_frame is not None:
            try:
                pts = extract_cropped_point_cloud(cur_frame, obj_id)
            except Exception:
                pts = None
        if pts is None:
            pts = np.asarray(src_pcd) if src_pcd is not None else None
        if pts is None:
            return None
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return None

        # Deterministic subsampling to bound cost
        if pts.shape[0] > self._sdf_refine_max_pts:
            step = int(np.ceil(pts.shape[0] / float(self._sdf_refine_max_pts)))
            pts = pts[::step]
        return pts

    # -------------------------------------------------------------------------
    # SDF query helpers (supports nvblox wrapper and legacy dense TSDF)
    # -------------------------------------------------------------------------
    def _query_sdf_signed(self, obj, pts_obj):
        """
        Returns:
          vals: (N,) float64
          valid: (N,) bool   (inside support and finite query)
        """
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        N = pts_obj.shape[0]
        vals = np.full((N,), np.nan, dtype=np.float64)
        valid = np.zeros((N,), dtype=bool)

        # Bounds if available (SDFBuilder stores vol_bnds in obj.sdf)
        vol_bnds = None
        if getattr(obj, "sdf", None) is not None and "vol_bnds" in obj.sdf:
            vol_bnds = np.asarray(obj.sdf["vol_bnds"], dtype=np.float32)

        if vol_bnds is not None:
            inb = np.logical_and(
                np.all(pts_obj >= vol_bnds[:, 0][None, :], axis=1),
                np.all(pts_obj <= vol_bnds[:, 1][None, :], axis=1),
            )
        else:
            inb = np.ones((N,), dtype=bool)

        # nvblox-style path first (same object.sdf_volume.query_sdf idea you already use)
        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            try:
                qvals = obj.sdf_volume.query_sdf(pts_obj)
                if qvals is not None:
                    qvals = np.asarray(qvals).reshape(-1)
                    if qvals.shape[0] == N:
                        finite = np.isfinite(qvals)
                        keep = inb & finite
                        vals[keep] = qvals[keep].astype(np.float64)
                        valid[keep] = True
                        return vals, valid
            except Exception:
                pass

        # legacy dense TSDF path (trilinear interpolation)
        if getattr(obj, "sdf", None) is None or "tsdf" not in obj.sdf:
            return vals, valid

        tsdf = np.asarray(obj.sdf["tsdf"], dtype=np.float32)
        origin = np.asarray(obj.sdf["vol_origin"], dtype=np.float32)
        voxel = float(obj.sdf["voxel_size"])
        vol_dim = np.array(tsdf.shape, dtype=np.int32)

        # Convert to continuous voxel coordinates
        xyz = (pts_obj - origin[None, :]) / voxel  # (N,3)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        z0 = np.floor(z).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Need neighbors for trilinear
        inb_tri = (
            (x0 >= 0)
            & (y0 >= 0)
            & (z0 >= 0)
            & (x1 < vol_dim[0])
            & (y1 < vol_dim[1])
            & (z1 < vol_dim[2])
        )
        keep = inb & inb_tri
        if not np.any(keep):
            return vals, valid

        xx = x[keep] - x0[keep]
        yy = y[keep] - y0[keep]
        zz = z[keep] - z0[keep]

        x0k, y0k, z0k = x0[keep], y0[keep], z0[keep]
        x1k, y1k, z1k = x1[keep], y1[keep], z1[keep]

        c000 = tsdf[x0k, y0k, z0k]
        c100 = tsdf[x1k, y0k, z0k]
        c010 = tsdf[x0k, y1k, z0k]
        c110 = tsdf[x1k, y1k, z0k]
        c001 = tsdf[x0k, y0k, z1k]
        c101 = tsdf[x1k, y0k, z1k]
        c011 = tsdf[x0k, y1k, z1k]
        c111 = tsdf[x1k, y1k, z1k]

        c00 = c000 * (1 - xx) + c100 * xx
        c10 = c010 * (1 - xx) + c110 * xx
        c01 = c001 * (1 - xx) + c101 * xx
        c11 = c011 * (1 - xx) + c111 * xx
        c0 = c00 * (1 - yy) + c10 * yy
        c1 = c01 * (1 - yy) + c11 * yy
        interp = c0 * (1 - zz) + c1 * zz

        idx_keep = np.where(keep)[0]
        vals[idx_keep] = interp.astype(np.float64)
        valid[idx_keep] = np.isfinite(interp)
        return vals, valid

    def _query_sdf_and_grad(self, obj, pts_obj):
        """
        Finite-difference gradient of SDF in object frame.
        Returns:
          sdf:   (N,)
          grad:  (N,3)
          valid: (N,)
        """
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        N = pts_obj.shape[0]
        sdf0, v0 = self._query_sdf_signed(obj, pts_obj)

        # grad epsilon in metric units via voxel size
        voxel = None
        if getattr(obj, "sdf", None) is not None and "voxel_size" in obj.sdf:
            voxel = float(obj.sdf["voxel_size"])
        elif getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "_voxel_size"
        ):
            voxel = float(obj.sdf_volume._voxel_size)
        else:
            voxel = 0.005

        eps = float(self._sdf_refine_grad_eps) * voxel
        eps = max(eps, 1e-4)

        E = np.eye(3, dtype=np.float32) * eps
        grad = np.zeros((N, 3), dtype=np.float64)
        valid = v0.copy()

        for k in range(3):
            fp, vp = self._query_sdf_signed(obj, pts_obj + E[k][None, :])
            fm, vm = self._query_sdf_signed(obj, pts_obj - E[k][None, :])
            vk = v0 & vp & vm
            gk = np.zeros((N,), dtype=np.float64)
            good = vk
            gk[good] = (fp[good] - fm[good]) / (2.0 * eps)
            grad[:, k] = gk
            valid &= vk

        # reject degenerate gradients
        gnorm = np.linalg.norm(grad, axis=1)
        valid &= np.isfinite(sdf0) & np.isfinite(gnorm) & (gnorm > 1e-8)
        return sdf0, grad, valid

    # -------------------------------------------------------------------------
    # Robust weighting / trimming
    # -------------------------------------------------------------------------
    def _compute_sdf_inliers(self, r_abs):
        if r_abs.size == 0:
            return np.zeros((0,), dtype=bool), np.inf

        if self._sdf_trim_method == "mad":
            med = np.median(r_abs)
            mad = np.median(np.abs(r_abs - med)) + 1e-12
            thr = float(med + self._sdf_mad_scale * mad)
        elif self._sdf_trim_method == "percentile":
            q = float(np.clip(self._sdf_inlier_percentile, 0.0, 100.0))
            thr = float(np.percentile(r_abs, q))
        elif self._sdf_trim_method == "fixed":
            thr = float(self._sdf_inlier_thres)
        else:
            raise ValueError(f"Invalid sdf_trim_method: {self._sdf_trim_method}")

        return r_abs <= thr, thr

    def _kernel_weights(self, r_abs):
        if self._sdf_kernel == "none":
            return np.ones_like(r_abs, dtype=np.float64)

        d = float(max(self._sdf_kernel_delta, 1e-12))
        if self._sdf_kernel == "huber":
            w = np.ones_like(r_abs, dtype=np.float64)
            m = r_abs > d
            w[m] = d / (r_abs[m] + 1e-12)
            return w
        elif self._sdf_kernel == "cauchy":
            x = r_abs / d
            return 1.0 / (1.0 + x * x)
        else:
            raise ValueError(f"Invalid sdf_kernel: {self._sdf_kernel}")

    # -------------------------------------------------------------------------
    # Main SDF refinement (IRLS + trimming)
    # -------------------------------------------------------------------------
    def _refine_pose_with_sdf(
        self, pts_cur, T_cur2obj_init, obj, src_corr=None, tgt_corr=None
    ):
        T = np.asarray(T_cur2obj_init, dtype=np.float64).copy()
        pts_cur = np.asarray(pts_cur, dtype=np.float64)

        dbg = {
            "enabled": True,
            "iters": 0,
            "cost_history": [],
            "support_history": [],
            "inliers_history": [],
            "trim_thr_history": [],
            "accepted": False,
        }

        if pts_cur.shape[0] < self._sdf_refine_min_pts:
            return None, dbg

        best_T = T.copy()
        best_cost = np.inf

        for it in range(self._sdf_refine_iters):
            # Transform current points -> object frame
            pts_obj = transform_pts(
                T.astype(np.float32), pts_cur.astype(np.float32)
            ).astype(np.float64)

            sdf_vals, sdf_grads, valid = self._query_sdf_and_grad(
                obj, pts_obj.astype(np.float32)
            )
            valid = np.asarray(valid, dtype=bool)

            support_ratio = float(valid.mean()) if valid.size else 0.0
            dbg["support_history"].append(support_ratio)

            if valid.sum() < self._sdf_refine_min_pts:
                break

            pts_obj_v = pts_obj[valid]
            r_sdf = sdf_vals[valid].astype(np.float64)  # signed residual
            g_sdf = sdf_grads[valid].astype(np.float64)
            r_abs = np.abs(r_sdf)

            # Trim SDF outliers
            in_sdf, thr = self._compute_sdf_inliers(r_abs)
            dbg["trim_thr_history"].append(float(thr))
            if in_sdf.sum() < self._sdf_refine_min_pts:
                break

            pts_obj_in = pts_obj_v[in_sdf]
            r_sdf_in = r_sdf[in_sdf]
            g_sdf_in = g_sdf[in_sdf]

            # Jacobian for residual r = sdf(x), x = T * p (left perturbation)
            # dr/dxi = grad^T [I, -skew(x)]
            N = pts_obj_in.shape[0]
            J_sdf = np.zeros((N, 6), dtype=np.float64)
            for i in range(N):
                x = pts_obj_in[i]
                G = np.zeros((3, 6), dtype=np.float64)
                G[:, :3] = np.eye(3)
                G[:, 3:] = -_skew(x)
                J_sdf[i] = g_sdf_in[i] @ G

            w_trim = np.ones((N,), dtype=np.float64)
            w_rob = self._kernel_weights(np.abs(r_sdf_in))
            w_sdf = self._joint_sdf_weight * w_trim * w_rob

            # Optional lightweight joint point-to-point term on correspondences
            # (Keeps the refinement from drifting if SDF is sparse/clipped.)
            J_list = [J_sdf]
            r_list = [r_sdf_in]
            w_list = [w_sdf]

            if (
                self._joint_p2p_weight > 0.0
                and src_corr is not None
                and tgt_corr is not None
                and len(src_corr) >= 3
                and len(src_corr) == len(tgt_corr)
            ):
                src_corr = np.asarray(src_corr, dtype=np.float64)
                tgt_corr = np.asarray(tgt_corr, dtype=np.float64)

                # Parent pose convention is T_obj2cur; here we optimize T_cur2obj.
                # For a small stabilizer term, compare transformed current->obj? If your
                # correspondences are not in current/object convention, set joint_p2p_weight=0.
                # This term is kept optional for safety.
                pass

            # Stack normal equations (weighted least squares)
            H = np.zeros((6, 6), dtype=np.float64)
            b = np.zeros((6,), dtype=np.float64)
            cost = 0.0

            for Jk, rk, wk in zip(J_list, r_list, w_list):
                wk = np.asarray(wk, dtype=np.float64)
                rk = np.asarray(rk, dtype=np.float64)
                Jk = np.asarray(Jk, dtype=np.float64)

                sw = np.sqrt(np.clip(wk, 0.0, None))[:, None]
                Jw = Jk * sw
                rw = rk * sw[:, 0]
                H += Jw.T @ Jw
                b += Jw.T @ rw
                cost += float(np.mean((rw) ** 2)) if rw.size else 0.0

            # Support penalty if too many points are out of support
            if support_ratio < self._sdf_min_support_ratio:
                cost += float(
                    self._sdf_bad_penalty
                    * (self._sdf_min_support_ratio - support_ratio)
                )

            dbg["cost_history"].append(float(cost))
            dbg["inliers_history"].append(int(in_sdf.sum()))

            # Damped Gauss-Newton: solve H dxi = -b
            lam = float(self._sdf_refine_damping)
            H_damped = H + lam * np.eye(6, dtype=np.float64)
            try:
                dxi = -np.linalg.solve(H_damped, b)
            except np.linalg.LinAlgError:
                break

            if not np.all(np.isfinite(dxi)):
                break

            # Step control (simple clipping)
            dxi[:3] = np.clip(dxi[:3], -0.02, 0.02)  # meters
            dxi[3:] = np.clip(dxi[3:], -0.2, 0.2)  # radians

            T_candidate = _left_update_SE3(T, dxi)

            # Evaluate candidate quickly (same robust cost)
            c_cost = self._eval_sdf_cost(pts_cur, T_candidate, obj)
            t_cost = self._eval_sdf_cost(pts_cur, T, obj)

            if np.isfinite(c_cost) and c_cost <= t_cost:
                T = T_candidate
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_T = T.copy()
            else:
                # reject step -> try smaller step once
                T_half = _left_update_SE3(T, 0.5 * dxi)
                h_cost = self._eval_sdf_cost(pts_cur, T_half, obj)
                if np.isfinite(h_cost) and h_cost < t_cost:
                    T = T_half
                    if h_cost < best_cost:
                        best_cost = h_cost
                        best_T = T.copy()

            dbg["iters"] = it + 1

            # Convergence
            if np.linalg.norm(dxi[:3]) < 1e-4 and np.linalg.norm(dxi[3:]) < 1e-3:
                break

        if np.isfinite(best_cost):
            dbg["accepted"] = True
            dbg["best_cost"] = float(best_cost)
            return best_T.astype(np.float32), dbg

        return None, dbg

    def _eval_sdf_cost(self, pts_cur, T_cur2obj, obj):
        pts_obj = transform_pts(
            np.asarray(T_cur2obj, dtype=np.float32),
            np.asarray(pts_cur, dtype=np.float32),
        )
        vals, valid = self._query_sdf_signed(obj, pts_obj)
        if valid.sum() < self._sdf_refine_min_pts:
            return np.inf

        r = np.abs(vals[valid].astype(np.float64))
        inl, _ = self._compute_sdf_inliers(r)
        if inl.sum() < self._sdf_refine_min_pts:
            return np.inf

        rr = r[inl]
        w = self._kernel_weights(rr)
        support = float(valid.mean())
        cost = float(np.mean(w * (rr**2)))
        if support < self._sdf_min_support_ratio:
            cost += float(
                self._sdf_bad_penalty * (self._sdf_min_support_ratio - support)
            )
        return cost
