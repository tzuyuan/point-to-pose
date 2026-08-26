import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.camera import extract_cropped_point_cloud
from point2pose.utils.transform import inverse_SE3, transform_pts


def _skew(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _left_update_SE3(T, dxi):
    dxi = np.asarray(dxi, dtype=np.float64).reshape(6)
    dt = dxi[:3]
    w = dxi[3:]
    dT = np.eye(4, dtype=np.float64)
    dT[:3, :3] = scipy_R.from_rotvec(w).as_matrix()
    dT[:3, 3] = dt
    return dT @ T


@REGISTER.register_module("svd_residual_outlier")
class SVDResidualOutlierRegister(Register):
    """ """

    def __init__(self, config=None):
        super().__init__(config)
        self._max_iter = config.get("max_iter", 5)
        self._threshold_method = config.get("threshold_method", "mad")
        self._inlier_thres = config.get("inlier_thres", 0.05)
        self._thres_reduce_factor = config.get("thres_reduce_factor", 0.01)
        self._mad_scale = config.get("mad_scale", 2.5)
        self._min_inliers = config.get("min_inliers", 3)

        # Optional post-registration SDF pose refinement.
        self._enable_sdf_refine = bool(config.get("enable_sdf_refine", False))
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
        self._sdf_trim_method = str(config.get("sdf_trim_method", "mad"))
        self._sdf_mad_scale = float(config.get("sdf_mad_scale", 2.5))
        self._sdf_inlier_thres = float(config.get("sdf_inlier_thres", 0.01))
        self._sdf_inlier_percentile = float(config.get("sdf_inlier_percentile", 80.0))
        self._sdf_kernel = str(config.get("sdf_kernel", "huber"))  # huber|cauchy|none
        self._sdf_kernel_delta = float(config.get("sdf_kernel_delta", 0.01))
        self._sdf_min_support_ratio = float(config.get("sdf_min_support_ratio", 0.15))
        self._sdf_bad_penalty = float(config.get("sdf_bad_penalty", 0.05))
        self._sdf_refine_step_trans_clip = float(
            config.get("sdf_refine_step_trans_clip", 0.02)
        )
        self._sdf_refine_step_rot_clip = float(
            config.get("sdf_refine_step_rot_clip", 0.2)
        )
        self._sdf_refine_min_depth = float(config.get("sdf_refine_min_depth", 0.08))
        self._sdf_refine_max_depth = float(config.get("sdf_refine_max_depth", 2.0))
        self._sdf_refine_fill_missing_depth = bool(
            config.get("sdf_refine_fill_missing_depth", True)
        )
        self._sdf_refine_window_size = int(config.get("sdf_refine_window_size", 5))
        self._sdf_refine_min_neighbors = int(config.get("sdf_refine_min_neighbors", 1))

    def register(self, src_pcd, tgt_pcd, init_pose=None, **kwargs):
        """
        Perform a single registration step using SVD.
        Args:
            src_pcd, tgt_pcd: (N,3) source/target with known correspondence (row-wise).
            init_pose: (4,4) initial pose.
        """
        obj = kwargs.get("obj", None)
        cur_frame = kwargs.get("cur_frame", None)
        obj_id = int(kwargs.get("obj_id", 0))

        stats = {}
        # number of points
        N = src_pcd.shape[0]

        # transform the points if the initial pose is given
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        # fit transformation using svd
        T = self._svd_fit(p0, tgt_pcd)

        inliers = np.ones(N, dtype=bool)

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)

            # choose threshold
            if self._threshold_method == "mad":
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-12
                thr = med + self._mad_scale * mad  # 1.4826 makes MAD ~ std for Gaussian
            elif self._threshold_method == "fixed":
                thr = float(self._inlier_thres)
            elif self._threshold_method == "reduce":
                thr = float(self._inlier_thres) - float(self._thres_reduce_factor * it)
                if thr < 0:
                    thr = 0.001
            else:
                raise ValueError(f"Invalid threshold method: {self._threshold_method}")

            new_inliers = residuals <= thr

            if self.debug_level > 1:
                print(f"[Register] iter {it} inliers: {new_inliers.sum()}")
                print(f"[Register] iter {it} thr: {thr}")
                print(f"[Register] iter {it} residuals: {residuals}")
                print(f"[Register] iter {it} res_median: {np.median(residuals)}")
                print(f"[Register] iter {it} res_mean: {np.mean(residuals)}")
                print(f"[Register] iter {it} res_max: {np.max(residuals)}")
                print(f"[Register] iter {it} num_inliers: {new_inliers.sum()}")

            # stop if no change or too few inliers
            if (
                np.array_equal(new_inliers, inliers)
                or new_inliers.sum() < self._min_inliers
                or it == self._max_iter - 1
            ):
                stats["iter"] = it
                stats["thr"] = thr
                stats["residuals"] = residuals
                stats["inliers"] = inliers

                print(f"[Register] iter {it} res_mean: {np.mean(residuals)}")

                break

            inliers = new_inliers

            # refit on inliers
            T = self._svd_fit(p0[inliers], tgt_pcd[inliers])

        # recover init pose if used
        if init_pose is not None:
            T = T @ init_pose

        # Recompute final residuals/inliers in the original correspondence frame.
        residuals_final = np.linalg.norm(transform_pts(T, src_pcd) - tgt_pcd, axis=1)
        thr_final = float(stats.get("thr", self._inlier_thres))
        stats["residuals"] = residuals_final
        stats["inliers"] = residuals_final <= thr_final

        sdf_refine_info = {"enabled": bool(self._enable_sdf_refine), "applied": False}
        if self._enable_sdf_refine:
            T, sdf_refine_info = self._maybe_refine_with_sdf(
                T_seed=T,
                tgt_corr=tgt_pcd,
                cur_frame=cur_frame,
                obj_id=obj_id,
                obj=obj,
            )
            residuals_refined = np.linalg.norm(transform_pts(T, src_pcd) - tgt_pcd, axis=1)
            stats["residuals"] = residuals_refined
            stats["inliers"] = residuals_refined <= thr_final
        stats["sdf_refine"] = sdf_refine_info

        return T, stats

    def _maybe_refine_with_sdf(self, T_seed, tgt_corr, cur_frame, obj_id, obj):
        info = {"enabled": True, "applied": False, "skip_reason": ""}
        if obj is None:
            info["skip_reason"] = "no_object"
            return np.asarray(T_seed, dtype=np.float64), info

        has_sdf = getattr(obj, "sdf_volume", None) is not None or (
            getattr(obj, "sdf", None) is not None and "tsdf" in obj.sdf
        )
        if not has_sdf:
            info["skip_reason"] = "no_sdf"
            return np.asarray(T_seed, dtype=np.float64), info

        pts_cur = self._get_sdf_source_points(
            tgt_corr=tgt_corr, cur_frame=cur_frame, obj_id=obj_id
        )
        if pts_cur is None or pts_cur.shape[0] < self._sdf_refine_min_pts:
            info["skip_reason"] = "too_few_points"
            return np.asarray(T_seed, dtype=np.float64), info

        T_cur2obj_seed = inverse_SE3(np.asarray(T_seed, dtype=np.float64))
        seed_cost = self._eval_sdf_cost(pts_cur, T_cur2obj_seed, obj)

        T_cur2obj_ref, dbg = self._refine_pose_with_sdf(
            pts_cur=pts_cur,
            T_cur2obj_init=T_cur2obj_seed,
            obj=obj,
        )
        info["debug"] = dbg

        if T_cur2obj_ref is None:
            info["skip_reason"] = "optimizer_failed"
            return np.asarray(T_seed, dtype=np.float64), info

        ref_cost = self._eval_sdf_cost(pts_cur, T_cur2obj_ref, obj)
        info["seed_cost"] = float(seed_cost) if np.isfinite(seed_cost) else np.inf
        info["refined_cost"] = float(ref_cost) if np.isfinite(ref_cost) else np.inf

        if np.isfinite(ref_cost) and (
            (not np.isfinite(seed_cost)) or ref_cost <= seed_cost
        ):
            info["applied"] = True
            return inverse_SE3(T_cur2obj_ref), info

        info["skip_reason"] = "not_improved"
        return np.asarray(T_seed, dtype=np.float64), info

    def _get_sdf_source_points(self, tgt_corr, cur_frame, obj_id):
        pts = None
        if self._sdf_refine_use_dense_cur and cur_frame is not None:
            try:
                pts = extract_cropped_point_cloud(
                    cur_frame,
                    obj_id,
                    min_depth=self._sdf_refine_min_depth,
                    max_depth=self._sdf_refine_max_depth,
                    fill_missing_depth=self._sdf_refine_fill_missing_depth,
                    window_size=self._sdf_refine_window_size,
                    min_neighbors=self._sdf_refine_min_neighbors,
                )
            except Exception:
                pts = None
        if pts is None and tgt_corr is not None:
            pts = np.asarray(tgt_corr)
        if pts is None:
            return None

        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return None
        if pts.shape[0] > self._sdf_refine_max_pts:
            step = int(np.ceil(pts.shape[0] / float(self._sdf_refine_max_pts)))
            pts = pts[::step]
        return pts

    def _query_sdf_signed(self, obj, pts_obj):
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        n = pts_obj.shape[0]
        vals = np.full((n,), np.nan, dtype=np.float64)
        valid = np.zeros((n,), dtype=bool)

        vol_bnds = None
        if getattr(obj, "sdf", None) is not None and "vol_bnds" in obj.sdf:
            vol_bnds = np.asarray(obj.sdf["vol_bnds"], dtype=np.float32)
        if vol_bnds is not None:
            inb = np.logical_and(
                np.all(pts_obj >= vol_bnds[:, 0][None, :], axis=1),
                np.all(pts_obj <= vol_bnds[:, 1][None, :], axis=1),
            )
        else:
            inb = np.ones((n,), dtype=bool)

        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            try:
                qvals = obj.sdf_volume.query_sdf(pts_obj)
                if qvals is not None:
                    qvals = np.asarray(qvals).reshape(-1)
                    if qvals.shape[0] == n:
                        finite = np.isfinite(qvals)
                        keep = inb & finite
                        vals[keep] = qvals[keep].astype(np.float64)
                        valid[keep] = True
                        return vals, valid
            except Exception:
                pass

        if getattr(obj, "sdf", None) is None or "tsdf" not in obj.sdf:
            return vals, valid

        tsdf = np.asarray(obj.sdf["tsdf"], dtype=np.float32)
        origin = np.asarray(obj.sdf["vol_origin"], dtype=np.float32)
        voxel = float(obj.sdf["voxel_size"])
        vol_dim = np.array(tsdf.shape, dtype=np.int32)

        xyz = (pts_obj - origin[None, :]) / voxel
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        z0 = np.floor(z).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

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
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        n = pts_obj.shape[0]
        sdf0, v0 = self._query_sdf_signed(obj, pts_obj)

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
        grad = np.zeros((n, 3), dtype=np.float64)
        valid = v0.copy()
        for k in range(3):
            fp, vp = self._query_sdf_signed(obj, pts_obj + E[k][None, :])
            fm, vm = self._query_sdf_signed(obj, pts_obj - E[k][None, :])
            vk = v0 & vp & vm
            gk = np.zeros((n,), dtype=np.float64)
            gk[vk] = (fp[vk] - fm[vk]) / (2.0 * eps)
            grad[:, k] = gk
            valid &= vk

        gnorm = np.linalg.norm(grad, axis=1)
        valid &= np.isfinite(sdf0) & np.isfinite(gnorm) & (gnorm > 1e-8)
        return sdf0, grad, valid

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
            thr = float(np.percentile(r_abs, 80.0))

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
        if self._sdf_kernel == "cauchy":
            x = r_abs / d
            return 1.0 / (1.0 + x * x)
        return np.ones_like(r_abs, dtype=np.float64)

    def _refine_pose_with_sdf(self, pts_cur, T_cur2obj_init, obj):
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
            r_sdf = sdf_vals[valid].astype(np.float64)
            g_sdf = sdf_grads[valid].astype(np.float64)
            r_abs = np.abs(r_sdf)

            in_sdf, thr = self._compute_sdf_inliers(r_abs)
            dbg["trim_thr_history"].append(float(thr))
            if in_sdf.sum() < self._sdf_refine_min_pts:
                break

            pts_obj_in = pts_obj_v[in_sdf]
            r_sdf_in = r_sdf[in_sdf]
            g_sdf_in = g_sdf[in_sdf]

            n = pts_obj_in.shape[0]
            J = np.zeros((n, 6), dtype=np.float64)
            for i in range(n):
                x = pts_obj_in[i]
                G = np.zeros((3, 6), dtype=np.float64)
                G[:, :3] = np.eye(3)
                G[:, 3:] = -_skew(x)
                J[i] = g_sdf_in[i] @ G

            w = self._kernel_weights(np.abs(r_sdf_in))
            sw = np.sqrt(np.clip(w, 0.0, None))[:, None]
            Jw = J * sw
            rw = r_sdf_in * sw[:, 0]
            H = Jw.T @ Jw
            b = Jw.T @ rw
            cost = float(np.mean(rw**2)) if rw.size else np.inf
            if support_ratio < self._sdf_min_support_ratio:
                cost += float(
                    self._sdf_bad_penalty * (self._sdf_min_support_ratio - support_ratio)
                )

            dbg["cost_history"].append(cost)
            dbg["inliers_history"].append(int(in_sdf.sum()))

            H += float(self._sdf_refine_damping) * np.eye(6, dtype=np.float64)
            try:
                dxi = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dxi)):
                break

            dxi[:3] = np.clip(
                dxi[:3], -self._sdf_refine_step_trans_clip, self._sdf_refine_step_trans_clip
            )
            dxi[3:] = np.clip(
                dxi[3:], -self._sdf_refine_step_rot_clip, self._sdf_refine_step_rot_clip
            )

            T_candidate = _left_update_SE3(T, dxi)
            c_cost = self._eval_sdf_cost(pts_cur, T_candidate, obj)
            t_cost = self._eval_sdf_cost(pts_cur, T, obj)

            if np.isfinite(c_cost) and c_cost <= t_cost:
                T = T_candidate
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_T = T.copy()
            else:
                T_half = _left_update_SE3(T, 0.5 * dxi)
                h_cost = self._eval_sdf_cost(pts_cur, T_half, obj)
                if np.isfinite(h_cost) and h_cost < t_cost:
                    T = T_half
                    if h_cost < best_cost:
                        best_cost = h_cost
                        best_T = T.copy()

            dbg["iters"] = it + 1
            if np.linalg.norm(dxi[:3]) < 1e-4 and np.linalg.norm(dxi[3:]) < 1e-3:
                break

        if np.isfinite(best_cost):
            dbg["accepted"] = True
            dbg["best_cost"] = float(best_cost)
            return best_T.astype(np.float32), dbg
        return None, dbg

    def _eval_sdf_cost(self, pts_cur, T_cur2obj, obj):
        pts_obj = transform_pts(
            np.asarray(T_cur2obj, dtype=np.float32), np.asarray(pts_cur, dtype=np.float32)
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
            cost += float(self._sdf_bad_penalty * (self._sdf_min_support_ratio - support))
        return cost

    def _svd_fit(self, pa, qa):
        """
        Rigid SVD fit (Procrustes) of Pa->Qa with known correspondence.
        """
        # compute the centroids
        cp = pa.mean(axis=0)
        cq = qa.mean(axis=0)
        # center the points
        P = pa - cp
        Q = qa - cq
        # compute the covariance matrix
        H = P.T @ Q
        # compute the SVD
        try:
            U, _S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            # Degenerate correspondence set (e.g. near-collinear/coincident points ->
            # rank-deficient H) -- SVD can fail to converge regardless of point count.
            # Fall back to translation-only (identity rotation) rather than crashing the
            # tracking loop; the residual/inlier logic in register() treats a bad fit like
            # any other iteration and will reject it via the usual thresholds.
            print("[Register] SVD did not converge in _svd_fit; falling back to identity rotation")
            T = np.eye(4)
            T[:3, 3] = cq - cp
            return T
        # rotation
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # translation
        t = cq - R @ cp
        # return SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _weighted_svd_fit(self, src, tgt, w):
        """
        Weighted rigid Procrustes (point-to-point). Returns 4x4 T (src->tgt).
        """
        w = np.clip(w, 0.0, None)
        if np.sum(w) < 1e-9:
            return np.eye(4)

        W = w / (np.sum(w) + 1e-12)
        mu_src = np.sum(src * W[:, None], axis=0)
        mu_tgt = np.sum(tgt * W[:, None], axis=0)

        Xc = src - mu_src
        Yc = tgt - mu_tgt

        # weighted cross-covariance
        S = (Xc * W[:, None]).T @ Yc
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        # det correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_tgt - R @ mu_src

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
