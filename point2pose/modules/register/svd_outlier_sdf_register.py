import numpy as np
from scipy.optimize import minimize

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER

from point2pose.utils.lie import exp_se3, vec_to_se3
from point2pose.utils.transform import inverse_SE3, transform_pts


@REGISTER.register_module("svd_outlier_sdf")
class SVDOutlierSDFRegister(Register):
    """SVD outlier rejection + local SDF-only pose optimization."""

    def __init__(self, config=None):
        super().__init__(config)
        self.type = "svd_outlier_sdf"

        self._max_iter = int(config.get("max_iter", 5))
        self._threshold_method = config.get("threshold_method", "mad")
        self._inlier_thres = float(config.get("inlier_thres", 0.05))
        self._thres_reduce_factor = float(config.get("thres_reduce_factor", 0.01))
        self._mad_scale = float(config.get("mad_scale", 2.5))
        self._min_inliers = int(config.get("min_inliers", 3))

        self._sdf_opt_max_iter = int(config.get("sdf_opt_max_iter", 25))
        self._sdf_opt_percentile = float(config.get("sdf_opt_percentile", 70.0))
        self._sdf_rot_bound = float(config.get("sdf_opt_rot_bound", 0.15))
        self._sdf_trans_bound = float(config.get("sdf_opt_trans_bound", 0.03))
        self._sdf_reg_lambda = float(config.get("sdf_opt_reg_lambda", 1e-3))
        self._sdf_reg_rot = float(config.get("sdf_opt_reg_rot", 1.0))
        self._sdf_reg_trans = float(config.get("sdf_opt_reg_trans", 1.0))

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
        obj=None,
        obj_id=0,
        mode="f2m",
    ):
        del sigma_src, sigma_tgt, sigma, prev_T, prev_frame, cur_frame, obj_id, mode
        stats = {}
        n_pts = int(src_pcd.shape[0])

        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        T_seed_local, inliers, residuals = self._svd_residual_outlier_fit(
            p0, tgt_pcd, n_pts
        )
        T_seed = T_seed_local @ init_pose if init_pose is not None else T_seed_local

        stats["inliers"] = inliers
        stats["residuals"] = residuals
        stats["sdf_optimized"] = False

        if obj is None:
            return T_seed, stats

        pts_for_sdf = tgt_pcd[inliers] if np.any(inliers) else tgt_pcd
        if pts_for_sdf.shape[0] < self._min_inliers:
            return T_seed, stats

        sdf_before = self._sdf_residual(
            pts_for_sdf,
            inverse_SE3(T_seed),
            obj,
            robust_percentile=self._sdf_opt_percentile,
        )
        if not np.isfinite(sdf_before):
            return T_seed, stats

        def objective(xi):
            dT = exp_se3(vec_to_se3(xi))
            T_cur = dT @ T_seed
            sdf_loss = self._sdf_residual(
                pts_for_sdf,
                inverse_SE3(T_cur),
                obj,
                robust_percentile=self._sdf_opt_percentile,
            )
            if not np.isfinite(sdf_loss):
                return 1e6
            reg = self._sdf_reg_lambda * (
                self._sdf_reg_rot * np.dot(xi[:3], xi[:3])
                + self._sdf_reg_trans * np.dot(xi[3:], xi[3:])
            )
            return float(sdf_loss)

        x0 = np.zeros(6, dtype=float)
        bounds = [(-self._sdf_rot_bound, self._sdf_rot_bound)] * 3 + [
            (-self._sdf_trans_bound, self._sdf_trans_bound)
        ] * 3
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self._sdf_opt_max_iter},
        )

        if not res.success:
            stats["sdf_before"] = float(sdf_before)
            stats["sdf_after"] = float(sdf_before)
            return T_seed, stats

        T_opt = exp_se3(vec_to_se3(res.x)) @ T_seed
        sdf_after = self._sdf_residual(
            pts_for_sdf,
            inverse_SE3(T_opt),
            obj,
            robust_percentile=self._sdf_opt_percentile,
        )

        if not np.isfinite(sdf_after) or sdf_after > sdf_before:
            stats["sdf_before"] = float(sdf_before)
            stats["sdf_after"] = float(sdf_before)
            return T_seed, stats

        stats["sdf_before"] = float(sdf_before)
        stats["sdf_after"] = float(sdf_after)
        stats["sdf_optimized"] = True
        stats["residuals"] = np.linalg.norm(
            transform_pts(T_opt, src_pcd) - tgt_pcd, axis=1
        )
        return T_opt, stats

    def _svd_residual_outlier_fit(self, p0, tgt_pcd, N):
        # fit transformation using svd
        T = self._svd_fit(p0, tgt_pcd)

        inliers = np.ones(N, dtype=bool)

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)

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

            # stop if no change or too few inliers
            if (
                np.array_equal(new_inliers, inliers)
                or new_inliers.sum() < self._min_inliers
                or it == self._max_iter - 1
            ):

                break

            inliers = new_inliers

            # refit on inliers
            T = self._svd_fit(p0[inliers], tgt_pcd[inliers])
        return T, inliers, residuals

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
        U, _S, Vt = np.linalg.svd(H)
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

    def _sdf_residual(self, pts_cur, T_cur2obj, obj, robust_percentile=70.0):
        if (
            obj is None
            or getattr(obj, "sdf", None) is None
            or pts_cur is None
            or pts_cur.shape[0] == 0
        ):
            return np.inf

        pts_obj = transform_pts(T_cur2obj, pts_cur)
        sdf_vals = None

        # nvblox-style query path
        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            qvals = obj.sdf_volume.query_sdf(pts_obj)
            if qvals is not None and qvals.shape[0] == pts_obj.shape[0]:
                sdf_vals = np.abs(qvals[np.isfinite(qvals)])

        # legacy dense grid path
        if (sdf_vals is None or sdf_vals.size == 0) and "tsdf" in obj.sdf:
            tsdf = obj.sdf["tsdf"]
            origin = obj.sdf["vol_origin"]
            voxel = float(obj.sdf["voxel_size"])
            vol_dim = np.array(tsdf.shape, dtype=np.int32)
            vox = np.floor((pts_obj - origin[None, :]) / voxel).astype(np.int32)

            inb = np.logical_and(
                np.all(vox >= 0, axis=1), np.all(vox < vol_dim[None, :], axis=1)
            )
            if not np.any(inb):
                return np.inf
            vox_in = vox[inb]
            sdf_vals = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]])

        if sdf_vals is None or sdf_vals.size == 0:
            return np.inf

        q = float(np.clip(robust_percentile, 0.0, 100.0))
        base = float(np.percentile(sdf_vals, q))
        support = float(np.mean(np.isfinite(sdf_vals)))
        return np.mean(sdf_vals)
