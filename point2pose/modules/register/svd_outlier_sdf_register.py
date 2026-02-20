import numpy as np
from scipy.optimize import minimize

from point2pose.core.module_registry import REGISTER
from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)
from point2pose.utils.lie import exp_se3, vec_to_se3
from point2pose.utils.transform import inverse_SE3, transform_pts


@REGISTER.register_module("svd_outlier_sdf")
class SVDOutlierSDFRegister(SVDClusterRANSACRegister):
    """SVD outlier rejection + local SDF-only pose optimization."""

    def __init__(self, config=None):
        super().__init__(config)
        self.type = "svd_outlier_sdf"
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
            return float(sdf_loss + reg)

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
