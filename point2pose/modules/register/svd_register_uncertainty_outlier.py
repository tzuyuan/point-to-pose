import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("svd_uncertainty_outlier")
class SVDUncertaintyOutlierRegister(Register):
    """SVD (weighted) point-to-point registration with residual outlier rejection."""

    def __init__(self, config=None):
        super().__init__(config)

        self.type = "svd_uncertainty_outlier"

        self._max_iter = config.get("max_iter", 5)
        self._threshold_method = config.get("threshold_method", "mad")
        self._inlier_thres = config.get("inlier_thres", 0.05)
        self._thres_reduce_factor = config.get("thres_reduce_factor", 0.01)
        self._mad_scale = config.get("mad_scale", 2.5)
        self._min_inliers = config.get("min_inliers", 3)
        # Numerical floor for variances to avoid inf weights / NaNs
        self._min_var = float(config.get("min_variance", 1e-2))

    def register(
        self,
        src_pcd,
        tgt_pcd,
        init_pose=None,
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
    ):
        """
        Perform a single registration step using (weighted) SVD.

        Args:
            src_pcd, tgt_pcd: (N,3) with known correspondences (row-wise).
            init_pose: optional (4,4) initial pose to pre-transform src_pcd.
            sigma_src: (N,) optional std of source points (isotropic).
            sigma_tgt: (N,) optional std of target points (isotropic).
            sigma:     (N,) optional combined std per pair (already aggregated).

        Returns:
            T: (4,4) src->tgt transform
            stats: dict with residuals, inliers, etc.
        """
        stats = {}
        N = src_pcd.shape[0]

        # -- Build inverse-variance weights if uncertainties are provided
        w = self._build_weights(N, sigma_src, sigma_tgt, sigma)

        # Pre-transform source by init pose if provided
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        # Initial fit (weighted if w is not None)
        T = (
            self._weighted_svd_fit(p0, tgt_pcd, w)
            if w is not None
            else self._svd_fit(p0, tgt_pcd)
        )

        inliers = np.ones(N, dtype=bool)

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)

            # --- choose threshold
            if self._threshold_method == "mad":
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-12
                thr = med + self._mad_scale * mad  # 1.4826 ~ std for Gaussian
            elif self._threshold_method == "fixed":
                thr = float(self._inlier_thres)
            elif self._threshold_method == "reduce":
                thr = float(self._inlier_thres) - float(self._thres_reduce_factor * it)
                thr = max(thr, 1e-3)
            else:
                raise ValueError(f"Invalid threshold method: {self._threshold_method}")

            new_inliers = residuals <= thr

            if self.debug_level > 1:
                print(f"[Register] iter {it} inliers: {new_inliers.sum()} thr: {thr}")
                print(
                    f"[Register] iter {it} res_median: {np.median(residuals):.6f} "
                    f"res_mean: {np.mean(residuals):.6f} res_max: {np.max(residuals):.6f}"
                )

            # stop if no change or too few inliers or last iter
            if (
                np.array_equal(new_inliers, inliers)
                or new_inliers.sum() < self._min_inliers
                or it == self._max_iter - 1
            ):
                stats["iter"] = it
                stats["thr"] = thr
                stats["residuals"] = residuals
                stats["inliers"] = inliers
                break

            inliers = new_inliers

            # Refit on inliers; subset weights if provided
            if w is not None:
                T = self._weighted_svd_fit(p0[inliers], tgt_pcd[inliers], w[inliers])
            else:
                T = self._svd_fit(p0[inliers], tgt_pcd[inliers])

        # Recover to world if we pre-transformed by init_pose
        if init_pose is not None:
            T = T @ init_pose

        return T, stats

    # ---------------- internal helpers ----------------

    def _build_weights(self, N, sigma_src, sigma_tgt, sigma):
        """
        Build inverse-variance weights from uncertainties.
        Returns:
            w: (N,) or None if no uncertainties provided.
        """
        any_sigma = (
            (sigma is not None) or (sigma_src is not None) or (sigma_tgt is not None)
        )
        if not any_sigma:
            return None

        # Normalize inputs to arrays
        def _as_arr(x):
            if x is None:
                return None
            x = np.asarray(x).astype(float)
            if x.ndim == 0:
                x = np.full((N,), float(x))
            assert x.shape[0] == N, f"uncertainty length {x.shape[0]} != N={N}"
            return x

        sigma = _as_arr(sigma)
        sigma_src = _as_arr(sigma_src)
        sigma_tgt = _as_arr(sigma_tgt)

        if sigma is None:
            # combine per correspondence: var = var_src + var_tgt
            var_src = (
                np.zeros(N)
                if sigma_src is None
                else np.maximum(sigma_src**2, self._min_var)
            )
            var_tgt = (
                np.zeros(N)
                if sigma_tgt is None
                else np.maximum(sigma_tgt**2, self._min_var)
            )
            var = var_src + var_tgt
        else:
            var = np.maximum(sigma**2, self._min_var)

        var = np.maximum(var, self._min_var)
        # inverse variance weights; safe for zeros via min_var
        w = 1.0 / var
        print(f"w: {w}")
        print(f"var: {var}")
        print(f"sigma_tgt: {sigma_tgt}")

        # Optional: normalize weights to sum to 1 to keep scales stable (not required)
        # w = w / (np.sum(w) + 1e-12)
        return w

    def _svd_fit(self, pa, qa):
        """
        Unweighted rigid SVD fit (Procrustes) of Pa->Qa with known correspondence.
        """
        cp = pa.mean(axis=0)
        cq = qa.mean(axis=0)
        P = pa - cp
        Q = qa - cq
        H = P.T @ Q
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = cq - R @ cp
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _weighted_svd_fit(self, src, tgt, w):
        """
        Weighted rigid Procrustes (point-to-point). Returns 4x4 T (src->tgt).
        w: (N,) inverse-variance weights (>=0), larger = more trust.
        """
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, None)
        if np.sum(w) < 1e-12:
            return np.eye(4)

        Wsum = np.sum(w) + 1e-12
        Wnorm = w / Wsum

        mu_src = np.sum(src * Wnorm[:, None], axis=0)
        mu_tgt = np.sum(tgt * Wnorm[:, None], axis=0)

        Xc = src - mu_src
        Yc = tgt - mu_tgt

        # weighted cross-covariance S = sum_i w_i Xc_i Yc_i^T
        S = (Xc * Wnorm[:, None]).T @ Yc
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
