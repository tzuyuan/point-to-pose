import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("svd_uncertainty_outlier")
class SVDUncertaintyOutlierRegister(Register):
    """
    Weighted SVD point-to-point registration with robust M-estimation + optional trimming.
    Keeps your uncertainties but adds IRLS (Cauchy/Huber/Tukey) and degeneracy checks.
    """

    def __init__(self, config=None):
        super().__init__(config or {})

        self.type = "svd_uncertainty_robust"

        # --- iteration / gating
        self._max_iter = int(config.get("max_iter", 10))
        self._min_inliers = int(config.get("min_inliers", 4))
        self._min_spatial_spread = float(config.get("min_spread", 1e-4))  # variance sum

        # --- robust kernel
        self._robust_kernel = str(config.get("robust_kernel", "cauchy")).lower()
        # c is the tuning constant. For Cauchy, 1.5~3 works well; for Huber/Tukey, 1.0~2.5*scale.
        self._robust_c = float(config.get("robust_c", 2.0))
        # Initial scale via MAD (Gaussian equiv. scale is 1.4826 * MAD)
        self._mad_scale = float(config.get("mad_scale", 1.4826))

        # --- optional trimming (keep fraction of best residuals each IRLS step)
        self._trim_frac = float(config.get("trim_frac", 0.0))  # 0.0 disables trimming
        self._enforce_trim_floor = bool(config.get("enforce_trim_floor", True))

        # --- uncertainties
        self._min_var = float(config.get("min_variance", 1e-3))

        # --- debugging
        self._print_every_iter = bool(config.get("print_every_iter", False))

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
        Args:
            src_pcd, tgt_pcd: (N,3) known correspondences
            init_pose: (4,4) optional prior
            sigma_src, sigma_tgt, sigma: optional std (isotropic) -> weights
        Returns:
            T (4,4), stats dict
        """
        stats = {}
        N = int(src_pcd.shape[0])
        if N < self._min_inliers:
            return np.eye(4), {"reason": "too_few_pairs", "inliers": np.zeros(N, bool)}

        # build uncertainty weights once
        base_w = self._build_weights(N, sigma_src, sigma_tgt, sigma)  # (N,) or None
        if base_w is None:
            base_w = np.ones(N, dtype=float)

        # pre-transform by init if any
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        # initial unweighted (but with base_w) fit
        T = self._weighted_svd_fit(p0, tgt_pcd, base_w)

        inliers = np.ones(N, dtype=bool)
        robust_w = np.ones(N, dtype=float)

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            residual_vec = p_T - tgt_pcd  # (N,3)
            residuals = np.linalg.norm(residual_vec, axis=1)  # (N,)

            # --- compute robust weights from residuals
            # robust scale via MAD on current inliers
            med = (
                np.median(residuals[inliers]) if inliers.any() else np.median(residuals)
            )
            mad = np.median(np.abs(residuals - med)) + 1e-12
            scale = (
                self._mad_scale * mad if mad > 0 else max(1e-6, np.std(residuals) * 0.5)
            )

            robust_w[:] = self._kernel_weights(
                residuals / max(scale, 1e-12), self._robust_kernel, self._robust_c
            )

            # --- optional trimming
            trim_mask = np.ones(N, dtype=bool)
            if self._trim_frac > 0.0:
                k = max(self._min_inliers, int(np.ceil((1.0 - self._trim_frac) * N)))
                order = np.argsort(residuals)
                keep_idx = order[:k]
                trim_mask[:] = False
                trim_mask[keep_idx] = True

            # final mask = previous inliers (start as all True), trimmed, positive weights
            new_mask = (trim_mask) & (robust_w > 1e-12)
            if new_mask.sum() < self._min_inliers and self._enforce_trim_floor:
                # fall back to no trimming if it got too aggressive
                new_mask = robust_w > 1e-12

            # spatial degeneracy check (avoid collinear/near-planar collapse without spread)
            if new_mask.sum() >= self._min_inliers:
                spread = (
                    p0[new_mask].var(axis=0).sum() + tgt_pcd[new_mask].var(axis=0).sum()
                )
                if spread < self._min_spatial_spread:
                    stats.update({"reason": "degenerate_geometry", "spread": spread})
                    return np.eye(4), stats

            # stop criteria: mask stable
            if np.array_equal(new_mask, inliers) and it > 0:
                stats.update(
                    {
                        "iter": it,
                        "residuals": residuals,
                        "inliers": inliers,
                        "robust_w": robust_w,
                    }
                )
                break

            inliers = new_mask

            if inliers.sum() < self._min_inliers:
                stats.update(
                    {"reason": "too_few_inliers", "iter": it, "inliers": inliers}
                )
                return np.eye(4), stats

            # combine weights: base inverse-variance * robust IRLS
            w_tot = base_w[inliers] * robust_w[inliers]
            # refit on the weighted inlier set
            T = self._weighted_svd_fit(p0[inliers], tgt_pcd[inliers], w_tot)

            if self._print_every_iter and self.debug_level > 0:
                print(
                    f"[Register] iter {it} | kept {inliers.sum()}/{N} | "
                    f"median={np.median(residuals[inliers]):.4f} max={np.max(residuals[inliers]):.4f}"
                )

            if it == self._max_iter - 1:
                stats.update(
                    {
                        "iter": it,
                        "residuals": residuals,
                        "inliers": inliers,
                        "robust_w": robust_w,
                    }
                )

        if init_pose is not None:
            T = T @ init_pose

        return T, stats

    # ---------------- helpers ----------------

    def _kernel_weights(self, x, kind="cauchy", c=2.0):
        """
        Return IRLS weights u(r)/r for robust M-estimation given standardized residual x = r/scale.
        """
        x = np.asarray(x, dtype=float)
        ax = np.abs(x) + 1e-12

        if kind == "cauchy":
            # rho = (c^2/2) * log(1 + (x/c)^2), psi = x / (1 + (x/c)^2)
            return 1.0 / (1.0 + (x / c) ** 2)
        elif kind == "huber":
            # psi = x if |x|<=c else c*sign(x)  => weight = min(1, c/|x|)
            return np.minimum(1.0, c / ax)
        elif kind == "tukey":
            # Tukey biweight: 0 outside |x|>c, (1 - (x/c)^2)^2 inside
            w = np.zeros_like(x)
            m = ax <= c
            r = x[m] / c
            w[m] = (1.0 - r * r) ** 2
            return w
        else:
            # fallback: no robustness
            return np.ones_like(x)

    def _build_weights(self, N, sigma_src, sigma_tgt, sigma):
        any_sigma = (
            (sigma is not None) or (sigma_src is not None) or (sigma_tgt is not None)
        )
        if not any_sigma:
            return None

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
        w = 1.0 / var
        return w

    def _svd_fit(self, pa, qa):
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
