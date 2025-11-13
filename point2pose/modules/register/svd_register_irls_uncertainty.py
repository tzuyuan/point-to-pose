import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("svd_uncertainty_irls")
class SVDUncertaintyIRLSRegister(Register):
    """
    Uncertainty-aware SVD + IRLS:
      - base weights = inverse variance from (sigma_src, sigma_tgt, sigma)
      - robust weights = IRLS from standardized residuals (z = r / scale)
      - optional hard gating by residuals (MAD/fixed/reduce) like your baseline
    """

    def __init__(self, config=None):
        super().__init__(config or {})

        # --- outer loop / gating (same as your baseline)
        self._max_iter = int(config.get("max_iter", 10))
        self._threshold_method = str(
            config.get("threshold_method", "mad")
        ).lower()  # "mad"|"fixed"|"reduce"|"none"
        self._inlier_thres = float(config.get("inlier_thres", 0.05))
        self._thres_reduce_factor = float(config.get("thres_reduce_factor", 0.01))
        self._mad_scale = float(config.get("mad_scale", 2.5))
        self._min_inliers = int(config.get("min_inliers", 3))
        self._min_var = float(
            config.get("min_variance", 1e-6)
        )  # smaller floor is ok here

        # --- IRLS controls
        self._irls_loss = str(
            config.get("irls_loss", "l1")
        ).lower()  # "l1"|"huber"|"cauchy"|"tukey"
        self._irls_c = float(
            config.get("irls_c", 1.345)
        )  # Huber default; try 1.345~2.0
        self._scale_source = str(
            config.get("scale_source", "mad")
        ).lower()  # "mad"|"std"
        self._irls_eps = float(config.get("irls_eps", 1e-12))  # avoid div-by-zero
        self._stop_dW = float(config.get("stop_dW", 1e-3))  # weight-change stop
        self._print_every_iter = bool(config.get("print_every_iter", False))

        # --- optional trimming in z-space (0 disables)
        self._trim_frac = float(config.get("trim_frac", 0.0))

    def register(
        self,
        src_pcd,
        tgt_pcd,
        init_pose=None,
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
    ):
        N = src_pcd.shape[0]
        stats = {}
        if N < self._min_inliers:
            return np.eye(4), {"reason": "too_few_pairs", "inliers": np.zeros(N, bool)}

        # base inverse-variance weights (heteroscedastic)
        w_ivar = self._build_inverse_variance_weights(N, sigma_src, sigma_tgt, sigma)
        if w_ivar is None:
            w_ivar = np.ones(N, float)

        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        # init fit using only inverse-variance weights
        T = self._weighted_svd_fit(p0, tgt_pcd, w_ivar)

        inliers = np.ones(N, dtype=bool)
        w_robust = np.ones(N, dtype=float)  # IRLS weights start as 1

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            r_vec = p_T - tgt_pcd
            r = np.linalg.norm(r_vec, axis=1)  # (N,)

            # ----- optional hard gating by raw residuals (same as your baseline)
            if self._threshold_method == "mad":
                med = np.median(r)
                mad = np.median(np.abs(r - med)) + 1e-12
                thr = med + self._mad_scale * 1.4826 * mad
                mask_gate = r <= thr
            elif self._threshold_method == "fixed":
                thr = float(self._inlier_thres)
                mask_gate = r <= thr
            elif self._threshold_method == "reduce":
                thr = max(self._inlier_thres - self._thres_reduce_factor * it, 1e-3)
                mask_gate = r <= thr
            elif self._threshold_method == "none":
                thr = None
                mask_gate = np.ones(N, bool)
            else:
                raise ValueError(f"Invalid threshold method: {self._threshold_method}")

            # ----- compute standardized residuals z = r / scale on the gated set
            if self._scale_source == "mad":
                rin = r[mask_gate]
                if rin.size == 0:
                    return np.eye(4), {"reason": "no_points_after_gate"}
                med = np.median(rin)
                mad = np.median(np.abs(rin - med)) + 1e-12
                scale = self._mad_scale * mad
            else:  # "std"
                rin = r[mask_gate]
                s = np.std(rin) if rin.size > 1 else 0.0
                scale = max(s, 1e-9)

            z = r / max(scale, 1e-12)

            # ----- IRLS robust weights (soft downweighting)
            w_prev = w_robust.copy()
            w_robust = self._irls_weights(
                z, self._irls_loss, self._irls_c, self._irls_eps
            )

            # optional trimming by z (keeps smallest z fraction)
            if self._trim_frac > 0.0:
                k = max(self._min_inliers, int(np.ceil((1.0 - self._trim_frac) * N)))
                idx = np.argsort(z)[:k]
                trim_mask = np.zeros(N, bool)
                trim_mask[idx] = True
                mask = mask_gate & trim_mask
            else:
                mask = mask_gate

            if mask.sum() < self._min_inliers:
                return np.eye(4), {"reason": "too_few_inliers_after_mask", "iter": it}

            # ----- combine weights and refit
            w_total = w_ivar[mask] * w_robust[mask]
            T = self._weighted_svd_fit(p0[mask], tgt_pcd[mask], w_total)

            # ----- stopping criteria
            dW = np.max(np.abs(w_robust - w_prev))
            if self._print_every_iter and self.debug_level > 0:
                msg = (
                    f"[IRLS] iter {it} kept {mask.sum()}/{N} "
                    f"r_med={np.median(r[mask]):.4f} r_max={np.max(r[mask]):.4f} "
                    f"scale={scale:.4f} dW={dW:.2e}"
                )
                if thr is not None:
                    msg += f" thr={thr:.4f}"
                print(msg)

            # stop if weights and gate both stabilized
            if np.array_equal(mask, inliers) and dW < self._stop_dW:
                stats.update(
                    {"iter": it, "residuals": r, "inliers": mask, "w_robust": w_robust}
                )
                break

            inliers = mask
            if it == self._max_iter - 1:
                stats.update(
                    {"iter": it, "residuals": r, "inliers": mask, "w_robust": w_robust}
                )

        if init_pose is not None:
            T = T @ init_pose
        return T, stats

    # ---------- helpers ----------

    def _irls_weights(self, z, loss, c, eps):
        """
        Given standardized residuals z (>=0), return IRLS weights psi(z)/z.
        We compute with |z| for symmetry.
        """
        z = np.asarray(z, float)
        a = np.abs(z) + eps

        if loss == "l1":
            # LAD: psi = sign(z) -> w = 1/|z|
            return 1.0 / a
        elif loss == "huber":
            # psi = z if |z|<=c else c*sign(z)  => w = min(1, c/|z|)
            return np.minimum(1.0, c / a)
        elif loss == "cauchy":
            # psi = z / (1 + (z/c)^2)  => w = 1 / (1 + (z/c)^2)
            return 1.0 / (1.0 + (z / c) ** 2)
        elif loss == "tukey":
            # psi = z*(1 - (z/c)^2)^2 for |z|<=c; 0 outside  => w = (1 - (z/c)^2)^2 inside, 0 outside
            w = np.zeros_like(z)
            m = a <= c
            r = z[m] / c
            w[m] = (1.0 - r * r) ** 2
            return w
        else:
            # default: no robustness
            return np.ones_like(z)

    def _build_inverse_variance_weights(self, N, sigma_src, sigma_tgt, sigma):
        any_sigma = (
            (sigma is not None) or (sigma_src is not None) or (sigma_tgt is not None)
        )
        if not any_sigma:
            return None

        def _as_arr(x):
            if x is None:
                return None
            x = np.asarray(x, float)
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

        return 1.0 / np.maximum(var, self._min_var)

    def _weighted_svd_fit(self, src, tgt, w):
        w = np.asarray(w, float)
        w = np.clip(w, 0.0, None)
        Wsum = np.sum(w)
        if not np.isfinite(Wsum) or Wsum < 1e-12:
            return np.eye(4)

        W = w / (Wsum + 1e-12)
        mu_src = np.sum(src * W[:, None], axis=0)
        mu_tgt = np.sum(tgt * W[:, None], axis=0)

        Xc = src - mu_src
        Yc = tgt - mu_tgt
        S = (Xc * W[:, None]).T @ Yc

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
