import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.lie import log_SE3


@REGISTER.register_module("svd_cluster_ransac")
class SVDClusterRANSACRegister(Register):
    def __init__(self, config=None):
        super().__init__(config)

        self.type = "svd_cluster_ransac"

        self._ransac_iters = config.get("ransac_iters", 30)
        self._sample_size = config.get("sample_size", 3)  # 3 or 4
        self._inlier_thres = config.get("inlier_thres", 0.01)
        self._min_inliers = config.get("min_inliers", 6)
        self._max_clusters = config.get("max_clusters", 10)

        self._trans_w = config.get("trans_weight", 1.0)
        self._min_var = float(config.get("min_variance", 1e-2))

    def register(
        self,
        src_pcd,
        tgt_pcd,
        init_pose=None,
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
        prev_T=None,
    ):
        stats = {}
        N = src_pcd.shape[0]
        w = self._build_weights(N, sigma_src, sigma_tgt, sigma)

        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        remaining = np.ones(N, dtype=bool)
        candidates = []

        for _c in range(self._max_clusters):
            candidate = self._RANSAC(
                p0=p0,
                tgt_pcd=tgt_pcd,
                w=w,
                remaining=remaining,
                init_pose=init_pose,
            )
            if candidate is None:
                break
            candidates.append(candidate)

        if len(candidates) == 0:
            T0 = init_pose if init_pose is not None else np.eye(4)
            stats["clusters"] = []
            stats["selected_idx"] = -1
            stats["remaining_mask"] = remaining
            stats["inliers"] = np.zeros(N, dtype=bool)
            stats["residuals"] = np.ones(N) * -1.0
            return T0, stats

        # choose: closest to previous transform (init_pose) if available
        if init_pose is not None:
            d = [self._pose_dist(c["T"], prev_T) for c in candidates]
            best_cluster_idx = int(np.argmin(d))
        else:
            # fallback: most inliers then lowest mean residual
            best_cluster_idx = int(
                np.lexsort(
                    (
                        [c["mean_res"] for c in candidates],
                        [-c["ninliers"] for c in candidates],
                    )
                )[0]
            )

        # recompute inliers and residuals from the best cluster
        inliers = np.zeros(N, dtype=bool)
        inliers[candidates[best_cluster_idx]["inliers"]] = True
        residuals = np.linalg.norm(
            transform_pts(candidates[best_cluster_idx]["T"], src_pcd) - tgt_pcd, axis=1
        )

        stats["clusters"] = candidates
        stats["best_cluster_idx"] = best_cluster_idx
        stats["remaining_mask"] = remaining
        stats["inliers"] = inliers
        stats["residuals"] = residuals
        return candidates[best_cluster_idx]["T"], stats

    def _RANSAC(self, p0, tgt_pcd, w, remaining, init_pose):
        """
        Extract one cluster using RANSAC+SVD on the remaining pool.

        Side-effect: updates `remaining` in-place by removing found inliers.

        Returns a candidate dict (same schema as `stats["candidates"]`) or None
        if no valid cluster can be found / remaining pool too small.
        """
        idx = np.where(remaining)[0]
        if idx.size < self._min_inliers or idx.size < self._sample_size:
            return None

        best_T = None
        best_inl = None
        best_score = -1e18
        best_mean = 1e18

        # 1) RANSAC on remaining pool
        for _k in range(self._ransac_iters):
            # sample a subset of the remaining pool
            samp = np.random.choice(idx, self._sample_size, replace=False)
            # fit a rigid transformation to the sampled points
            Tk = (
                self._weighted_svd_fit(p0[samp], tgt_pcd[samp], w[samp])
                if w is not None
                else self._svd_fit(p0[samp], tgt_pcd[samp])
            )
            # compute the residuals between the transformed source points and the target points
            r = np.linalg.norm(transform_pts(Tk, p0[idx]) - tgt_pcd[idx], axis=1)

            # compute the inliers
            inl = r <= self._inlier_thres
            # count the number of inliers
            ninl = int(inl.sum())

            if ninl < self._min_inliers:
                continue

            mean_r = float(r[inl].mean())
            score = ninl - 0.5 * mean_r

            if score > best_score:
                best_score = score
                best_T = Tk
                best_inl = inl
                best_mean = mean_r

        if best_T is None or best_inl is None:
            return None

        # 2) compute inlier set under threshold
        inlier_idx = idx[best_inl]
        if inlier_idx.size < self._min_inliers:
            return None

        # 3) refine on inliers
        Tr = (
            self._weighted_svd_fit(p0[inlier_idx], tgt_pcd[inlier_idx], w[inlier_idx])
            if w is not None
            else self._svd_fit(p0[inlier_idx], tgt_pcd[inlier_idx])
        )

        # re-evaluate refined residuals and (optionally) tighten inliers once
        rr = np.linalg.norm(
            transform_pts(Tr, p0[inlier_idx]) - tgt_pcd[inlier_idx], axis=1
        )
        mean_rr = float(rr.mean()) if rr.size else 1e18

        # 4) remove those inliers from pool
        remaining[inlier_idx] = False

        Tw = Tr @ init_pose if init_pose is not None else Tr
        return {
            "T": Tw,
            "inliers": inlier_idx,
            "ninliers": int(inlier_idx.size),
            "mean_res": mean_rr,
            "score": float(best_score),
            "mean_ransac": float(best_mean),
        }

    def _pose_dist(self, Ta, Tb):
        return np.linalg.norm(log_SE3(Ta @ inverse_SE3(Tb)))

    def _build_weights(self, N, sigma_src, sigma_tgt, sigma):
        if sigma is None and sigma_src is None and sigma_tgt is None:
            return None

        def arr(x):
            if x is None:
                return None
            x = np.asarray(x).astype(float)
            if x.ndim == 0:
                x = np.full((N,), float(x))
            return x

        sigma = arr(sigma)
        sigma_src = arr(sigma_src)
        sigma_tgt = arr(sigma_tgt)

        if sigma is None:
            var_src = (
                0.0 if sigma_src is None else np.maximum(sigma_src**2, self._min_var)
            )
            var_tgt = (
                0.0 if sigma_tgt is None else np.maximum(sigma_tgt**2, self._min_var)
            )
            var = var_src + var_tgt
        else:
            var = np.maximum(sigma**2, self._min_var)

        var = np.maximum(var, self._min_var)
        return 1.0 / var

    def _svd_fit(self, pa, qa):
        cp = pa.mean(axis=0)
        cq = qa.mean(axis=0)
        P = pa - cp
        Q = qa - cq
        H = P.T @ Q
        U, _, Vt = np.linalg.svd(H)
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
        s = np.sum(w) + 1e-12
        wn = w / s

        mu_src = np.sum(src * wn[:, None], axis=0)
        mu_tgt = np.sum(tgt * wn[:, None], axis=0)

        Xc = src - mu_src
        Yc = tgt - mu_tgt

        S = (Xc * wn[:, None]).T @ Yc
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
