import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("svd_ransac")
class SVDRansacRegister(Register):
    """
    Simple RANSAC rigid registration with known correspondences,
    designed to be with SVDResidualOutlierRegister.

    - Uniform sampling
    - Configurable sample_size >= 3
    - Score by inlier count
    - Refit with SVD on inliers
    - Returns stats: residuals, thr, inliers, iter
    """

    def __init__(self, config=None):
        super().__init__(config or {})
        cfg = config or {}

        self._max_trials = int(cfg.get("max_trials", 200))
        self._inlier_thres = float(cfg.get("inlier_thres", 0.05))
        self._min_inliers = int(cfg.get("min_inliers", 3))

        self._sample_size = int(cfg.get("sample_size", 3))
        if self._sample_size < 3:
            raise ValueError("sample_size must be >= 3")

        # keep degeneracy very mild to avoid over-rejection
        self._check_degeneracy = bool(cfg.get("check_degeneracy", False))
        self._min_triangle_area = float(cfg.get("min_triangle_area", 0.0))

        self._refine_iters = int(cfg.get("refine_iters", 1))

    def register(self, src_pcd, tgt_pcd, init_pose=None, **kwargs):
        stats = {}
        N = src_pcd.shape[0]

        if N < self._sample_size:
            T_small = init_pose.copy() if init_pose is not None else np.eye(4)
            stats.update(
                {
                    "iter": 0,
                    "thr": self._inlier_thres,
                    "residuals": np.array([]),
                    "inliers": np.zeros(N, dtype=bool),
                    "reason": "too_few_points_for_sample_size",
                }
            )
            return T_small, stats

        # pre-transform source if init pose is given
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        best_T = None
        best_inliers = None
        best_num_in = -1

        all_idx = np.arange(N)

        for trial in range(self._max_trials):
            sample_idx = np.random.choice(
                all_idx, size=self._sample_size, replace=False
            )

            if self._check_degeneracy and self._sample_size == 3:
                a, b, c = p0[sample_idx]
                area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
                if area < self._min_triangle_area:
                    continue

            T_h = self._svd_fit(p0[sample_idx], tgt_pcd[sample_idx])

            p_h = transform_pts(T_h, p0)
            residuals = np.linalg.norm(p_h - tgt_pcd, axis=1)
            inliers = residuals <= self._inlier_thres
            num_in = int(np.sum(inliers))

            if num_in > best_num_in:
                best_num_in = num_in
                best_T = T_h
                best_inliers = inliers

        # if no hypothesis found, fallback
        if best_T is None:
            T = self._svd_fit(p0, tgt_pcd)
            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)
            inliers = residuals <= self._inlier_thres

            if init_pose is not None:
                T = T @ init_pose

            stats.update(
                {
                    "iter": 0,
                    "thr": self._inlier_thres,
                    "residuals": residuals,
                    "inliers": inliers,
                    "reason": "ransac_no_hypothesis_fallback_svd",
                }
            )
            return T, stats

        # refine on inliers (unweighted)
        T = best_T
        inliers = best_inliers

        for it in range(self._refine_iters):
            if np.sum(inliers) < self._min_inliers:
                break

            T = self._svd_fit(p0[inliers], tgt_pcd[inliers])

            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)
            new_inliers = residuals <= self._inlier_thres

            if np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers

        # compute final residuals for stats
        p_T = transform_pts(T, p0)
        residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)

        # recover init pose if used
        if init_pose is not None:
            T = T @ init_pose

        stats.update(
            {
                "iter": self._refine_iters,
                "thr": self._inlier_thres,
                "residuals": residuals,
                "inliers": inliers,
                "num_inliers": int(np.sum(inliers)),
                "inlier_ratio": float(np.sum(inliers)) / max(N, 1),
                "trials_used": self._max_trials,
                "sample_size": self._sample_size,
                "reason": (
                    "ransac_ok"
                    if np.sum(inliers) >= self._min_inliers
                    else "ransac_weak_consensus"
                ),
            }
        )

        return T, stats

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
