import numpy as np


from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


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

    def register(self, src_pcd, tgt_pcd, init_pose=None):
        """
        Perform a single registration step using SVD.
        Args:
            src_pcd, tgt_pcd: (N,3) source/target with known correspondence (row-wise).
            init_pose: (4,4) initial pose.
        """
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

        return T, stats

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
