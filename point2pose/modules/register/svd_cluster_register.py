import numpy as np

from scipy.spatial import cKDTree

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts, inverse_SE3, to_homo
from point2pose.utils.lie import log_SE3
from point2pose.utils.camera import (
    compute_projection_consistency,
    compute_projection_consistency_iou,
    extract_cropped_point_cloud,
)


@REGISTER.register_module("svd_cluster")
class SVDClusterRegister(Register):
    """
    Deterministic multi-cluster extraction:
    - For each cluster: robustly fit T on the remaining pool using residual trimming (MAD/fixed/reduce),
      refine on inliers, then remove those inliers from the pool.
    - Finally pick the best cluster (closest to prev_T / init_pose, or by inlier count & mean residual).
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.type = "svd_cluster"

        self._max_clusters = int(config.get("max_clusters", 10))

        # residual-outlier params (same as SVDResidualOutlierRegister)
        self._max_iter = int(config.get("max_iter", 5))
        self._threshold_method = config.get("threshold_method", "mad")
        self._inlier_thres = float(config.get("inlier_thres", 0.05))
        self._thres_reduce_factor = float(config.get("thres_reduce_factor", 0.01))
        self._mad_scale = float(config.get("mad_scale", 2.5))
        self._min_inliers = int(config.get("min_inliers", 3))
        self._select_method = config.get("select_method", "reproj_error")

        # uncertainty weights (optional)
        self._use_uncertainty = bool(config.get("use_uncertainty", False))
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
        prev_frame=None,
        cur_frame=None,
        obj_id=0,
        obj=None,
        mode="f2m",
    ):
        stats = {}
        N = int(src_pcd.shape[0])

        # weights
        w = None
        if self._use_uncertainty:
            w = self._build_weights(N, sigma_src, sigma_tgt, sigma)

        # apply init_pose once to source points
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        remaining = np.ones(N, dtype=bool)
        candidates = []

        for _c in range(self._max_clusters):
            cand = self._register_one_cluster(
                p0=p0,
                tgt_pcd=tgt_pcd,
                w=w,
                remaining=remaining,
                init_pose=init_pose,
            )
            if cand is None:
                break
            candidates.append(cand)

        if len(candidates) == 0:
            T0 = init_pose if init_pose is not None else np.eye(4)
            stats["clusters"] = []
            stats["best_cluster_idx"] = -1
            stats["remaining_mask"] = remaining
            stats["inliers"] = np.zeros(N, dtype=bool)
            stats["residuals"] = np.ones(N) * -1.0
            return T0, stats

        # choose: reprojection error if cur_frame available, otherwise fallback
        reproj_errors = None
        if self._select_method == "reproj_error":
            reproj_errors = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)

            for c in candidates:
                src_pcd_full_cur = src_pcd_full.copy()
                if mode == "f2f":
                    T_cur2prev = inverse_SE3(c["T"])
                elif mode == "f2m":
                    T_cur2prev = prev_T @ inverse_SE3(c["T"])

                err = compute_projection_consistency(
                    src_pcd_full_cur, T_cur2prev, prev_frame, obj_id=0
                )
                reproj_errors.append(err)
                c["reproj_error"] = float(err)
            best_cluster_idx = int(np.argmin(reproj_errors))
        elif self._select_method == "reproj_error_iou":
            reproj_errors = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)

            for c in candidates:
                src_pcd_full_cur = src_pcd_full.copy()
                T_cur2prev = prev_T @ inverse_SE3(c["T"])
                err = compute_projection_consistency_iou(
                    src_pcd_full_cur, T_cur2prev, prev_frame, obj_id=0
                )
                reproj_errors.append(err)
                c["reproj_error_iou"] = float(err)
            best_cluster_idx = int(np.argmax(reproj_errors))
        elif self._select_method == "3d_dist":
            dist_errors = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            tgt_pcd_full = extract_cropped_point_cloud(prev_frame, obj_id)
            for c in candidates:
                src_pcd_full_cur = src_pcd_full.copy()
                T_cur2prev = prev_T @ inverse_SE3(c["T"])
                dists = self.icp_like_point2point_residuals(
                    src_pcd_full_cur, tgt_pcd_full, T_cur2prev
                )

                dist_errors.append(dists)
                c["3d_dist"] = float(dists)
            best_cluster_idx = int(np.argmin(dist_errors))

        elif self._select_method == "3d_dist_prev_and_kf":
            dist_errors = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            tgt_pcd_full_prev = extract_cropped_point_cloud(prev_frame, obj_id)
            tgt_pcd_full_kf = extract_cropped_point_cloud(
                obj.keyframes[-1].frame, obj_id
            )

            for c in candidates:
                src_pcd_full_cur = src_pcd_full.copy()

                T_cur2prev = prev_T @ inverse_SE3(c["T"])
                T_cur2kf = obj.keyframes[-1].pose @ inverse_SE3(c["T"])

                dists_prev = self.icp_like_point2point_residuals(
                    src_pcd_full_cur, tgt_pcd_full_prev, T_cur2prev
                )
                dists_kf = self.icp_like_point2point_residuals(
                    src_pcd_full_cur, tgt_pcd_full_kf, T_cur2kf
                )

                dist_errors.append(dists_prev + dists_kf)
                c["3d_dist_prev_and_kf"] = float(dists_prev + dists_kf)
            best_cluster_idx = int(np.argmin(dist_errors))

        elif self._select_method == "dist_to_prev":
            ref_T = prev_T if prev_T is not None else init_pose
            d = [self._pose_dist(c["T"], ref_T) for c in candidates]
            best_cluster_idx = int(np.argmin(d))
        elif self._select_method == "inlier_count":
            best_cluster_idx = int(np.argmax([c["ninliers"] for c in candidates]))
        elif self._select_method == "mean_residual":
            best_cluster_idx = int(np.argmin([c["mean_res"] for c in candidates]))
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

        # recompute inliers/residuals for the selected cluster (for full N)
        inliers_full = np.zeros(N, dtype=bool)
        inliers_full[candidates[best_cluster_idx]["inliers"]] = True
        residuals_full = np.linalg.norm(
            transform_pts(candidates[best_cluster_idx]["T"], src_pcd) - tgt_pcd, axis=1
        )

        stats["clusters"] = candidates
        stats["best_cluster_idx"] = best_cluster_idx
        stats["remaining_mask"] = remaining
        stats["inliers"] = inliers_full
        stats["residuals"] = residuals_full
        if reproj_errors is not None:
            stats["reproj_errors"] = np.array(reproj_errors, dtype=float)
        return candidates[best_cluster_idx]["T"], stats

    def _register_one_cluster(self, p0, tgt_pcd, w, remaining, init_pose):
        """
        Extract one cluster by robust trimming on the remaining pool.
        Side-effect: updates `remaining` in-place by removing found inliers.

        Returns candidate dict or None if no valid cluster can be found.
        """
        idx = np.where(remaining)[0]
        if idx.size < self._min_inliers:
            return None

        P = p0[idx]
        Q = tgt_pcd[idx]
        ww = w[idx] if (w is not None) else None

        # initial fit (deterministic)
        T_rel = (
            self._weighted_svd_fit(P, Q, ww) if ww is not None else self._svd_fit(P, Q)
        )

        inliers = np.ones(idx.size, dtype=bool)
        last_thr = None
        last_res = None
        last_it = 0

        for it in range(self._max_iter):
            p_T = transform_pts(T_rel, P)
            residuals = np.linalg.norm(p_T - Q, axis=1)

            # threshold selection
            if self._threshold_method == "mad":
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-12
                thr = float(med + self._mad_scale * mad)
            elif self._threshold_method == "fixed":
                thr = float(self._inlier_thres)
            elif self._threshold_method == "reduce":
                thr = float(self._inlier_thres) - float(self._thres_reduce_factor * it)
                if thr < 0:
                    thr = 0.001
            else:
                raise ValueError(f"Invalid threshold method: {self._threshold_method}")

            new_inliers = residuals <= thr

            last_thr = thr
            last_res = residuals
            last_it = it

            # stop if stable or too few inliers
            if np.array_equal(new_inliers, inliers) or (
                new_inliers.sum() < self._min_inliers
            ):
                break

            inliers = new_inliers

            # refit on inliers
            Pin = P[inliers]
            Qin = Q[inliers]
            if Pin.shape[0] < self._min_inliers:
                break

            if ww is not None:
                win = ww[inliers]
                T_rel = self._weighted_svd_fit(Pin, Qin, win)
            else:
                T_rel = self._svd_fit(Pin, Qin)

        # finalize inlier set under the last threshold
        if last_res is None:
            return None

        final_inliers_local = last_res <= float(last_thr)
        if final_inliers_local.sum() < self._min_inliers:
            return None

        inlier_idx = idx[final_inliers_local]

        # refine one more time on final inliers
        Pin = p0[inlier_idx]
        Qin = tgt_pcd[inlier_idx]
        if w is not None:
            T_rel = self._weighted_svd_fit(Pin, Qin, w[inlier_idx])
        else:
            T_rel = self._svd_fit(Pin, Qin)

        # mean residual after refinement (on the inlier set)
        rr = np.linalg.norm(transform_pts(T_rel, Pin) - Qin, axis=1)
        mean_rr = float(rr.mean()) if rr.size else 1e18

        # peel off
        remaining[inlier_idx] = False

        # map back to transform that acts on original src_pcd
        T_world = (T_rel @ init_pose) if init_pose is not None else T_rel

        return {
            "T": T_world,
            "inliers": inlier_idx,
            "ninliers": int(inlier_idx.size),
            "mean_res": mean_rr,
            "iter": int(last_it),
            "thr": float(last_thr),
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

    def icp_like_point2point_residuals(
        self, source_pcd, target_pcd, T_src2tgt, max_corr_dist=0.01
    ):
        """
        ICP-style point-to-point residuals WITHOUT running ICP:
        - transform source points by T_src2tgt (CPU numpy)
        - 1-NN search in target (Cupoch KDTree)
        - residual = distance to nearest neighbor

        Returns:
        dists: (Ns,)
        inlier_mask: (Ns,) if max_corr_dist is not None
        rmse: float if max_corr_dist is not None
        """
        # source = cph.geometry.PointCloud()
        # source.points = cph.utility.Vector3fVector(source_pcd.astype(np.float32))
        # target = cph.geometry.PointCloud()
        # target.points = cph.utility.Vector3fVector(target_pcd.astype(np.float32))

        # estimation_method = cph.registration.TransformationEstimationPointToPoint()

        # # Set up convergence criteria
        # criteria = cph.registration.ICPConvergenceCriteria()
        # criteria.max_iteration = 0

        # # Perform ICP registration
        # result = cph.registration.registration_icp(
        #     source=source,
        #     target=target,
        #     max_correspondence_distance=max_corr_dist,
        #     init=T_src2tgt,
        #     estimation_method=estimation_method,
        #     criteria=criteria,
        # )

        # dists = result.inlier_rmse

        source_pcd = (T_src2tgt @ to_homo(source_pcd).T).T[:, :3]
        nn_index = cKDTree(source_pcd)
        nn_dists, _ = nn_index.query(target_pcd, k=1, workers=-1)
        return nn_dists.mean()
