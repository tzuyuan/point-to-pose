import numpy as np
import cupoch as cph
import copy

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts, inverse_SE3, to_homo
from point2pose.utils.lie import log_SE3
from point2pose.utils.camera import (
    compute_projection_consistency,
    compute_projection_consistency_mask_counting,
    compute_projection_consistency_iou,
    extract_cropped_point_cloud,
)


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
        self._max_iter = config.get("max_iter", 15)
        self._use_uncertainty = config.get("use_uncertainty", False)
        self._select_method = config.get("select_method", "reproj_error")
        self._select_w_prev = float(config.get("select_w_prev", 0.55))
        self._select_w_kf = float(config.get("select_w_kf", 0.30))
        self._select_w_motion = float(config.get("select_w_motion", 0.15))
        self._select_w_sparse_map = float(config.get("select_w_sparse_map", 0.7))
        self._select_w_sdf = float(config.get("select_w_sdf", 0.3))
        self._select_overlap_dist = float(config.get("select_overlap_dist", 0.02))
        self._select_min_overlap_ratio = float(
            config.get("select_min_overlap_ratio", 0.15)
        )
        self._select_robust_percentile = float(
            config.get("select_robust_percentile", 70.0)
        )
        self._select_bad_penalty = float(config.get("select_bad_penalty", 0.02))

        self._min_var = float(config.get("min_variance", 1e-2))

        np.random.seed(0)
        self._last_selected_T = None

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
        stats = {}
        N = src_pcd.shape[0]
        if self._use_uncertainty:
            w = self._build_weights(N, sigma_src, sigma_tgt, sigma)
        else:
            w = None

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

                # dists, _, _ = self.icp_like_point2point_score(
                #     src_pcd_full_cur, tgt_pcd_full, T_cur2prev
                # )

                dist_errors.append(dists)
                c["3d_dist"] = float(dists)
            best_cluster_idx = int(np.argmin(dist_errors))
            # xi = log_SE3(candidates[best_cluster_idx]["T"] @ inverse_SE3(prev_T))

            # dt, ddeg = self._se3_delta(candidates[best_cluster_idx]["T"], prev_T)

            # if dist_errors[best_cluster_idx] > 0.01:
            #     T, inliers, residuals = self._svd_residual_outlier_fit(p0, tgt_pcd, N)
            #     stats["clusters"] = candidates
            #     stats["best_cluster_idx"] = -1
            #     stats["remaining_mask"] = np.array(
            #         [False] * N
            #     )  # all points are inliers for this fallback
            #     stats["inliers"] = inliers
            #     stats["residuals"] = residuals
            #     return prev_T, stats  # fallback to prev_T if all candidates are bad
            # if dist_errors[best_cluster_idx] > 0.01 or dt > 0.05 or ddeg > 15:
            #     stats["clusters"] = candidates
            #     stats["best_cluster_idx"] = best_cluster_idx
            #     stats["remaining_mask"] = remaining
            #     stats["inliers"] = np.array([])
            #     stats["residuals"] = np.array([])
            #     return prev_T, stats  # fallback to prev_T if all candidates are bad
        elif self._select_method == "3d_dist_sparse_map":
            dist_errors = []
            trg_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            # map_pts = obj.key_points
            for c in candidates:
                map_pts = obj.key_points.copy()
                T_map2cur = c["T"]
                dists = self.icp_like_point2point_residuals(
                    map_pts, trg_pcd_full, T_map2cur
                )
                sdf_score = self._sdf_residual(
                    trg_pcd_full, inverse_SE3(T_map2cur), obj
                )
                if np.isfinite(sdf_score):
                    dists = self._select_w_sparse_map * float(
                        dists
                    ) + self._select_w_sdf * float(sdf_score)
                    c["sdf_dist"] = float(sdf_score)

                dist_errors.append(dists)
                c["3d_dist"] = float(dists)
            best_cluster_idx = int(np.argmin(dist_errors))

        elif self._select_method == "3d_dist_dense_map":
            dist_errors = []
            trg_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            # map_pts = obj.key_points
            for c in candidates:
                T_map2cur = c["T"]
                sdf_score = self._sdf_residual(
                    trg_pcd_full, inverse_SE3(T_map2cur), obj
                )
                if np.isfinite(sdf_score):
                    dists = float(sdf_score)
                    c["3d_dist"] = float(dists)
                else:
                    dists = np.finfo(np.float32).max
                    c["3d_dist"] = float(-1)

                dist_errors.append(float(dists))
            best_cluster_idx = int(np.argmin(dist_errors))

        elif self._select_method == "3d_dist_kf":
            dist_errors = []
            pcd_kf = []
            num_kf = 1
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            for k in obj.keyframes[-num_kf:]:
                if k.frame is None:
                    continue
                tgt_kf = extract_cropped_point_cloud(k.frame, obj_id)
                pcd_kf.append(tgt_kf)

            nn_index = cKDTree(src_pcd_full)
            for c in candidates:
                dists = []
                for i, k in enumerate(obj.keyframes[-num_kf:]):

                    trg_pcd = pcd_kf[i].copy()

                    T_cur2kf = k.pose @ inverse_SE3(c["T"])
                    nn_dists, _ = nn_index.query(trg_pcd, k=1, workers=-1)
                    dists.append(float(np.median(nn_dists)))

                dist_errors.append(float(np.mean(dists)))
                c["3d_dist_kf"] = float(np.mean(dists))
            best_cluster_idx = int(np.argmin(dist_errors))

        elif self._select_method == "3d_dist_prev_kf":
            scores = []
            scores_prev = []
            scores_kf = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            tgt_prev = extract_cropped_point_cloud(prev_frame, obj_id)
            tgt_kf = extract_cropped_point_cloud(obj.keyframes[-1].frame, obj_id)

            # tune these to your depth noise / object scale
            max_corr = 0.05  # e.g. 2cm if inlier_thres=1cm
            min_ratio = 0.15
            min_inl = 200

            lam_smooth = float(self.config.get("select_smooth_lambda", 0.05))  # small
            eps = 1e-6

            for c in candidates:
                T_cur2prev = prev_T @ inverse_SE3(c["T"])
                T_cur2kf = obj.keyframes[-1].pose @ inverse_SE3(c["T"])

                s_prev = self.icp_like_point2point_residuals(
                    src_pcd_full.copy(), tgt_prev, T_cur2prev
                )

                s_kf = self.icp_like_point2point_residuals(
                    src_pcd_full.copy(), tgt_kf, T_cur2kf
                )

                s = np.min([s_prev, s_kf])
                scores.append(s)
                scores_prev.append(s_prev)
                scores_kf.append(s_kf)
                c["3d_score_prev_kf"] = float(s)

            best_cluster_idx = int(np.argmin(scores))
        elif self._select_method == "3d_dist_prev_kf_motion":
            scores = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)
            tgt_prev = (
                extract_cropped_point_cloud(prev_frame, obj_id)
                if prev_frame is not None
                else None
            )
            kf = (
                obj.keyframes[-1]
                if (obj is not None and len(obj.keyframes) > 0)
                else None
            )
            tgt_kf = (
                extract_cropped_point_cloud(kf.frame, obj_id)
                if (kf is not None and kf.frame is not None)
                else None
            )

            ref_T = (
                self._last_selected_T
                if self._last_selected_T is not None
                else (prev_T if prev_T is not None else init_pose)
            )

            for c in candidates:
                score = 0.0
                used_geom_term = False
                T_inv = inverse_SE3(c["T"])

                if (
                    prev_T is not None
                    and tgt_prev is not None
                    and tgt_prev.shape[0] > 0
                ):
                    T_cur2prev = prev_T @ T_inv
                    d_prev, overlap_prev = self._robust_symmetric_nn_residual(
                        src_pcd_full,
                        tgt_prev,
                        T_cur2prev,
                        robust_percentile=self._select_robust_percentile,
                        overlap_dist=self._select_overlap_dist,
                    )
                    if overlap_prev < self._select_min_overlap_ratio:
                        d_prev += self._select_bad_penalty
                    score += self._select_w_prev * d_prev
                    used_geom_term = True
                    c["3d_dist_prev_motion"] = float(d_prev)
                    c["overlap_prev_motion"] = float(overlap_prev)

                if (
                    kf is not None
                    and getattr(kf, "pose", None) is not None
                    and tgt_kf is not None
                    and tgt_kf.shape[0] > 0
                ):
                    T_cur2kf = kf.pose @ T_inv
                    d_kf, overlap_kf = self._robust_symmetric_nn_residual(
                        src_pcd_full,
                        tgt_kf,
                        T_cur2kf,
                        robust_percentile=self._select_robust_percentile,
                        overlap_dist=self._select_overlap_dist,
                    )
                    if overlap_kf < self._select_min_overlap_ratio:
                        d_kf += self._select_bad_penalty
                    score += self._select_w_kf * d_kf
                    used_geom_term = True
                    c["3d_dist_kf_motion"] = float(d_kf)
                    c["overlap_kf_motion"] = float(overlap_kf)

                if ref_T is not None:
                    d_motion = self._pose_dist(c["T"], ref_T)
                    score += self._select_w_motion * d_motion
                    c["pose_motion_prior"] = float(d_motion)

                # fallback if geometric terms are unavailable
                if not used_geom_term:
                    score += float(c["mean_res"])

                scores.append(float(score))
                c["3d_score_prev_kf_motion"] = float(score)

            best_cluster_idx = int(np.argmin(scores))

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

        stats["clusters"] = candidates
        stats["best_cluster_idx"] = best_cluster_idx
        stats["remaining_mask"] = remaining
        stats["inliers"] = inliers
        stats["residuals"] = residuals
        if reproj_errors is not None:
            stats["reproj_errors"] = np.array(reproj_errors, dtype=float)

        self._last_selected_T = candidates[best_cluster_idx]["T"]
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
            # Tk = self._svd_fit(p0[samp], tgt_pcd[samp])
            # compute the residuals between the transformed source points and the target points
            r = np.linalg.norm(transform_pts(Tk, p0[idx]) - tgt_pcd[idx], axis=1)

            # compute the inliers
            inl = r <= self._inlier_thres

            # count the number of inliers
            ninl = int(inl.sum())

            if ninl < self._min_inliers:
                continue

            mean_r = float(r[inl].mean())
            score = ninl
            # rob_r = float(np.percentile(r[inl], 80))  # or np.median(r[inl])
            # score = ninl - 100 * rob_r

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

        r_all = np.linalg.norm(transform_pts(Tr, p0[idx]) - tgt_pcd[idx], axis=1)

        inl_all = r_all <= self._inlier_thres

        inlier_idx = idx[inl_all]
        if inlier_idx.size < self._min_inliers:
            return None

        mean_rr = float(r_all[inl_all].mean())

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

    # def _build_weights(self, N, sigma_src, sigma_tgt, sigma):
    #     if sigma is None and sigma_src is None and sigma_tgt is None:
    #         return None

    #     def arr(x):
    #         if x is None:
    #             return None
    #         x = np.asarray(x).astype(float)
    #         if x.ndim == 0:
    #             x = np.full((N,), float(x))
    #         return x

    #     sigma = arr(sigma)
    #     sigma_src = arr(sigma_src)
    #     sigma_tgt = arr(sigma_tgt)

    #     if sigma is None:
    #         var_src = (
    #             0.0 if sigma_src is None else np.maximum(sigma_src**2, self._min_var)
    #         )
    #         var_tgt = (
    #             0.0 if sigma_tgt is None else np.maximum(sigma_tgt**2, self._min_var)
    #         )
    #         var = var_src + var_tgt
    #     else:
    #         var = np.maximum(sigma**2, self._min_var)

    #     var = np.maximum(var, self._min_var)
    #     return 1.0 / var

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
            var_src = 0.0 if sigma_src is None else np.maximum(sigma_src, self._min_var)
            var_tgt = 0.0 if sigma_tgt is None else np.maximum(sigma_tgt, self._min_var)
            var = var_src + var_tgt
        else:
            var = np.maximum(sigma, self._min_var)

        var = np.maximum(var, self._min_var)
        return np.exp(-var / 0.1)

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

        source_pcd = (T_src2tgt @ to_homo(source_pcd).T).T[:, :3]
        nn_index = cKDTree(source_pcd)
        nn_dists, _ = nn_index.query(target_pcd, k=1, workers=-1)
        return nn_dists.mean()

    def icp_like_point2point_score(
        self,
        source_pcd,
        target_pcd,
        T_src2tgt,
        max_corr_dist=0.05,
        min_inlier_ratio=0.25,
        min_inliers=200,
    ):
        src = (T_src2tgt @ to_homo(source_pcd).T).T[:, :3]

        tree = cKDTree(target_pcd)
        dists, _ = tree.query(src, k=1, workers=-1)

        inl = dists <= max_corr_dist
        ninl = int(inl.sum())
        ratio = ninl / max(1, src.shape[0])

        if ninl < min_inliers or ratio < min_inlier_ratio:
            return np.inf, ratio, ninl

        # robust score: median or p80 works great
        score = float(np.median(dists[inl]))  # or np.percentile(dists[inl], 80)
        return score, ratio, ninl

    def _robust_symmetric_nn_residual(
        self,
        source_pcd,
        target_pcd,
        T_src2tgt,
        robust_percentile=70.0,
        overlap_dist=0.02,
    ):
        if (
            source_pcd is None
            or target_pcd is None
            or source_pcd.shape[0] == 0
            or target_pcd.shape[0] == 0
        ):
            return np.inf, 0.0

        q = float(np.clip(robust_percentile, 0.0, 100.0))

        src_t = (T_src2tgt @ to_homo(source_pcd).T).T[:, :3]

        tree_tgt = cKDTree(target_pcd)
        d_src_tgt, _ = tree_tgt.query(src_t, k=1, workers=-1)

        tree_src = cKDTree(src_t)
        d_tgt_src, _ = tree_src.query(target_pcd, k=1, workers=-1)

        robust_res = 0.5 * (np.percentile(d_src_tgt, q) + np.percentile(d_tgt_src, q))
        overlap_ratio = 0.5 * (
            float(np.mean(d_src_tgt <= overlap_dist))
            + float(np.mean(d_tgt_src <= overlap_dist))
        )
        return float(robust_res), float(overlap_ratio)

    def _se3_delta(self, T_new, T_old):
        dT = T_new @ inverse_SE3(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        dR = scipy_R.from_matrix(dT[:3, :3]).magnitude()
        ddeg = float(np.degrees(dR))
        return dt, ddeg

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
        return base + 0.05 * (1.0 - support)

    def _svd_residual_outlier_fit(self, p0, tgt_pcd, N):
        # fit transformation using svd
        T = self._svd_fit(p0, tgt_pcd)

        inliers = np.ones(N, dtype=bool)

        for it in range(self._max_iter):
            p_T = transform_pts(T, p0)
            residuals = np.linalg.norm(p_T - tgt_pcd, axis=1)

            thr = float(self._inlier_thres)

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
