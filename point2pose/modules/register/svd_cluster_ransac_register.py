import numpy as np
import cupoch as cph
import copy

from scipy.spatial import cKDTree

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
        self._use_uncertainty = config.get("use_uncertainty", False)
        self._select_method = config.get("select_method", "reproj_error")

        self._min_var = float(config.get("min_variance", 1e-2))

        np.random.seed(0)

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
        # elif self._selected_method == "3d_dist"
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

                # s_prev, r_prev, _ = self.icp_like_point2point_score(
                #     src_pcd_full,
                #     tgt_prev,
                #     T_cur2prev,
                #     max_corr_dist=max_corr,
                #     min_inlier_ratio=min_ratio,
                #     min_inliers=min_inl,
                # )
                # s_kf, r_kf, _ = self.icp_like_point2point_score(
                #     src_pcd_full,
                #     tgt_kf,
                #     T_cur2kf,
                #     max_corr_dist=max_corr,
                #     min_inlier_ratio=min_ratio,
                #     min_inliers=min_inl,
                # )

                s_prev = self.icp_like_point2point_residuals(
                    src_pcd_full, tgt_prev, T_cur2prev
                )

                s_kf = self.icp_like_point2point_residuals(
                    src_pcd_full, tgt_kf, T_cur2kf
                )

                # if np.isinf(s_prev) and np.isinf(s_kf):
                #     # fallback that doesn't depend on prev_T
                #     s = c["mean_res"] - 0.001 * c["ninliers"]
                # else:
                #     w_prev = r_prev / (r_prev + r_kf + eps)
                #     s = w_prev * s_prev + (1.0 - w_prev) * s_kf

                # # hysteresis: discourage switching when scores are close
                # if getattr(self, "_last_selected_T", None) is not None:
                #     s += lam_smooth * self._pose_dist(c["T"], self._last_selected_T)

                # scores.append(float(s))
                s = np.min([s_prev, s_kf])
                scores.append(s)
                c["3d_score_prev_kf"] = float(s)

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

        r_all = np.linalg.norm(transform_pts(Tr, p0[idx]) - tgt_pcd[idx], axis=1)

        # optional: adaptive threshold from MAD (see next section)
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
