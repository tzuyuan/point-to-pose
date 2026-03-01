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


def _skew(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _left_update_SE3(T, dxi):
    dxi = np.asarray(dxi, dtype=np.float64).reshape(6)
    dt = dxi[:3]
    w = dxi[3:]
    dT = np.eye(4, dtype=np.float64)
    dT[:3, :3] = scipy_R.from_rotvec(w).as_matrix()
    dT[:3, 3] = dt
    return dT @ T


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
        self._select_3d_dist_min_depth = float(
            config.get("select_3d_dist_min_depth", 0.08)
        )
        self._select_3d_dist_max_depth = float(
            config.get("select_3d_dist_max_depth", 0.5)
        )
        self._select_3d_dist_fill_missing_depth = bool(
            config.get("select_3d_dist_fill_missing_depth", False)
        )
        self._select_3d_dist_window_size = int(
            config.get("select_3d_dist_window_size", 3)
        )
        self._select_3d_dist_min_neighbors = int(
            config.get("select_3d_dist_min_neighbors", 1)
        )
        self._select_dense_map_close_margin = float(
            config.get("select_dense_map_close_margin", 1e-3)
        )
        self._select_dense_map_hybrid_robust_percentile = float(
            config.get("select_dense_map_hybrid_robust_percentile", 90.0)
        )
        self._select_dense_map_hybrid_sdf_tau = float(
            config.get("select_dense_map_hybrid_sdf_tau", 0.01)
        )
        self._select_dense_map_hybrid_min_support = float(
            config.get("select_dense_map_hybrid_min_support", 0.05)
        )
        self._select_dense_map_hybrid_min_inlier_ratio = float(
            config.get("select_dense_map_hybrid_min_inlier_ratio", 0.05)
        )
        self._select_dense_map_hybrid_sdf_gate_margin = float(
            config.get("select_dense_map_hybrid_sdf_gate_margin", 0.005)
        )
        self._select_dense_map_hybrid_sdf_w_support = float(
            config.get("select_dense_map_hybrid_sdf_w_support", 0.03)
        )
        self._select_dense_map_hybrid_sdf_w_inlier = float(
            config.get("select_dense_map_hybrid_sdf_w_inlier", 0.03)
        )
        self._select_dense_map_hybrid_w_sdf = float(
            config.get("select_dense_map_hybrid_w_sdf", 0.6)
        )
        self._select_dense_map_hybrid_w_prev = float(
            config.get("select_dense_map_hybrid_w_prev", 0.2)
        )
        self._select_dense_map_hybrid_w_motion = float(
            config.get("select_dense_map_hybrid_w_motion", 0.15)
        )
        self._select_dense_map_hybrid_w_ransac = float(
            config.get("select_dense_map_hybrid_w_ransac", 0.05)
        )
        self._select_support_guard_enable = bool(
            config.get("select_support_guard_enable", True)
        )
        self._select_support_guard_min_rel_ninliers = float(
            config.get("select_support_guard_min_rel_ninliers", 0.25)
        )
        self._select_support_guard_abs_min_ninliers = int(
            config.get(
                "select_support_guard_abs_min_ninliers",
                max(6, int(self._min_inliers)),
            )
        )
        self._select_support_guard_keep_rel_ninliers = float(
            config.get("select_support_guard_keep_rel_ninliers", 0.70)
        )
        self._select_support_guard_pose_weight = float(
            config.get("select_support_guard_pose_weight", 0.05)
        )

        # Optional SDF pose refinement after best-hypothesis selection.
        self._enable_sdf_refine = bool(config.get("enable_sdf_refine", False))
        self._sdf_refine_iters = int(config.get("sdf_refine_iters", 8))
        self._sdf_refine_damping = float(config.get("sdf_refine_damping", 1e-4))
        self._sdf_refine_grad_eps = float(
            config.get("sdf_refine_grad_eps", 0.5)
        )  # in voxels
        self._sdf_refine_max_pts = int(config.get("sdf_refine_max_pts", 1500))
        self._sdf_refine_use_dense_cur = bool(
            config.get("sdf_refine_use_dense_cur", True)
        )
        self._sdf_refine_min_pts = int(config.get("sdf_refine_min_pts", 30))
        self._sdf_trim_method = str(config.get("sdf_trim_method", "mad"))
        self._sdf_mad_scale = float(config.get("sdf_mad_scale", 2.5))
        self._sdf_inlier_thres = float(config.get("sdf_inlier_thres", 0.01))
        self._sdf_inlier_percentile = float(config.get("sdf_inlier_percentile", 80.0))
        self._sdf_kernel = str(config.get("sdf_kernel", "huber"))  # huber|cauchy|none
        self._sdf_kernel_delta = float(config.get("sdf_kernel_delta", 0.01))
        self._sdf_min_support_ratio = float(config.get("sdf_min_support_ratio", 0.15))
        self._sdf_bad_penalty = float(config.get("sdf_bad_penalty", 0.05))
        self._sdf_refine_step_trans_clip = float(
            config.get("sdf_refine_step_trans_clip", 0.02)
        )
        self._sdf_refine_step_rot_clip = float(
            config.get("sdf_refine_step_rot_clip", 0.2)
        )

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
            src_pcd_full = extract_cropped_point_cloud(
                cur_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )
            tgt_pcd_full = extract_cropped_point_cloud(
                prev_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )
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
            trg_pcd_full = extract_cropped_point_cloud(
                cur_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )
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
            trg_pcd_full = extract_cropped_point_cloud(
                cur_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )
            # map_pts = obj.key_points
            for c in candidates:
                T_map2cur = c["T"]
                sdf_score = self._sdf_residual(
                    trg_pcd_full, inverse_SE3(T_map2cur), obj, robust_percentile=90.0
                )
                if np.isfinite(sdf_score):
                    dists = float(sdf_score)
                    c["3d_dist"] = float(dists)
                else:
                    dists = np.finfo(np.float32).max
                    c["3d_dist"] = float(-1)

                dist_errors.append(float(dists))
            ranked_idx = np.argsort(np.asarray(dist_errors, dtype=float))
            best_cluster_idx = int(ranked_idx[0])

            # Tie-break close top-2 SDF scores with a motion prior to previous estimate.
            if ranked_idx.size >= 2:
                second_idx = int(ranked_idx[1])
                best_score = float(dist_errors[best_cluster_idx])
                second_score = float(dist_errors[second_idx])
                if (second_score - best_score) <= self._select_dense_map_close_margin:
                    ref_T = prev_T if prev_T is not None else init_pose
                    if ref_T is not None:
                        d0 = self._pose_dist(candidates[best_cluster_idx]["T"], ref_T)
                        d1 = self._pose_dist(candidates[second_idx]["T"], ref_T)
                        candidates[best_cluster_idx]["dense_map_prev_pose_dist"] = (
                            float(d0)
                        )
                        candidates[second_idx]["dense_map_prev_pose_dist"] = float(d1)
                        if d1 < d0:
                            best_cluster_idx = second_idx

        elif self._select_method == "3d_dist_dense_map_hybrid":
            trg_pcd_full = extract_cropped_point_cloud(
                cur_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )

            best_cluster_idx = self._select_dense_map_hybrid(
                candidates=candidates,
                src_dense_cur=trg_pcd_full,
                prev_T=prev_T,
                prev_frame=prev_frame,
                init_pose=init_pose,
                obj=obj,
                obj_id=obj_id,
                mode=mode,
            )

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

        support_guard_info = {"applied": False}
        if self._select_support_guard_enable and self._select_method in (
            "3d_dist_dense_map",
            "3d_dist_dense_map_hybrid",
        ):
            ref_T = prev_T if prev_T is not None else init_pose
            best_cluster_idx, support_guard_info = self._apply_support_guard(
                candidates=candidates,
                best_cluster_idx=best_cluster_idx,
                ref_T=ref_T,
            )

        selected_T = np.asarray(candidates[best_cluster_idx]["T"], dtype=np.float64)
        sdf_refine_info = {"enabled": bool(self._enable_sdf_refine), "applied": False}
        if self._enable_sdf_refine:
            selected_T, sdf_refine_info = self._maybe_refine_with_sdf(
                T_seed=selected_T,
                src_corr=src_pcd,
                tgt_corr=tgt_pcd,
                cur_frame=cur_frame,
                obj_id=obj_id,
                obj=obj,
            )
            candidates[best_cluster_idx]["T"] = selected_T
            candidates[best_cluster_idx]["sdf_refine"] = sdf_refine_info

        # Recompute residuals / inliers from final selected pose (after optional SDF refine).
        residuals = np.linalg.norm(transform_pts(selected_T, src_pcd) - tgt_pcd, axis=1)
        inliers = residuals <= self._inlier_thres
        candidates[best_cluster_idx]["ninliers"] = int(np.count_nonzero(inliers))
        candidates[best_cluster_idx]["mean_res"] = float(
            np.mean(residuals[inliers]) if np.any(inliers) else np.mean(residuals)
        )

        stats["clusters"] = candidates
        stats["best_cluster_idx"] = best_cluster_idx
        stats["remaining_mask"] = remaining
        stats["inliers"] = inliers
        stats["residuals"] = residuals
        stats["support_guard"] = support_guard_info
        stats["sdf_refine"] = sdf_refine_info
        if reproj_errors is not None:
            stats["reproj_errors"] = np.array(reproj_errors, dtype=float)

        self._last_selected_T = selected_T
        return selected_T, stats

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
            try:
                Tk = (
                    self._weighted_svd_fit(p0[samp], tgt_pcd[samp], w[samp])
                    if w is not None
                    else self._svd_fit(p0[samp], tgt_pcd[samp])
                )
            except np.linalg.LinAlgError:
                continue
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

    def _normalize_cost(self, arr, candidate_indices):
        values = np.asarray(arr, dtype=float)
        out = np.ones_like(values, dtype=float)
        idx = np.asarray(candidate_indices, dtype=int)
        if idx.size == 0:
            return out

        sel = values[idx]
        finite = np.isfinite(sel)
        if not np.any(finite):
            return out

        finite_vals = sel[finite]
        lo = float(np.percentile(finite_vals, 5.0))
        hi = float(np.percentile(finite_vals, 95.0))
        if hi <= lo + 1e-12:
            out[idx[finite]] = 0.0
            return out

        out[idx[finite]] = np.clip((sel[finite] - lo) / (hi - lo), 0.0, 1.0)
        return out

    def _compute_T_cur2prev(self, T_map2cur, prev_T, mode):
        T_inv = inverse_SE3(T_map2cur)
        if mode == "f2f":
            return T_inv
        if mode == "f2m":
            if prev_T is None:
                return None
            return prev_T @ T_inv
        if prev_T is None:
            return T_inv
        return prev_T @ T_inv

    def _apply_support_guard(self, candidates, best_cluster_idx, ref_T=None):
        info = {
            "applied": False,
            "old_best_idx": int(best_cluster_idx),
            "new_best_idx": int(best_cluster_idx),
            "old_best_ninliers": -1,
            "new_best_ninliers": -1,
            "max_ninliers": -1,
            "required_ninliers": -1,
        }

        m = len(candidates)
        if m <= 1 or best_cluster_idx < 0 or best_cluster_idx >= m:
            return int(best_cluster_idx), info

        nin = np.asarray([float(c.get("ninliers", 0)) for c in candidates], dtype=float)
        if nin.size == 0:
            return int(best_cluster_idx), info

        max_n = int(np.max(nin))
        if max_n <= 0:
            return int(best_cluster_idx), info

        req_rel = int(
            np.ceil(
                max(0.0, float(self._select_support_guard_min_rel_ninliers))
                * float(max_n)
            )
        )
        req_abs = int(
            max(
                int(self._min_inliers),
                int(self._select_support_guard_abs_min_ninliers),
            )
        )
        req_n = int(max(req_rel, req_abs))
        best_n = int(nin[int(best_cluster_idx)])

        info["old_best_ninliers"] = int(best_n)
        info["max_ninliers"] = int(max_n)
        info["required_ninliers"] = int(req_n)

        if best_n >= req_n:
            info["new_best_ninliers"] = int(best_n)
            return int(best_cluster_idx), info

        keep_rel = float(
            np.clip(float(self._select_support_guard_keep_rel_ninliers), 0.0, 1.0)
        )
        keep_n = int(max(int(self._min_inliers), np.ceil(keep_rel * float(max_n))))
        pool = np.flatnonzero(nin >= keep_n)
        if pool.size == 0:
            pool = np.array([int(np.argmax(nin))], dtype=int)

        d3_vals = []
        for i in pool:
            v = float(candidates[i].get("3d_dist", np.inf))
            if (not np.isfinite(v)) or v < 0.0:
                v = np.inf
            d3_vals.append(v)
        d3 = np.asarray(d3_vals, dtype=float)

        chosen = int(pool[0])
        finite_d3 = np.isfinite(d3)
        if np.any(finite_d3):
            pool_f = pool[finite_d3]
            d3_f = d3[finite_d3]
            if ref_T is not None:
                pose_d = np.asarray(
                    [self._pose_dist(candidates[i]["T"], ref_T) for i in pool_f],
                    dtype=float,
                )
                fused = d3_f + float(self._select_support_guard_pose_weight) * pose_d
                chosen = int(pool_f[int(np.argmin(fused))])
            else:
                chosen = int(pool_f[int(np.argmin(d3_f))])
        elif ref_T is not None:
            pose_d = np.asarray(
                [self._pose_dist(candidates[i]["T"], ref_T) for i in pool], dtype=float
            )
            chosen = int(pool[int(np.argmin(pose_d))])
        else:
            chosen = int(np.argmax(nin))

        if chosen != int(best_cluster_idx):
            info["applied"] = True
            candidates[int(best_cluster_idx)]["support_guard_rejected"] = True
            candidates[int(best_cluster_idx)]["support_guard_reason"] = "low_support"
            candidates[int(chosen)]["support_guard_selected"] = True
            candidates[int(chosen)]["support_guard_reason"] = "low_support"

        info["new_best_idx"] = int(chosen)
        info["new_best_ninliers"] = int(nin[int(chosen)])
        return int(chosen), info

    def _select_dense_map_hybrid(
        self,
        candidates,
        src_dense_cur,
        prev_T,
        prev_frame,
        init_pose,
        obj,
        obj_id,
        mode,
    ):
        m = len(candidates)
        if m == 1:
            c0 = candidates[0]
            c0["dense_map_hybrid_score"] = 0.0
            return 0

        sdf_costs = np.full((m,), np.inf, dtype=float)
        prev_costs = np.full((m,), np.inf, dtype=float)
        motion_costs = np.full((m,), np.inf, dtype=float)
        ransac_res_costs = np.zeros((m,), dtype=float)
        ransac_inv_inlier_costs = np.zeros((m,), dtype=float)

        ref_T = (
            self._last_selected_T
            if self._last_selected_T is not None
            else (prev_T if prev_T is not None else init_pose)
        )

        tgt_prev = None
        if prev_frame is not None:
            tgt_prev = extract_cropped_point_cloud(
                prev_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )

        for i, c in enumerate(candidates):
            T_map2cur = c["T"]
            T_cur2obj = inverse_SE3(T_map2cur)

            sdf_raw, sdf_support_ratio, sdf_inlier_ratio = self._sdf_residual_stats(
                src_dense_cur,
                T_cur2obj,
                obj,
                robust_percentile=self._select_dense_map_hybrid_robust_percentile,
                tau=self._select_dense_map_hybrid_sdf_tau,
            )
            c["dense_map_hybrid_sdf_raw"] = (
                float(sdf_raw) if np.isfinite(sdf_raw) else -1.0
            )
            c["dense_map_hybrid_sdf_support"] = float(sdf_support_ratio)
            c["dense_map_hybrid_sdf_inlier_ratio"] = float(sdf_inlier_ratio)

            if (
                np.isfinite(sdf_raw)
                and sdf_support_ratio >= self._select_dense_map_hybrid_min_support
                and sdf_inlier_ratio >= self._select_dense_map_hybrid_min_inlier_ratio
            ):
                sdf_cost = (
                    float(sdf_raw)
                    + self._select_dense_map_hybrid_sdf_w_support
                    * (1.0 - float(sdf_support_ratio))
                    + self._select_dense_map_hybrid_sdf_w_inlier
                    * (1.0 - float(sdf_inlier_ratio))
                )
                sdf_costs[i] = float(sdf_cost)
                c["3d_dist"] = float(sdf_cost)
            else:
                c["3d_dist"] = float(-1)

            if ref_T is not None:
                motion_cost = self._pose_dist(T_map2cur, ref_T)
                motion_costs[i] = float(motion_cost)
                c["dense_map_hybrid_motion"] = float(motion_cost)
            else:
                c["dense_map_hybrid_motion"] = -1.0

            if (
                tgt_prev is not None
                and tgt_prev.shape[0] > 0
                and src_dense_cur is not None
                and src_dense_cur.shape[0] > 0
            ):
                T_cur2prev = self._compute_T_cur2prev(T_map2cur, prev_T, mode)
                if T_cur2prev is not None:
                    d_prev, overlap_prev = self._robust_symmetric_nn_residual(
                        src_dense_cur,
                        tgt_prev,
                        T_cur2prev,
                        robust_percentile=self._select_robust_percentile,
                        overlap_dist=self._select_overlap_dist,
                    )
                    if overlap_prev < self._select_min_overlap_ratio:
                        d_prev += self._select_bad_penalty
                    prev_costs[i] = float(d_prev)
                    c["dense_map_hybrid_prev_dist"] = float(d_prev)
                    c["dense_map_hybrid_prev_overlap"] = float(overlap_prev)
                else:
                    c["dense_map_hybrid_prev_dist"] = -1.0
                    c["dense_map_hybrid_prev_overlap"] = 0.0
            else:
                c["dense_map_hybrid_prev_dist"] = -1.0
                c["dense_map_hybrid_prev_overlap"] = 0.0

            ransac_res_costs[i] = float(c.get("mean_res", 0.0))
            ransac_inv_inlier_costs[i] = 1.0 / max(1.0, float(c.get("ninliers", 1)))

        finite_sdf = np.where(np.isfinite(sdf_costs))[0]
        if finite_sdf.size > 0:
            best_sdf = float(np.min(sdf_costs[finite_sdf]))
            candidate_pool = np.where(
                sdf_costs <= best_sdf + self._select_dense_map_hybrid_sdf_gate_margin
            )[0]
            if candidate_pool.size == 0:
                candidate_pool = finite_sdf
        else:
            candidate_pool = np.arange(m, dtype=int)

        sdf_norm = self._normalize_cost(sdf_costs, candidate_pool)
        prev_norm = self._normalize_cost(prev_costs, candidate_pool)
        motion_norm = self._normalize_cost(motion_costs, candidate_pool)
        ransac_res_norm = self._normalize_cost(ransac_res_costs, candidate_pool)
        ransac_inv_inlier_norm = self._normalize_cost(
            ransac_inv_inlier_costs, candidate_pool
        )
        ransac_norm = 0.5 * (ransac_res_norm + ransac_inv_inlier_norm)

        w_sdf = float(self._select_dense_map_hybrid_w_sdf)
        w_prev = float(self._select_dense_map_hybrid_w_prev)
        w_motion = float(self._select_dense_map_hybrid_w_motion)
        w_ransac = float(self._select_dense_map_hybrid_w_ransac)

        if not np.any(np.isfinite(sdf_costs[candidate_pool])):
            w_sdf = 0.0
        if not np.any(np.isfinite(prev_costs[candidate_pool])):
            w_prev = 0.0
        if not np.any(np.isfinite(motion_costs[candidate_pool])):
            w_motion = 0.0

        wsum = w_sdf + w_prev + w_motion + w_ransac
        if wsum <= 1e-12:
            for i, c in enumerate(candidates):
                c["dense_map_hybrid_score"] = float(c.get("mean_res", 0.0))
            return int(
                np.argmin(np.asarray([c.get("mean_res", np.inf) for c in candidates]))
            )

        w_sdf /= wsum
        w_prev /= wsum
        w_motion /= wsum
        w_ransac /= wsum

        fused = np.ones((m,), dtype=float) * np.finfo(np.float32).max
        for i in candidate_pool:
            fused[i] = (
                w_sdf * sdf_norm[i]
                + w_prev * prev_norm[i]
                + w_motion * motion_norm[i]
                + w_ransac * ransac_norm[i]
            )

        for i, c in enumerate(candidates):
            c["dense_map_hybrid_score"] = float(fused[i])
            c["dense_map_hybrid_sdf_term"] = float(sdf_norm[i])
            c["dense_map_hybrid_prev_term"] = float(prev_norm[i])
            c["dense_map_hybrid_motion_term"] = float(motion_norm[i])
            c["dense_map_hybrid_ransac_term"] = float(ransac_norm[i])

        return int(np.argmin(fused))

    def _se3_delta(self, T_new, T_old):
        dT = T_new @ inverse_SE3(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        dR = scipy_R.from_matrix(dT[:3, :3]).magnitude()
        ddeg = float(np.degrees(dR))
        return dt, ddeg

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

    def _maybe_refine_with_sdf(
        self,
        T_seed,
        src_corr,
        tgt_corr,
        cur_frame,
        obj_id,
        obj,
    ):
        info = {"enabled": True, "applied": False, "skip_reason": ""}
        if obj is None:
            info["skip_reason"] = "no_object"
            return np.asarray(T_seed, dtype=np.float64), info

        has_sdf = getattr(obj, "sdf_volume", None) is not None or (
            getattr(obj, "sdf", None) is not None and "tsdf" in obj.sdf
        )
        if not has_sdf:
            info["skip_reason"] = "no_sdf"
            return np.asarray(T_seed, dtype=np.float64), info

        pts_cur = self._get_sdf_source_points(
            src_pcd=src_corr, cur_frame=cur_frame, obj_id=obj_id
        )
        if pts_cur is None or pts_cur.shape[0] < self._sdf_refine_min_pts:
            info["skip_reason"] = "too_few_points"
            return np.asarray(T_seed, dtype=np.float64), info

        T_cur2obj_seed = inverse_SE3(np.asarray(T_seed, dtype=np.float64))
        seed_cost = self._eval_sdf_cost(pts_cur, T_cur2obj_seed, obj)

        T_cur2obj_ref, dbg = self._refine_pose_with_sdf(
            pts_cur=pts_cur,
            T_cur2obj_init=T_cur2obj_seed,
            obj=obj,
            src_corr=src_corr,
            tgt_corr=tgt_corr,
        )
        info["debug"] = dbg

        if T_cur2obj_ref is None:
            info["skip_reason"] = "optimizer_failed"
            return np.asarray(T_seed, dtype=np.float64), info

        ref_cost = self._eval_sdf_cost(pts_cur, T_cur2obj_ref, obj)
        info["seed_cost"] = float(seed_cost) if np.isfinite(seed_cost) else np.inf
        info["refined_cost"] = float(ref_cost) if np.isfinite(ref_cost) else np.inf

        if np.isfinite(ref_cost) and (
            (not np.isfinite(seed_cost)) or ref_cost <= seed_cost
        ):
            info["applied"] = True
            return inverse_SE3(T_cur2obj_ref), info

        info["skip_reason"] = "not_improved"
        return np.asarray(T_seed, dtype=np.float64), info

    def _get_sdf_source_points(self, src_pcd, cur_frame, obj_id):
        pts = None
        if self._sdf_refine_use_dense_cur and cur_frame is not None:
            try:
                pts = extract_cropped_point_cloud(
                    cur_frame,
                    obj_id,
                    min_depth=self._select_3d_dist_min_depth,
                    max_depth=self._select_3d_dist_max_depth,
                    fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                    window_size=self._select_3d_dist_window_size,
                    min_neighbors=self._select_3d_dist_min_neighbors,
                )
            except Exception:
                pts = None
        if pts is None and src_pcd is not None:
            pts = np.asarray(src_pcd)
        if pts is None:
            return None

        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return None
        if pts.shape[0] > self._sdf_refine_max_pts:
            step = int(np.ceil(pts.shape[0] / float(self._sdf_refine_max_pts)))
            pts = pts[::step]
        return pts

    def _query_sdf_signed(self, obj, pts_obj):
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        n = pts_obj.shape[0]
        vals = np.full((n,), np.nan, dtype=np.float64)
        valid = np.zeros((n,), dtype=bool)

        vol_bnds = None
        if getattr(obj, "sdf", None) is not None and "vol_bnds" in obj.sdf:
            vol_bnds = np.asarray(obj.sdf["vol_bnds"], dtype=np.float32)
        if vol_bnds is not None:
            inb = np.logical_and(
                np.all(pts_obj >= vol_bnds[:, 0][None, :], axis=1),
                np.all(pts_obj <= vol_bnds[:, 1][None, :], axis=1),
            )
        else:
            inb = np.ones((n,), dtype=bool)

        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            try:
                qvals = obj.sdf_volume.query_sdf(pts_obj)
                if qvals is not None:
                    qvals = np.asarray(qvals).reshape(-1)
                    if qvals.shape[0] == n:
                        finite = np.isfinite(qvals)
                        keep = inb & finite
                        vals[keep] = qvals[keep].astype(np.float64)
                        valid[keep] = True
                        return vals, valid
            except Exception:
                pass

        if getattr(obj, "sdf", None) is None or "tsdf" not in obj.sdf:
            return vals, valid

        tsdf = np.asarray(obj.sdf["tsdf"], dtype=np.float32)
        origin = np.asarray(obj.sdf["vol_origin"], dtype=np.float32)
        voxel = float(obj.sdf["voxel_size"])
        vol_dim = np.array(tsdf.shape, dtype=np.int32)

        xyz = (pts_obj - origin[None, :]) / voxel
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        z0 = np.floor(z).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        inb_tri = (
            (x0 >= 0)
            & (y0 >= 0)
            & (z0 >= 0)
            & (x1 < vol_dim[0])
            & (y1 < vol_dim[1])
            & (z1 < vol_dim[2])
        )
        keep = inb & inb_tri
        if not np.any(keep):
            return vals, valid

        xx = x[keep] - x0[keep]
        yy = y[keep] - y0[keep]
        zz = z[keep] - z0[keep]

        x0k, y0k, z0k = x0[keep], y0[keep], z0[keep]
        x1k, y1k, z1k = x1[keep], y1[keep], z1[keep]

        c000 = tsdf[x0k, y0k, z0k]
        c100 = tsdf[x1k, y0k, z0k]
        c010 = tsdf[x0k, y1k, z0k]
        c110 = tsdf[x1k, y1k, z0k]
        c001 = tsdf[x0k, y0k, z1k]
        c101 = tsdf[x1k, y0k, z1k]
        c011 = tsdf[x0k, y1k, z1k]
        c111 = tsdf[x1k, y1k, z1k]

        c00 = c000 * (1 - xx) + c100 * xx
        c10 = c010 * (1 - xx) + c110 * xx
        c01 = c001 * (1 - xx) + c101 * xx
        c11 = c011 * (1 - xx) + c111 * xx
        c0 = c00 * (1 - yy) + c10 * yy
        c1 = c01 * (1 - yy) + c11 * yy
        interp = c0 * (1 - zz) + c1 * zz

        idx_keep = np.where(keep)[0]
        vals[idx_keep] = interp.astype(np.float64)
        valid[idx_keep] = np.isfinite(interp)
        return vals, valid

    def _query_sdf_and_grad(self, obj, pts_obj):
        pts_obj = np.asarray(pts_obj, dtype=np.float32)
        n = pts_obj.shape[0]
        sdf0, v0 = self._query_sdf_signed(obj, pts_obj)

        if getattr(obj, "sdf", None) is not None and "voxel_size" in obj.sdf:
            voxel = float(obj.sdf["voxel_size"])
        elif getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "_voxel_size"
        ):
            voxel = float(obj.sdf_volume._voxel_size)
        else:
            voxel = 0.005

        eps = float(self._sdf_refine_grad_eps) * voxel
        eps = max(eps, 1e-4)

        E = np.eye(3, dtype=np.float32) * eps
        grad = np.zeros((n, 3), dtype=np.float64)
        valid = v0.copy()
        for k in range(3):
            fp, vp = self._query_sdf_signed(obj, pts_obj + E[k][None, :])
            fm, vm = self._query_sdf_signed(obj, pts_obj - E[k][None, :])
            vk = v0 & vp & vm
            gk = np.zeros((n,), dtype=np.float64)
            gk[vk] = (fp[vk] - fm[vk]) / (2.0 * eps)
            grad[:, k] = gk
            valid &= vk

        gnorm = np.linalg.norm(grad, axis=1)
        valid &= np.isfinite(sdf0) & np.isfinite(gnorm) & (gnorm > 1e-8)
        return sdf0, grad, valid

    def _compute_sdf_inliers(self, r_abs):
        if r_abs.size == 0:
            return np.zeros((0,), dtype=bool), np.inf

        if self._sdf_trim_method == "mad":
            med = np.median(r_abs)
            mad = np.median(np.abs(r_abs - med)) + 1e-12
            thr = float(med + self._sdf_mad_scale * mad)
        elif self._sdf_trim_method == "percentile":
            q = float(np.clip(self._sdf_inlier_percentile, 0.0, 100.0))
            thr = float(np.percentile(r_abs, q))
        elif self._sdf_trim_method == "fixed":
            thr = float(self._sdf_inlier_thres)
        else:
            thr = float(np.percentile(r_abs, 80.0))

        return r_abs <= thr, thr

    def _kernel_weights(self, r_abs):
        if self._sdf_kernel == "none":
            return np.ones_like(r_abs, dtype=np.float64)

        d = float(max(self._sdf_kernel_delta, 1e-12))
        if self._sdf_kernel == "huber":
            w = np.ones_like(r_abs, dtype=np.float64)
            m = r_abs > d
            w[m] = d / (r_abs[m] + 1e-12)
            return w
        if self._sdf_kernel == "cauchy":
            x = r_abs / d
            return 1.0 / (1.0 + x * x)
        return np.ones_like(r_abs, dtype=np.float64)

    def _refine_pose_with_sdf(self, pts_cur, T_cur2obj_init, obj, src_corr=None, tgt_corr=None):
        T = np.asarray(T_cur2obj_init, dtype=np.float64).copy()
        pts_cur = np.asarray(pts_cur, dtype=np.float64)

        dbg = {
            "enabled": True,
            "iters": 0,
            "cost_history": [],
            "support_history": [],
            "inliers_history": [],
            "trim_thr_history": [],
            "accepted": False,
        }

        if pts_cur.shape[0] < self._sdf_refine_min_pts:
            return None, dbg

        best_T = T.copy()
        best_cost = np.inf

        for it in range(self._sdf_refine_iters):
            pts_obj = transform_pts(
                T.astype(np.float32), pts_cur.astype(np.float32)
            ).astype(np.float64)

            sdf_vals, sdf_grads, valid = self._query_sdf_and_grad(
                obj, pts_obj.astype(np.float32)
            )
            valid = np.asarray(valid, dtype=bool)
            support_ratio = float(valid.mean()) if valid.size else 0.0
            dbg["support_history"].append(support_ratio)

            if valid.sum() < self._sdf_refine_min_pts:
                break

            pts_obj_v = pts_obj[valid]
            r_sdf = sdf_vals[valid].astype(np.float64)
            g_sdf = sdf_grads[valid].astype(np.float64)
            r_abs = np.abs(r_sdf)

            in_sdf, thr = self._compute_sdf_inliers(r_abs)
            dbg["trim_thr_history"].append(float(thr))
            if in_sdf.sum() < self._sdf_refine_min_pts:
                break

            pts_obj_in = pts_obj_v[in_sdf]
            r_sdf_in = r_sdf[in_sdf]
            g_sdf_in = g_sdf[in_sdf]

            n = pts_obj_in.shape[0]
            J = np.zeros((n, 6), dtype=np.float64)
            for i in range(n):
                x = pts_obj_in[i]
                G = np.zeros((3, 6), dtype=np.float64)
                G[:, :3] = np.eye(3)
                G[:, 3:] = -_skew(x)
                J[i] = g_sdf_in[i] @ G

            w = self._kernel_weights(np.abs(r_sdf_in))
            sw = np.sqrt(np.clip(w, 0.0, None))[:, None]
            Jw = J * sw
            rw = r_sdf_in * sw[:, 0]
            H = Jw.T @ Jw
            b = Jw.T @ rw
            cost = float(np.mean(rw**2)) if rw.size else np.inf
            if support_ratio < self._sdf_min_support_ratio:
                cost += float(
                    self._sdf_bad_penalty * (self._sdf_min_support_ratio - support_ratio)
                )

            dbg["cost_history"].append(cost)
            dbg["inliers_history"].append(int(in_sdf.sum()))

            H += float(self._sdf_refine_damping) * np.eye(6, dtype=np.float64)
            try:
                dxi = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dxi)):
                break

            dxi[:3] = np.clip(
                dxi[:3], -self._sdf_refine_step_trans_clip, self._sdf_refine_step_trans_clip
            )
            dxi[3:] = np.clip(
                dxi[3:], -self._sdf_refine_step_rot_clip, self._sdf_refine_step_rot_clip
            )

            T_candidate = _left_update_SE3(T, dxi)
            c_cost = self._eval_sdf_cost(pts_cur, T_candidate, obj)
            t_cost = self._eval_sdf_cost(pts_cur, T, obj)

            if np.isfinite(c_cost) and c_cost <= t_cost:
                T = T_candidate
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_T = T.copy()
            else:
                T_half = _left_update_SE3(T, 0.5 * dxi)
                h_cost = self._eval_sdf_cost(pts_cur, T_half, obj)
                if np.isfinite(h_cost) and h_cost < t_cost:
                    T = T_half
                    if h_cost < best_cost:
                        best_cost = h_cost
                        best_T = T.copy()

            dbg["iters"] = it + 1
            if np.linalg.norm(dxi[:3]) < 1e-4 and np.linalg.norm(dxi[3:]) < 1e-3:
                break

        if np.isfinite(best_cost):
            dbg["accepted"] = True
            dbg["best_cost"] = float(best_cost)
            return best_T.astype(np.float32), dbg
        return None, dbg

    def _eval_sdf_cost(self, pts_cur, T_cur2obj, obj):
        pts_obj = transform_pts(
            np.asarray(T_cur2obj, dtype=np.float32), np.asarray(pts_cur, dtype=np.float32)
        )
        vals, valid = self._query_sdf_signed(obj, pts_obj)
        if valid.sum() < self._sdf_refine_min_pts:
            return np.inf

        r = np.abs(vals[valid].astype(np.float64))
        inl, _ = self._compute_sdf_inliers(r)
        if inl.sum() < self._sdf_refine_min_pts:
            return np.inf

        rr = r[inl]
        w = self._kernel_weights(rr)
        support = float(valid.mean())
        cost = float(np.mean(w * (rr**2)))
        if support < self._sdf_min_support_ratio:
            cost += float(self._sdf_bad_penalty * (self._sdf_min_support_ratio - support))
        return cost

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
        return base

    def _sdf_residual_stats(
        self,
        pts_cur,
        T_cur2obj,
        obj,
        robust_percentile=90.0,
        tau=0.01,
    ):
        if (
            obj is None
            or getattr(obj, "sdf", None) is None
            or pts_cur is None
            or pts_cur.shape[0] == 0
        ):
            return np.inf, 0.0, 0.0

        pts_obj = transform_pts(T_cur2obj, pts_cur)
        sdf_vals = None
        support_ratio = 0.0

        # nvblox-style query path
        if getattr(obj, "sdf_volume", None) is not None and hasattr(
            obj.sdf_volume, "query_sdf"
        ):
            qvals = obj.sdf_volume.query_sdf(pts_obj)
            if qvals is not None and qvals.shape[0] == pts_obj.shape[0]:
                finite = np.isfinite(qvals)
                support_ratio = float(np.mean(finite))
                if np.any(finite):
                    sdf_vals = np.abs(qvals[finite])

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
            support_ratio = float(np.mean(inb))
            if np.any(inb):
                vox_in = vox[inb]
                sdf_vals = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]])

        if sdf_vals is None or sdf_vals.size == 0:
            return np.inf, float(support_ratio), 0.0

        q = float(np.clip(robust_percentile, 0.0, 100.0))
        base = float(np.percentile(sdf_vals, q))
        tau = float(max(tau, 1e-9))
        inlier_ratio = float(np.mean(sdf_vals <= tau))
        return base, float(support_ratio), float(inlier_ratio)

    # def _sdf_residual(
    #     self,
    #     pts_cur,
    #     T_cur2obj,
    #     obj,
    #     robust_percentile=70.0,
    #     tau=0.02,
    #     min_support=0.2,
    #     min_inliers=10,
    #     w_support=0.05,
    #     w_inlier=0.05,
    # ):
    #     if (
    #         obj is None
    #         or getattr(obj, "sdf", None) is None
    #         or pts_cur is None
    #         or pts_cur.shape[0] == 0
    #     ):
    #         return np.inf

    #     pts_obj = transform_pts(T_cur2obj, pts_cur)

    #     # --- NVBlox / ESDF-style ---
    #     if getattr(obj, "sdf_volume", None) is not None and hasattr(
    #         obj.sdf_volume, "query_sdf"
    #     ):
    #         qvals = obj.sdf_volume.query_sdf(pts_obj)
    #         if qvals is None or qvals.shape[0] != pts_obj.shape[0]:
    #             return np.inf

    #         finite = np.isfinite(qvals)
    #         support_ratio = float(np.mean(finite))
    #         if support_ratio < min_support:
    #             return np.inf

    #         abs_sdf = np.abs(qvals[finite])
    #         inl = abs_sdf <= tau
    #         if int(np.count_nonzero(inl)) < min_inliers:
    #             return np.inf

    #         q = float(np.clip(robust_percentile, 0.0, 100.0))
    #         base = float(np.percentile(abs_sdf[inl], q))
    #         inlier_ratio = float(np.mean(inl))
    #         return (
    #             base
    #             + w_support * (1.0 - support_ratio)
    #             + w_inlier * (1.0 - inlier_ratio)
    #         )

    #     # --- Dense TSDF grid ---
    #     if "tsdf" in obj.sdf:
    #         tsdf = obj.sdf["tsdf"]
    #         origin = obj.sdf["vol_origin"]
    #         voxel = float(obj.sdf["voxel_size"])
    #         vol_dim = np.array(tsdf.shape, dtype=np.int32)

    #         vox = np.floor((pts_obj - origin[None, :]) / voxel).astype(np.int32)
    #         inb = np.logical_and(
    #             np.all(vox >= 0, axis=1), np.all(vox < vol_dim[None, :], axis=1)
    #         )
    #         support_ratio = float(np.mean(inb))
    #         if support_ratio < min_support:
    #             return np.inf

    #         vox_in = vox[inb]
    #         abs_sdf = np.abs(tsdf[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]])

    #         inl = abs_sdf <= tau
    #         if int(np.count_nonzero(inl)) < min_inliers:
    #             return np.inf

    #         q = float(np.clip(robust_percentile, 0.0, 100.0))
    #         base = float(np.percentile(abs_sdf[inl], q))
    #         inlier_ratio = float(np.mean(inl))
    #         return (
    #             base
    #             + w_support * (1.0 - support_ratio)
    #             + w_inlier * (1.0 - inlier_ratio)
    #         )

    #     return np.inf
