import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.lie import log_SE3
from point2pose.utils.camera import (
    compute_projection_consistency,
    extract_cropped_point_cloud,
)

try:
    import cv2
except Exception as e:
    cv2 = None


@REGISTER.register_module("pnp_cluster_ransac")
class PnPClusterRANSACRegister(Register):
    """
    Robust PnP + multi-consensus extraction (cluster peeling).

    What you must provide:
      - `img_pts`: (N,2) 2D pixels in the *current* image for each correspondence
      - `K`: (3,3) camera intrinsics
      - optionally `dist`: (k,) distortion coeffs (OpenCV order), default zeros

    Coordinate convention:
      OpenCV solvePnP estimates T_obj2cam (object/world -> camera).
      This class can return either:
        - "cam2obj": T_cam2obj  (camera -> object/world)  [default]
        - "obj2cam": T_obj2cam  (object/world -> camera)

    IMPORTANT:
      Your original SVD code returns a transform that is used elsewhere in your stack
      (e.g., `compute_projection_consistency`). If you’re unsure which convention your
      downstream expects, keep BOTH from stats and swap via `T_convention`.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.type = "pnp_cluster_ransac"

        # RANSAC & inlier settings (pixel domain)
        self._ransac_iters = int(config.get("ransac_iters", 200))
        self._reproj_thres_px = float(config.get("reproj_thres_px", 3.0))
        self._ransac_conf = float(config.get("ransac_confidence", 0.999))
        self._min_inliers = int(config.get("min_inliers", 12))
        self._max_clusters = int(config.get("max_clusters", 10))

        # Refinement
        self._refine = bool(config.get("refine", True))
        self._refine_method = str(config.get("refine_method", "ITERATIVE")).upper()

        # PnP method
        # Good defaults:
        #   EPNP is fast for RANSAC hypotheses; ITERATIVE is good for refinement.
        self._pnp_ransac_method = str(config.get("pnp_ransac_method", "EPNP")).upper()

        # Returned transform convention
        self._T_convention = str(config.get("T_convention", "obj2cam")).lower()
        assert self._T_convention in ("cam2obj", "obj2cam")

        # Small degeneracy guards
        self._min_image_span_px = float(config.get("min_image_span_px", 20.0))
        self._min_3d_span_m = float(config.get("min_3d_span_m", 0.02))

        np.random.seed(0)

    def register(
        self,
        src_pcd,  # kept for interface compatibility (you may not need it for PnP)
        tgt_pcd,  # object/world 3D points: (N,3)  (must correspond to img_pts)
        init_pose=None,  # optional prior (used only for fallback selection)
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
        prev_T=None,
        prev_frame=None,
        cur_frame=None,
        obj_id=0,
        mode="f2m",
        img_pts=None,  # (N,2) required
        dist=None,  # optional
    ):

        K = cur_frame.intrinsics

        stats = {}

        img_pts = np.asarray(img_pts, dtype=np.float64)
        src_pcd = np.asarray(src_pcd, dtype=np.float64)
        tgt_pcd = np.asarray(tgt_pcd, dtype=np.float64)

        if dist is None:
            dist = np.zeros((5,), dtype=np.float64)
        else:
            dist = np.asarray(dist, dtype=np.float64).reshape(-1)

        N = src_pcd.shape[0]
        if img_pts.shape[0] != N:
            raise ValueError(
                f"img_pts has {img_pts.shape[0]} rows but tgt_pcd has {N}."
            )

        remaining = np.ones(N, dtype=bool)
        candidates = []

        for _c in range(self._max_clusters):
            cand = self._RANSAC_PnP(
                obj_pts=src_pcd,
                img_pts=img_pts,
                K=K,
                dist=dist,
                remaining=remaining,
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
            stats["reproj_errors"] = np.ones(N) * -1.0
            stats["residuals"] = np.ones(N) * -1.0
            stats["T_obj2cam"] = None
            stats["T_cam2obj"] = None
            return T0, stats

        # Choose best cluster:
        # 1) If you have your dense projection consistency machinery, keep it.
        # 2) Otherwise default to (most inliers, lowest mean reprojection).
        reproj_consistency_scores = None
        if cur_frame is not None and prev_frame is not None:
            reproj_consistency_scores = []
            src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id)

            for c in candidates:
                # c["T"] follows your chosen convention; for projection consistency
                # your existing code expects some T_cur2prev-ish transform.
                #
                # We keep your original logic, but store both forms for debugging.
                if self._T_convention == "cam2obj":
                    # T is cam->obj/world; invert gives obj/world->cam
                    T_used = c["T"]
                else:
                    # T is obj/world->cam; invert gives cam->obj/world
                    T_used = inverse_SE3(c["T"])

                # keep the original code path (may be swapped in your project)
                if mode == "f2f":
                    T_cur2prev = T_used
                elif mode == "f2m":
                    T_cur2prev = prev_T @ T_used if prev_T is not None else T_used
                else:
                    T_cur2prev = T_used

                err = compute_projection_consistency(
                    src_pcd_full.copy(), T_cur2prev, prev_frame, obj_id=0
                )
                reproj_consistency_scores.append(float(err))
                c["reproj_error_dense"] = float(err)

            best_cluster_idx = int(np.argmin(reproj_consistency_scores))
        elif init_pose is not None:
            ref_T = prev_T if prev_T is not None else init_pose
            d = [self._pose_dist(c["T"], ref_T) for c in candidates]
            best_cluster_idx = int(np.argmin(d))
        else:
            # fallback: most inliers, then lowest mean reprojection
            best_cluster_idx = int(
                np.lexsort(
                    (
                        [c["mean_reproj"] for c in candidates],
                        [-c["ninliers"] for c in candidates],
                    )
                )[0]
            )

        # Build output stats
        inliers = np.zeros(N, dtype=bool)
        inliers[candidates[best_cluster_idx]["inliers"]] = True

        # Per-point reprojection error under the chosen best pose (for all points)
        best = candidates[best_cluster_idx]
        per_pt_reproj = self._compute_reproj_errors(
            obj_pts=tgt_pcd,
            img_pts=img_pts,
            K=K,
            dist=dist,
            rvec=best["rvec"],
            tvec=best["tvec"],
        )

        stats["clusters"] = candidates
        stats["best_cluster_idx"] = best_cluster_idx
        stats["remaining_mask"] = remaining
        stats["inliers"] = inliers
        stats["reproj_errors"] = per_pt_reproj
        if reproj_consistency_scores is not None:
            stats["reproj_errors_dense"] = np.array(
                reproj_consistency_scores, dtype=float
            )

        # Always store both transforms for sanity/debug
        stats["T_obj2cam"] = best["T_obj2cam"]
        stats["T_cam2obj"] = best["T_cam2obj"]

        # Compute 3D residuals
        src_cam = transform_pts(best["T_obj2cam"], src_pcd)
        residuals = np.linalg.norm(src_cam - tgt_pcd, axis=1)
        # ignore points without depth
        residuals[tgt_pcd[:, 2] <= 1e-6] = -1.0
        stats["residuals"] = residuals

        return best["T"], stats

    def _RANSAC_PnP(self, obj_pts, img_pts, K, dist, remaining):
        idx = np.where(remaining)[0]
        if idx.size < self._min_inliers or idx.size < 4:
            return None

        obj = obj_pts[idx]
        img = img_pts[idx]

        if not self._passes_spread_checks(obj, img):
            # if the remaining pool is degenerate, stop peeling clusters
            return None

        obj_cv = obj.reshape(-1, 1, 3).astype(np.float64)
        img_cv = img.reshape(-1, 1, 2).astype(np.float64)

        flags = self._cv_pnp_flag(self._pnp_ransac_method)

        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            objectPoints=obj_cv,
            imagePoints=img_cv,
            cameraMatrix=K,
            distCoeffs=dist,
            iterationsCount=int(self._ransac_iters),
            reprojectionError=float(self._reproj_thres_px),
            confidence=float(self._ransac_conf),
            flags=flags,
        )

        if (not ok) or (inl is None):
            return None

        inl = inl.reshape(-1).astype(int)
        inlier_idx = idx[inl]
        if inlier_idx.size < self._min_inliers:
            return None

        # Optional refine on inliers
        if self._refine:
            rvec, tvec = self._refine_pnp(
                obj_pts=obj_pts[inlier_idx],
                img_pts=img_pts[inlier_idx],
                K=K,
                dist=dist,
                rvec=rvec,
                tvec=tvec,
            )

        # Evaluate mean reprojection on inliers
        reproj_inl = self._compute_reproj_errors(
            obj_pts=obj_pts[inlier_idx],
            img_pts=img_pts[inlier_idx],
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
        )
        mean_reproj = float(np.mean(reproj_inl)) if reproj_inl.size else 1e18

        # Score: inliers - alpha * mean_reproj (pixel)
        score = float(inlier_idx.size) - 0.5 * mean_reproj

        # Build transforms
        T_obj2cam = self._rt_to_T(rvec, tvec)  # world/object -> camera
        T_cam2obj = inverse_SE3(T_obj2cam)  # camera -> world/object

        # Choose what to return as "T"
        T = T_cam2obj if self._T_convention == "cam2obj" else T_obj2cam

        # Peel this cluster
        remaining[inlier_idx] = False

        return {
            "T": T,
            "inliers": inlier_idx,
            "ninliers": int(inlier_idx.size),
            "mean_reproj": mean_reproj,
            "score": score,
            "rvec": rvec.reshape(3, 1).astype(np.float64),
            "tvec": tvec.reshape(3, 1).astype(np.float64),
            "T_obj2cam": T_obj2cam,
            "T_cam2obj": T_cam2obj,
        }

    def _refine_pnp(self, obj_pts, img_pts, K, dist, rvec, tvec):
        obj_cv = obj_pts.reshape(-1, 1, 3).astype(np.float64)
        img_cv = img_pts.reshape(-1, 1, 2).astype(np.float64)

        # ITERATIVE is usually best for refinement
        if self._refine_method == "ITERATIVE":
            flag = cv2.SOLVEPNP_ITERATIVE
        elif self._refine_method == "EPNP":
            flag = cv2.SOLVEPNP_EPNP
        else:
            flag = cv2.SOLVEPNP_ITERATIVE

        ok, rvec2, tvec2 = cv2.solvePnP(
            objectPoints=obj_cv,
            imagePoints=img_cv,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=flag,
        )
        if ok:
            return rvec2, tvec2
        return rvec, tvec

    def _compute_reproj_errors(self, obj_pts, img_pts, K, dist, rvec, tvec):
        obj_cv = obj_pts.reshape(-1, 1, 3).astype(np.float64)
        proj, _ = cv2.projectPoints(obj_cv, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_pts.astype(np.float64), axis=1)
        return err

    def _rt_to_T(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        t = tvec.reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _cv_pnp_flag(self, name):
        name = name.upper()
        if name == "EPNP":
            return cv2.SOLVEPNP_EPNP
        if name == "AP3P":
            return cv2.SOLVEPNP_AP3P
        if name == "P3P":
            return cv2.SOLVEPNP_P3P
        if name == "ITERATIVE":
            return cv2.SOLVEPNP_ITERATIVE
        # default
        return cv2.SOLVEPNP_EPNP

    def _passes_spread_checks(self, obj_pts, img_pts):
        # quick degeneracy checks: need enough 2D/3D spread
        if obj_pts.shape[0] < 4:
            return False
        span_img = float(np.max(np.linalg.norm(img_pts - img_pts.mean(axis=0), axis=1)))
        span_obj = float(np.max(np.linalg.norm(obj_pts - obj_pts.mean(axis=0), axis=1)))
        if span_img < self._min_image_span_px:
            return False
        if span_obj < self._min_3d_span_m:
            return False
        return True

    def _pose_dist(self, Ta, Tb):
        return np.linalg.norm(log_SE3(Ta @ inverse_SE3(Tb)))
