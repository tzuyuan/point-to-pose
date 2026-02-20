import bisect
import numpy as np

import gtsam

from point2pose.core.module_registry import OPTIMIZER
from point2pose.core.base_optimizer import Optimizer
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.utils.transform import inverse_SE3


@OPTIMIZER.register_module("lm_graph_sdf")
class LMGraphSDFOptimizer(Optimizer):
    """
    A generic graph-based optimizer using Levenberg-Marquardt.

    This module maintains an internal GTSAM NonlinearFactorGraph and Values
    container. Each call to `optimize()` adds new variables and factors based on
    incoming data and then performs a batch nonlinear optimization.

    This class mirrors the structure of the ISAM2-based optimizer but solves the
    problem using LM instead of incremental updates.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "lm_graph_sdf_optimizer"

        # Persistent factor graph and variable initializations
        self._graph = gtsam.NonlinearFactorGraph()
        self._values = gtsam.Values()

        # Prior on the first pose
        prior_noise_param = config.get(
            "prior_noise_param", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_noise_param, dtype=float)
        )

        # LM parameters
        self._lm_params = gtsam.LevenbergMarquardtParams()
        self._lm_params.setMaxIterations(config.get("max_iterations", 20))
        self._lm_params.setRelativeErrorTol(config.get("relative_error_tol", 1e-5))
        self._lm_params.setAbsoluteErrorTol(config.get("absolute_error_tol", 1e-5))
        self._lm_params.setlambdaInitial(config.get("lambda_initial", 1e-1))
        # self._lm_params.setlambdaFactor(config.get("lambda_factor", 2.0))

        # self._lm_params.setVerbosityLM("TRYLAMBDA")
        self._lm_params.setVerbosityLM("SUMMARY")

        # ------------------------------------------------------------------
        # Optional formulation knobs (all default to preserving old behavior)
        # ------------------------------------------------------------------
        # 1) Pose-chain constraint (odometry) to reduce pose freedom so map/landmarks
        #    have to absorb inconsistency across frames.
        self._use_between_factor = bool(config.get("use_between_factor", False))
        between_noise_param = config.get("between_noise_param", None)
        self._between_noise = None
        if between_noise_param is not None:
            self._between_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(between_noise_param, dtype=float)
            )

        # 2) Landmark measurement noise (BearingRangeFactor3D has 3 residual dims:
        #    2 for bearing + 1 for range). Previously this was effectively [1,1,1].
        self._landmark_noise_param = config.get("landmark_noise_param", None)  # (3,)
        self._landmark_use_robust = bool(config.get("landmark_use_robust", True))
        self._landmark_huber_k = float(config.get("landmark_huber_k", 1.345))
        self._landmark_sigma_scale_by = str(
            config.get("landmark_sigma_scale_by", "none")
        ).lower()  # {"none","residual","uncertainty"}
        self._landmark_sigma_min = float(config.get("landmark_sigma_min", 1e-6))

        # 3) Optional SDF factors on newly created landmarks.
        self._use_sdf_factors = bool(config.get("use_sdf_factors", True))
        self._sdf_noise_sigma = float(config.get("sdf_noise_sigma", 0.02))
        self._sdf_use_robust = bool(config.get("sdf_use_robust", True))
        self._sdf_huber_k = float(config.get("sdf_huber_k", 1.345))
        self._sdf_fd_step = float(config.get("sdf_fd_step", 0.5))
        self._sdf_truncation_abs = float(config.get("sdf_truncation_abs", 0.2))

        sdf_base_noise = gtsam.noiseModel.Isotropic.Sigma(1, self._sdf_noise_sigma)
        if self._sdf_use_robust:
            self._sdf_noise = gtsam.noiseModel.Robust(
                gtsam.noiseModel.mEstimator.Huber(self._sdf_huber_k), sdf_base_noise
            )
        else:
            self._sdf_noise = sdf_base_noise

        # Bookkeeping
        self._inserted_poses = set()
        self._inserted_landmarks = set()
        self.inserted_landmark_ids = []

        self._initialized = False
        self._prev_pose_inv = np.eye(4)
        self._prev_frame_id = -1
        self._prev_rel_T = None

        self.Xo = gtsam.symbol("x", 0)

        # Active SDF context for this frame; populated in optimize().
        self._active_sdf = None
        self._active_sdf_volume = None

    def _trilinear_interpolate_tsdf(self, pts_obj: np.ndarray):
        """Trilinear interpolation on dense TSDF grid; returns (vals, valid_mask)."""
        if self._active_sdf is None or "tsdf" not in self._active_sdf:
            n = pts_obj.shape[0]
            return np.zeros((n,), dtype=float), np.zeros((n,), dtype=bool)

        tsdf = np.asarray(self._active_sdf["tsdf"], dtype=float)
        origin = np.asarray(self._active_sdf["vol_origin"], dtype=float).reshape(3)
        voxel = float(self._active_sdf["voxel_size"])
        dims = np.asarray(tsdf.shape, dtype=np.int32)

        gx = (pts_obj[:, 0] - origin[0]) / voxel
        gy = (pts_obj[:, 1] - origin[1]) / voxel
        gz = (pts_obj[:, 2] - origin[2]) / voxel

        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        z0 = np.floor(gz).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        valid = (
            (x0 >= 0)
            & (y0 >= 0)
            & (z0 >= 0)
            & (x1 < dims[0])
            & (y1 < dims[1])
            & (z1 < dims[2])
        )
        vals = np.zeros((pts_obj.shape[0],), dtype=float)
        if not np.any(valid):
            return vals, valid

        xv = gx[valid] - x0[valid]
        yv = gy[valid] - y0[valid]
        zv = gz[valid] - z0[valid]

        v000 = tsdf[x0[valid], y0[valid], z0[valid]]
        v100 = tsdf[x1[valid], y0[valid], z0[valid]]
        v010 = tsdf[x0[valid], y1[valid], z0[valid]]
        v110 = tsdf[x1[valid], y1[valid], z0[valid]]
        v001 = tsdf[x0[valid], y0[valid], z1[valid]]
        v101 = tsdf[x1[valid], y0[valid], z1[valid]]
        v011 = tsdf[x0[valid], y1[valid], z1[valid]]
        v111 = tsdf[x1[valid], y1[valid], z1[valid]]

        c00 = v000 * (1.0 - xv) + v100 * xv
        c10 = v010 * (1.0 - xv) + v110 * xv
        c01 = v001 * (1.0 - xv) + v101 * xv
        c11 = v011 * (1.0 - xv) + v111 * xv
        c0 = c00 * (1.0 - yv) + c10 * yv
        c1 = c01 * (1.0 - yv) + c11 * yv
        vals[valid] = c0 * (1.0 - zv) + c1 * zv
        return vals, valid

    def _query_sdf_only(self, pts_obj: np.ndarray):
        """Query SDF values from accelerated backend if available, else dense TSDF."""
        pts = np.asarray(pts_obj, dtype=float).reshape(-1, 3)
        n = pts.shape[0]
        vals = np.zeros((n,), dtype=float)
        valid = np.zeros((n,), dtype=bool)

        if self._active_sdf_volume is not None:
            for name in ["query_sdf", "query_tsdf", "interpolate_tsdf"]:
                fn = getattr(self._active_sdf_volume, name, None)
                if fn is None:
                    continue
                try:
                    out = fn(pts.astype(np.float32))
                except Exception:
                    continue
                out = np.asarray(out).reshape(-1)
                if out.shape[0] != n:
                    continue
                finite = np.isfinite(out)
                vals[finite] = out[finite]
                valid[finite] = True
                return vals, valid

            mapper = getattr(self._active_sdf_volume, "_mapper", None)
            if mapper is not None:
                for name in ["query_sdf", "query_tsdf", "interpolate_tsdf"]:
                    fn = getattr(mapper, name, None)
                    if fn is None:
                        continue
                    try:
                        out = fn(pts.astype(np.float32))
                    except Exception:
                        continue
                    out = np.asarray(out).reshape(-1)
                    if out.shape[0] != n:
                        continue
                    finite = np.isfinite(out)
                    vals[finite] = out[finite]
                    valid[finite] = True
                    return vals, valid

        tsdf_vals, tsdf_valid = self._trilinear_interpolate_tsdf(pts)
        vals[tsdf_valid] = tsdf_vals[tsdf_valid]
        valid[tsdf_valid] = True
        return vals, valid

    def _query_sdf_and_grad(self, pts_obj: np.ndarray):
        """
        Query SDF and gradient.
        Prefers backend accelerated gradient APIs; falls back to finite-difference
        using batched SDF queries (still accelerated when backend query exists).
        """
        pts = np.asarray(pts_obj, dtype=float).reshape(-1, 3)
        n = pts.shape[0]
        vals = np.zeros((n,), dtype=float)
        grads = np.zeros((n, 3), dtype=float)
        valid = np.zeros((n,), dtype=bool)

        # 1) direct gradient query from backend if available
        targets = []
        if self._active_sdf_volume is not None:
            targets.append(self._active_sdf_volume)
            mapper = getattr(self._active_sdf_volume, "_mapper", None)
            if mapper is not None:
                targets.append(mapper)

        query_names = [
            "query_sdf_with_grad",
            "query_sdf_with_gradient",
            "query_sdf_and_gradient",
            "query_tsdf_with_grad",
            "query_tsdf_and_gradient",
        ]
        for target in targets:
            for name in query_names:
                fn = getattr(target, name, None)
                if fn is None:
                    continue
                try:
                    out = fn(pts.astype(np.float32))
                except Exception:
                    continue

                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    vv = np.asarray(out[0]).reshape(-1)
                    gg = np.asarray(out[1]).reshape(-1, 3)
                    if vv.shape[0] != n or gg.shape[0] != n:
                        continue
                    finite = np.isfinite(vv) & np.all(np.isfinite(gg), axis=1)
                    vals[finite] = vv[finite]
                    grads[finite] = gg[finite]
                    valid[finite] = True
                    return vals, grads, valid

        # 2) fallback: central-difference gradient with batched queries
        vals0, valid0 = self._query_sdf_only(pts)
        if not np.any(valid0):
            return vals, grads, valid

        if self._active_sdf is not None and "voxel_size" in self._active_sdf:
            h = float(self._active_sdf["voxel_size"]) * self._sdf_fd_step
        else:
            h = 0.5 * max(float(self._sdf_noise_sigma), 1e-3)

        h = max(h, 1e-5)
        offsets = np.array(
            [[h, 0.0, 0.0], [-h, 0.0, 0.0], [0.0, h, 0.0], [0.0, -h, 0.0], [0.0, 0.0, h], [0.0, 0.0, -h]],
            dtype=float,
        )
        pts6 = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        vals6, valid6 = self._query_sdf_only(pts6)
        vals6 = vals6.reshape(n, 6)
        valid6 = valid6.reshape(n, 6)

        good = valid0 & np.all(valid6, axis=1)
        grads[good, 0] = (vals6[good, 0] - vals6[good, 1]) / (2.0 * h)
        grads[good, 1] = (vals6[good, 2] - vals6[good, 3]) / (2.0 * h)
        grads[good, 2] = (vals6[good, 4] - vals6[good, 5]) / (2.0 * h)
        vals[good] = vals0[good]
        valid[good] = True
        return vals, grads, valid

    def _make_sdf_factor(self, landmark_key):
        key_vec = gtsam.KeyVector()
        key_vec.push_back(landmark_key)
        trunc = abs(self._sdf_truncation_abs)

        def sdf_error(_factor, values, jacobians):
            p = np.asarray(values.atPoint3(landmark_key), dtype=float).reshape(1, 3)
            sdf_val_arr, sdf_grad_arr, valid_arr = self._query_sdf_and_grad(p)
            if not valid_arr[0]:
                if jacobians is not None:
                    jacobians[0] = np.zeros((1, 3), dtype=float)
                return np.array([0.0], dtype=float)

            sdf_val = float(np.clip(sdf_val_arr[0], -trunc, trunc))
            sdf_grad = np.asarray(sdf_grad_arr[0], dtype=float)
            if jacobians is not None:
                jacobians[0] = sdf_grad.reshape(1, 3)
            return np.array([sdf_val], dtype=float)

        return gtsam.CustomFactor(self._sdf_noise, key_vec, sdf_error)

    def get_num_poses(self) -> int:
        """Return the number of pose variables currently in the graph."""
        return len(self._inserted_poses)

    def optimize(self, data: ObjectFrameData):
        """
        Add new variables and factors to the graph based on incoming data, then
        perform a nonlinear optimization using Levenberg–Marquardt.
        """
        frame_id = data.frame_id
        self._active_sdf = getattr(data, "sdf", None)
        self._active_sdf_volume = getattr(data, "sdf_volume", None)

        # input pose is world_T_cam; optimization uses its inverse
        cur_pose_c2w = inverse_SE3(data.pose)
        cur_pose_c2w_gtsam = gtsam.Pose3(cur_pose_c2w)

        Xi = gtsam.symbol("x", frame_id)

        # Insert or update pose variable
        if Xi not in self._inserted_poses:
            self._values.insert(Xi, cur_pose_c2w_gtsam)
            self._inserted_poses.add(Xi)
            self.Xo = Xi

        # Add prior on first pose or between factor afterward
        if not self._initialized:
            self._graph.push_back(
                gtsam.PriorFactorPose3(Xi, cur_pose_c2w_gtsam, self._prior_noise)
            )
            # print(
            #     "initializing with :",
            #     cur_pose_c2w_gtsam,
            #     "and noise",
            #     self._prior_noise,
            # )
            # self._prev_rel_T = gtsam.Pose3(np.eye(4))
        elif data.rel_pose is not None:
            prev_id = self._prev_frame_id
            X_prev = gtsam.symbol("x", prev_id)

            # relative pose is from previous frame to current frame
            # we need to invert it to get the current frame to the previous frame
            rel_T_cim12ci = gtsam.Pose3(inverse_SE3(data.rel_pose))

            T_ci2o = gtsam.Pose3(inverse_SE3(data.pose))

            # Optional: BetweenFactorPose3 to constrain pose chain
            # (this makes landmark/map correction more likely, because pose can't
            # independently absorb all per-frame inconsistencies).
            if self._use_between_factor and X_prev in self._inserted_poses:
                res = (
                    np.mean(data.reg_residuals)
                    if getattr(data, "reg_residuals", None) is not None
                    and np.size(data.reg_residuals) > 0
                    else 1.0
                )
                if self._between_noise is not None:
                    between_noise = self._between_noise
                else:
                    # Safe fallback (very loose) if user enabled between factors
                    # but didn't provide between_noise_param.
                    between_scale = max(1.0, res / 0.005)
                    between_noise = gtsam.noiseModel.Diagonal.Sigmas(
                        between_scale
                        * np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05], dtype=float)
                    )

                    between_noise = gtsam.noiseModel.Robust(
                        gtsam.noiseModel.mEstimator.Huber(1.0),
                        between_noise,
                    )

                if res < 0.01:
                    self._graph.push_back(
                        gtsam.BetweenFactorPose3(
                            X_prev, Xi, rel_T_cim12ci, between_noise
                        )
                    )

            # between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            #     np.array(
            #         [
            #             sigma_between * 100,
            #             sigma_between * 100,
            #             sigma_between * 100,
            #             sigma_between * 50,
            #             sigma_between * 50,
            #             sigma_between * 50,
            #         ],
            #         dtype=float,
            #     )
            # )

            # self._graph.push_back(
            #     gtsam.BetweenFactorPose3(Xi, self.Xo, T_ci2o, between_noise)
            # )

            # # between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            # #     np.array([5, 5, 5, 0.5, 0.5, 0.5], dtype=float)
            # # )
            # # between_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)

            # self._graph.push_back(
            #     gtsam.BetweenFactorPose3(X_prev, Xi, rel_T_cim12ci, between_noise)
            # )

            # self._prev_rel_T = rel_T_cim12ci

        self._prev_pose_inv = data.pose
        self._prev_frame_id = frame_id

        # Insert landmark variables and their factors
        if data.reg_inliers.size > 0:  # and self._initialized:
            # Choose pose used for seeding new landmarks
            if self._values.exists(Xi):
                try:
                    seed_pose = self._values.atPose3(Xi)
                except RuntimeError:
                    seed_pose = cur_pose_c2w_gtsam
            else:
                seed_pose = cur_pose_c2w_gtsam

            assert (
                len(data.reg_valid_idx)
                == len(data.reg_inliers)
                == len(data.reg_residuals)
            ), "inliers/residuals must be aligned with reg_valid_idx if indexed by mask"

            for m, lid in enumerate(data.reg_valid_idx):
                if not data.reg_inliers[m] or np.isnan(data.reg_cur_3d[m]).any():
                    continue

                z_cam = data.reg_cur_3d[m]
                Lj = gtsam.symbol("l", int(lid))

                # z_range = float(np.linalg.norm(z_cam))
                # sigma_bearing = 0.007 / max(z_range, 1e-3)
                # --- Landmark noise model ---
                diag = (
                    np.asarray(self._landmark_noise_param, dtype=float)
                    if self._landmark_noise_param is not None
                    else np.array([0.04, 0.04, 0.07], dtype=float)
                )

                sigma_scale = 1.0
                if self._landmark_sigma_scale_by == "residual":
                    if (
                        getattr(data, "reg_residuals", None) is not None
                        and data.reg_residuals.size > m
                    ):
                        res = float(data.reg_residuals[m])
                        # reference residual in meters (tune this)
                        res_ref = getattr(
                            self, "_landmark_residual_ref", 0.01
                        )  # 1 cm default
                        sigma_scale = max(0.3, res / max(res_ref, 1e-6))

                elif self._landmark_sigma_scale_by == "uncertainty":
                    if (
                        getattr(data, "reg_uncertainties", None) is not None
                        and data.reg_uncertainties.size > m
                    ):
                        u = float(data.reg_uncertainties[m])
                        u_ref = getattr(self, "_landmark_uncert_ref", 0.1)
                        sigma_scale = max(1.5, u / max(u_ref, 1e-6))

                elif self._landmark_sigma_scale_by == "residual_and_uncertainty":
                    res_scale = 1.0
                    unc_scale = 1.0
                    if (
                        getattr(data, "reg_residuals", None) is not None
                        and data.reg_residuals.size > m
                    ):
                        res = float(data.reg_residuals[m])
                        res_ref = getattr(
                            self, "_landmark_residual_ref", 0.01
                        )  # 1 cm default
                        res_scale = max(0.3, res / max(res_ref, 1e-6))

                    if (
                        getattr(data, "reg_uncertainties", None) is not None
                        and data.reg_uncertainties.size > m
                    ):
                        u = float(data.reg_uncertainties[m])
                        u_ref = getattr(self, "_landmark_uncert_ref", 0.1)
                        unc_scale = max(1.5, u / max(u_ref, 1e-6))

                    sigma_scale = res_scale * unc_scale

                # sigma_scale = 1
                # if self._landmark_sigma_scale_by == "residual":
                #     if (
                #         getattr(data, "reg_residuals", None) is not None
                #         and data.reg_residuals.size > m
                #     ):
                #         sigma_scale = float(
                #             max(
                #                 self._landmark_sigma_min,
                #                 float(data.reg_residuals[m]),
                #             )
                #         )
                # elif self._landmark_sigma_scale_by == "uncertainty":
                #     if (
                #         getattr(data, "reg_uncertainties", None) is not None
                #         and data.reg_uncertainties.size > m
                #     ):
                #         sigma_scale = float(
                #             max(
                #                 self._landmark_sigma_min,
                #                 float(data.reg_uncertainties[m]),
                #             )
                #         )
                # elif self._landmark_sigma_scale_by == "residual_and_uncertainty":
                #     res_scale = 1
                #     unc_scale = 1
                #     if (
                #         getattr(data, "reg_residuals", None) is not None
                #         and data.reg_residuals.size > m
                #     ):
                #         res_scale = float(
                #             max(
                #                 self._landmark_sigma_min,
                #                 float(data.reg_residuals[m]),
                #             )
                #         )
                #     if (
                #         getattr(data, "reg_uncertainties", None) is not None
                #         and data.reg_uncertainties.size > m
                #     ):
                #         unc_scale = float(
                #             max(
                #                 self._landmark_sigma_min,
                #                 float(data.reg_uncertainties[m]),
                #             )
                #         )
                #     sigma_scale = res_scale * unc_scale

                base_noise = gtsam.noiseModel.Diagonal.Sigmas(diag * sigma_scale)
                if self._landmark_use_robust:
                    point_noise = gtsam.noiseModel.Robust(
                        gtsam.noiseModel.mEstimator.Huber(self._landmark_huber_k),
                        base_noise,
                    )
                    # point_noise = gtsam.noiseModel.Robust(
                    #     gtsam.noiseModel.mEstimator.Cauchy(1.0),
                    #     base_noise,
                    # )
                else:
                    point_noise = base_noise

                # point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
                # point_noise = gtsam.noiseModel.Diagonal.Sigmas(
                #     np.array([0.1, 0.1, 0.3], dtype=float)
                # )
                # # )

                # Create landmark if missing
                if Lj not in self._inserted_landmarks:
                    pw = seed_pose.transformFrom(gtsam.Point3(*z_cam))
                    self._values.insert(Lj, pw)
                    self._inserted_landmarks.add(Lj)
                    bisect.insort(self.inserted_landmark_ids, int(lid))

                    # Add SDF unary constraint only when landmark is first added.
                    if self._use_sdf_factors and (
                        self._active_sdf_volume is not None or self._active_sdf is not None
                    ):
                        self._graph.push_back(self._make_sdf_factor(Lj))

                z_range = float(np.linalg.norm(z_cam))
                if z_range <= 1e-9:
                    continue

                z_bearing = gtsam.Unit3(z_cam / z_range)

                self._graph.push_back(
                    gtsam.BearingRangeFactor3D(Xi, Lj, z_bearing, z_range, point_noise)
                )

        # Skip LM on first call
        if not self._initialized:
            self._initialized = True
            return None

        # Run LM optimization
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self._graph, self._values, self._lm_params
            )
            result = optimizer.optimize()
        except RuntimeError:
            print(f"[LMGraphSDFOptimizer] Optimization failed for frame {frame_id}")
            return None

        # Update state with optimized values
        self._values = result

        # Extract optimized pose
        try:
            Xi_hat = result.atPose3(Xi)
        except RuntimeError:
            return None

        pose_opt = inverse_SE3(Xi_hat.matrix())

        ### Extract optimized landmarks
        # landmark_xyz = gtsam.utilities.extractPoint3(result)

        num_L = len(self.inserted_landmark_ids)
        landmark_xyz = np.empty((num_L, 3), dtype=float)
        ids = np.empty((num_L,), dtype=np.int64)

        k = 0
        for lid in self.inserted_landmark_ids:
            Lj = gtsam.symbol("l", int(lid))
            if not result.exists(Lj):
                continue

            p = np.asarray(result.atPoint3(Lj), dtype=float).reshape(
                3,
            )  # (3,)
            landmark_xyz[k, :] = p
            ids[k] = int(lid)
            k += 1

        landmark_xyz = landmark_xyz[:k, :]
        ids = ids[:k]  # keep as np array (or ids.tolist() if you prefer a list)

        print(
            f"[LMGraphSDFOptimizer] Optimized frame {frame_id} for object {data.obj_id}, "
            f"{len(self._inserted_poses)} frames and {landmark_xyz.shape[0]} landmarks"
        )

        # Export all optimized poses so downstream can update every keyframe state.
        pose_frame_ids = []
        poses = []
        for Xj in sorted(
            self._inserted_poses, key=lambda k: int(gtsam.Symbol(k).index())
        ):
            if not result.exists(Xj):
                continue
            fid = int(gtsam.Symbol(Xj).index())
            pose_j = inverse_SE3(result.atPose3(Xj).matrix())
            pose_frame_ids.append(fid)
            poses.append(pose_j)

        if poses:
            poses_optimized = np.asarray(poses, dtype=float)
            pose_frame_ids_optimized = np.asarray(pose_frame_ids, dtype=np.int64)
        else:
            poses_optimized = np.zeros((0, 4, 4), dtype=float)
            pose_frame_ids_optimized = np.zeros((0,), dtype=np.int64)

        return OptimizerResult(
            obj_id=data.obj_id,
            frame_id=frame_id,
            pose_optimized=pose_opt,
            key_points_optimized=landmark_xyz,
            key_points_idx_optimized=ids,
            poses_optimized=poses_optimized,
            pose_frame_ids_optimized=pose_frame_ids_optimized,
        )
