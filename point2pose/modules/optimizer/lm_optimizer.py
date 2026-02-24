import bisect
import numpy as np

import gtsam

from point2pose.core.module_registry import OPTIMIZER
from point2pose.core.base_optimizer import Optimizer
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.utils.transform import transform_pts, inverse_SE3


@OPTIMIZER.register_module("lm_graph")
class LMGraphOptimizer(Optimizer):
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
        self.name = "lm_graph_optimizer"

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

        # Bookkeeping
        self._inserted_poses = set()
        self._inserted_landmarks = set()
        self.inserted_landmark_ids = []

        self._initialized = False
        self._prev_pose_inv = np.eye(4)
        self._prev_frame_id = -1
        self._prev_rel_T = None

        self.Xo = gtsam.symbol("x", 0)

    def get_num_poses(self) -> int:
        """Return the number of pose variables currently in the graph."""
        return len(self._inserted_poses)

    def optimize(self, data: ObjectFrameData):
        """
        Add new variables and factors to the graph based on incoming data, then
        perform a nonlinear optimization using Levenberg–Marquardt.
        """
        frame_id = data.frame_id

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
                if self._between_noise is not None:
                    between_noise = self._between_noise
                else:
                    # Safe fallback (very loose) if user enabled between factors
                    # but didn't provide between_noise_param.
                    res = (
                        np.mean(data.reg_residuals)
                        if data.reg_residuals.size > 0
                        else 1.0
                    )

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
            # return None

        # Run LM optimization
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self._graph, self._values, self._lm_params
            )
            result = optimizer.optimize()
        except RuntimeError:
            print(f"[LMGraphOptimizer] Optimization failed for frame {frame_id}")
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
            f"[LMGraphOptimizer] Optimized frame {frame_id} for object {data.obj_id}, "
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
