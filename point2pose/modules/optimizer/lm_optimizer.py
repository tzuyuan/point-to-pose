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

        self._lm_params.setVerbosityLM("SUMMARY")

        # Bookkeeping
        self._inserted_poses = set()
        self._inserted_landmarks = set()
        self.inserted_landmark_ids = []

        self._initialized = False
        self._prev_pose_inv = np.eye(4)
        self._prev_frame_id = -1
        self._prev_rel_T = None

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

        # Add prior on first pose or between factor afterward
        if not self._initialized:
            self._graph.push_back(
                gtsam.PriorFactorPose3(Xi, cur_pose_c2w_gtsam, self._prior_noise)
            )
            # self._prev_rel_T = gtsam.Pose3(np.eye(4))
        elif data.rel_pose is not None:
            prev_id = self._prev_frame_id
            X_prev = gtsam.symbol("x", prev_id)

            # relative pose is from previous frame to current frame
            # we need to invert it to get the current frame to the previous frame
            rel_T_cim12ci = gtsam.Pose3(inverse_SE3(data.rel_pose))

            # Noise scaled by measurement residuals
            base_sigma = (
                float(max(1e-4, np.mean(data.residuals)))
                if data.residuals.size > 0
                else 0.01
            )
            # sigma_between = base_sigma  # Relax odometry relative to features

            # between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            #     np.array(
            #         [
            #             sigma_between * 1000,
            #             sigma_between * 1000,
            #             sigma_between * 1000,
            #             sigma_between * 500,
            #             sigma_between * 500,
            #             sigma_between * 500,
            #         ],
            #         dtype=float,
            #     )
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
        if data.inliers.size > 0:  # and self._initialized:
            # Choose pose used for seeding new landmarks
            if self._values.exists(Xi):
                try:
                    seed_pose = self._values.atPose3(Xi)
                except RuntimeError:
                    seed_pose = cur_pose_c2w_gtsam
            else:
                seed_pose = cur_pose_c2w_gtsam

            assert (
                len(data.valid_idx) == len(data.inliers) == len(data.residuals)
            ), "inliers/residuals must be aligned with valid_idx if indexed by mask"

            for m, lid in enumerate(data.valid_idx):
                if not data.inliers[m] or np.isnan(data.cur_3d[lid]).any():
                    continue

                z_cam = data.cur_3d[lid]
                Lj = gtsam.symbol("l", int(lid))

                # sigma_point = float(max(1e-4, data.residuals[m]))
                sigma_point = float(max(1e-4, data.uncertainties[m]))
                # base_noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma_point)
                base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array(
                        [sigma_point, sigma_point, sigma_point],
                        dtype=float,
                    )
                )
                # point_noise = base_noise
                point_noise = gtsam.noiseModel.Robust(
                    gtsam.noiseModel.mEstimator.Huber(1.345), base_noise
                )

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
            return None

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

        return OptimizerResult(
            obj_id=data.obj_id,
            frame_id=frame_id,
            pose_optimized=pose_opt,
            key_points_optimized=landmark_xyz,
            key_points_idx_optimized=ids,
        )
