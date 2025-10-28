import bisect
import numpy as np

import gtsam

from point2pose.core.module_registry import OPTIMIZER
from point2pose.core.base_optimizer import Optimizer
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.utils.transform import transform_pts, inverse_SE3


@OPTIMIZER.register_module("isam2")
class ISAM2Optimizer(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.name = "isam2_optimizer"

        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(config.get("relinearize_threshold", 0.1))
        isam_params.relinearizeSkip = config.get("relinearize_skip", 1)

        self._isam = gtsam.ISAM2(isam_params)

        self._graph = gtsam.NonlinearFactorGraph()
        self._values = gtsam.Values()

        prior_noise_param = config.get(
            "prior_noise_param", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_noise_param)
        )

        self._inserted_poses = set()
        self._inserted_landmarks = set()
        self.inserted_landmark_ids = []

        self._initialized = False
        self._prev_pose_inv = np.eye(4)
        self._prev_frame_id = -1

    def optimize(self, data: ObjectFrameData):
        """Perform the optimization operation."""
        # the pose takes the first frame to the current camera frame
        # in optmization, we use the inverse of the pose
        frame_id = data.frame_id
        cur_pose = inverse_SE3(data.pose)
        # registration stats
        cur_3d = data.cur_3d
        cur_3d_idx = data.cur_3d_idx
        inliers = data.inliers
        residuals = data.residuals

        cur_pose_gtsam = gtsam.Pose3(cur_pose)

        # -------- update pose factor -----------
        Xi = gtsam.symbol("x", frame_id)
        if Xi not in self._inserted_poses:
            self._values.insert(Xi, cur_pose_gtsam)
            self._inserted_poses.add(Xi)

        if frame_id == 0:
            self._graph.push_back(
                gtsam.PriorFactorPose3(Xi, cur_pose_gtsam, self._prior_noise)
            )
            self._initialized = True
        else:
            Xim1 = gtsam.symbol("x", self._prev_frame_id)
            rel_T = gtsam.Pose3(inverse_SE3(self._prev_pose_inv) @ cur_pose)

            # noise scaled by the residuals
            sigma_between = (
                float(max(1e-4, np.mean(residuals))) if residuals.size else 0.01
            )
            between_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma_between)

            self._graph.push_back(
                gtsam.BetweenFactorPose3(Xim1, Xi, rel_T, between_noise)
            )
            # self._values.insert(Xi, cur_pose)
            # self._inserted_poses.add(Xi)

        self._prev_pose_inv = data.pose
        self._prev_frame_id = frame_id

        # -------- update keypoints factor -----------
        # We use the *current* initial/estimated pose to seed new landmarks
        # If we haven't run iSAM yet, fall back to our inserted guess
        Xi_seed = gtsam.Pose3(cur_pose)

        try:
            Xi_seed = self._isam.calculateEstimate().atPose3(Xi)
        except RuntimeError:
            Xi_seed = gtsam.Pose3(cur_pose)

        # Loop over measurements
        if inliers.size > 0 and frame_id > 0:
            for m, lid in enumerate(cur_3d_idx):
                if not inliers[m] or np.isnan(cur_3d[m]).any():
                    continue
                z_cam = cur_3d[m]  # 3D point in camera_i frame
                Lj = gtsam.symbol("l", int(lid))
                point_noise = gtsam.noiseModel.Isotropic.Sigma(3, residuals[m])
                # Seed landmark once (in world), using Xi_seed * z_cam
                if Lj not in self._inserted_landmarks:
                    p_w = Xi_seed.transformFrom(gtsam.Point3(*z_cam))
                    self._values.insert(Lj, p_w)
                    self._inserted_landmarks.add(Lj)
                    bisect.insort(self.inserted_landmark_ids, int(lid))

                # form bearing/range in the camera/body frame (your code does this)
                z_range = float(np.linalg.norm(z_cam))
                if z_range <= 1e-9:
                    continue
                z_bearing = gtsam.Unit3(z_cam / z_range)

                self._graph.push_back(
                    gtsam.BearingRangeFactor3D(Xi, Lj, z_bearing, z_range, point_noise)
                )

        # --- Incremental update ---
        self._isam.update(self._graph, self._values)

        # (Optional) occasionally force an extra relinearization sweep
        # if i % 10 == 0:
        #     isam.update()

        # print(f"[ISAM2Optimizer] frame id: {frame_id}")
        # print(f"[ISAM2Optimizer] Graph: {self._graph}")
        # print(f"[ISAM2Optimizer] Values: {self._values}")
        self._graph.resize(0)
        self._values.clear()

        if frame_id > 0:
            # Get rolling estimate if you want to use it online
            current_estimate = self._isam.calculateEstimate()

            # Example: read back pose_i now
            Xi_hat = current_estimate.atPose3(Xi)
            pose_optimized = inverse_SE3(Xi_hat.matrix())
            key_points_optimized = gtsam.utilities.extractPoint3(current_estimate)
            # key_points_optimized = transform_pts(pose_optimized, key_points_optimized)

            print(
                f"[ISAM2Optimizer] Optimized frame {frame_id} for object {data.obj_id}, number of key points: {key_points_optimized.shape[0]}"
            )
            return OptimizerResult(
                obj_id=data.obj_id,
                frame_id=frame_id,
                pose_optimized=pose_optimized,
                key_points_optimized=key_points_optimized,
                key_points_idx_optimized=self.inserted_landmark_ids,
            )
        else:
            return None
