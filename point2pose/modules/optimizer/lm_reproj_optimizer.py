import bisect
import numpy as np
import gtsam

from point2pose.core.module_registry import OPTIMIZER
from point2pose.core.base_optimizer import Optimizer
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.utils.transform import inverse_SE3


@OPTIMIZER.register_module("lm_graph_reproj")
class LMGraphReprojOptimizer(Optimizer):
    """
    ## TODO: this is not working yet, need to fix it
    Same structure as your LMGraphOptimizer, but uses reprojection factors.
    Adds support for 2D-only tracks via:
      - data.visible_pts_2d: (M,2)
      - data.visible_pts_2d_idx: (M,)
    We store 2D observations in a pending buffer and only create landmarks when
    depth is available (via your existing inlier+valid-depth cur_3d).
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "lm_graph_reproj_optimizer"

        self._graph = gtsam.NonlinearFactorGraph()
        self._values = gtsam.Values()

        prior_noise_param = config.get(
            "prior_noise_param", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_noise_param, dtype=float)
        )

        self._uncertainty_min = config.get("uncertainty_min", 1e-2)
        self._uncertainty_thres = config.get("uncertainty_thres", 0.3)

        # --- reprojection gating ---
        self._reproj_gate_px = config.get("reproj_gate_px", 20.0)
        self._reproj_gate_sigma = config.get("reproj_gate_sigma", 100.0)
        self._reproj_gate_z_min = config.get("reproj_gate_z_min", 1e-6)

        # avoid duplicate (pose, landmark) reproj factors
        self._added_reproj_pairs = set()

        self._lm_params = gtsam.LevenbergMarquardtParams()
        self._lm_params.setMaxIterations(config.get("max_iterations", 20))
        self._lm_params.setRelativeErrorTol(config.get("relative_error_tol", 1e-5))
        self._lm_params.setAbsoluteErrorTol(config.get("absolute_error_tol", 1e-5))
        self._lm_params.setlambdaInitial(config.get("lambda_initial", 1e-1))
        self._lm_params.setVerbosityLM("SUMMARY")

        # fx, fy, s, cx, cy = config["cal3_s2"]
        self._K = None

        self._inserted_poses = set()
        self._inserted_landmarks = set()
        self.inserted_landmark_ids = []

        self._initialized = False
        self._prev_pose_inv = np.eye(4)
        self._prev_frame_id = -1

        self.Xo = gtsam.symbol("x", 0)

        # lid -> {frame_id: (u,v)}
        self._pending_uv = {}
        self._pending_uncertainties = {}

    def get_num_poses(self) -> int:
        return len(self._inserted_poses)

    def _add_reproj_factor(self, Xi, Lj, uv, sigma_pix):
        # Dedup: don't add same (pose, landmark) twice
        pair = (int(Xi), int(Lj))
        if pair in self._added_reproj_pairs:
            return False

        # Gate before adding
        if not self._passes_reproj_gate(Xi, Lj, uv, sigma_pix):
            return False

        base_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(sigma_pix))
        pix_noise = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(1.345), base_noise
        )
        meas = gtsam.Point2(float(uv[0]), float(uv[1]))

        self._graph.push_back(
            gtsam.GenericProjectionFactorCal3_S2(meas, pix_noise, Xi, Lj, self._K)
        )

        self._added_reproj_pairs.add(pair)
        return True

    # def _add_reproj_factor(self, Xi, Lj, uv, sigma_pix):
    #     base_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(sigma_pix))
    #     pix_noise = gtsam.noiseModel.Robust(
    #         gtsam.noiseModel.mEstimator.Huber(1.345), base_noise
    #     )
    #     meas = gtsam.Point2(float(uv[0]), float(uv[1]))
    #     # print(
    #     #     f"adding reproj factor at frame {gtsam.Symbol(Xi)}, landmark {gtsam.Symbol(Lj)}: {meas}"
    #     # )
    #     # print(f"sigma_pix: {sigma_pix}")
    #     # print(f"K: {self._K}")

    #     self._graph.push_back(
    #         gtsam.GenericProjectionFactorCal3_S2(meas, pix_noise, Xi, Lj, self._K)
    #     )

    #     # cam = gtsam.PinholeCameraCal3_S2(self._values.atPose3(Xi), self._K)
    #     # uv_hat = cam.project(self._values.atPoint3(Lj))
    #     # print(
    #     #     "meas",
    #     #     uv,
    #     #     "pred",
    #     #     uv_hat,
    #     #     "err",
    #     #     np.linalg.norm(np.array(uv_hat) - np.array(uv)),
    #     # )

    def optimize(self, data: ObjectFrameData):
        frame_id = data.frame_id

        # data.pose is world_T_cam; we need cam_T_world
        cur_pose_c2w = inverse_SE3(data.pose)
        cur_pose_c2w_gtsam = gtsam.Pose3(cur_pose_c2w)

        Xi = gtsam.symbol("x", frame_id)

        # insert value for the pose if not already inserted
        if Xi not in self._inserted_poses:
            self._values.insert(Xi, cur_pose_c2w_gtsam)
            self._inserted_poses.add(Xi)
            self.Xo = Xi

        if not self._initialized:
            # insert the camera intrinsics
            self._K = gtsam.Cal3_S2(
                float(data.intrinsics[0, 0]),  # fx
                float(data.intrinsics[1, 1]),  # fy
                float(data.intrinsics[0, 1]),  # s
                float(data.intrinsics[0, 2]),  # cx
                float(data.intrinsics[1, 2]),  # cy
            )
            # insert the prior factor for the first pose
            self._graph.push_back(
                gtsam.PriorFactorPose3(Xi, cur_pose_c2w_gtsam, self._prior_noise)
            )

        self._prev_pose_inv = data.pose
        self._prev_frame_id = frame_id

        if self._values.exists(Xi):
            # get the pose from the values if it exists
            try:
                seed_pose_c2w = self._values.atPose3(Xi)
            except RuntimeError:
                # if the pose does not exist, use the current pose
                seed_pose_c2w = cur_pose_c2w_gtsam
        else:
            # if the pose does not exist, use the current pose
            seed_pose_c2w = cur_pose_c2w_gtsam

        # 1) Store all visible 2D (including those without depth)
        if data.visible_pts_2d_idx.size > 0:
            for k, lid in enumerate(data.visible_pts_2d_idx):
                lid = int(lid)
                if lid not in self._pending_uv:
                    self._pending_uv[lid] = {}
                    self._pending_uncertainties[lid] = {}

                if data.visible_uncertainties[k] > self._uncertainty_thres:
                    continue
                self._pending_uv[lid][frame_id] = data.visible_pts_2d[k]
                # self._pending_uncertainties[lid][frame_id] = (
                #     max(self._uncertainty_min, float(data.visible_uncertainties[k]))
                #     * 50.0
                # )
                self._pending_uncertainties[lid][frame_id] = 5.0  # placeholder for now

        # 2) Promote depth-valid inliers to landmarks; add reprojection factors
        if data.reg_inliers.size > 0:
            assert (
                len(data.reg_valid_idx)
                == len(data.reg_inliers)
                == len(data.reg_residuals)
            )

            for m, lid in enumerate(data.reg_valid_idx):
                if not data.reg_inliers[m] or np.isnan(data.reg_cur_3d[m]).any():
                    continue

                lid_int = int(lid)
                Lj = gtsam.symbol("l", lid_int)

                # pixel sigma: use your existing uncertainty (same as your 3D factor used)
                sigma_pix = float(max(1e-2, data.reg_uncertainties[m]))

                # create landmark once (seed from current depth)
                if Lj not in self._inserted_landmarks:
                    z_cam = data.reg_cur_3d[m]
                    pw = seed_pose_c2w.transformFrom(gtsam.Point3(*z_cam))
                    self._values.insert(Lj, pw)
                    self._inserted_landmarks.add(Lj)
                    bisect.insort(self.inserted_landmark_ids, lid_int)

                    # when first created, add all stored 2D observations for this lid
                    if lid_int in self._pending_uv:
                        # iterate over a copy since we may delete
                        for fid, uv in list(self._pending_uv[lid_int].items()):
                            uncertainty = self._pending_uncertainties[lid_int][fid]
                            Xk = gtsam.symbol("x", int(fid))
                            ok = self._add_reproj_factor(Xk, Lj, uv, uncertainty)
                            if not ok:
                                # drop this observation so it doesn't keep coming back
                                del self._pending_uv[lid_int][fid]
                                del self._pending_uncertainties[lid_int][fid]

                else:
                    # landmark already exists: add factor for this frame if we have visible uv stored
                    if (
                        lid_int in self._pending_uv
                        and frame_id in self._pending_uv[lid_int]
                    ):
                        uv = self._pending_uv[lid_int][frame_id]
                        uncertainty = self._pending_uncertainties[lid_int][frame_id]
                        self._add_reproj_factor(Xi, Lj, uv, uncertainty)

        if not self._initialized:
            self._initialized = True
            return None

        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self._graph, self._values, self._lm_params
            )
            result = optimizer.optimize()
        except RuntimeError:
            print(f"[LMGraphReprojOptimizer] Optimization failed for frame {frame_id}")
            return None

        self._values = result

        try:
            Xi_hat = result.atPose3(Xi)
        except RuntimeError:
            return None

        pose_opt = inverse_SE3(Xi_hat.matrix())

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
            )
            landmark_xyz[k, :] = p
            ids[k] = int(lid)
            k += 1

        landmark_xyz = landmark_xyz[:k, :]
        ids = ids[:k]

        print(
            f"[LMGraphReprojOptimizer] Optimized frame {frame_id} for object {data.obj_id}, "
            f"{len(self._inserted_poses)} frames and {landmark_xyz.shape[0]} landmarks"
        )

        return OptimizerResult(
            obj_id=data.obj_id,
            frame_id=frame_id,
            pose_optimized=pose_opt,
            key_points_optimized=landmark_xyz,
            key_points_idx_optimized=ids,
        )

    def _passes_reproj_gate(self, Xi, Lj, uv, sigma_pix) -> bool:
        """
        Returns True if reprojection is valid and error is within gate.
        Uses current self._values as the gating estimate.
        """
        if self._K is None:
            return False
        if (not self._values.exists(Xi)) or (not self._values.exists(Lj)):
            return False

        uv = np.asarray(uv, dtype=float).reshape(
            2,
        )
        sigma_pix = float(sigma_pix)

        pose = self._values.atPose3(
            Xi
        )  # Pose3 (whatever convention you're using consistently)
        pw = self._values.atPoint3(Lj)  # Point3 in world

        # Cheirality / in-front check (world -> camera)
        pc = pose.transformTo(pw)  # point in camera frame
        if pc[2] <= self._reproj_gate_z_min:
            return False

        cam = gtsam.PinholeCameraCal3_S2(pose, self._K)
        uv_hat = cam.project(pw)
        # uv_hat = np.array([uv_hat, uv_hat.y()], dtype=float)

        err = float(np.linalg.norm(uv_hat - uv))

        gate = max(self._reproj_gate_px, self._reproj_gate_sigma * sigma_pix)
        return err <= gate
