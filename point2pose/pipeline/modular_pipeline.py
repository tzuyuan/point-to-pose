import os
import time
import numpy as np
import torch
import copy
from collections import deque
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import OPTIMIZER  # for config check
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.modules.object.object import Object
from point2pose.io.outputs.logger import DataLogger
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.logger_fields import RAGGED_FIELDS

# Components
from point2pose.pipeline.components.front_end import FrontEnd
from point2pose.pipeline.components.key_frame_manager import KeyFrameManager
from point2pose.pipeline.components.local_optimizer import LocalOptimizer
from point2pose.pipeline.components.key_frame_graph import KeyFrameGraph
from point2pose.pipeline.components.recovery_manager import RecoveryManager
from point2pose.modules.reconstruction import SDFBuilder


class ModularPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        # Components
        self.frontend = FrontEnd(cfg)
        self.kf_manager = KeyFrameManager(cfg)
        self.local_optimizer = LocalOptimizer(cfg)
        self.kf_graph = KeyFrameGraph(cfg)
        self.recovery_manager = RecoveryManager(cfg)
        self.sdf_builder = SDFBuilder(cfg.reconstructor.params)

        # State
        self.track_table = PointTrackTable.new(n0=0)
        self.objects = []
        self.num_obj = self.pipeline_cfg.get("max_num_obj", 1)
        self._initialized = False

        # depth estimate related
        # TODO: Make this a class
        self.use_depth_estimate = self.pipeline_cfg.get("use_depth_estimate", False)
        self.depth_estimator_type = self.pipeline_cfg.get(
            "depth_estimator_type", "depth_anything"
        )
        self.depth_encoder = self.pipeline_cfg.get("depth_encoder", "vits")
        self._device = self.pipeline_cfg.get("device", "cpu")
        self.depth_estimator = None
        self._depth_model_cfg = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        # module settings
        self._estimate_init_pose = self.pipeline_cfg.get("estimate_init_pose", False)
        self.use_local_graph = self.pipeline_cfg.get("use_local_graph", False)
        self.min_depth = self.pipeline_cfg.get("min_depth", 0.05)
        self.max_depth = self.pipeline_cfg.get("max_depth", 1.0)
        self.max_rel_rotation_deg = self.pipeline_cfg.get("max_rel_rotation_deg", 20)
        self.max_rel_translation = self.pipeline_cfg.get("max_rel_translation", 0.05)
        self.local_opt_type = self.cfg.local_optimizer.get("type", "isam2")
        self.local_graph_max_num_frames = self.cfg.local_optimizer.params.get(
            "local_graph_max_num_frames", -1
        )
        self.reg_residual_thres = self.cfg.register.params.get("residual_thres", 0.07)

        self.num_hist = self.cfg.register.params.get("num_hist", 5)
        self.hist_frames = deque(maxlen=self.num_hist)
        self.hist_fe_results = deque(maxlen=self.num_hist)
        self.hist_track_tables = deque(maxlen=self.num_hist)

        # Logging
        self.save_pose = self.pipeline_cfg.get("save_pose", False)
        self.pose_save_path = self.pipeline_cfg.get("pose_save_path", "./poses")
        self.pose_log_files = []
        self.debug_level = self.pipeline_cfg.get("debug_level", 0)
        self.debug_dir = self.pipeline_cfg.get("debug_dir", None)
        self.sdf_mesh_save_every = int(self.pipeline_cfg.get("sdf_mesh_save_every", 1))
        self.sdf_mesh_save_dir = None

        if self.debug_dir:
            self.sdf_mesh_save_dir = os.path.join(self.debug_dir, "sdf_mesh")
        else:
            self.sdf_mesh_save_dir = "./results/sdf_mesh"
        os.makedirs(self.sdf_mesh_save_dir, exist_ok=True)

        if self.save_pose:
            os.makedirs(self.pose_save_path, exist_ok=True)

        self.save_meta_data = self.pipeline_cfg.get("save_meta_data", False)
        self.meta_data_save_path = self.pipeline_cfg.get(
            "meta_data_save_path", "./meta_data"
        )
        if self.save_meta_data:
            os.makedirs(self.meta_data_save_path, exist_ok=True)
            self.data_logger = DataLogger(
                out_dir=self.meta_data_save_path,
                base_name="meta_data",
                ragged_fields=RAGGED_FIELDS,
                also_save_h5=False,
            )

    # -------- one-time init with user clicks ----------
    def add_user_points(self, obj_points: list[list[int]], labels: list[int]):
        """
        points (List[List[int]]): List of points, each defined by [x, y].
            labels (List[int]): List of labels, each defined by 1 or 0.
                                1 means positive point, 0 means negative point.
        """
        # forward to segmenter; it will start tracking objects internally

        self.frontend.segmenter.add_input_points(obj_points, labels)

    def initialize_first_frame(self, frame):

        # TODO: make this a class
        if self.depth_estimator_type == "depth_anything":
            # local import to avoid import-time CUDA/BLAS side-effects
            from third_party.depth_anything_v2_metric.dpt import DepthAnythingV2

            m = DepthAnythingV2(
                **self._depth_model_cfg[self.depth_encoder], max_depth=20
            )
            state = torch.load(
                f"checkpoints/depth_anything/depth_anything_v2_metric_hypersim_{self.depth_encoder}.pth",
                map_location="cpu",
            )
            m.load_state_dict(state)
            self.depth_estimator = m.to(self._device).eval()
        elif self.depth_estimator_type == "promptda":
            from promptda.promptda import PromptDA

            self.depth_estimator = (
                PromptDA.from_pretrained(
                    f"checkpoints/promptda/{self.depth_encoder}.ckpt"
                )
                .to(self._device)
                .eval()
            )

        # 1. Initialize FrontEnd (Segmentation + Tracker)
        self.frontend.initialize(frame)

        # 2. Setup Objects based on segmentation
        if self.frontend.use_segmenter:
            # Prefer segmenter's object count if available (supports mask init), otherwise fall back to point labels
            if (
                hasattr(self.frontend.segmenter, "num_obj")
                and self.frontend.segmenter.num_obj > 0
            ):
                self.num_obj = self.frontend.segmenter.num_obj
            else:
                self.num_obj = np.sum(
                    np.asarray(self.frontend.segmenter.input_labels) == 1
                )
        print(f"Initialized with {self.num_obj} objects based on segmenter output.")
        for obj_id in range(self.num_obj):
            self.objects.append(Object(obj_id))
            self.objects[obj_id].pose = np.eye(4)

            if self.save_pose:
                pose_log_path = os.path.join(
                    self.pose_save_path, f"obj_{obj_id}_pose.txt"
                )
                f = open(pose_log_path, "w", encoding="utf-8")
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                f.write(
                    self._pose_matrix_to_tum_format(np.eye(4), timestamp=time.time())
                )
                self.pose_log_files.append(f)
            else:
                self.pose_log_files.append(None)

        # 3. Initialize KeyFrameManager (Sampling)
        # This will populate track_table and object keypoints
        new_kfs = self.kf_manager.initialize(
            frame, self.track_table, self.objects, self.frontend.tracker
        )

        self.kf_graph.update(new_kfs)

        for kf in new_kfs:
            obj = self.objects[kf.obj_id]
            sdf_ok = self.sdf_builder.integrate_keyframe(obj=obj, keyframe=kf)
            if not sdf_ok:
                print(
                    f"Frame {frame.id}: Failed to integrate keyframe for obj {obj.id}"
                )
                continue

        # 4. Initial Optimization
        if self.use_local_graph:
            for obj_id in range(self.num_obj):
                valid_idx = self.objects[obj_id].valid
                pts_2d_visible, pts_2d_visible_idx, visible_uncertainties = (
                    self.track_table.get_visible_2d_and_uncertainties_for_obj(obj_id)
                )
                self.local_optimizer.optimize(
                    ObjectFrameData(
                        obj_id=obj_id,
                        frame_id=frame.id,
                        intrinsics=frame.intrinsics,
                        pose=np.eye(4),
                        rel_pose=np.eye(4),
                        visible_pts_2d=pts_2d_visible,
                        visible_pts_2d_idx=pts_2d_visible_idx,
                        visible_uncertainties=visible_uncertainties,
                        reg_cur_3d=self.objects[obj_id].key_points,
                        reg_cur_3d_idx=self.objects[obj_id].kp_track_indices,
                        reg_valid_idx=self.objects[obj_id].kp_track_indices,
                        reg_inliers=np.ones(
                            len(self.objects[obj_id].key_points), dtype=bool
                        ),
                        reg_residuals=0.00001
                        * np.ones(len(self.objects[obj_id].key_points)),
                        reg_uncertainties=0.000001
                        * np.ones(len(self.objects[obj_id].key_points)),
                    )
                )

        # 5. Estimate Initial Pose (if configured)
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        if self._estimate_init_pose:
            self._estimate_init_pose_and_bbox_for_all_obj(frame)
            for i in range(self.num_obj):
                out_pose[i] = np.eye(4)

        # Log initial state
        if self.save_meta_data:
            # Get keyframe camera frame keypoints if available
            kf_kp_3d_camera_init = np.array([])
            if len(self.kf_manager.keyframes.get(0, [])) > 0:
                first_kf = self.kf_manager.keyframes[0][0]
                if first_kf.kp_3d_camera is not None:
                    kf_kp_3d_camera_init = first_kf.kp_3d_camera

            self.data_logger.log(
                {
                    "timestamp": frame.timestamp,
                    "frame_id": 0,
                    "track2d": self.track_table.track_2d,
                    "uncertainties": self.track_table.uncertainty,
                    "visibles": self.track_table.visible,
                    "track3d": self.track_table.track_3d,
                    "valid": self.track_table.valid,
                    "valid_depth": frame.depth,
                    "obj_init_pose": self.objects[0].init_pose,
                    "obj_pose": self.objects[0].pose,
                    "obj_key_point_frames": self.objects[0].key_point_frames,
                    "obj_key_points": self.objects[0].key_points,
                    "obj_uncertainties": self.objects[0].uncertainties,
                    "obj_valid": self.objects[0].valid,
                    "obj_kp_3d_camera": kf_kp_3d_camera_init,  # Newly initialized keypoints in camera frame
                    "is_key_frame": True,
                    "pose_frontend": np.eye(4),
                    "pose_local": np.eye(4),
                    "reg_curr3d": self.objects[obj_id].key_points,
                    "reg_residuals": np.zeros(len(self.objects[obj_id].key_points)),
                    "reg_inliers": np.ones(
                        len(self.objects[obj_id].key_points), dtype=bool
                    ),
                    "reg_valid_idx": self.objects[0].valid,
                    # Dense recovery info (initialized for first frame - no recovery possible)
                    "dense_recovery_triggered": False,
                    "dense_recovery_pose_before": np.eye(4),
                    "dense_recovery_pose_after": np.eye(4),
                    "dense_recovery_rel_before": np.eye(4),
                    "dense_recovery_rel_after": np.eye(4),
                    "dense_recovery_inliers_before": np.array([]),
                    "dense_recovery_residuals_before": np.array([]),
                    "dense_recovery_inliers_after": np.array([]),
                    "dense_recovery_residuals_after": np.array([]),
                }
            )

        self._initialized = True

        return out_pose

    def step(self, frame):
        if not self._initialized:
            return self.initialize_first_frame(frame)

        # TODO: make this a class
        if self.use_depth_estimate:
            if self.depth_estimator_type == "depth_anything":

                def align_scale(pred, gt, mask):
                    eps = 1e-6
                    ratio = gt[mask] / np.clip(pred[mask], eps, None)
                    s = np.median(ratio)
                    return pred * s

                mask = frame.depth > 0  # or a tighter validity mask

                raw = frame.rgb
                if raw.dtype != np.uint8:
                    # if it's float [0,1], convert back
                    raw = (np.clip(raw, 0, 1) * 255).astype(np.uint8)

                raw_bgr = raw[..., ::-1].copy()  # RGB -> BGR
                depth_pred = self.depth_estimator.infer_image(raw_bgr)

                depth_pred_aligned = align_scale(
                    depth_pred, frame.depth / frame.depth_factor, mask
                )

                frame.depth_factor = 1.0
            elif self.depth_estimator_type == "promptda":
                depth_resize = cv.resize(
                    frame.depth, (630, 476), interpolation=cv.INTER_LINEAR
                )
                rgb_resize = cv.resize(
                    frame.rgb, (630, 476), interpolation=cv.INTER_LINEAR
                )
                depth_torch = (
                    torch.from_numpy(depth_resize / frame.depth_factor)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self._device)
                )
                rgb_torch = (
                    torch.from_numpy(rgb_resize)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self._device)
                )
                depth_troch = self.depth_estimator.predict(rgb_torch, depth_torch)
                print("depth_troch shape: ", type(depth_troch))
                from promptda.utils.io_wrapper import load_image, load_depth, save_depth

                save_depth(depth_troch, prompt_depth=depth_torch, image=rgb_torch)
                depth = depth_troch.squeeze(0).squeeze(0).cpu().numpy()
                depth = depth.astype(np.float32, copy=False)
                depth = cv.resize(
                    depth,
                    (frame.depth.shape[1], frame.depth.shape[0]),
                    interpolation=cv.INTER_LINEAR,
                )

            if isinstance(depth_pred_aligned, torch.Tensor):
                depth_pred_aligned = depth_pred_aligned.detach().cpu().numpy()
                depth_pred_aligned = depth_pred_aligned.astype(np.float32, copy=False)

            frame.depth = depth_pred_aligned.astype(np.float32, copy=False)

        iter_start_time = time.time()
        module_times = {
            "frontend": 0.0,
            "track_table": 0.0,
            "recovery": 0.0,
            "local_opt": 0.0,
            "keyframe": 0.0,
            "global_opt": 0.0,
            "logging": 0.0,
        }
        #################################################################
        ##                        Front End                            ##
        #################################################################
        t0 = time.time()
        fe_result = self.frontend.step(frame, self.track_table, self.objects)
        module_times["frontend"] = time.time() - t0

        # per-object update
        for obj_id in range(self.num_obj):
            self._update_object_from_frontend(obj_id, fe_result)

        pose_frontend = {}
        for obj_id in range(self.num_obj):
            if self.objects[obj_id].pose is not None:
                pose_frontend[obj_id] = self.objects[obj_id].pose.copy()
            else:
                pose_frontend[obj_id] = np.eye(4)

        #################################################################
        ##                     Track Table Update                      ##
        #################################################################
        # FrontEndResult contains the data needed to update the table
        t0 = time.time()
        self.track_table.update_track_table(
            fe_result.tracks,
            fe_result.track_3d,
            fe_result.track_valid,
            fe_result.uncertainties,
            fe_result.visibles,
        )

        # update history
        self._update_history_frames(frame, fe_result, self.track_table)
        module_times["track_table"] = time.time() - t0

        #################################################################
        ##                      Recovery Manager                       ##
        #################################################################
        # perform frame to map registration if the object is marked as lost
        t0 = time.time()
        # self.recovery_manager.update(
        #     frame,
        #     fe_result,
        #     self.objects,
        #     self.track_table,
        #     self.kf_manager.keyframes,
        #     self.frontend.register,
        # )
        module_times["recovery"] = time.time() - t0

        #################################################################
        ##                     Local Optimization                      ##
        #################################################################
        # perform graph optimization of frames between keyframes
        t0 = time.time()
        pose_local = {}
        if self.use_local_graph:
            for obj_id in range(self.num_obj):

                if obj_id not in fe_result.valid_indices or self.objects[obj_id].lost:
                    continue

                if self.local_graph_max_num_frames > 0:
                    # get the number of frames in the local graph
                    num_frames_in_local_graph = self.local_optimizer.get_num_frames(
                        obj_id
                    )
                    # print(
                    #     f"Frame {frame.id}: Number of frames in local graph: {num_frames_in_local_graph}"
                    # )
                    if num_frames_in_local_graph >= self.local_graph_max_num_frames:
                        self.local_optimizer.reset(obj_id)

                # Only optimize if registration was good enough
                # if self.objects[obj_id].mean_residual < self.reg_residual_thres:

                # get the visible points in the object
                pts_2d_visible, pts_2d_visible_idx, visible_uncertainties = (
                    self.track_table.get_visible_2d_and_uncertainties_for_obj(obj_id)
                )

                object_frame_data = ObjectFrameData(
                    obj_id=obj_id,
                    frame_id=frame.id,
                    intrinsics=frame.intrinsics,
                    pose=self.objects[obj_id].pose,
                    rel_pose=fe_result.rel_poses[obj_id],
                    visible_pts_2d=pts_2d_visible,
                    visible_pts_2d_idx=pts_2d_visible_idx,
                    visible_uncertainties=visible_uncertainties,
                    reg_cur_3d=fe_result.valid_curr_3d[obj_id],
                    reg_cur_3d_idx=fe_result.valid_indices[obj_id],
                    reg_valid_idx=fe_result.reg_stats[obj_id].get(
                        "valid_idx", np.array([])
                    ),
                    reg_inliers=fe_result.reg_stats[obj_id].get(
                        "inliers", np.array([])
                    ),
                    reg_residuals=fe_result.reg_stats[obj_id].get(
                        "residuals", np.array([])
                    ),
                    reg_uncertainties=fe_result.uncertainties[
                        fe_result.valid_indices[obj_id]
                    ],
                )

                opt_result = self.local_optimizer.optimize(object_frame_data)
                self.local_optimizer.update_object_state(
                    self.objects[obj_id], opt_result, self.track_table
                )

                pose_local[obj_id] = self.objects[obj_id].pose

        module_times["local_opt"] = time.time() - t0

        #################################################################
        ##                     Key Frame Manager                       ##
        #################################################################
        # check, initialize new keyframes, and sample new keypoints
        t0 = time.time()

        new_keyframes = self.kf_manager.update(
            hist_frames=self.hist_frames,
            hist_fe_results=self.hist_fe_results,
            hist_track_tables=self.hist_track_tables,
            track_table=self.track_table,
            objects=self.objects,
            tracker=self.frontend.tracker,
            conservative=True,
        )
        module_times["keyframe"] = time.time() - t0

        # If new keyframe, reset local optimizer
        for kf in new_keyframes:
            print(f"Frame {frame.id}: Keyframe triggered. Resetting local optimizer.")
            self.local_optimizer.reset(kf.obj_id)

        #################################################################
        ##                     Global Optimization                     ##
        #################################################################
        # perform global optimization of key frames only
        t0 = time.time()
        if new_keyframes:
            updated_global_poses, updated_landmarks = self.kf_graph.update(
                new_keyframes
            )

        for kf in new_keyframes:
            obj = self.objects[kf.obj_id]
            key = (kf.obj_id, kf.kf_idx)
            if key in updated_global_poses:
                global_pose = updated_global_poses[key]

                dt, ddeg = self._se3_delta(global_pose, obj.pose)
                if dt > self.max_rel_translation or ddeg > self.max_rel_rotation_deg:
                    print(
                        f"Frame {frame.id}: Warning: Large pose change after global optimization for obj {kf.obj_id} kf {kf.kf_idx}. Δt={dt:.3f}m, ΔR={ddeg:.2f}deg"
                    )
                else:
                    obj.pose = global_pose.copy()
                    kf.pose = global_pose.copy()

                if kf.obj_id in updated_landmarks:
                    lm_pts, lm_global_track_ids = updated_landmarks[kf.obj_id]
                    lm_idx = obj.track_idx_2_obj_idx[lm_global_track_ids]
                    lm_ok = (lm_idx >= 0) & (lm_idx < len(obj.key_points))
                    if np.any(lm_ok):
                        obj.key_points[lm_idx[lm_ok]] = lm_pts[lm_ok].copy()
            else:
                print(
                    f"Frame {frame.id}: No optimized global pose for obj {kf.obj_id} kf {kf.kf_idx}; using keyframe pose for SDF fusion."
                )

            sdf_ok = self.sdf_builder.integrate_keyframe(obj=obj, keyframe=kf)
            if sdf_ok:
                print(
                    f"Frame {frame.id}: Updated SDF for obj {kf.obj_id} from keyframe {kf.kf_idx} (num_integrated={obj.sdf_num_integrated})."
                )
                if (
                    self.sdf_mesh_save_every > 0
                    and (obj.sdf_num_integrated % self.sdf_mesh_save_every) == 0
                ):
                    mesh_path = os.path.join(
                        self.sdf_mesh_save_dir,
                        f"obj_{kf.obj_id}_frame_{frame.id}_kf_{kf.kf_idx}.ply",
                    )
                    saved = self.sdf_builder.export_debug_mesh(obj, mesh_path)
                    if saved and self.debug_level > 1:
                        print(f"Frame {frame.id}: Saved SDF mesh to {mesh_path}")

        module_times["global_opt"] = time.time() - t0

        #################################################################
        ##                         Logging                             ##
        #################################################################
        # log the results
        t0 = time.time()
        self._log_step(frame, fe_result, new_keyframes, pose_frontend, pose_local)
        module_times["logging"] = time.time() - t0

        iter_total_time = time.time() - iter_start_time
        print(
            f"Frame {frame.id}: "
            f"frontend={module_times['frontend']:.4f}s, "
            f"track_table={module_times['track_table']:.4f}s, "
            f"recovery={module_times['recovery']:.4f}s, "
            f"local_opt={module_times['local_opt']:.4f}s, "
            f"keyframe={module_times['keyframe']:.4f}s, "
            f"global_opt={module_times['global_opt']:.4f}s, "
            f"logging={module_times['logging']:.4f}s, "
            f"total={iter_total_time:.4f}s"
        )

        # Return poses
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        for i in range(self.num_obj):
            out_pose[i] = self.objects[i].pose

        return out_pose

    def _log_step(
        self,
        frame,
        fe_result,
        new_keyframes,
        pose_frontend=None,
        pose_local=None,
    ):
        if self.save_pose:
            for obj_id, f in enumerate(self.pose_log_files):
                if f:
                    f.write(self._pose_matrix_to_tum_format(self.objects[obj_id].pose))
                    f.flush()

        if self.save_meta_data:
            # ---------------- log meta data ----------------
            # Assuming logging for obj_id=0 as in pipeline_single_process.py
            # TODO: Extend to support multi-object logging if needed
            obj_id = 0

            reg_stats = fe_result.reg_stats.get(obj_id, {})
            valid_stats = fe_result.valid_stats.get(obj_id, {})

            # Extract dense recovery info
            dense_recovery_triggered = fe_result.dense_recovery_triggered.get(
                obj_id, False
            )
            dense_stats_before = fe_result.dense_recovery_stats_before.get(obj_id, {})
            dense_stats_after = fe_result.dense_recovery_stats_after.get(obj_id, {})

            # Get newly initialized keypoints in camera frame if this is a keyframe
            kf_kp_3d_camera = np.array([])
            if self.kf_manager.is_key_frame.get(obj_id, False):
                # Find the keyframe for this frame
                current_kf = None
                # First check if it's in new_keyframes
                for kf in new_keyframes:
                    if kf.frame_id == frame.id and kf.obj_id == obj_id:
                        current_kf = kf
                        break
                # If not found, get the last keyframe for this object
                if current_kf is None and obj_id in self.kf_manager.keyframes:
                    kf_list = self.kf_manager.keyframes[obj_id]
                    if len(kf_list) > 0 and kf_list[-1].frame_id == frame.id:
                        current_kf = kf_list[-1]

                if current_kf is not None and current_kf.kp_3d_camera is not None:
                    kf_kp_3d_camera = current_kf.kp_3d_camera

            log_payload = {
                "timestamp": frame.timestamp,
                "frame_id": frame.id,
                "track2d": fe_result.tracks,
                "uncertainties": fe_result.uncertainties,
                "visibles": fe_result.visibles,
                "track3d": fe_result.track_3d,
                "valid": fe_result.track_valid,
                "valid_depth": fe_result.track_valid,  # Assuming this meant track_valid in single_process
                "obj_init_pose": self.objects[obj_id].init_pose,
                "obj_pose": self.objects[obj_id].pose,
                "obj_key_points": self.objects[obj_id].key_points,
                "obj_uncertainties": self.objects[obj_id].uncertainties,
                "obj_valid": self.objects[obj_id].valid,
                "obj_key_point_frames": self.objects[obj_id].key_point_frames,
                "obj_kp_3d_camera": kf_kp_3d_camera,  # Newly initialized keypoints in camera frame (only for keyframes)
                "is_key_frame": self.kf_manager.is_key_frame.get(obj_id, False),
                # Registration stats
                "reg_key_points_idx": fe_result.valid_indices.get(obj_id),
                "reg_key_points": fe_result.valid_key_points.get(
                    obj_id
                ),  # Not exactly same but conceptually
                "reg_prev3d": fe_result.valid_key_points.get(
                    obj_id
                ),  # For f2f, prev3d is key_points
                "reg_curr3d": fe_result.valid_curr_3d.get(obj_id),
                "reg_residuals": reg_stats.get("residuals", np.array([])),
                "reg_inliers": reg_stats.get("inliers", np.array([])),
                "reg_valid_idx": reg_stats.get("valid_idx", np.array([])),
                # Clustered registration diagnostics (if produced by the register module)
                # Keep the raw cluster candidate list for later analysis.
                "reg_clusters": reg_stats.get("clusters", []),
                "reg_best_cluster_idx": int(reg_stats.get("best_cluster_idx", -1)),
                "reg_sdf_refine": reg_stats.get("sdf_refine", {}),
                "reg_pose_jump_guard_rejected": bool(
                    reg_stats.get("pose_jump_guard_info", {}).get("rejected", False)
                ),
                "reg_pose_jump_guard_dt": float(
                    reg_stats.get("pose_jump_guard_info", {}).get("dt", -1.0)
                ),
                "reg_pose_jump_guard_ddeg": float(
                    reg_stats.get("pose_jump_guard_info", {}).get("ddeg", -1.0)
                ),
                "reg_pose_jump_guard_used": int(
                    reg_stats.get("pose_jump_guard_info", {}).get("used", 0)
                ),
                "reg_pose_jump_guard_ninliers": int(
                    reg_stats.get("pose_jump_guard_info", {}).get("ninliers", 0)
                ),
                "reg_pose_jump_guard_inlier_ratio": float(
                    reg_stats.get("pose_jump_guard_info", {}).get("inlier_ratio", 0.0)
                ),
                "reg_pose_jump_guard_num_clusters": int(
                    reg_stats.get("pose_jump_guard_info", {}).get("num_clusters", 0)
                ),
                "iter": reg_stats.get("iter", -1),
                # Intermediate Poses
                "pose_frontend": (
                    pose_frontend[obj_id] if pose_frontend else np.eye(4)
                ),
                "pose_local": pose_local[obj_id] if pose_local else np.eye(4),
                # Dense recovery info
                "dense_recovery_triggered": dense_recovery_triggered,
                "dense_recovery_pose_before": fe_result.dense_recovery_pose_before.get(
                    obj_id, np.eye(4)
                ),
                "dense_recovery_pose_after": fe_result.dense_recovery_pose_after.get(
                    obj_id, np.eye(4)
                ),
                "dense_recovery_rel_before": fe_result.dense_recovery_rel_before.get(
                    obj_id, np.eye(4)
                ),
                "dense_recovery_rel_after": fe_result.dense_recovery_rel_after.get(
                    obj_id, np.eye(4)
                ),
                "dense_recovery_inliers_before": dense_stats_before.get(
                    "inliers", np.array([])
                ),
                "dense_recovery_residuals_before": dense_stats_before.get(
                    "residuals", np.array([])
                ),
                "dense_recovery_inliers_after": dense_stats_after.get(
                    "inliers", np.array([])
                ),
                "dense_recovery_residuals_after": dense_stats_after.get(
                    "residuals", np.array([])
                ),
                # Valid extraction stats
                **valid_stats,
            }

            if new_keyframes:
                log_payload["new_keyframes"] = [
                    {
                        "frame_id": kf.frame_id,
                        "obj_id": kf.obj_id,
                        "num_kp": kf.get_num_kps(),
                        "num_obs": kf.get_num_obs(),
                    }
                    for kf in new_keyframes
                ]

            self.data_logger.log(log_payload)

    def _pose_matrix_to_tum_format(self, pose_matrix, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        tx, ty, tz = pose_matrix[:3, 3]
        rotation = scipy_R.from_matrix(pose_matrix[:3, :3])
        qx, qy, qz, qw = rotation.as_quat()
        return f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"

    def _estimate_init_pose_and_bbox_for_all_obj(self, frame):
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        # estimate initial pose
        for obj_id in range(self.num_obj):
            # get the initial 3d points from frame.mask
            # mask = frame.mask[obj_id, 0]
            # y_coords, x_coords = np.where(mask > 0)
            # valid_pxl_in_mask = np.stack([x_coords, y_coords], axis=1)
            # Keep masks on GPU
            mask = frame.mask[obj_id, 0]  # [H, W] on cuda, dtype=bool/uint8

            # Get (y,x) indices on GPU; switch to (x,y) like your NumPy code
            coords_yx = torch.nonzero(mask > 0, as_tuple=False)  # [N, 2], (y,x)
            valid_pxl_in_mask_g = coords_yx[
                :, [1, 0]
            ].contiguous()  # [N, 2], (x,y), still on GPU

            valid_pxl_in_mask = valid_pxl_in_mask_g.cpu().numpy()
            ## TODO: remove this potentially
            mean_mask_pixel = np.mean(valid_pxl_in_mask, axis=0)

            ## TODO: Add outlier removal
            initial_3d_points, _ = convert_pixel_to_world(
                pixel=valid_pxl_in_mask,
                depth_image=frame.depth,
                cam_intrinsics=frame.intrinsics,
                depth_factor=frame.depth_factor,
                remove_invalid=True,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            # estimate initial bbox
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(initial_3d_points)

            ## Temporary outlier removal
            ## TODO: Make this a class
            pcd_stat_outlier_removal = False
            pcd_stat_outlier_removal_nb_neighbors = 20
            pcd_stat_outlier_removal_std_ratio = 2.0
            pcd_radius_outlier_removal = True
            pcd_radius_outlier_removal_radius = 0.1

            if pcd_stat_outlier_removal:
                _, ind_stat = pcd.remove_statistical_outlier(
                    nb_neighbors=pcd_stat_outlier_removal_nb_neighbors,
                    std_ratio=pcd_stat_outlier_removal_std_ratio,
                )
            close_indices = None
            if pcd_radius_outlier_removal:
                # Distance-based outlier removal if clicked point is provided
                mean_mask_pixel_world, _ = convert_pixel_to_world(
                    pixel=mean_mask_pixel,
                    depth_image=frame.depth,
                    cam_intrinsics=frame.intrinsics,
                    cam2world=np.eye(4),
                    depth_factor=frame.depth_factor,
                )
                # Calculate distances from clicked point to all points
                points_array = initial_3d_points
                distances = np.linalg.norm(points_array - mean_mask_pixel_world, axis=1)

                # Keep points within a reasonable distance (e.g., 0.2 meters)
                max_distance = pcd_radius_outlier_removal_radius
                close_indices = np.where(distances <= max_distance)[0]

            ind_union = None
            if pcd_stat_outlier_removal and pcd_radius_outlier_removal:
                ind_union = np.intersect1d(ind_stat, close_indices)
            elif pcd_stat_outlier_removal:
                ind_union = ind_stat
            elif pcd_radius_outlier_removal:
                ind_union = close_indices

            if ind_union is not None:
                pcd_world_clean = pcd.select_by_index(ind_union)
            else:
                pcd_world_clean = pcd

            self.objects[obj_id].init_bbox = pcd_world_clean.get_oriented_bounding_box()
            # self.objects[obj_id].init_bbox = pcd.get_axis_aligned_bounding_box()
            self.objects[obj_id].bbox = self.objects[obj_id].init_bbox
            # set the initial pose
            out_pose[obj_id, :3, :3] = self.objects[obj_id].init_bbox.R
            out_pose[obj_id, :3, 3] = self.objects[obj_id].init_bbox.center
            # the first frame pose is the identity
            self.objects[obj_id].pose = np.eye(4)
            # initial pose is the pose from the object frame to the first frame
            self.objects[obj_id].init_pose = out_pose[obj_id]

            # Log initial pose in TUM format
            if self.save_pose and self.pose_log_files[obj_id] is not None:
                tum_pose = self._pose_matrix_to_tum_format(
                    out_pose[obj_id], timestamp=time.time()
                )
                self.pose_log_files[obj_id].write(tum_pose)
                self.pose_log_files[obj_id].flush()
            # self.objects[obj_id].pose = np.eye(4)
            # out_pose[obj_id] = np.eye(4)
            # -----------------------------------------

            if self.debug_level > 1:

                # ensure debug directory exists
                debug_bbx_dir = os.path.join(self.debug_dir, "pipeline/initial_bbx")
                os.makedirs(debug_bbx_dir, exist_ok=True)

                # get color from frame.rgb
                # color = frame.rgb[y_coords, x_coords]
                color = frame.rgb[valid_pxl_in_mask[:, 1], valid_pxl_in_mask[:, 0]]
                pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
                o3d.io.write_point_cloud(
                    os.path.join(debug_bbx_dir, f"initial_pcd_{obj_id}.ply"),
                    pcd,
                )

                # get the initial bbox
                obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(
                    self.objects[obj_id].init_bbox
                )
                # change the line color to green and width to 2
                obb_ls.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
                obb_ls.lines = o3d.utility.Vector2iVector(obb_ls.lines)
                o3d.io.write_line_set(
                    os.path.join(debug_bbx_dir, f"initial_bbx_{obj_id}.ply"),
                    obb_ls,
                )
        return out_pose

    def _update_object_from_frontend(self, obj_id, fe_result):
        obj = self.objects[obj_id]

        # if obj.lost:
        #     return

        # 1. Pose update from front end
        if obj_id in fe_result.obj_poses:
            obj.pose = fe_result.obj_poses[obj_id]

        # 2. Save 3D correspondences (f2f)
        obj.curr_frame_points_3d = fe_result.valid_curr_3d.get(obj_id, None)
        obj.curr_frame_indices = fe_result.valid_indices.get(obj_id, None)

        # 3. Save registration stats
        stats = fe_result.reg_stats.get(obj_id, {})
        obj.inliers = stats.get("inliers", None)
        obj.residuals = stats.get("residuals", None)

        # 4. Cache uncertainty for gating & KF decision
        if obj.curr_frame_indices is not None:
            obj.curr_uncertainties = fe_result.uncertainties[obj.curr_frame_indices]
        else:
            obj.curr_uncertainties = None

        # 5. Compute mean residual
        obj.mean_residual = fe_result.mean_residuals[obj_id]

        # 6. Lost condition
        # obj.lost = obj.mean_residual > self.reg_residual_thres

    def _update_history_frames(self, frame, fe_result, track_table):
        self.hist_frames.append(frame)
        self.hist_fe_results.append(fe_result)
        self.hist_track_tables.append(copy.deepcopy(track_table))

    def _se3_delta(self, T_new, T_old):
        dT = T_new @ inverse_SE3(T_old)
        dt = float(np.linalg.norm(dT[:3, 3]))
        dR = scipy_R.from_matrix(dT[:3, :3]).magnitude()
        ddeg = float(np.degrees(dR))
        return dt, ddeg

    def __del__(self):
        for f in self.pose_log_files:
            if f:
                f.close()
