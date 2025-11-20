import os
import time
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import OPTIMIZER  # for config check
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.modules.object.object import Object
from point2pose.io.outputs.logger import DataLogger
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.logger_fields import RAGGED_FIELDS

# Components
from point2pose.pipeline.components.front_end import FrontEnd
from point2pose.pipeline.components.key_frame_manager import KeyFrameManager
from point2pose.pipeline.components.local_optimizer import LocalOptimizer
from point2pose.pipeline.components.key_frame_graph import KeyFrameGraph


class ModularPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        # Components
        self.frontend = FrontEnd(cfg)
        self.kf_manager = KeyFrameManager(cfg)
        self.local_optimizer = LocalOptimizer(cfg)
        self.kf_graph = KeyFrameGraph(cfg)

        # State
        self.track_table = PointTrackTable.new(n0=0)
        self.objects = []
        self.num_obj = self.pipeline_cfg.get("max_num_obj", 1)
        self.frame_id = 0

        # Logging
        self.save_pose = self.pipeline_cfg.get("save_pose", False)
        self.pose_save_path = self.pipeline_cfg.get("pose_save_path", "./poses")
        self.pose_log_files = []

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

        self._estimate_init_pose = self.pipeline_cfg.get("estimate_init_pose", False)
        self.use_graph_optimization = self.pipeline_cfg.get(
            "use_graph_optimization", False
        )
        self.reg_residual_thres = self.cfg.register.params.get("residual_thres", 0.07)

    def initialize_first_frame(self, frame):
        # 1. Initialize FrontEnd (Segmentation + Tracker)
        self.frontend.initialize(frame)

        # 2. Setup Objects based on segmentation
        if self.frontend.use_segmenter:
            self.num_obj = np.sum(np.asarray(self.frontend.segmenter.input_labels) == 1)

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
        self.kf_manager.initialize(
            frame, self.track_table, self.objects, self.frontend.tracker
        )

        # 4. Initial Optimization (optional, typically just prior)
        for obj_id in range(self.num_obj):
            self.local_optimizer.optimize(
                ObjectFrameData(
                    obj_id=obj_id,
                    frame_id=0,
                    pose=np.eye(4),
                    rel_pose=np.eye(4),
                    cur_3d=self.objects[obj_id].key_points,
                    cur_3d_idx=np.arange(len(self.objects[obj_id].key_points)),
                    inliers=np.ones(len(self.objects[obj_id].key_points), dtype=bool),
                    residuals=np.zeros(len(self.objects[obj_id].key_points)),
                    uncertainties=0.01 * np.ones(len(self.objects[obj_id].key_points)),
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
                    "is_key_frame": False,
                }
            )

        self.frame_id += 1
        return out_pose

    def step(self, frame):
        if self.frame_id == 0:
            return self.initialize_first_frame(frame)

        # 1. Front End (Tracking + Registration)
        fe_result = self.frontend.step(frame, self.track_table, self.objects)

        # per-object update
        for obj_id in range(self.num_obj):
            self._update_object_from_frontend(obj_id, fe_result)

        # 2. Track Table Update
        # FrontEndResult contains the data needed to update the table
        self.track_table.update_track_table(
            fe_result.tracks,
            fe_result.track_3d,
            fe_result.track_valid,
            fe_result.uncertainties,
            fe_result.visibles,
        )

        # 3. Key Frame Manager (Check & Sample)
        new_keyframes = self.kf_manager.update(
            frame, fe_result, self.track_table, self.objects, self.frontend.tracker
        )

        # If new keyframe, reset local optimizer
        if new_keyframes:
            print(
                f"Frame {self.frame_id}: Keyframe triggered. Resetting local optimizer."
            )
            self.local_optimizer.reset()

        # 4. Local Optimization
        if self.use_graph_optimization:
            for obj_id in range(self.num_obj):
                if obj_id not in fe_result.valid_indices:
                    continue

                # Only optimize if registration was good enough
                # if self.objects[obj_id].mean_residual < self.reg_residual_thres:
                if True:

                    object_frame_data = ObjectFrameData(
                        obj_id=obj_id,
                        frame_id=self.frame_id,
                        pose=self.objects[obj_id].pose,
                        rel_pose=fe_result.rel_poses[obj_id],
                        cur_3d=fe_result.valid_curr_3d[obj_id],
                        cur_3d_idx=fe_result.valid_indices[obj_id],
                        inliers=fe_result.reg_stats[obj_id].get(
                            "inliers", np.array([])
                        ),
                        residuals=fe_result.reg_stats[obj_id].get(
                            "residuals", np.array([])
                        ),
                        uncertainties=fe_result.uncertainties[
                            fe_result.valid_indices[obj_id]
                        ],
                    )

                    opt_result = self.local_optimizer.optimize(object_frame_data)
                    self.local_optimizer.update_object_state(
                        self.objects[obj_id], opt_result, self.track_table
                    )

        # 5. Logging
        self._log_step(frame, fe_result, new_keyframes)

        self.frame_id += 1

        # Return poses
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        for i in range(self.num_obj):
            out_pose[i] = self.objects[i].pose

        return out_pose

    def _log_step(self, frame, fe_result, new_keyframes):
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

            log_payload = {
                "timestamp": frame.timestamp,
                "frame_id": self.frame_id,
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
                "iter": reg_stats.get("iter", -1),
                # Valid extraction stats
                **valid_stats,
            }

            if new_keyframes:
                log_payload["new_keyframes"] = [
                    {
                        "frame_id": kf.frame_id,
                        "obj_id": kf.obj_id,
                        "num_kp": len(kf.keypoints_3d_camera),
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
        # Ported from PipelineSingleProcess
        for obj_id in range(self.num_obj):
            mask = frame.mask[obj_id, 0]
            coords_yx = torch.nonzero(mask > 0, as_tuple=False)
            valid_pxl_in_mask_g = coords_yx[:, [1, 0]].contiguous()
            valid_pxl_in_mask = valid_pxl_in_mask_g.cpu().numpy()

            initial_3d_points, _ = convert_pixel_to_world(
                pixel=valid_pxl_in_mask,
                depth_image=frame.depth,
                cam_intrinsics=frame.intrinsics,
                depth_factor=frame.depth_factor,
                remove_invalid=True,
            )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(initial_3d_points)

            # Simple outlier removal
            if True:  # pcd_radius_outlier_removal
                pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

            self.objects[obj_id].init_bbox = pcd.get_oriented_bounding_box()
            self.objects[obj_id].bbox = self.objects[obj_id].init_bbox

            # Set initial pose from bbox
            # This logic depends on whether init_pose is relative to camera or world.
            # Assuming standard object frame is aligned with bbox
            # self.objects[obj_id].pose = np.eye(4)
            init_pose = np.eye(4)
            init_pose[:3, :3] = self.objects[obj_id].init_bbox.R
            init_pose[:3, 3] = self.objects[obj_id].init_bbox.center
            self.objects[obj_id].init_pose = init_pose

            # Typically first frame pose is identity in camera frame,
            # and we define object frame relative to that?
            # Original code:
            # out_pose[obj_id, :3, :3] = ...R
            # out_pose[obj_id, :3, 3] = ...center
            # self.objects[obj_id].pose = np.eye(4)
            # self.objects[obj_id].init_pose = out_pose[obj_id]

            # So the object's pose in world (camera 0) is init_pose.
            # The object's current pose (T_c_o) might be what we are tracking?
            # PipelineSingleProcess tracks T_w_c or T_c_o?
            # "pose" usually means T_c_o (object in camera) or T_w_o (object in world/camera0).
            # Given "pose = pose @ objects[obj_id].pose", it seems to accumulate relative motion?
            # Let's stick to the original assignment:
            self.objects[obj_id].pose = np.eye(4)

    def _update_object_from_frontend(self, obj_id, fe_result):
        obj = self.objects[obj_id]

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
        obj.lost = obj.mean_residual > self.reg_residual_thres

    def __del__(self):
        for f in self.pose_log_files:
            if f:
                f.close()
