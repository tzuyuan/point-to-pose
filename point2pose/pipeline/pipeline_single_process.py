import os

import time

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as scipy_R

import cupoch as cph

# from point2pose.third_party.depth_anything_v2.dpt import DepthAnythingV2

from point2pose.core.build import build_from_cfg

import point2pose.modules as _modules  # trigger registrations
from point2pose.core.module_registry import (
    REGISTER,
    TRACKER,
    SAMPLER,
    CRITERION,
    SEGMENTER,
    OPTIMIZER,
)

# from point2pose.modules.optimizer_manager.optimizer_manager import OptimizerManager
from point2pose.data_types.criterion_context import CriterionContext
from point2pose.data_types.sampler_context import SamplerContext
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.data_types.object_frame_data import ObjectFrameData
from point2pose.modules.object.object import Object
from point2pose.modules.optimizer.isam2_optimizer import ISAM2Optimizer
from point2pose.data_types.optimizer_result import OptimizerResult
from point2pose.io.outputs.logger import DataLogger
from point2pose.io.outputs.point_cloud_io import save_reg_pcd, save_pcd
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import transform_pts, inverse_SE3
from point2pose.utils.se3_low_pass_filter import SE3LowPassFilter
from point2pose.utils.lie import se3_to_vec, log_SE3


class PipelineSingleProcess:
    """
    Note: This version supports a single object only for now.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        self.debug_level = self.pipeline_cfg.get("debug_level", 0)
        self.debug_dir = self.pipeline_cfg.get("debug_dir", None)

        # Pose logging configuration
        self.save_pose = self.pipeline_cfg.get("save_pose", False)
        self.pose_save_path = self.pipeline_cfg.get("pose_save_path", "./poses")

        # Registration statistics logging
        self.reg_stats_log = None
        self.reg_stats_log_path = None

        if self.debug_level > 0 and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            self._reg_debug_dir = os.path.join(self.debug_dir, "register")
            if not os.path.exists(self._reg_debug_dir):
                os.makedirs(self._reg_debug_dir, exist_ok=True)

        # Cropped point cloud debug saving
        self.save_cropped_pcd = self.pipeline_cfg.get("save_cropped_pcd", False)
        self.cropped_pcd_dir = self.pipeline_cfg.get(
            "cropped_pcd_dir",
            os.path.join(self.debug_dir if self.debug_dir else "./debug", "pcd"),
        )
        if self.save_cropped_pcd:
            os.makedirs(self.cropped_pcd_dir, exist_ok=True)

        # Create pose save directory if pose logging is enabled
        if self.save_pose:
            os.makedirs(self.pose_save_path, exist_ok=True)
            # Initialize registration stats log
            self.reg_stats_log_path = os.path.join(
                self.pose_save_path, "registration_stats.txt"
            )
            with open(self.reg_stats_log_path, "w", encoding="utf-8") as f:
                f.write(
                    "timestamp\tframe_id\tobj_id\tnum_points\titer\tthr\tres_mean\tres_median\tres_max\tnum_inliers\ttotal_points\tmean_residual_inliers\tmean_residual_outliers\n"
                )

        # Key points saving configuration
        self.save_key_points = self.pipeline_cfg.get("save_key_points", False)
        self.key_points_save_path = self.pipeline_cfg.get(
            "key_points_save_path", "./key_points"
        )

        # Create key points save directory if key point saving is enabled
        if self.save_key_points:
            os.makedirs(self.key_points_save_path, exist_ok=True)

        self.register = build_from_cfg(cfg.register, REGISTER)
        self.segmenter = build_from_cfg(cfg.segmenter, SEGMENTER)
        self.tracker = build_from_cfg(cfg.tracker, TRACKER)
        # self.state = build_from_cfg(cfg.state, STATE)
        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)
        # self.optimizer = build_from_cfg(cfg.optimizer, OPTIM)

        # from point2pose.third_party.depth_anything_v2.dpt import DepthAnythingV2

        ## TODO: make this a class

        self.use_depth_estimate = self.pipeline_cfg.get("use_depth_estimate", False)
        self.depth_encoder = self.pipeline_cfg.get("depth_encoder", "vits")
        self._device = self.pipeline_cfg.get("device", "cpu")
        self.depth_estimator = None  # lazy init
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

        max_num_obj = self.pipeline_cfg.get("max_num_obj", 1)
        self.optimizer = ISAM2Optimizer(cfg.optimizer)

        self.frame_id = 0

        self.crit_ctx = CriterionContext()
        self.samp_ctx = SamplerContext(frame=None)
        self.track_table = PointTrackTable.new(n0=0)

        self.num_obj = 0
        self.objects = []
        self.is_key_frame = {}

        # Pose log file handles for each object
        self.pose_log_files = []

        self._estimate_init_pose = self.pipeline_cfg.get("estimate_init_pose", False)
        self._frame2map_reg = self.pipeline_cfg.get("frame_to_map_reg", False)
        self._reg_remove_outside_mask = self.cfg.register.params.get(
            "remove_outside_mask", False
        )
        self.save_meta_data = self.pipeline_cfg.get("save_meta_data", False)
        self.meta_data_save_path = self.pipeline_cfg.get(
            "meta_data_save_path", "./meta_data"
        )
        if self.save_meta_data:
            os.makedirs(self.meta_data_save_path, exist_ok=True)
            self.data_logger = DataLogger(
                out_dir=self.meta_data_save_path,
                base_name="meata_data",
                # ragged fields include data that is not fixed shape
                ragged_fields={
                    # tracker stats
                    "track2d",
                    "uncertainties",
                    "visibles",
                    "track3d",
                    "valid_depth",
                    # object stats
                    "obj_key_points",
                    "obj_uncertainties",
                    "obj_valid",
                    "obj_key_point_frames",
                    # registeration stats
                    "reg_key_points_idx",
                    "reg_key_points",
                    "reg_curr3d",
                    "reg_inliers",
                    "reg_residuals",
                },
                also_save_h5=True,
            )

    # -------- one-time init with user clicks ----------
    def add_user_points(self, obj_points: list[list[int]], labels: list[int]):
        """
        points (List[List[int]]): List of points, each defined by [x, y].
            labels (List[int]): List of labels, each defined by 1 or 0.
                                1 means positive point, 0 means negative point.
        """
        # forward to segmenter; it will start tracking objects internally

        self.segmenter.add_input_points(obj_points, labels)

    def initialize_first_frame(self, frame):
        """
        Initialize the pipeline for the first frame.
        """
        # ------------- segmentation -------------
        self.segmenter.initialize(frame.rgb)

        # get number of objects
        self.num_obj = np.sum(np.asarray(self.segmenter.input_labels) == 1)

        # get segmentation mask
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits

        self.samp_ctx.frame = frame

        # ------------- sampler -------------
        self._sample_for_all_obj(self.samp_ctx)

        # ------------- tracker -------------
        self.tracker.initialize(frame)

        # initialize objects
        for obj_id in range(self.num_obj):
            self.objects.append(Object(obj_id))
            self.objects[obj_id].pose = np.eye(4)
            # assign the key points to the object
            self.objects[obj_id].key_points = self.track_table.track_3d[
                self.track_table.obj2track_map[obj_id]
            ]
            self.objects[obj_id].valid = self.track_table.valid[
                self.track_table.obj2track_map[obj_id]
            ]
            # Initialize frame tracking for initial key points (frame 0)
            self.objects[obj_id].key_point_frames = np.full(
                len(self.track_table.obj2track_map[obj_id]), 0, dtype=int
            )

            opt_results = self.optimizer.optimize(
                ObjectFrameData(
                    obj_id=obj_id,
                    frame_id=0,
                    pose=np.eye(4),
                    cur_3d=self.objects[obj_id].key_points,
                    cur_3d_idx=np.arange(len(self.objects[obj_id].key_points)),
                    inliers=np.ones(len(self.objects[obj_id].key_points), dtype=bool),
                    residuals=np.zeros(len(self.objects[obj_id].key_points)),
                    uncertainties=0.01 * np.ones(len(self.objects[obj_id].key_points)),
                )
            )

            self.is_key_frame[obj_id] = False

            # Initialize pose log file for this object if pose saving is enabled
            if self.save_pose:
                pose_log_path = os.path.join(
                    self.pose_save_path, f"obj_{obj_id}_pose.txt"
                )
                pose_log_file = open(pose_log_path, "w", encoding="utf-8")
                pose_log_file.write("# timestamp tx ty tz qx qy qz qw\n")
                # add initial pose which is the identity
                tum_pose = self._pose_matrix_to_tum_format(
                    np.eye(4), timestamp=time.time()
                )
                pose_log_file.write(tum_pose)
                self.pose_log_files.append(pose_log_file)
            else:
                self.pose_log_files.append(None)

        # Logger
        if self.save_meta_data:
            self.data_logger.log(
                {
                    "timestamp": frame.timestamp,
                    "frame_id": self.frame_id,
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
                }
            )

        self.crit_ctx.objects = self.objects
        self.criterion.initialize(self.crit_ctx)

        # estimate initial pose
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        if self._estimate_init_pose:
            self._estimate_init_pose_and_bbox_for_all_obj(frame)

        self.frame_id += 1
        return out_pose

    # -------- main loop per frame ----------
    def step(self, frame):
        """
        Step the pipeline for one frame.
        """
        if self.use_depth_estimate:
            self._ensure_depth_model()
            depth = self.depth_estimator.infer_image(frame.rgb)
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.astype(np.uint8)
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
                depth = depth.astype(np.float32, copy=False)

            frame.depth = self._fit_affine_depth_to_rs(
                est_depth=depth, rs_depth=frame.depth
            )

        # frame.depth = depth

        # if it's the first frame, initialize the pipeline
        if self.frame_id == 0:
            return self.initialize_first_frame(frame)

        # Start timing the entire step
        step_start_time = time.time()

        # ------------- segmenter -------------
        segmenter_start = time.time()
        _, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits  # mask is a torch tensor on gpu
        segmenter_time = time.time() - segmenter_start
        print(f"Frame {self.frame_id} - Segmentation: {segmenter_time:.4f}s")

        # Optionally save cropped point clouds per object using mask (with RGB)
        if self.save_cropped_pcd:
            num_objs = len(self.objects)
            for obj_id in range(num_objs):
                try:
                    mask = frame.mask[obj_id, 0]
                    coords_yx = torch.nonzero(mask > 0, as_tuple=False)
                    if coords_yx.numel() == 0:
                        continue
                    pxl_xy = coords_yx[:, [1, 0]].cpu().numpy()
                    world_pts, valid = convert_pixel_to_world(
                        pixel=pxl_xy,
                        depth_image=frame.depth,
                        cam_intrinsics=frame.intrinsics,
                        depth_factor=frame.depth_factor,
                    )
                    if world_pts.size == 0 or not np.any(valid):
                        continue
                    world_pts = world_pts[valid]
                    pxl_xy_valid = pxl_xy[valid]
                    rgb_vals = frame.rgb[pxl_xy_valid[:, 1], pxl_xy_valid[:, 0]]
                    save_pcd(
                        world_pts,
                        rgb_vals,
                        self.cropped_pcd_dir,
                        f"obj_{obj_id}_frame_{self.frame_id}",
                    )
                except Exception:
                    pass

        self.samp_ctx.frame = frame

        # ------------- tracker -------------
        tracker_start = time.time()
        tracks, uncertainties, visibles = self.tracker.track_once(frame)

        print(f"num tracks: {len(tracks)}")
        # Convert visibles to boolean since TAPIR returns float32
        # visibles = visibles.astype(bool)
        visibles = visibles > 0.5
        tracker_time = time.time() - tracker_start
        print(f"Frame {self.frame_id} - Tracking: {tracker_time:.4f}s")

        # convert tracks into 3D points using depth and intrinsics
        conversion_start = time.time()
        track_3d, track_valid = convert_pixel_to_world(
            pixel=tracks,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
        )
        conversion_time = time.time() - conversion_start
        print(
            f"Frame {self.frame_id} - Pixel to World conversion: {conversion_time:.4f}s"
        )

        # ------------- register -------------
        register_start = time.time()
        reg_stats = {}
        reg_stats_obj0 = {}
        for obj_id in range(self.num_obj):

            # check and update the keypoints from the graph optimizer

            # frame to map registration
            if self._frame2map_reg:
                t_extract_start = time.time()

                if self._reg_remove_outside_mask:
                    idx, key_points, curr3d, new_visibles = (
                        self._extract_valid_key_points_mask_remove(
                            self.objects[obj_id],
                            self.track_table.obj2track_map[obj_id],
                            tracks,
                            track_3d,
                            visibles,
                            track_valid,
                            uncertainties,
                            frame.mask[obj_id, 0],
                            uncertainty_thres=0.6,
                        )
                    )
                    self.track_table.visible = new_visibles
                else:
                    idx, key_points, curr3d = self._extract_valid_key_points(
                        self.objects[obj_id],
                        self.track_table.obj2track_map[obj_id],
                        track_3d,
                        visibles,
                        track_valid,
                        uncertainties,
                        uncertainty_thres=0.2,
                    )

                t_extract_end = time.time()
                print(
                    f"Extract valid key points time: {t_extract_end - t_extract_start:.4f}s"
                )

                # TODO: put it at better place

                if self.save_meta_data and obj_id == 0:
                    reg_stats_obj0.update(
                        {
                            "reg_key_points_idx": idx,
                            "reg_key_points": key_points,
                            "reg_curr3d": curr3d,
                        }
                    )

                if key_points.shape[0] < 3 or curr3d.shape[0] < 3:
                    # self.data_logger.log({"too_few_points": 1})
                    reg_stats_obj0.update({"too_few_points": 1})
                    print(
                        f"Frame {self.frame_id} - Object {obj_id} - Too few points: {key_points.shape[0]}"
                    )
                    continue

                # self.data_logger.log({"too_few_points": 0})
                reg_stats_obj0.update({"too_few_points": 0})

                # TODO optimize this by saving the previous pose directly
                prev_pose = self.objects[obj_id].pose

                # key points are represented in the first frame coordinate system
                # pose_i_0 is the transformation from the first frame to the current frame
                # init_pose here is the warm start for the optimization, not the initial pose of the object
                if self.register.type == "svd_uncertainty_outlier":
                    pose_i_0, reg_stats = self.register.register(
                        key_points,
                        curr3d,
                        init_pose=prev_pose,
                        sigma_tgt=uncertainties[idx],
                    )
                else:
                    pose_i_0, reg_stats = self.register.register(
                        key_points, curr3d, init_pose=prev_pose
                    )

                # xi se(3) lie algebra. (omega, v)
                xi_im1_i = se3_to_vec(log_SE3(inverse_SE3(pose_i_0) @ prev_pose))

                # omega_norm = np.linalg.norm(xi_im1_i[:3])
                # v_norm = np.linalg.norm(xi_im1_i[3:])
                inliers = reg_stats["inliers"]
                mean_residual = np.mean(reg_stats["residuals"][inliers])
                self.objects[obj_id].omega = xi_im1_i[:3]
                self.objects[obj_id].v = xi_im1_i[3:]
                self.objects[obj_id].mean_residual = mean_residual
                if mean_residual < 0.07:
                    pose = pose_i_0

                    # update object info
                    self.objects[obj_id].pose = pose

                    # update object frame data and send to optimizer manager
                    object_frame_data = ObjectFrameData(
                        obj_id=obj_id,
                        frame_id=self.frame_id,
                        pose=pose,
                        cur_3d=curr3d,
                        cur_3d_idx=idx,
                        inliers=reg_stats.get("inliers", np.array([])),
                        residuals=reg_stats.get("residuals", np.array([])),
                        uncertainties=uncertainties[idx],
                    )

                    # once the input is set, the optimizer manager will start the optimization process
                    opt_results = self.optimizer.optimize(object_frame_data)

                    # if we have optmized results, update the object info
                    if opt_results is not None:
                        ## TODO: do we update pose here...?
                        # self.objects[obj_id].pose = opt_results.pose_optimized
                        # print(
                        #     f"key points before optimization: {self.objects[obj_id].key_points.shape}"
                        # )
                        # print(self.objects[obj_id].key_points)
                        kp_idx = opt_results.key_points_idx_optimized
                        # kp_idx = self.track_table.obj2track_map[obj_id][kp_idx]
                        kp_optimized = opt_results.key_points_optimized
                        self.objects[obj_id].key_points[kp_idx] = kp_optimized
                        # print(
                        #     f"key points after optimization: {self.objects[obj_id].key_points.shape}"
                        # )
                        # print(self.objects[obj_id].key_points)
                else:
                    pose = self.objects[obj_id].pose

                # Log pose in TUM format
                if self.save_pose and self.pose_log_files[obj_id] is not None:
                    tum_pose = self._pose_matrix_to_tum_format(
                        self.objects[obj_id].pose, timestamp=time.time()
                    )
                    self.pose_log_files[obj_id].write(tum_pose)
                    self.pose_log_files[obj_id].flush()

                # Log registration statistics
                self._log_registration_stats(
                    self.frame_id, obj_id, key_points.shape[0], reg_stats
                )

                # TODO: put it at better place
                if self.save_meta_data and obj_id == 0:
                    reg_stats_obj0.update(
                        {
                            "reg_iter": reg_stats.get("iter", -1),
                            "reg_thr": reg_stats.get("thr", -1.0),
                            "reg_residuals": reg_stats.get("residuals", np.array([])),
                            "reg_inliers": reg_stats.get("inliers", np.array([])),
                        }
                    )

                print(f"pose at frame {self.frame_id}: {pose}")

                if self.debug_level > 1:

                    self.save_optimizer_key_points(obj_id)

                    save_reg_pcd(
                        key_points,
                        curr3d,
                        pose_i_0,
                        self._reg_debug_dir,
                        f"obj_{obj_id}_frame_{self.frame_id}",
                        reg_stats,
                    )

            # frame to frame registration
            else:
                idx, prev3d, curr3d = self._extract_valid_idx_points_for_obj(
                    obj_id, self.track_table, track_3d, track_valid, visibles
                )
                # self.prev3d_before = self.track_table.track_3d
                # self.curr3d_before = track_3d
                # self.prev3d_after = prev3d
                # self.curr3d_after = curr3d

                pose, reg_stats = self.register.register(prev3d, curr3d)
                self.objects[obj_id].pose = pose @ self.objects[obj_id].pose

                # Log pose in TUM format
                if self.save_pose and self.pose_log_files[obj_id] is not None:
                    tum_pose = self._pose_matrix_to_tum_format(
                        self.objects[obj_id].pose, timestamp=time.time()
                    )
                    self.pose_log_files[obj_id].write(tum_pose)
                    self.pose_log_files[obj_id].flush()

                # Log registration statistics
                if self.debug_level > 0:
                    self._log_registration_stats(
                        self.frame_id, obj_id, prev3d.shape[0], reg_stats
                    )

                print(f"Frame {self.frame_id} - Object {obj_id} - Pose: {pose}")
                if self.debug_level > 1:
                    save_reg_pcd(
                        prev3d,
                        curr3d,
                        pose,
                        self._reg_debug_dir,
                        f"obj_{obj_id}_frame_{self.frame_id}",
                    )
        register_time = time.time() - register_start
        # print(f"Frame {self.frame_id} - Registration: {register_time:.4f}s")

        # update track table to the current frame info
        table_update_start = time.time()
        self.track_table.update_track_table(
            tracks, track_3d, track_valid, uncertainties, visibles
        )
        table_update_time = time.time() - table_update_start
        # print(f"Frame {self.frame_id} - Track table update: {table_update_time:.4f}s")

        # ---------------- log meta data ----------------
        if self.save_meta_data:
            print("obj_id: 0---------------------------")
            print(self.objects[0].pose)
            self.data_logger.log(
                {
                    "timestamp": frame.timestamp,
                    "frame_id": self.frame_id,
                    "track2d": tracks,
                    "uncertainties": uncertainties,
                    "visibles": visibles,
                    "track3d": track_3d,
                    "valid": track_valid,
                    "valid_depth": frame.depth,
                    "obj_init_pose": self.objects[0].init_pose,
                    "obj_pose": self.objects[0].pose,
                    "obj_key_points": self.objects[0].key_points,
                    "obj_uncertainties": self.objects[0].uncertainties,
                    "obj_valid": self.objects[0].valid,
                    "obj_key_point_frames": self.objects[0].key_point_frames,
                    **reg_stats_obj0,
                }
            )

        # ------------- criterion and sample -------------
        sampling_start = time.time()
        self.crit_ctx.update_criterion_context(
            cur_iter=self.frame_id,
            uncertainty=uncertainties,
            reg_stats=reg_stats,
            track_table=self.track_table,
            frame=frame,
        )
        self.samp_ctx.update_sampler_context(frame=frame, track_table=self.track_table)
        self._check_and_sample_for_all_obj(self.samp_ctx)
        sampling_time = time.time() - sampling_start
        # print(f"Frame {self.frame_id} - Criterion and sampling: {sampling_time:.4f}s")

        self.frame_id += 1

        # Print total step time
        total_step_time = time.time() - step_start_time
        print(
            f"Frame {self.frame_id-1} - Total step time: {total_step_time:.4f}s - Breakdown: seg={segmenter_time:.3f}s, track={tracker_time:.3f}s, conv={conversion_time:.3f}s, reg={register_time:.3f}s, table={table_update_time:.3f}s, sample={sampling_time:.3f}s"
        )
        print("-" * 60)

        return np.tile(np.eye(4), (self.num_obj, 1, 1))

    def _sample_for_all_obj(self, context: SamplerContext):
        """
        Sample points for all objects, and add the query points to the tracker.
        """
        all_sampled_points = np.empty((0, 2))
        frame = context.frame
        for obj_id in range(self.num_obj):
            # Sample points
            new_sampled_points = self.sampler.sample(context, obj_id)
            all_sampled_points = np.vstack((all_sampled_points, new_sampled_points))
            print(f"new_sampled_points shape: {new_sampled_points.shape}")
            # add new sampled points to the tracker
            new_indices = self.tracker.add_query_points(frame, new_sampled_points)

            # update track2obj_map and obj2track_map
            self.track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

        # convert tracks into 3D points using depth and intrinsics
        track_3d, track_valid = convert_pixel_to_world(
            pixel=all_sampled_points,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
        )

        self.track_table.update_track_table(
            track_2d=all_sampled_points,
            track_3d=track_3d,
            valid=track_valid,
            uncertainties=0.5 * np.ones(len(all_sampled_points), dtype=float),
            visibles=np.ones(len(all_sampled_points), dtype=bool),
        )

    def _check_and_sample_for_all_obj(self, samp_context: SamplerContext):
        """
        Check sample criteria for all objects and sample points for all objects if the criterion is met.
        """
        # Check sample criteria
        frame = samp_context.frame
        for obj_id in range(self.num_obj):
            ## Temp outlier rejection
            ## TODO: make outlier rejection better organized
            omega_norm = np.linalg.norm(self.objects[obj_id].omega)
            v_norm = np.linalg.norm(self.objects[obj_id].v)
            if (
                omega_norm > 0.1
                and v_norm > 1
                and self.objects[obj_id].mean_residual > 0.07
            ):
                continue

            ## TODO: make crit_ctx to be object-specific?
            if self.criterion.check_sample_criterion(self.crit_ctx, obj_id):
                self.is_key_frame[obj_id] = True
                # Sample points
                new_sampled_points = self.sampler.sample(samp_context, obj_id)

                # add new sampled points to the tracker
                new_indices = self.tracker.add_query_points(frame, new_sampled_points)

                # update track2obj_map and obj2track_map
                self.track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

                # update the key points of the objects
                new_points_3d, valid_new_points_3d = convert_pixel_to_world(
                    pixel=new_sampled_points,
                    depth_image=frame.depth,
                    cam_intrinsics=frame.intrinsics,
                    depth_factor=frame.depth_factor,
                )

                # key points are represented in the first frame coordinate system
                new_points_3d_frame_0 = transform_pts(
                    inverse_SE3(self.objects[obj_id].pose),
                    new_points_3d,
                )

                # add new key points to the object with 0.5 uncertainty
                self.objects[obj_id].add_key_points(
                    new_points_3d_frame_0,
                    0.5 * np.ones(len(new_points_3d_frame_0), dtype=float),
                    valid_new_points_3d,
                    self.frame_id,
                )

                # Save key points with frame-based coloring if enabled
                if self.save_key_points:
                    self._save_key_points_for_object(obj_id)

                self.track_table.append_track_table(
                    track_2d=new_sampled_points,
                    track_3d=new_points_3d,
                    valid=valid_new_points_3d,
                    uncertainties=0.5 * np.ones(len(new_sampled_points), dtype=float),
                    visibles=np.ones(len(new_sampled_points), dtype=bool),
                )

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

    def _extract_valid_key_points(
        self,
        obj,
        obj_idx,
        cur_pts_3d,
        cur_visible,
        cur_valid,
        cur_uncertainties,
        uncertainty_thres=0.3,
    ):
        """
        Args:
            obj_idx:     (N,) global indices for this object's points
            cur_pts_3d:  (N,3) global 3D point array
            cur_visible: (N,) bool visibility mask for all points
            cur_valid:   (N,) bool validity mask for all points
            obj.key_points: (M,3) per-object key points (aligned with obj_idx)
        Returns:
            idx:        (K,) global indices of valid & visible points
            key_points: (K,3) subset from obj.key_points
            curr3d:     (K,3) subset from cur_pts_3d
        """
        obj_idx = np.asarray(obj_idx)

        valid_kp_bool = np.asarray(obj.valid, dtype=bool)  # M bool

        # object-local visibility mask (aligned with obj_idx)
        vis_obj = np.asarray(cur_visible, dtype=bool)[obj_idx]
        val_obj = np.asarray(cur_valid, dtype=bool)[obj_idx]
        uncer_obj = (
            np.asarray(cur_uncertainties, dtype=float)[obj_idx] < uncertainty_thres
        )
        both_mask = vis_obj & val_obj & valid_kp_bool & uncer_obj  # (M,)

        # global indices for valid+visible points
        idx = obj_idx[both_mask]  # (K,)

        # per-object arrays use local mask; global arrays use global idx
        key_points = obj.key_points[both_mask].copy()
        curr3d = cur_pts_3d[idx].copy()

        return idx, key_points, curr3d

    # def _extract_valid_key_points(
    #     self,
    #     obj,
    #     obj_idx,  # (M,) global indices for this object's points
    #     cur_pts_2d,  # (N,2) global 2D pixel coords (float ok)
    #     cur_pts_3d,  # (N,3) global 3D points
    #     cur_visible,  # (N,) bool  -- UPDATED IN-PLACE
    #     cur_valid,  # (N,) bool
    #     cur_uncertainties,  # (N,) float
    #     frame_mask_np,  # (H,W) or (1,H,W) NumPy array; >0 means inside
    #     uncertainty_thres=0.3,
    # ):
    #     """
    #     Side effect:
    #         - Updates cur_visible (N,) to False for points OUTSIDE the mask
    #         or with non-finite 2D coordinates.

    #     Returns:
    #         idx:        (K,) np int64 global indices of valid points
    #         key_points: (K,3) np array from obj.key_points
    #         curr3d:     (K,3) np array from cur_pts_3d
    #     """
    #     # Normalize mask to boolean HxW
    #     if frame_mask_np.ndim == 3 and frame_mask_np.shape[0] == 1:
    #         mask_bool = frame_mask_np[0] > 0
    #     elif frame_mask_np.ndim == 2:
    #         mask_bool = frame_mask_np > 0
    #     else:
    #         raise ValueError(
    #             "frame_mask_np must be (H,W) or (1,H,W) for a single object."
    #         )
    #     H, W = mask_bool.shape

    #     # Per-object views
    #     val_obj = cur_valid[obj_idx]  # (M,)
    #     uncer_obj = cur_uncertainties[obj_idx] < float(uncertainty_thres)  # (M,)

    #     valid_kp_obj = getattr(obj, "valid", np.ones(len(obj_idx), dtype=bool))
    #     valid_kp_obj = np.asarray(valid_kp_obj, dtype=bool)  # (M,)

    #     # 2D coords for these points
    #     pts2d_obj = cur_pts_2d[obj_idx]  # (M,2)
    #     finite_xy = np.isfinite(pts2d_obj).all(axis=1)  # (M,)

    #     # Convert to pixel indices (round to nearest) and clamp
    #     x = np.rint(pts2d_obj[:, 0]).astype(np.int64)
    #     y = np.rint(pts2d_obj[:, 1]).astype(np.int64)
    #     np.clip(x, 0, W - 1, out=x)
    #     np.clip(y, 0, H - 1, out=y)

    #     # Mask lookup
    #     inside_mask = mask_bool[y, x]  # (M,) bool

    #     # ====== Update global visibility (N,) in-place ======
    #     # Mark points OUTSIDE mask or with bad 2D as NOT visible
    #     outside_or_bad = (~inside_mask) | (~finite_xy)  # (M,)
    #     if outside_or_bad.any():
    #         cur_visible[obj_idx[outside_or_bad]] = False

    #     # Recompute vis_obj after the update
    #     vis_obj = cur_visible[obj_idx]  # (M,)

    #     # Final selection (keep inside the mask)
    #     both_mask = (
    #         vis_obj & val_obj & uncer_obj & valid_kp_obj & finite_xy & inside_mask
    #     )  # (M,)

    #     # Return subsets
    #     idx = obj_idx[both_mask]  # (K,)
    #     key_points = obj.key_points[both_mask].copy()  # (K,3)
    #     curr3d = cur_pts_3d[idx].copy()  # (K,3)

    #     return idx, key_points, curr3d, cur_visible

    def _extract_valid_key_points_mask_remove(
        self,
        obj,
        obj_idx,  # (M,) global indices for this object's points (np int)
        cur_pts_2d,  # (N,2) global 2D pixel coords (np float ok)
        cur_pts_3d,  # (N,3) global 3D points (np)
        cur_visible,  # (N,) bool  -- UPDATED IN-PLACE (np)
        cur_valid,  # (N,) bool (np)
        cur_uncertainties,  # (N,) float (np)
        frame_mask_gpu,  # (H,W) or (1,H,W) torch.Tensor on GPU; >0 means inside
        uncertainty_thres=0.3,
    ):
        """
        Side effect:
            - Updates cur_visible (N,) to False for points OUTSIDE the mask
            or with non-finite 2D coordinates.

        Returns:
            idx:        (K,) np int64 global indices of valid points
            key_points: (K,3) np array from obj.key_points
            curr3d:     (K,3) np array from cur_pts_3d
            cur_visible: updated visibility (same array, returned for convenience)
        """
        # --- normalize mask to boolean 2D on GPU ---
        if frame_mask_gpu.ndim == 3 and frame_mask_gpu.shape[0] == 1:
            mask2d = frame_mask_gpu[0]
        elif frame_mask_gpu.ndim == 2:
            mask2d = frame_mask_gpu
        else:
            raise ValueError(
                "frame_mask_gpu must be (H,W) or (1,H,W) for a single object."
            )
        # make sure it's boolean without allocating extra tensors if already bool
        mask_bool = (mask2d > 0) if mask2d.dtype != torch.bool else mask2d
        H, W = mask_bool.shape

        # --- numpy views/arrays ---
        obj_idx = np.asarray(obj_idx, dtype=np.int64)
        cur_pts_2d = np.asarray(cur_pts_2d, dtype=np.float32)
        cur_pts_3d = np.asarray(cur_pts_3d)
        cur_visible = np.asarray(cur_visible, dtype=bool)
        cur_valid = np.asarray(cur_valid, dtype=bool)
        cur_uncertainties = np.asarray(cur_uncertainties, dtype=np.float32)

        # per-object
        val_obj = cur_valid[obj_idx]
        uncer_obj = cur_uncertainties[obj_idx] < float(uncertainty_thres)
        valid_kp_obj = np.asarray(
            getattr(obj, "valid", np.ones(len(obj_idx), dtype=bool)), dtype=bool
        )

        pts2d_obj = cur_pts_2d[obj_idx]  # (M,2)
        finite_xy = np.isfinite(pts2d_obj).all(axis=1)  # (M,)

        # round -> int pixel indices, clamp to image
        x = np.rint(pts2d_obj[:, 0]).astype(np.int64)
        y = np.rint(pts2d_obj[:, 1]).astype(np.int64)
        np.clip(x, 0, W - 1, out=x)
        np.clip(y, 0, H - 1, out=y)

        # --- sample GPU mask with torch indexing (fast & minimal conversion) ---
        dev = mask_bool.device
        x_t = torch.from_numpy(x).to(device=dev, dtype=torch.long)
        y_t = torch.from_numpy(y).to(device=dev, dtype=torch.long)
        inside_mask_np = mask_bool[y_t, x_t].detach().cpu().numpy()  # (M,) bool

        # ====== update global visibility (N,) in-place ======
        # mark points outside the mask or with bad 2D as NOT visible
        outside_or_bad = (~inside_mask_np) | (~finite_xy)
        if outside_or_bad.any():
            cur_visible[obj_idx[outside_or_bad]] = False

        # recompute vis after update
        vis_obj = cur_visible[obj_idx]

        # final selection (keep inside mask)
        both_mask = (
            vis_obj & val_obj & uncer_obj & valid_kp_obj & finite_xy & inside_mask_np
        )

        idx = obj_idx[both_mask]  # (K,)
        key_points = obj.key_points[both_mask].copy()
        curr3d = cur_pts_3d[idx].copy()

        return idx, key_points, curr3d, cur_visible

    def _extract_valid_idx_points_for_obj(
        self,
        obj_id: int,
        track_table,  # holds previous-frame state
        curr_pts_3d: np.ndarray,  # (N+k, 3) current 3D points from convert_pixel_to_world
        curr_valid: np.ndarray,  # (N+k,) bool
        curr_visible: np.ndarray,  # (N+k,) bool
    ):
        """
        Returns:
            idx:     (M,) int indices where correspondence holds and both frames say valid&visible
            prev3d:  (M,3) 3D points from previous frame
            curr3d:  (M,3) 3D points from current frame
        """

        # indices belonging to this object
        obj_idx = track_table.obj2track_map[obj_id]  # np.ndarray of indices

        # restrict to indices that existed in the previous frame (exclude newly-sampled k)
        # also guard against any array-length mismatch
        n_prev = len(track_table.valid)  # length of previous arrays
        n_curr = len(curr_valid)
        common_idx = obj_idx[(obj_idx < n_prev) & (obj_idx < n_curr)]

        # intersection mask: valid & visible in BOTH frames
        both_mask = (
            curr_visible[common_idx]
            & curr_valid[common_idx]
            & track_table.visible[common_idx]
            & track_table.valid[common_idx]
        )

        idx = common_idx[both_mask]
        prev3d = track_table.track_3d[idx].copy()
        curr3d = curr_pts_3d[idx].copy()
        return idx, prev3d, curr3d

    # def _refine_registration(self):
    # pose_i_0, reg_stats = self.register.register(
    #     key_points, curr3d, init_pose=prev_pose
    # )

    # pose_i_0 = pose_init_guess

    # refiner_t = time.time()
    # # refine the solution using another registration?
    # criteria = cph.registration.ICPConvergenceCriteria()
    # criteria.max_iteration = 5

    # key_points_pcd = cph.geometry.PointCloud()
    # key_points_pcd.points = cph.utility.Vector3fVector(key_points)
    # masked_pcd = cph.geometry.PointCloud()

    # mask = frame.mask[obj_id, 0]  # [H, W] on cuda, dtype=bool/uint8

    # # Get (y,x) indices on GPU; switch to (x,y) like your NumPy code
    # coords_yx = torch.nonzero(mask > 0, as_tuple=False)  # [N, 2], (y,x)
    # valid_pxl_in_mask_g = coords_yx[
    #     :, [1, 0]
    # ].contiguous()  # [N, 2], (x,y), still on GPU

    # valid_pxl_in_mask = valid_pxl_in_mask_g.cpu().numpy()

    # masked_pts, _ = convert_pixel_to_world(
    #     pixel=valid_pxl_in_mask,
    #     depth_image=frame.depth,
    #     cam_intrinsics=frame.intrinsics,
    #     depth_factor=frame.depth_factor,
    #     remove_invalid=True,
    # )

    # masked_pcd.points = cph.utility.Vector3fVector(masked_pts)

    # refine_result = cph.registration.registration_icp(
    #     key_points_pcd,
    #     masked_pcd,
    #     max_correspondence_distance=0.015,
    #     init=pose_i_0.astype(np.float32),
    #     estimation_method=cph.registration.TransformationEstimationPointToPlane(),
    #     criteria=criteria,
    # )

    # pose_i_0 = refine_result.transformation

    # # temp saving
    # if self.debug_level > 1:

    #     save_reg_pcd(
    #         key_points,
    #         masked_pts,
    #         pose_to_key,
    #         self._reg_debug_dir,
    #         f"obj_{obj_id}_frame_{self.frame_id}_refine",
    #         reg_stats,
    #     )

    def _pose_matrix_to_tum_format(self, pose_matrix, timestamp=None):
        """
        Convert a 4x4 pose matrix to TUM format: timestamp tx ty tz qx qy qz qw
        Args:
            pose_matrix: 4x4 numpy array
            timestamp: float timestamp (if None, uses current time)
        Returns:
            str: TUM format pose string
        """
        if timestamp is None:
            timestamp = time.time()

        # Extract translation
        tx, ty, tz = pose_matrix[:3, 3]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = pose_matrix[:3, :3]
        rotation = scipy_R.from_matrix(rotation_matrix)
        qx, qy, qz, qw = rotation.as_quat()  # Returns [x, y, z, w]

        return f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"

    def _log_registration_stats(self, frame_id, obj_id, num_points, reg_stats):
        """
        Log registration statistics to file
        Args:
            frame_id: int current frame ID
            obj_id: int object ID
            num_points: int number of points used for registration
            reg_stats: dict registration statistics from register method
        """
        if self.reg_stats_log_path is None:
            return

        # Get current timestamp
        timestamp = time.time()

        # Extract statistics with defaults
        iter_count = reg_stats.get("iter", -1)
        threshold = reg_stats.get("thr", -1.0)
        residuals = reg_stats.get("residuals", np.array([]))
        inliers = reg_stats.get("inliers", np.array([]))

        # Calculate residual statistics
        if len(residuals) > 0:
            res_mean = np.mean(residuals)
            res_median = np.median(residuals)
            res_max = np.max(residuals)
        else:
            res_mean = res_median = res_max = -1.0

        num_inliers = np.sum(inliers) if len(inliers) > 0 else -1

        # Additional stats
        total_points = int(len(residuals)) if residuals is not None else 0
        if total_points > 0 and len(inliers) == total_points:
            inlier_mask = inliers.astype(bool)
            outlier_mask = ~inlier_mask
            mean_residual_inliers = (
                float(np.mean(residuals[inlier_mask])) if np.any(inlier_mask) else -1.0
            )
            mean_residual_outliers = (
                float(np.mean(residuals[outlier_mask]))
                if np.any(outlier_mask)
                else -1.0
            )
        else:
            mean_residual_inliers = -1.0
            mean_residual_outliers = -1.0

        # Write to file
        with open(self.reg_stats_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{timestamp:.6f}\t{frame_id}\t{obj_id}\t{num_points}\t{iter_count}\t{threshold:.6f}\t{res_mean:.6f}\t{res_median:.6f}\t{res_max:.6f}\t{num_inliers}\t{total_points}\t{mean_residual_inliers:.6f}\t{mean_residual_outliers:.6f}\n"
            )

    # def _log_stats(self, frame_id, obj_id, num_points, reg_stats):
    #     """
    #     Log registration statistics to file
    #     Args:
    #         frame_id: int current frame ID
    #         obj_id: int object ID
    #         num_points: int number of points used for registration
    #         reg_stats: dict registration statistics from register method
    #     """
    #     self.data_logger.log(
    #         {
    #             "timestamp": time.time(),
    #             "frame_id": frame_id,
    #             "obj_id": obj_id,
    #             "num_points": num_points,
    #             # tracker stats
    #             "track2d": track2d,
    #             "track3d": track3d,
    #             "valid": valid,
    #             "uncertainties": uncertainties,
    #             "visibles": visibles,
    #             "valid_depth": valid_depth,
    #             # object stats
    #             "obj_key_points": obj_key_points,
    #             # registration stats
    #             "reg_iter": reg_stats.get("iter", -1),
    #             "reg_thr": reg_stats.get("thr", -1.0),
    #             "reg_residuals": reg_stats.get("residuals", np.array([])),
    #             "reg_inliers": reg_stats.get("inliers", np.array([])),
    #         }
    #     )

    def _save_key_points_for_object(self, obj_id: int):
        """
        Save key points for a specific object with frame-based coloring.

        Args:
            obj_id (int): Object ID to save key points for
        """
        if not self.save_key_points or obj_id >= len(self.objects):
            return

        # Create filename with frame ID
        filename = f"obj_{obj_id}_keypoints_frame_{self.frame_id}.ply"
        save_path = os.path.join(self.key_points_save_path, filename)

        # Save key points with frame-based coloring
        self.objects[obj_id].save_key_points_with_colors(save_path, self.frame_id)

        print(
            f"Saved key points for object {obj_id} at frame {self.frame_id} to {save_path}"
        )

    def save_optimizer_key_points(self, obj_id: int):
        """
        Save key points after optimization into `debug/optimizer` using the
        existing `_save_key_points_for_object()` method.

        This temporarily overrides `self.key_points_save_path` and
        `self.save_key_points` to ensure saving occurs in the requested
        debug directory without changing global configuration.

        Args:
            obj_id (int): Object ID whose key points to save
        """
        # Determine optimizer debug directory
        optimizer_dir = (
            os.path.join(self.debug_dir, "optimizer")
            if self.debug_dir
            else "debug/optimizer"
        )
        os.makedirs(optimizer_dir, exist_ok=True)

        # Stash original settings
        original_save_flag = self.save_key_points
        original_save_path = self.key_points_save_path

        try:
            # Force-save into optimizer directory
            self.save_key_points = True
            self.key_points_save_path = optimizer_dir
            self._save_key_points_for_object(obj_id)
        finally:
            # Restore original settings
            self.save_key_points = original_save_flag
            self.key_points_save_path = original_save_path

    def __del__(self):
        """Cleanup method to close pose log files"""
        for log_file in self.pose_log_files:
            if log_file is not None:
                log_file.close()

    def _ensure_depth_model(self):
        if self.depth_estimator is not None:
            return
        # local import to avoid import-time CUDA/BLAS side-effects
        from third_party.depth_anything_v2.dpt import DepthAnythingV2

        m = DepthAnythingV2(**self._depth_model_cfg[self.depth_encoder])

        # load on CPU first (faster, avoids early CUDA init), then move to device
        state = torch.load(
            f"checkpoints/depth_anything/depth_anything_v2_{self.depth_encoder}.pth",
            map_location="cpu",
        )
        m.load_state_dict(state)
        self.depth_estimator = m.to(self._device).eval()

    def _fit_affine_depth_to_rs(
        self,
        est_depth,  # HxW, float32/float64 or uint8; relative depth from estimator
        rs_depth,  # HxW, uint16 (mm) or float (m) from RealSense
        depth_factor=None,  # e.g. 1000.0 if rs_depth is uint16 in millimeters
        max_range_m=8.0,  # use only RS pixels < 8 meters for the fit
        mask=None,  # optional [H,W] or [N,1,H,W] torch.BoolTensor: restrict fit region
        min_samples=64,  # require at least this many samples to fit
    ):
        """
        Fit s,b by least-squares on valid pixels where 0 < RS < max_range_m.
        Returns d_metric (meters) as float32 with d_metric = s * est_depth + b.
        """
        # ---- to numpy ----
        est_is_torch = isinstance(est_depth, torch.Tensor)
        rs_is_torch = isinstance(rs_depth, torch.Tensor)

        est = (
            est_depth.detach().cpu().numpy() if est_is_torch else np.asarray(est_depth)
        )
        rs = rs_depth.detach().cpu().numpy() if rs_is_torch else np.asarray(rs_depth)

        # If estimator accidentally produced uint8, undo to [0,1] so LS still works
        if est.dtype == np.uint8:
            est = est.astype(np.float32) / 255.0
        else:
            est = est.astype(np.float32, copy=False)

        # RealSense -> meters
        if depth_factor is None:
            depth_factor = 1000.0 if np.issubdtype(rs.dtype, np.integer) else 1.0
        rs_m = rs.astype(np.float32, copy=False) / float(depth_factor)

        # Valid pixels: finite, 0 < RS < max_range_m, finite est
        valid = (
            np.isfinite(rs_m)
            & (rs_m > 0)
            & (rs_m < float(max_range_m))
            & np.isfinite(est)
        )

        # Optional mask restriction (union if [N,1,H,W])
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 4:
                    m2d = (mask[:, 0] > 0).any(dim=0)
                elif mask.ndim == 2:
                    m2d = mask > 0
                else:
                    m2d = None
                if m2d is not None:
                    valid &= m2d.detach().cpu().numpy().astype(bool)
            elif isinstance(mask, np.ndarray) and mask.ndim == 2:
                valid &= mask.astype(bool)

        # Need enough samples
        if valid.sum() < min_samples:
            # Not enough overlap to fit; just return the estimator (in its current scale) shifted to be non-negative
            d_metric = est.copy()
            d_metric[~np.isfinite(d_metric)] = 0.0
            d_metric = np.maximum(d_metric, 0.0).astype(np.float32, copy=False)
            return d_metric

        # Build LS system: y ~ s*x + b
        x = est[valid].astype(np.float64, copy=False).reshape(-1, 1)
        y = rs_m[valid].astype(np.float64, copy=False).reshape(-1, 1)
        A = np.hstack([x, np.ones_like(x)])
        s, b = np.linalg.lstsq(A, y, rcond=None)[0].ravel()

        # Apply to full map
        d_metric = (s * est + b).astype(np.float32, copy=False)
        # Clamp tiny negatives (can happen with noise)
        np.maximum(d_metric, 0.0, out=d_metric)
        return d_metric
