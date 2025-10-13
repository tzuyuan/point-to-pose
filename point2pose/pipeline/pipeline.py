import os

import time

import numpy as np
import open3d as o3d

import torch

from point2pose.core.build import build_from_cfg

import point2pose.modules as _modules  # trigger registrations
from point2pose.core.module_registry import (
    REGISTER,
    TRACKER,
    SAMPLER,
    CRITERION,
    SEGMENTER,
    OPTIM,
)
from point2pose.data_types.criterion_context import CriterionContext
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.modules.object.object import Object
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.point_cloud_io import save_reg_pcd
from point2pose.utils.transform import transform_pts


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        self.debug_level = self.pipeline_cfg.get("debug_level", 0)
        self.debug_dir = self.pipeline_cfg.get("debug_dir", None)

        if self.debug_level > 0 and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.register = build_from_cfg(cfg.register, REGISTER)
        self.segmenter = build_from_cfg(cfg.segmenter, SEGMENTER)
        self.tracker = build_from_cfg(cfg.tracker, TRACKER)
        # self.state = build_from_cfg(cfg.state, STATE)
        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)
        # self.optimizer = build_from_cfg(cfg.optimizer, OPTIM)

        self.frame_id = 0

        self.crit_ctx = CriterionContext()
        self.track_table = PointTrackTable.new(n0=0)

        self.num_obj = 0
        self.objects = []

        # self.prev3d_way_before = None
        # self.prev3d_before = None
        # self.curr3d_before = None
        self.prev3d = None
        self.curr3d = None

        self._estimate_init_pose = self.pipeline_cfg.get("estimate_init_pose", False)
        self._frame2map_reg = self.pipeline_cfg.get("frame_to_map_reg", False)

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

        # ------------- sampler -------------
        self._sample_for_all_obj(frame)

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

        # estimate initial pose
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        if self._estimate_init_pose:
            out_pose = self._estimate_init_pose_and_bbox_for_all_obj(frame)

        self.frame_id += 1
        return out_pose

    # -------- main loop per frame ----------
    def step(self, frame):
        """
        Step the pipeline for one frame.
        """
        # if it's the first frame, initialize the pipeline
        if self.frame_id == 0:
            return self.initialize_first_frame(frame)

        # Start timing the entire step
        step_start_time = time.time()

        # ------------- segmenter -------------
        segmenter_start = time.time()
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits  # mask is a torch tensor on gpu
        segmenter_time = time.time() - segmenter_start
        print(f"Frame {self.frame_id} - Segmentation: {segmenter_time:.4f}s")

        # ------------- tracker -------------
        tracker_start = time.time()
        tracks, uncertainties, visibles = self.tracker.track_once(frame)
        print(f"num tracks: {len(tracks)}")
        # Convert visibles to boolean since TAPIR returns float32
        visibles = visibles.astype(bool)
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
        for obj_id in range(self.num_obj):
            # frame to map registration
            if self._frame2map_reg:

                idx, key_points, curr3d = self.extract_valid_key_points(
                    self.objects[obj_id],
                    self.track_table.obj2track_map[obj_id],
                    track_3d,
                    visibles,
                    track_valid,
                )

                if key_points.shape[0] < 3 or curr3d.shape[0] < 3:
                    continue

                pose_to_key = self.register.register(key_points, curr3d)

                pose = pose_to_key @ self.objects[obj_id].init_pose
                self.objects[obj_id].pose = pose

                self.prev3d = key_points
                self.curr3d = curr3d
                print(f"pose at frame {self.frame_id}: {pose}")

                if self.debug_level > 1:
                    reg_debug_dir = os.path.join(self.debug_dir, "register")
                    if not os.path.exists(reg_debug_dir):
                        os.makedirs(reg_debug_dir, exist_ok=True)
                    save_reg_pcd(
                        key_points,
                        curr3d,
                        pose_to_key,
                        reg_debug_dir,
                        f"obj_{obj_id}_frame_{self.frame_id}",
                    )

            # frame to frame registration
            else:
                idx, prev3d, curr3d = self.extract_valid_idx_points_for_obj(
                    obj_id, self.track_table, track_3d, track_valid, visibles
                )
                # self.prev3d_before = self.track_table.track_3d
                # self.curr3d_before = track_3d
                # self.prev3d_after = prev3d
                # self.curr3d_after = curr3d

                pose = self.register.register(prev3d, curr3d)
                self.objects[obj_id].pose = pose @ self.objects[obj_id].pose
                print(f"Frame {self.frame_id} - Object {obj_id} - Pose: {pose}")
                if self.debug_level > 1:
                    reg_debug_dir = os.path.join(self.debug_dir, "register")
                    if not os.path.exists(reg_debug_dir):
                        os.makedirs(reg_debug_dir, exist_ok=True)
                    save_reg_pcd(
                        prev3d,
                        curr3d,
                        pose,
                        reg_debug_dir,
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

        # ------------- criterion and sample -------------
        sampling_start = time.time()
        self.crit_ctx.update_criterion_context(
            cur_iter=self.frame_id, uncertainty=uncertainties
        )
        self._check_and_sample_for_all_obj(frame)
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

    def _sample_for_all_obj(self, frame):
        """
        Sample points for all objects, and add the query points to the tracker.
        """
        all_sampled_points = np.empty((0, 2))
        for obj_id in range(self.num_obj):
            # Sample points
            new_sampled_points = self.sampler.sample(frame, obj_id)
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

    def _check_and_sample_for_all_obj(self, frame):
        """
        Check sample criteria for all objects and sample points for all objects if the criterion is met.
        """
        # Check sample criteria
        for obj_id in range(self.num_obj):
            ## TODO: make crit_ctx to be object-specific?
            if self.criterion.check_sample_criterion(self.crit_ctx):
                # Sample points
                new_sampled_points = self.sampler.sample(frame, obj_id)

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

                new_points_3d_obj_frame = transform_pts(
                    np.linalg.inv(
                        self.objects[obj_id].pose
                        @ np.linalg.inv(self.objects[obj_id].init_pose)
                    ),
                    new_points_3d,
                )
                self.objects[obj_id].key_points = np.concatenate(
                    (self.objects[obj_id].key_points, new_points_3d_obj_frame)
                )
                self.objects[obj_id].valid = np.concatenate(
                    (
                        self.objects[obj_id].valid,
                        valid_new_points_3d,
                    )
                )

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
            self.objects[obj_id].pose = out_pose[obj_id]
            self.objects[obj_id].init_pose = out_pose[obj_id]
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

    def extract_valid_key_points(
        self, obj, obj_idx, cur_pts_3d, cur_visible, cur_valid
    ):
        """
        Args:
            obj_idx:     (M,) global indices for this object's points
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
        both_mask = vis_obj & val_obj & valid_kp_bool  # (M,)

        # global indices for valid+visible points
        idx = obj_idx[both_mask]  # (K,)

        # per-object arrays use local mask; global arrays use global idx
        key_points = obj.key_points[both_mask].copy()
        curr3d = cur_pts_3d[idx].copy()

        return idx, key_points, curr3d

    def extract_valid_idx_points_for_obj(
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
