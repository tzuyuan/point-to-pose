import os

import time

import numpy as np
import open3d as o3d

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

        self.prev3d_way_before = None
        self.prev3d_before = None
        self.curr3d_before = None
        self.prev3d_after = None
        self.curr3d_after = None

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

        # initialize objects
        for obj_id in range(self.num_obj):
            self.objects.append(Object(obj_id))

        # get segmentation mask
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits.cpu().numpy()

        # ------------- sampler -------------
        tracks = self._sample_for_all_obj(frame)

        # ------------- tracker -------------
        self.tracker.initialize(frame)
        # tracks = self.tracker.query_points.clone().cpu().numpy()

        # convert tracks into 3D points using depth and intrinsics
        track_3d, track_valid = convert_pixel_to_world(
            pixel=tracks,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
        )
        self.prev3d_way_before = track_3d
        # self.track_table.append(
        #     k=len(tracks),
        #     frame_id=frame.id,
        #     track_3d=track_3d,
        #     visible=np.ones(len(tracks), dtype=bool),
        #     valid=track_valid,
        #     uncertainty=0.5 * np.ones(len(tracks), dtype=float),
        # )

        self.track_table.update_track_table(
            track_3d=track_3d,
            valid=track_valid,
            uncertainties=0.5 * np.ones(len(tracks), dtype=float),
            visibles=np.ones(len(tracks), dtype=bool),
        )

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
        frame.mask = mask_logits.cpu().numpy()
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
                pass
            # frame to frame registration
            else:
                idx, prev3d, curr3d = self.extract_valid_idx_points_for_obj(
                    obj_id, self.track_table, track_3d, track_valid, visibles
                )
                self.prev3d_before = self.track_table.track_3d
                self.curr3d_before = track_3d
                self.prev3d_after = prev3d
                self.curr3d_after = curr3d

                pose = self.register.register(prev3d, curr3d)
                self.objects[obj_id].pose = pose
                print(f"pose at frame {self.frame_id}: {pose}")
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
        print(f"Frame {self.frame_id} - Registration: {register_time:.4f}s")

        # update track table to the current frame info
        table_update_start = time.time()
        self.track_table.update_track_table(
            track_3d, track_valid, uncertainties, visibles
        )
        table_update_time = time.time() - table_update_start
        print(f"Frame {self.frame_id} - Track table update: {table_update_time:.4f}s")

        # ------------- criterion and sample -------------
        sampling_start = time.time()
        self.crit_ctx.cur_iter = self.frame_id
        self._check_and_sample_for_all_obj(frame)
        sampling_time = time.time() - sampling_start
        print(f"Frame {self.frame_id} - Criterion and sampling: {sampling_time:.4f}s")

        self.frame_id += 1

        # Print total step time
        total_step_time = time.time() - step_start_time
        print(f"Frame {self.frame_id-1} - Total step time: {total_step_time:.4f}s")
        print(
            f"Frame {self.frame_id-1} - Breakdown: seg={segmenter_time:.3f}s, track={tracker_time:.3f}s, conv={conversion_time:.3f}s, reg={register_time:.3f}s, table={table_update_time:.3f}s, sample={sampling_time:.3f}s"
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
            # add new sampled points to the tracker
            new_indices = self.tracker.add_query_points(frame, new_sampled_points)

            # update track2obj_map and obj2track_map
            self.track_table.add_new_points_to_track_obj_maps(new_indices, obj_id)

        return all_sampled_points

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

    def _estimate_init_pose_and_bbox_for_all_obj(self, frame):
        out_pose = np.tile(np.eye(4), (self.num_obj, 1, 1))
        # estimate initial pose
        for obj_id in range(self.num_obj):
            # get the initial 3d points from frame.mask
            mask = frame.mask[obj_id, 0]
            y_coords, x_coords = np.where(mask > 0)
            valid_pxl_in_mask = np.stack([x_coords, y_coords], axis=1)
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
            self.objects[obj_id].init_bbox = pcd.get_oriented_bounding_box()
            self.objects[obj_id].bbox = self.objects[obj_id].init_bbox
            # set the initial pose
            out_pose[obj_id, :3, :3] = self.objects[obj_id].init_bbox.R
            out_pose[obj_id, :3, 3] = self.objects[obj_id].init_bbox.center
            self.objects[obj_id].pose = out_pose[obj_id]

            if self.debug_level > 1:

                # ensure debug directory exists
                debug_bbx_dir = os.path.join(self.debug_dir, "pipeline/initial_bbx")
                os.makedirs(debug_bbx_dir, exist_ok=True)

                # get color from frame.rgb
                color = frame.rgb[y_coords, x_coords]
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
