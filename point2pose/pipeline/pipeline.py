import numpy as np


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


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

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
        # track is the 2d tracked points in tracker
        # obj is the object id in the pipeline
        self.track2obj_map = {}  # index -> obj_id. (int -> int)
        self.obj2track_map = {}  # obj_id -> indices. (int -> np.ndarray)

        self._prev_track_3d = None
        self._prev_track_valid = None

        self._seg_initialized = False
        self._track_initialized = False

        self._estimate_init_pose = self.pipeline_cfg.get("estimate_init_pose", False)

    # -------- one-time init with user clicks ----------
    def add_user_points(self, obj_points: list[list[int]], labels: list[int]):
        """
        points (List[List[int]]): List of points, each defined by [x, y].
            labels (List[int]): List of labels, each defined by 1 or 0.
                                1 means positive point, 0 means negative point.
        """
        # forward to segmenter; it will start tracking objects internally

        self.segmenter.add_input_points(obj_points, labels)

    # -------- main loop per frame ----------
    def step(self, frame):
        # initialize objects
        if not self._seg_initialized:
            self.segmenter.initialize(frame.rgb)
            # obj_ids, _ = self.segmenter.segment(frame.rgb)

            # find out number of 1 in labels, which is a list
            self.num_obj = np.sum(np.asarray(self.segmenter.input_labels) == 1)
            for obj_id in range(self.num_obj):
                self.objects.append(Object(obj_id))

            self._seg_initialized = True

        # Segmentation -> masks per object
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits.cpu().numpy()

        # set frame id in criterion context
        self.crit_ctx.cur_iter = self.frame_id
        new_indices_all = []
        # Check sample criteria
        for obj_id in range(self.num_obj):
            ## TODO: make crit_ctx to be object-specific?
            if self.criterion.check_sample_criterion(self.crit_ctx):
                # Sample points
                new_sampled_points = self.sampler.sample(frame, obj_id)

                # add new sampled points to the tracker
                new_indices = self.tracker.add_query_points(
                    frame.id, new_sampled_points
                )

                new_indices_all.extend(new_indices.tolist())

                # update track2obj_map and obj2track_map
                self.track2obj_map.update(
                    {int(idx): obj_id for idx in new_indices.tolist()}
                )
                # update obj2track_map
                if obj_id in self.obj2track_map:
                    self.obj2track_map[obj_id] = np.concatenate(
                        (self.obj2track_map[obj_id], new_indices)
                    )
                else:
                    self.obj2track_map[obj_id] = new_indices

                # update track_table
                self.track_table.append(len(new_indices), obj_id, frame.id)
        print(len(self.track_table))
        # Track points using 2d tracker
        if not self._track_initialized:
            self.tracker.initialize(frame)
            self._track_initialized = True

        else:
            tracks, uncertainties, visibles = self.tracker.track_once(frame)
            print(f"[Pipeline] Tracked {tracks.shape[0]} points.")
            N = len(self.track_table)
            print(N, len(uncertainties), len(visibles))
            self.track_table.visible[:N] = visibles  # or scatter by returned ids
            self.track_table.last_seen[visibles] = self.frame_id
            self.track_table.uncertainty[:N] = uncertainties

        # convert tracks into 3D points using depth and intrinsics
        track_3d, track_valid = convert_pixel_to_world(
            pixel=tracks,
            depth_image=frame.depth,
            cam_intrinsics=frame.intrinsics,
            depth_factor=frame.depth_factor,
        )

        if self.frame_id == 0:
            self._prev_track_3d = track_3d
            self._prev_track_valid = track_valid

            self.frame_id += 1

            if self._estimate_init_pose:
                # estimate initial pose
                # for obj_id in range(self.num_obj):
                pass

            # return np.eye(4)

        self.frame_id += 1

        return track_3d, track_valid
