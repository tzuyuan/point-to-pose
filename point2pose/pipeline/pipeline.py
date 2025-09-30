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
from point2pose.modules.object.object import Object
from point2pose.utils.camera import convert_pixel_to_world


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        # reference to ensure the imported module is considered used
        print("REGISTER items:")

        print(REGISTER._items)
        print("TRACKER items:")
        print(TRACKER._items)
        print("SAMPLER items:")
        print(SAMPLER._items)
        print("CRITERION items:")
        print(CRITERION._items)
        print("SEGMENTER items:")
        print(SEGMENTER._items)
        print("OPTIM items:")
        print(OPTIM._items)

        self.register = build_from_cfg(cfg.register, REGISTER)
        self.segmenter = build_from_cfg(cfg.segmenter, SEGMENTER)
        self.tracker = build_from_cfg(cfg.tracker, TRACKER)
        # self.state = build_from_cfg(cfg.state, STATE)
        self.criterion = build_from_cfg(cfg.criterion, CRITERION)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)
        # self.optimizer = build_from_cfg(cfg.optimizer, OPTIM)

        self.frame_id = 0

        self.crit_ctx = CriterionContext()

        self.num_obj = 0
        self.objects = []
        # track is the 2d tracked points in tracker
        # obj is the object id in the pipeline
        self.track2obj_map = {}  # index -> obj_id. (int -> int)
        self.obj2track_map = {}  # obj_id -> indices. (int -> np.ndarray)

        self.initialized = False

    # -------- one-time init with user clicks ----------
    def add_user_points(
        self, obj_points: dict[int, list[tuple[int, int]]], labels: dict[int, list[int]]
    ):
        # forward to segmenter; it will start tracking objects internally
        for obj_id, pts in obj_points.items():
            self.segmenter.add_input_points(pts, labels[obj_id])

    # -------- main loop per frame ----------
    def step(self, frame):
        # Segmentation -> masks per object
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
        frame.mask = mask_logits.cpu().numpy()

        # initialize objects
        if not self.initialized:
            self.initialized = True
            self.num_obj = len(obj_ids)
            for obj_id in obj_ids:
                self.objects.append(Object(obj_id))

        # Check sample criteria
        for obj_id in range(self.num_obj):
            ## TODO: make crit_ctx to be object-specific?
            if self.criterion.check_sample_criterion(self.crit_ctx):
                # Sample points
                new_sampled_points = self.sampler.sample(frame)

                # add new sampled points to the tracker
                new_indices = self.tracker.add_query_points(
                    frame.id, new_sampled_points
                )

                # update track2obj_map and obj2track_map
                self.track2obj_map[new_indices] = obj_id
                self.obj2track_map[obj_id] = np.concatenate(
                    (self.obj2track_map[obj_id], new_indices)
                )

        # Track points
        tracks, uncertainties, visibles = self.tracker.track_once(frame)

        tracks_3d, valid_idx = convert_pixel_to_world(
            tracks,
            frame.depth,
            frame.intrinsics,
            frame.depth_factor,
        )

        self.frame_id += 1
