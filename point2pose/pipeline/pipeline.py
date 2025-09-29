from point2pose.io.sources.base import BaseDatasetSource, BaseLiveSource
from point2pose.core.build import build_from_cfg
import point2pose.modules  # trigger registrations
from point2pose.core.module_registry import (
    REGISTER,
    TRACKER,
    STATE,
    SAMPLER,
    OPTIM,
)


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.register = build_from_cfg(cfg.register, REGISTER)
        self.segmenter = build_from_cfg(cfg.segmenter, REGISTER)
        self.tracker = build_from_cfg(cfg.tracker, TRACKER)
        self.state = build_from_cfg(cfg.state, STATE)
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)
        self.optimizer = build_from_cfg(cfg.optimizer, OPTIM)

        self.frame_id = 0

    # -------- one-time init with user clicks ----------
    def add_user_points(
        self, obj_points: dict[int, list[tuple[int, int]]], labels: dict[int, list[int]]
    ):
        # forward to segmenter; it will start tracking objects internally
        for obj_id, pts in obj_points.items():
            self.segmenter.add_input_points(pts, labels[obj_id])

    # -------- main loop per frame ----------
    def step(self, frame):
        self.frame_id += 1

        # 1) Segmentation -> masks per object
        obj_ids, mask_logits = self.segmenter.segment(frame.rgb)
