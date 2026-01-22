import numpy as np
import torch

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("rotation_threshold_and_min_num")
class RotationThresholdAndMinNumCriterion(SampleCriterion):
    """
    Rotation threshold criterion checks for every N iterations and returns true.
    """

    def __init__(self, config):
        super().__init__()
        self._max_angle_deg = config.get("max_angle_deg", 60)
        self._min_num_pts = config.get("min_num_pts", 10)
        self._min_mask_area = config.get("min_mask_area", 100)

        # v is a list of direction vectors. (num_dirs, 3)
        # we assume the first vector to be the initial direction as (0, 0, 1)
        self._num_obj = 0
        self._v_list = []

    def initialize(self, crit_ctx: CriterionContext):
        self._num_obj = len(crit_ctx.objects)
        # v is a list of direction vectors. (num_dirs, 3)
        self._v_list = []
        for obj_id in range(self._num_obj):
            self._v_list.append(np.array([[0, 0, 1]]))

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        obj = context.objects[obj_id]

        reg_stats = context.reg_stats[obj_id]
        # if reg_stats is None:
        #     return True
        num_pts = reg_stats["correspond_curr3d"].shape[0]
        print(f"[Criterion] num visible points: {num_pts}")

        # sum number of pixel being in mask being 1
        mask_area = torch.sum(context.frame.mask[obj_id, 0])

        if (num_pts < self._min_num_pts) and (mask_area > self._min_mask_area):
            return True

        # pts_id = context.track_table.obj2track_map[obj_id]
        # cur_visible = context.track_table.visible[pts_id]
        # if cur_visible.sum() < self._min_num_pts:
        #     return True

        R = obj.pose[:3, :3]

        R_relative = R

        u = R_relative @ self._v_list[obj_id][0, :]

        # compute inner product of u and v
        inner_product = u @ self._v_list[obj_id].T
        angle = np.arccos(inner_product)
        angle_deg = np.rad2deg(angle)

        print(f"[Criterion] angle_deg: {angle_deg}")

        if np.any(angle_deg < self._max_angle_deg):
            return False
        else:
            self._v_list[obj_id] = np.concatenate(
                [self._v_list[obj_id], u.reshape(1, -1)], axis=0
            )
            return True
