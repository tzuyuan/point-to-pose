import numpy as np

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("rotation_threshold")
class RotationThresholdCriterion(SampleCriterion):
    """
    Rotation threshold criterion checks for every N iterations and returns true.
    """

    def __init__(self, config):
        super().__init__()
        self._max_angle_deg = config.get("max_angle_deg", 60)

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
