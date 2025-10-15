import numpy as np

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("uncertainty_number")
class UncertaintyNumberCriterion(SampleCriterion):
    """
    Uncertainty ratio criterion checks if the ratio of uncertainty to total uncertainty
    exceeds a specified threshold.
    """

    def __init__(self, config):
        super().__init__()
        self._uncer_thres = config.get("uncer_thres", 0.5)
        self._min_num_pts = config.get("min_num_pts", 5)

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:

        if context.uncertainty is None:
            print("[Warning][Criterion] Uncertainty is none, not checking criterion.")
            return False

        num_pts = context.uncertainty.shape[0]

        if num_pts == 0:
            print("[Warning][Criterion] No points, not checking criterion.")
            return False

        # count number of points with uncertainty less than threshold
        valid = context.uncertainty < self._uncer_thres
        valid_count = np.sum(valid)

        print(
            f"[Uncertainty Number Criterion] Valid count: {valid_count}, Threshold: {self._min_num_pts}"
        )

        # Ensure native Python bool is returned (not numpy.bool_)
        return bool(valid_count < self._min_num_pts)
