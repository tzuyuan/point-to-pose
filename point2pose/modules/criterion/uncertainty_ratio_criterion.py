import numpy as np

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("uncertainty_ratio")
class UncertaintyRatioCriterion(SampleCriterion):
    """
    Uncertainty ratio criterion checks if the ratio of uncertainty to total uncertainty
    exceeds a specified threshold.
    """

    def __init__(self, config):
        super().__init__()
        self._uncer_thres = config.get("uncer_thres", 0.5)
        self._ratio_thres = config.get("ratio_thres", 0.5)

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        """
        Check the ratio of (number of points with uncertainty > threshold) to (total number of points).
        If the ratio falls below a specified threshold, return True to indicate that resampling is needed.

        Args:
            crit_context (CriterionContext): The context containing the current iteration information.

        Returns:
            bool: True if the ratio of certain points to total number of points is smaller than a specified threshold, False otherwise.
        """
        if context.uncertainty is None:
            print("[Warning][Criterion] Uncertainty is none, not checking criterion.")
            return False

        num_pts = context.uncertainty.shape[0]

        if num_pts == 0:
            print("[Warning][Criterion] No points, not checking criterion.")
            return False

        # count number of points with uncertainty less than threshold
        valid = context.uncertainty > self._uncer_thres
        valid_count = np.sum(valid)

        ratio = valid_count / num_pts

        print(
            f"[Uncertainty Ratio Criterion] Valid count: {valid_count}, Total points: {num_pts}, Ratio: {ratio}, Threshold: {self._ratio_thres}"
        )

        # Ensure native Python bool is returned (not numpy.bool_)
        return bool(ratio > self._ratio_thres)
