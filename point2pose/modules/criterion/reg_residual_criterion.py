import numpy as np

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("registration_residual")
class RegistrationResidualCriterion(SampleCriterion):
    """
    Registration residual criterion checks for every N iterations and returns true.
    """

    def __init__(self, config):
        super().__init__()
        self._residual_thres = config.get("residual_thres", 0.01)

    def check_sample_criterion(self, context: CriterionContext) -> bool:
        """
        Check if the current iteration is a multiple of the return every n iterations.

        Args:
            crit_context (CriterionContext): The context containing the current iteration information.

        Returns:
            bool: True if the current iteration is a multiple of the return every n iterations, False otherwise.
        """
        # Always return true for the first iteration
        if context.reg_stats is None or "residuals" not in context.reg_stats:
            return False
        inliers = context.reg_stats["inliers"]
        rs_mean = np.mean(context.reg_stats["residuals"][inliers])
        return rs_mean > self._residual_thres
