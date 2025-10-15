from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("iteration")
class IterationCriterion(SampleCriterion):
    """
    Iteration criterion checks for every N iterations and returns true.
    """

    def __init__(self, config):
        super().__init__()
        self._cur_iter = 0
        self._return_every_n_iterations = config.get("update_per_iter", 5)

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        """
        Check if the current iteration is a multiple of the return every n iterations.

        Args:
            crit_context (CriterionContext): The context containing the current iteration information.

        Returns:
            bool: True if the current iteration is a multiple of the return every n iterations, False otherwise.
        """
        # Always return true for the first iteration
        if context.cur_iter == 0:
            return True

        if context.cur_iter is not None:
            self._cur_iter = context.cur_iter
        else:
            self._cur_iter += 1
        return self._cur_iter % self._return_every_n_iterations == 0
