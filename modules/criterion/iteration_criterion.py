from core.base_criterion import SampleCriterion
from core.module_registry import CRITERION
from data_types.criterion_context import CriterionContext


@CRITERION.register_module("iteration")
class IterationCriterion(SampleCriterion):
    """
    Iteration criterion checks for every N iterations and returns true.
    """

    def __init__(self, return_every_n_iterations=1):
        super().__init__()
        self._cur_iter = 0
        self._return_every_n_iterations = return_every_n_iterations

    def check_sample_criterion(self, context: CriterionContext) -> bool:
        """
        Check if the current iteration is a multiple of the return every n iterations.

        Args:
            crit_context (CriterionContext): The context containing the current iteration information.

        Returns:
            bool: True if the current iteration is a multiple of the return every n iterations, False otherwise.
        """
        if context.cur_iter is not None:
            self._cur_iter = context.cur_iter
        else:
            self._cur_iter += 1
        return self._cur_iter % self._return_every_n_iterations == 0
