from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("rotation")
class RotationCriterion(SampleCriterion):
    """
    Rotation criterion checks for every N iterations and returns true.
    """

    def __init__(self, config):
        super().__init__()
        self._discretization_angle = config.get("discretization_angle", 60)

    def check_sample_criterion(self, context: CriterionContext) -> bool:
        

        
