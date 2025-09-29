from abc import ABC, abstractmethod

from point2pose.data_types.criterion_context import CriterionContext


class SampleCriterion(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check_sample_criterion(self, context: CriterionContext) -> bool:
        """Check if the sampling criterion is met and return a boolean value."""
        return False
