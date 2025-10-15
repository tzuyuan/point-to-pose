from abc import ABC, abstractmethod

from point2pose.data_types.criterion_context import CriterionContext


class SampleCriterion(ABC):
    def __init__(self):
        self._num_obj = 0

    def initialize(self, context: CriterionContext):
        """Initialize the criterion."""
        self._num_obj = len(context.objects)

    @abstractmethod
    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        """Check if the sampling criterion is met and return a boolean value."""
        return False
