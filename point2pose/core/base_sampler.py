import os
from abc import ABC, abstractmethod

from point2pose.data_types.sampler_context import SamplerContext


class Sampler(ABC):

    def __init__(self, config):
        self.name = "base_sampler"
        self.debug_level = config.get("debug_level", 0)
        self.debug_dir = config.get("debug_dir", None)

        if self.debug_level > 0 and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    @abstractmethod
    def sample(self, context: SamplerContext):
        """Perform the sampling operation."""
        pass
