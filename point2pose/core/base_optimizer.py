import os
from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, config):
        self.name = "base_optimizer"
        self.debug_level = config.get("debug_level", 0)
        self.debug_dir = config.get("debug_dir", None)

        if self.debug_level > 0 and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    @abstractmethod
    def optimize(self):
        """Perform the optimization operation."""
        pass
