from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Perform the sampling operation."""
        pass
