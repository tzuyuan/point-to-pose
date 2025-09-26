from abc import ABC, abstractmethod


class Segmenter(ABC):
    def __init__(self, config):
        self.name = "base_segmenter"

    @abstractmethod
    def segment(self, *args, **kwargs):
        """Perform the segmentation operation."""
        pass
