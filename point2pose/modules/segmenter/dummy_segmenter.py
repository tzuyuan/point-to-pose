from point2pose.core.base_segmenter import Segmenter
from point2pose.core.module_registry import SEGMENTER


@SEGMENTER.register_module("dummy")
class DummySegmenter(Segmenter):
    def __init__(self, config):
        super().__init__(config)

    def initialize(self, frame):
        """Initialize the segmenter."""
        pass

    def segment(self, image):
        """Segment the frame."""
        return [], None
