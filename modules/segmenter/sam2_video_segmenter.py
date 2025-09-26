from sam2.sam2_video_predictor import Sam2VideoPredictor

from core.base_segmenter import Segmenter
from core.module_registry import SEGMENTER


class Sam2VideoSegmenter(Segmenter):
    def __init__(self, config):
        super().__init__(config)
        self.name = "sam2_video"
        self.device = config.device

        self.predictor = Sam2VideoPredictor.from_pretrained("sam_vit_h_4b8939.pth")

    def segment(self, image, boxes):
        """
        Segment the image using SAM2 with the provided bounding boxes.

        Args:
            image (np.ndarray): The input image.
            boxes (List[List[int]]): List of bounding boxes, each defined by [x1, y1, x2, y2].

        Returns:
            List[np.ndarray]: List of binary masks corresponding to each bounding box.
        """
        masks = self.predictor.predict(image, boxes)
        return masks
