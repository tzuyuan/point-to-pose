import numpy as np

from sam2.build_sam import build_sam2_camera_predictor

from point2pose.core.base_segmenter import Segmenter
from point2pose.core.module_registry import SEGMENTER


class Sam2RealTimeSegmenter(Segmenter):
    def __init__(self, config):
        super().__init__(config)
        self.name = "sam2_real_time"
        self.device = config.device

        model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        sam2_checkpoint = config.get(
            "sam2_checkpoint", "checkpoints/sam2.1/sam2.1_hiera_large.pt"
        )

        self.predictor = build_sam2_camera_predictor(
            model_cfg,
            sam2_checkpoint,
        )

        self.input_points = []
        self.input_labels = []
        self.num_obj = 0
        self.tracking_started = False
        self.frame_count = 0

    def add_input_points(self, points, labels):
        """
        Add new points to be tracked.
        Args:
            points (List[List[int]]): List of points, each defined by [x, y].
            labels (List[int]): List of labels, each defined by 1 or 0.
                                1 means positive point, 0 means negative point.
        """
        self.input_points.extend(points)
        self.input_labels.extend(labels)

    def initialize(self, image):
        """
        Initialize the segmenter with the provided points.
        """
        self.predictor.load_first_frame(image)

        # Add all points to predictor.
        # Return True if at least one object is added.
        self.tracking_started = self._add_all_points_to_predictor()

    def segment(self, image):
        """
        Segment the image using SAM2 with the provided bounding boxes.

        Args:
            image (np.ndarray): The input RGB image.

        Returns:
            out_obj_ids (List[int]): List of object IDs.
            out_mask_logits (torch.Tensor): mask logits. [num_obj, 1, height, width]
        """
        out_obj_ids, out_mask_logits = self.predictor.track(image)
        self.frame_count += 1
        return out_obj_ids, out_mask_logits

    def _add_all_points_to_predictor(self) -> bool:
        """
        Add all points to SAM2 predictor.
        Return True if at least one object is added.
        """
        # Add points for each object (grouping points by consecutive positive labels)
        obj_id = 0
        current_points = []
        current_labels = []

        for point, label in zip(self.input_points, self.input_labels):
            if label == 1:  # Positive point: start new object
                if current_points:  # Save previous object if exists
                    self.predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=obj_id,
                        points=np.array(current_points, dtype=np.float32),
                        labels=np.array(current_labels, dtype=np.int32),
                    )
                    obj_id += 1
                    current_points = []
                    current_labels = []

                current_points.append(point)
                current_labels.append(label)
            else:  # Negative point: add to current object
                current_points.append(point)
                current_labels.append(label)

        # Add the last object
        if current_points:
            self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                points=np.array(current_points, dtype=np.float32),
                labels=np.array(current_labels, dtype=np.int32),
            )

        self.num_obj = obj_id + 1
        if self.num_obj > 0:
            print(
                f"[SAM2] Added {self.num_obj} object(s) with {len(self.input_points)} points"
            )
            return True
        else:
            print(
                "[SAM2] No objects added. Please call add_input_points() to add prompt points before calling initialize()"
            )
            return False
