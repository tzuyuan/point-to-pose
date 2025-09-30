import numpy as np

from point2pose.data_types.frame import Frame
from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER


@SAMPLER.register_module("random")
class RandomSampler:
    def __init__(self, config):
        self.num_points = config.get("num_points", 10)

    def sample(self, frame: Frame):
        """

        Args:
        frame: Frame object class
        """
        # sample random points from the mask
        # mask [N,1,H,W] N is the number of objects
        mask = frame.mask
        y_coords, x_coords = np.where(mask > 0)
        valid_pixels = np.stack([x_coords, y_coords], axis=1)
        if len(valid_pixels) < self.num_points:
            print(
                f"Warning: Only {len(valid_pixels)} valid pixels available, returning all"
            )
            return valid_pixels

        idx = np.random.choice(len(valid_pixels), size=self.num_points, replace=False)
        return valid_pixels[idx]
