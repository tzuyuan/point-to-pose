import numpy as np
import cv2 as cv

import torch

from point2pose.data_types.frame import Frame
from point2pose.core.base_sampler import Sampler
from point2pose.core.module_registry import SAMPLER


@SAMPLER.register_module("random")
class RandomSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        self.num_points = config.get("num_points", 10)
        self._i = 0

    def sample(self, frame: Frame, obj_id: int) -> np.ndarray:
        """

        Args:
        frame: Frame object class
        """
        # sample random points from the mask
        # mask [N,1,H,W] N is the number of objects

        # mask = frame.mask[obj_id, 0]
        # y_coords, x_coords = np.where(mask > 0)
        # valid_pixels = np.stack([x_coords, y_coords], axis=1)
        mask = frame.mask[obj_id, 0]  # [H, W] on cuda, dtype=bool/uint8

        # Get (y,x) indices on GPU; switch to (x,y) like your NumPy code
        coords_yx = torch.nonzero(mask > 0, as_tuple=False)  # [N, 2], (y,x)
        valid_pxl_in_mask_g = coords_yx[
            :, [1, 0]
        ].contiguous()  # [N, 2], (x,y), still on GPU

        valid_pixels = valid_pxl_in_mask_g.cpu().numpy()
        if len(valid_pixels) < self.num_points:
            print(
                f"Warning: Only {len(valid_pixels)} valid pixels available, returning all"
            )
            return valid_pixels

        idx = np.random.choice(len(valid_pixels), size=self.num_points, replace=False)

        if self.debug_level >= 1:
            # use opencv to plot the points on the mask, as well as the rgb image and save them sperately
            # please save them under self.debug_dir

            mask_img = cv.cvtColor(mask.cpu().numpy(), cv.COLOR_GRAY2BGR)
            rgb_img = cv.cvtColor(frame.rgb, cv.COLOR_RGB2BGR)
            for i in idx:
                cv.circle(
                    mask_img,
                    (valid_pixels[i, 0], valid_pixels[i, 1]),
                    5,
                    (0, 0, 255),
                    -1,
                )
                cv.circle(
                    rgb_img,
                    (valid_pixels[i, 0], valid_pixels[i, 1]),
                    5,
                    (0, 0, 255),
                    -1,
                )
            cv.imwrite(f"{self.debug_dir}/{frame.id}_mask_{obj_id}.png", mask_img)
            cv.imwrite(f"{self.debug_dir}/{frame.id}_rgb_{obj_id}.png", rgb_img)
            print(f"[Random Sampler] Saved debug images to {self.debug_dir}")

        print(f"[Random Sampler] Sampled {len(idx)} points for object {obj_id}")

        return valid_pixels[idx]
