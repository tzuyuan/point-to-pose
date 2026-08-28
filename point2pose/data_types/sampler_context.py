from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from point2pose.data_types.frame import Frame


@dataclass
class SamplerContext:
    """
    The class stores possible context information for sampler.
    """

    frame: Frame

    track_table: Optional[dict] = None

    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

    def update_sampler_context(self, **kwargs):
        """
        Update the criterion context with the given keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
