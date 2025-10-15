from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from point2pose.modules.object.object import Object


@dataclass
class CriterionContext:
    """
    The class stores possible context information for criterion checking.
    """

    cur_iter: Optional[int] = None
    uncertainty: Optional[np.ndarray] = None

    objects: Optional[List[Object]] = None

    reg_stats: Optional[dict] = None

    def update_criterion_context(self, **kwargs):
        """
        Update the criterion context with the given keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
