from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CriterionContext:
    """
    The class stores possible context information for criterion checking.
    """

    cur_iter: Optional[int] = None
    uncertainty: Optional[np.ndarray] = None
