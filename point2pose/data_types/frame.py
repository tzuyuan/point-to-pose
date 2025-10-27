from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import numpy as np
import torch


@dataclass(slots=True)
class Frame:
    id: int
    rgb: np.ndarray  # HxWx3 uint8 (BGR or RGB—pick one convention)
    depth: Optional[np.ndarray] = None  # HxW float32 (meters) or uint16 raw
    mask: Optional[torch.Tensor] = None  #  [N,1,H,W] N is the number of objects

    # camera related
    intrinsics: Optional[np.ndarray] = None  # 3x3 float64
    depth_factor: Optional[float] = None

    timestamp: Optional[float] = None

    # TODO: move this potentially
    # TODO: doesn't work for multiple objects yet
    # convex hull of the current visible points in this frame
    convex_hull_xy: Optional[np.ndarray] = None  # Nx2 float32 (x,y) coordinates
