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

    # dataset related
    gt_pose: Optional[np.ndarray] = (
        None  # 4x4 float64 (world_T_cam or cam_T_obj—document it)
    )
    occ_mask: Optional[np.ndarray] = None  # HxW uint8/bool
    xyz_map: Optional[np.ndarray] = None  # HxWx3 float32 (if provided)

    meta: Dict[str, Any] = field(default_factory=dict)  # arbitrary extras
