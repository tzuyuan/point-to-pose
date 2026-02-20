from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ObjectFrameData:
    obj_id: int
    frame_id: int

    # camera intrinsics
    intrinsics: np.ndarray

    # pose of the object
    pose: np.ndarray

    rel_pose: np.ndarray

    visible_pts_2d: np.ndarray  # (N_v, 2) pixel coordinates
    visible_pts_2d_idx: np.ndarray  # (N_v,) global tracker indices
    visible_uncertainties: np.ndarray  # (N_v,) visible point uncertainties

    # registration stats
    reg_cur_3d: np.ndarray
    reg_cur_3d_idx: np.ndarray
    reg_valid_idx: np.ndarray
    reg_inliers: np.ndarray
    reg_residuals: np.ndarray
    reg_uncertainties: np.ndarray

    # Optional SDF context for optimization-time SDF constraints.
    # `sdf` is the dense metadata dict (legacy path), `sdf_volume` is the
    # live backend object (nvblox / TSDF adapter) for accelerated queries.
    sdf: Optional[dict] = None
    sdf_volume: Optional[Any] = None
