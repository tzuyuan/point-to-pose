from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectFrameData:
    obj_id: int
    frame_id: int

    # pose of the object
    pose: np.ndarray

    rel_pose: np.ndarray

    # registration stats
    cur_3d: np.ndarray
    cur_3d_idx: np.ndarray
    inliers: np.ndarray
    residuals: np.ndarray
    uncertainties: np.ndarray
