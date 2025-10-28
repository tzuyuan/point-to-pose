from dataclasses import dataclass
import numpy as np


@dataclass
class OptimizerResult:
    obj_id: int
    frame_id: int
    pose_optimized: np.ndarray
    key_points_optimized: np.ndarray
    key_points_idx_optimized: np.ndarray
