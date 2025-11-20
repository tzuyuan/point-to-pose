from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class KeyFrame:
    """
    Container for storing key information about a key frame.
    """

    frame_id: int
    obj_id: int
    timestamp: Optional[float]

    # Pose of the tracked object in this frame (4x4 SE(3))
    pose: np.ndarray

    # Newly associated keypoints sampled for this key frame
    keypoint_track_indices: np.ndarray  # global tracker indices
    keypoints_2d: np.ndarray  # Nx2 pixel coordinates
    keypoints_3d_camera: np.ndarray  # Nx3 in current camera frame
    keypoints_3d_object: np.ndarray  # Nx3 in object (frame-0) coordinates
    keypoints_valid_mask: np.ndarray  # bool mask for valid depth

    # Dense cropped point cloud for the object in this frame
    dense_pcd_points: np.ndarray  # Mx3 world/camera coordinates
    dense_pcd_colors: Optional[np.ndarray] = None  # Mx3 uint8 RGB

    # Extra metadata (registration stats, masks, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
