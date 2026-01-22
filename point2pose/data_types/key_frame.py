from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from point2pose.utils.transform import transform_pts, inverse_SE3

from point2pose.data_types.frame import Frame


@dataclass
class KeyFrame:
    """
    Container for storing key information about a key frame.
    """

    frame_id: int  # frame id when the key frame was sampled
    obj_id: int  # object id
    kf_idx: int  # index of the key frame in the key frame list for this object
    timestamp: Optional[float]
    frame: Frame

    # Pose of the tracked object in this frame (4x4 SE(3))
    pose: np.ndarray

    # Newly sampled keypoints associated with this key frame
    kp_track_indices: np.ndarray  # global tracker indices (N_k,)
    kp_2d: np.ndarray  # (N_k, 2) pixel coordinates
    kp_3d_camera: np.ndarray  # (N_k, 3) in current camera frame
    kp_3d_object: np.ndarray  # (N_k, 3) in object (frame-0) coordinates
    kp_valid: np.ndarray  # (N_k,) bool mask for valid depth

    # All observed points for this key frame (may include kp + older points)
    obs_track_indices: np.ndarray  # (N_o,) global tracker indices
    obs_2d: np.ndarray  # (N_o, 2) pixel coordinates
    obs_3d_camera: np.ndarray  # (N_o, 3) in current camera frame
    obs_3d_object: np.ndarray  # (N_o, 3) in object (frame-0) coordinates
    obs_valid: np.ndarray  # (N_o,) bool mask for valid depth
    obs_visible: np.ndarray  # (N_o,) bool mask for visible points
    obs_uncertainties: np.ndarray  # (N_o,) float uncertainties for the points

    # registration stats
    reg_correspond_curr3d: np.ndarray  # (N_o, 3) current 3D points
    reg_inliers: np.ndarray  # (N_o,) bool mask for inliers
    reg_residuals: np.ndarray  # (N_o,) float residuals for the points
    reg_valid_idx: np.ndarray  # (N_o,) valid indices for the points

    # Dense cropped point cloud for the object in this frame
    dense_pts: np.ndarray  # (M, 3) world/camera coordinates
    dense_colors: Optional[np.ndarray] = None  # (M, 3) uint8 RGB

    # Extra metadata (registration stats, masks, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_kps(self):
        """
        with the new pose, update the key points and observed points in the object frame
        """
        T = inverse_SE3(self.pose)
        self.kp_3d_object = transform_pts(T, self.kp_3d_camera)

    def get_num_kps(self):
        """
        get the number of key points
        """
        return self.kp_3d_object.shape[0]

    def get_num_obs(self):
        """
        get the number of observed points
        """
        return self.obs_3d_object.shape[0]
