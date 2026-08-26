import numpy as np
from collections import deque
from typing import Optional

from point2pose.utils.lie import log_SE3, vec_to_se3, se3_to_vec, exp_se3


class SE3LowPassFilter:
    """
    A low pass filter for SE(3) poses that works in the Lie algebra se(3).

    This filter properly handles the non-Euclidean structure of SE(3) by:
    1. Converting poses to the Lie algebra se(3) using logarithm
    2. Applying low pass filtering in the vector space of se(3)
    3. Converting back to SE(3) using exponential mapping

    Parameters
    ----------
    alpha : float
        Smoothing factor (0 < alpha < 1). Smaller values = more smoothing.
    window_size : int, optional
        Window size for moving average. If None, uses exponential smoothing only.
    """

    def __init__(self, alpha: float = 0.1, window_size: Optional[int] = None):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self.window_size = window_size
        self._initialized = False
        self._filtered_pose: Optional[np.ndarray] = None

        # For moving average if window_size is specified
        if window_size is not None and window_size > 0:
            self._se3_buffer = deque(maxlen=window_size)
            self._use_moving_average = True
        else:
            self._use_moving_average = False

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new SE(3) pose.

        Parameters
        ----------
        pose : np.ndarray
            A 4x4 SE(3) transformation matrix

        Returns
        -------
        np.ndarray
            The filtered 4x4 SE(3) transformation matrix
        """
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix")

        # Ensure the pose is a valid SE(3) matrix
        # if not self._is_valid_se3(pose):
        #     raise ValueError("Input pose is not a valid SE(3) matrix")

        if not self._initialized:
            self._filtered_pose = pose.copy()
            self._initialized = True
            return self._filtered_pose

        if self._use_moving_average:
            return self._update_moving_average(pose)
        else:
            return self._update_exponential_diff(pose)

    def _update_exponential_smoothing(self, pose: np.ndarray) -> np.ndarray:
        """Update using exponential smoothing in se(3)."""
        # Convert current filtered pose to se(3)
        current_se3 = log_SE3(self._filtered_pose)
        current_se3_vec = se3_to_vec(current_se3)

        # Convert new pose to se(3)
        new_se3 = log_SE3(pose)
        new_se3_vec = se3_to_vec(new_se3)

        # Apply exponential smoothing in se(3)
        filtered_se3_vec = (1 - self.alpha) * current_se3_vec + self.alpha * new_se3_vec

        # Convert back to SE(3)
        filtered_se3 = vec_to_se3(filtered_se3_vec)
        self._filtered_pose = exp_se3(filtered_se3)

        return self._filtered_pose

    def _update_exponential_diff(self, pose: np.ndarray) -> np.ndarray:
        """Update using exponential smoothing in se(3)."""
        # Convert current filtered pose to se(3)
        self._filtered_pose = self._filtered_pose @ exp_se3(
            self.alpha * log_SE3(np.linalg.inv(self._filtered_pose) @ pose)
        )
        return self._filtered_pose

    def _update_moving_average(self, pose: np.ndarray) -> np.ndarray:
        """Update using moving average in se(3)."""
        # Convert pose to se(3)
        se3 = log_SE3(pose)
        se3_vec = se3_to_vec(se3)

        # Add to buffer
        self._se3_buffer.append(se3_vec)

        # Compute average in se(3)
        if len(self._se3_buffer) > 0:
            avg_se3_vec = np.mean(self._se3_buffer, axis=0)
            avg_se3 = vec_to_se3(avg_se3_vec)
            self._filtered_pose = exp_se3(avg_se3)

        return self._filtered_pose

    def _is_valid_se3(self, pose: np.ndarray) -> bool:
        """Check if a matrix is a valid SE(3) transformation."""
        # Check dimensions
        if pose.shape != (4, 4):
            return False

        # Check that bottom row is [0, 0, 0, 1]
        if not np.allclose(pose[3, :], [0, 0, 0, 1]):
            return False

        # Check that rotation part is orthogonal
        R = pose[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            return False

        # Check determinant of rotation part is 1
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            return False

        return True

    def reset(self):
        """Reset the filter state."""
        self._initialized = False
        self._filtered_pose = None
        if self._use_moving_average:
            self._se3_buffer.clear()

    def get_filtered_pose(self) -> Optional[np.ndarray]:
        """Get the current filtered pose."""
        return self._filtered_pose.copy() if self._filtered_pose is not None else None
