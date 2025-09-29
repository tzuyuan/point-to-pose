import numpy as np


class Object:
    """
    Object class that keeps tracks of the 3D points belonging to the object.
    """

    def __init__(self, name: str):
        self.name = name

        # 3D points belonging to the object, represented in the initial object frame
        self.key_points = np.empty((0, 3))  # Nx3
        self.uncertainties = np.empty((0,))

        # pose of the object
        self.pose = np.eye(
            4
        )  # 4x4 transformation matrix from object frame to world frame

        self.bbox = (
            None  # 3D bounding box of the object, represented in the object frame
        )

    def add_key_points(self, new_key_points: np.ndarray, new_uncertainties: np.ndarray):
        """
        Add new 3D points to the object.

        Args:
            new_points_3d (np.ndarray): New 3D points to add, shape (M, 3).
            new_uncertainties (np.ndarray): Uncertainties associated with the new points, shape (M,).
        """
        assert new_key_points.shape[1] == 3, "new_key_points should have shape (M, 3)"
        assert (
            new_key_points.shape[0] == new_uncertainties.shape[0]
        ), "new_key_points and new_uncertainties should have the same number of points"

        self.key_points = np.vstack((self.key_points, new_key_points))
        self.uncertainties = np.hstack((self.uncertainties, new_uncertainties))
