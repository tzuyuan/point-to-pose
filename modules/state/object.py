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

    ##############################################################
    ########################## Getters ###########################
    ##############################################################

    def get_key_points_and_uncertainty(self):
        """
        Get the 3D points and their uncertainties.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 3D points (shape (N, 3)) and their uncertainties (shape (N,)).
        """
        return self.key_points, self.uncertainties

    def get_key_points(self) -> np.ndarray:
        """
        Get the 3D points of the object.

        Returns:
            np.ndarray: The 3D points, shape (N, 3).
        """
        return self.key_points

    def get_uncertainties(self) -> np.ndarray:
        """
        Get the uncertainties associated with the 3D points.

        Returns:
            np.ndarray: The uncertainties, shape (N,).
        """
        return self.uncertainties

    def get_pose(self) -> np.ndarray:
        """
        Get the pose of the object.

        Returns:
            np.ndarray: The pose as a 4x4 transformation matrix.
        """
        return self.pose

    ##############################################################
    ########################## Setters ###########################
    ##############################################################
    def set_key_points(self, new_key_points: np.ndarray):
        """
        Set the 3D points and their uncertainties.

        Args:
            new_key_points (np.ndarray): New 3D points to set, shape (N, 3).
            new_uncertainties (np.ndarray): Uncertainties associated with the new points, shape (N,).
        """
        assert new_key_points.shape[1] == 3, "new_key_points should have shape (N, 3)"

        self.key_points = new_key_points

    def set_uncertainties(self, new_uncertainties: np.ndarray):
        """
        Set the uncertainties associated with the 3D points.

        Args:
            new_uncertainties (np.ndarray): New uncertainties to set, shape (N,).
        """
        assert (
            self.key_points.shape[0] == new_uncertainties.shape[0]
        ), "key_points and new_uncertainties should have the same number of points"

        self.uncertainties = new_uncertainties

    def set_key_points_and_uncertainty(
        self, new_key_points: np.ndarray, new_uncertainties: np.ndarray
    ):
        """
        Set the 3D points and their uncertainties.

        Args:
            new_points_3d (np.ndarray): New 3D points to set, shape (N, 3).
            new_uncertainties (np.ndarray): Uncertainties associated with the new points, shape (N,).
        """
        assert new_key_points.shape[1] == 3, "new_key_points should have shape (N, 3)"
        assert (
            new_key_points.shape[0] == new_uncertainties.shape[0]
        ), "new_key_points and new_uncertainties should have the same number of points"

        self.key_points = new_key_points
        self.uncertainties = new_uncertainties

    def set_pose(self, new_pose: np.ndarray):
        """
        Set the pose of the object.

        Args:
            new_pose (np.ndarray): New pose as a 4x4 transformation matrix.
        """
        assert new_pose.shape == (
            4,
            4,
        ), "new_pose should be a 4x4 transformation matrix"
        self.pose = new_pose

    def set_bbox(self, bbox):
        self.bbox = bbox
