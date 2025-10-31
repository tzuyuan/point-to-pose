import numpy as np
import open3d as o3d


class Object:
    """
    Object class that keeps tracks of the 3D points belonging to the object.
    """

    def __init__(self, id: int):
        self.id = id

        # 3D points belonging to the object, represented in the first frame coordinate system
        self.key_points = np.empty((0, 3))  # Mx3
        self.uncertainties = np.empty((0,))
        self.valid = np.empty((0,))  # M, bool
        self.key_point_frames = np.empty(
            (0,), dtype=int
        )  # M, frame IDs when points were added

        # pose of the object
        # init_pose is the transformation from the object frame to the first frame
        # e.g. the estimated pose of the object in the first frame
        self.init_pose = np.eye(4)

        # pose is the transformation from the first frame to the current frame
        # T_i_0: transformation points from frame 0 -> frame i
        self.pose = np.eye(
            4
        )  # 4x4 transformation matrix from object frame to world frame

        self.init_bbox = None
        self.bbox = (
            None  # 3D bounding box of the object, represented in the object frame
        )

        omega = np.zeros(3)
        v = np.zeros(3)
        mean_residual = 0.0

    def add_key_points(
        self,
        new_key_points: np.ndarray,
        new_uncertainties: np.ndarray,
        new_valid: np.ndarray,
        frame_id: int = None,
    ):
        """
        Add new 3D points to the object.

        Args:
            new_key_points (np.ndarray): New 3D points to add, shape (M, 3).
            new_uncertainties (np.ndarray): Uncertainties associated with the new points, shape (M,).
            frame_id (int): Frame ID when these points were added. If None, uses -1.
        """
        assert new_key_points.shape[1] == 3, "new_key_points should have shape (M, 3)"
        assert (
            new_key_points.shape[0] == new_uncertainties.shape[0]
        ), "new_key_points and new_uncertainties should have the same number of points"

        if frame_id is None:
            frame_id = -1

        self.key_points = np.vstack((self.key_points, new_key_points))
        self.uncertainties = np.hstack((self.uncertainties, new_uncertainties))
        self.valid = np.hstack((self.valid, new_valid))
        self.key_point_frames = np.hstack(
            (self.key_point_frames, np.full(new_key_points.shape[0], frame_id))
        )

    def save_key_points_with_colors(self, save_path: str, current_frame_id: int = None):
        """
        Save key points as a colored point cloud with different colors for different frames.

        Args:
            save_path (str): Path to save the point cloud file (.ply format)
            current_frame_id (int): Current frame ID for coloring (if None, uses frame_id from key_point_frames)
        """

        if len(self.key_points) == 0:
            return

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.key_points)

        # Generate colors based on frame IDs
        if current_frame_id is None:
            frame_ids = self.key_point_frames
        else:
            frame_ids = self.key_point_frames.copy()
            # Update points added in current frame
            frame_ids[frame_ids == -1] = current_frame_id

        # Normalize frame IDs to [0, 1] for color mapping
        if len(np.unique(frame_ids)) > 1:
            min_frame = np.min(frame_ids)
            max_frame = np.max(frame_ids)
            normalized_frames = (frame_ids - min_frame) / (max_frame - min_frame)
        else:
            normalized_frames = np.ones_like(frame_ids) * 0.5

        # Create color map (using HSV to get distinct colors)
        colors = np.zeros((len(self.key_points), 3))
        for i, norm_frame in enumerate(normalized_frames):
            # Use HSV color space: hue varies with frame, saturation=1, value=1
            hue = norm_frame * 0.8  # Use 0.8 to avoid wrapping back to red
            # Convert HSV to RGB
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors[i] = [r, g, b]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud(save_path, pcd)
