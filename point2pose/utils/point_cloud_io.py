import os
import open3d as o3d
import numpy as np

from point2pose.utils.transform import transform_pts


def save_reg_pcd(src_pcd, trg_pcd, tf, save_dir, file_name="registered_pcd"):
    """
    Save a combined registered point cloud containing both target and transformed source points.
    Args:
        src_pcd: np.ndarray (N, 3) float 3dn source points
        trg_pcd: np.ndarray (M, 3) float 3d target points
        tf: (4, 4) float transformation matrix
        save_dir: (str) save directory
        file_name: (str) name for the saved file
    """
    print(f"number of source points: {src_pcd.shape[0]}")
    print(f"number of target points: {trg_pcd.shape[0]}")
    src_pcd_transformed = transform_pts(tf, src_pcd)

    # Michigan maize color (yellow) for source point cloud
    maize_color = np.array([1.0, 0.796, 0.020])  # RGB normalized to [0,1]

    # Michigan blue color for target point cloud
    blue_color = np.array([0.0, 0.153, 0.463])  # RGB normalized to [0,1]

    # Combine both point clouds
    combined_points = np.vstack([trg_pcd, src_pcd_transformed])

    # Create combined colors: blue for target, maize for transformed source
    trg_colors = np.tile(blue_color, (trg_pcd.shape[0], 1))
    src_colors = np.tile(maize_color, (src_pcd_transformed.shape[0], 1))
    combined_colors = np.vstack([trg_colors, src_colors])

    # Create and save combined point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"{file_name}.ply"), pcd)
