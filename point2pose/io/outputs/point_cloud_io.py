import os
import open3d as o3d
import numpy as np

from point2pose.utils.transform import transform_pts


def save_pcd(points, colors, save_dir, file_name="pcd"):
    """
    Save a point cloud with corresponding RGB colors.
    Args:
        points: np.ndarray (N, 3) float XYZ in meters
        colors: np.ndarray (N, 3) RGB; accepts uint8 [0,255] or float [0,1]
        save_dir: (str) output directory
        file_name: (str) base file name without extension
    """
    if points is None or len(points) == 0:
        return

    pts = np.asarray(points, dtype=float)

    # Normalize colors to float in [0,1]
    cols = np.asarray(colors)
    if cols.dtype == np.uint8:
        cols = cols.astype(np.float32) / 255.0
    else:
        cols = cols.astype(np.float32)
        cols = np.clip(cols, 0.0, 1.0)

    # Guard shape mismatches by trimming to min length
    n = min(pts.shape[0], cols.shape[0])
    if n == 0:
        return
    pts = pts[:n]
    cols = cols[:n]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"{file_name}.ply"), pcd)


def save_reg_pcd(
    src_pcd, trg_pcd, tf, save_dir, file_name="registered_pcd", reg_stats=None
):
    """
    Save a combined registered point cloud containing both target and transformed source points.
    Args:
        src_pcd: np.ndarray (N, 3) float 3dn source points
        trg_pcd: np.ndarray (M, 3) float 3d target points
        tf: (4, 4) float transformation matrix
        save_dir: (str) save directory
        file_name: (str) name for the saved file
        reg_stats: (optional dict) may include key "inliers" as (K,) bool mask.
            Indices where inliers is False (outliers) will be colored darker.
    """
    print(f"number of source points: {src_pcd.shape[0]}")
    print(f"number of target points: {trg_pcd.shape[0]}")
    src_pcd_transformed = transform_pts(tf, src_pcd)

    # Michigan maize color (yellow) for source point cloud
    maize_color = np.array([1.0, 0.796, 0.020])  # RGB normalized to [0,1]

    # Michigan blue color for target point cloud
    blue_color = np.array([0.0, 0.153, 0.463])  # RGB normalized to [0,1]

    # Darker variants for outliers
    # darker_maize = np.clip(maize_color * darken, 0.0, 1.0)
    # darker_blue = np.clip(blue_color * darken, 0.0, 1.0)
    darker_maize = np.array([1.0, 0.5, 0.0])
    darker_blue = np.array([1.0, 0.0, 0.5])
    # Combine both point clouds
    combined_points = np.vstack([trg_pcd, src_pcd_transformed])

    # Create combined colors: blue for target, maize for transformed source
    trg_colors = np.tile(blue_color, (trg_pcd.shape[0], 1))
    src_colors = np.tile(maize_color, (src_pcd_transformed.shape[0], 1))

    # If provided, apply outlier coloring using inliers mask
    if isinstance(reg_stats, dict) and "inliers" in reg_stats:
        inliers = reg_stats["inliers"]
        if (
            isinstance(inliers, np.ndarray)
            and inliers.ndim == 1
            and inliers.dtype == bool
        ):
            # Determine how many corresponded points the mask applies to
            n_match = min(
                inliers.shape[0], src_pcd_transformed.shape[0], trg_pcd.shape[0]
            )
            if n_match > 0:
                outlier_mask = ~inliers[:n_match]
                # Darken outliers on both clouds for corresponding indices
                if np.any(outlier_mask):
                    trg_colors[:n_match][outlier_mask] = darker_blue
                    src_colors[:n_match][outlier_mask] = darker_maize
    combined_colors = np.vstack([trg_colors, src_colors])

    # Create and save combined point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"{file_name}.ply"), pcd)
