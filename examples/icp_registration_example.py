#!/usr/bin/env python3
"""
Example script demonstrating Open3D ICP registration usage.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append("/home/justin/code/point-to-pose")

from point2pose.modules.register.open3d_icp_register import (
    Open3DICPRegister,
    create_icp_register,
)


def create_test_point_cloud(n_points: int = 1000, shape: str = "cube") -> np.ndarray:
    """
    Create test point clouds for registration.

    Args:
        n_points: Number of points to generate
        shape: Shape type ("cube", "sphere", "plane")

    Returns:
        Point cloud as numpy array (N, 3)
    """
    if shape == "cube":
        # Create a cube
        points = np.random.rand(n_points, 3) * 2 - 1  # Points in [-1, 1]^3
    elif shape == "sphere":
        # Create a sphere
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        u = np.random.uniform(0, 1, n_points)

        theta = np.arccos(costheta)
        r = u ** (1 / 3)  # Uniform distribution on sphere

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        points = np.column_stack([x, y, z])
    elif shape == "plane":
        # Create a plane
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.zeros(n_points)  # Flat plane
        points = np.column_stack([x, y, z])
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return points


def apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation matrix to point cloud."""
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Apply transformation
    transformed_homo = (transformation @ points_homo.T).T
    # Convert back to 3D
    return transformed_homo[:, :3]


def test_icp_registration():
    """Test ICP registration with different scenarios."""

    print("=== Open3D ICP Registration Test ===\n")

    # Test 1: Simple translation
    print("Test 1: Simple Translation")
    src_pcd = create_test_point_cloud(500, "cube")

    # Create target by applying a known transformation
    true_transform = np.eye(4)
    true_transform[:3, 3] = [0.5, 0.3, 0.2]  # Translation
    tgt_pcd = apply_transformation(src_pcd, true_transform)

    # Add some noise
    tgt_pcd += np.random.normal(0, 0.01, tgt_pcd.shape)

    # Test point-to-point ICP
    icp = create_icp_register(
        "point_to_point", max_iterations=50, max_correspondence_distance=0.1
    )
    T, stats = icp.register(src_pcd, tgt_pcd)

    print(f"True translation: {true_transform[:3, 3]}")
    print(f"Estimated translation: {T[:3, 3]}")
    print(f"Translation error: {np.linalg.norm(T[:3, 3] - true_transform[:3, 3]):.6f}")
    print(f"Fitness: {stats['fitness']:.4f}")
    print(f"RMSE: {stats['inlier_rmse']:.6f}")
    print(f"Converged: {stats['converged']}\n")

    # Test 2: Rotation and translation
    print("Test 2: Rotation and Translation")
    src_pcd = create_test_point_cloud(500, "sphere")

    # Create target with rotation and translation
    true_transform = np.eye(4)
    true_transform[:3, 3] = [0.2, 0.1, 0.3]  # Translation
    # Add rotation around Z-axis
    angle = np.pi / 6  # 30 degrees
    true_transform[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    tgt_pcd = apply_transformation(src_pcd, true_transform)
    tgt_pcd += np.random.normal(0, 0.01, tgt_pcd.shape)

    # Test point-to-plane ICP
    icp_ptp = create_icp_register(
        "point_to_plane", max_iterations=50, max_correspondence_distance=0.1
    )
    T, stats = icp_ptp.register(src_pcd, tgt_pcd)

    print(f"True transformation:\n{true_transform}")
    print(f"Estimated transformation:\n{T}")
    print(f"Transformation error: {np.linalg.norm(T - true_transform):.6f}")
    print(f"Fitness: {stats['fitness']:.4f}")
    print(f"RMSE: {stats['inlier_rmse']:.6f}")
    print(f"Converged: {stats['converged']}\n")

    # Test 3: With initial pose
    print("Test 3: With Initial Pose")
    src_pcd = create_test_point_cloud(300, "plane")

    # Create target with large transformation
    true_transform = np.eye(4)
    true_transform[:3, 3] = [1.0, 0.5, 0.3]  # Large translation
    tgt_pcd = apply_transformation(src_pcd, true_transform)
    tgt_pcd += np.random.normal(0, 0.02, tgt_pcd.shape)

    # Provide initial guess
    init_pose = np.eye(4)
    init_pose[:3, 3] = [0.8, 0.4, 0.2]  # Close to true transformation

    icp = create_icp_register(
        "point_to_point", max_iterations=50, max_correspondence_distance=0.2
    )
    T, stats = icp.register(src_pcd, tgt_pcd, init_pose)

    print(f"True translation: {true_transform[:3, 3]}")
    print(f"Initial guess: {init_pose[:3, 3]}")
    print(f"Estimated translation: {T[:3, 3]}")
    print(f"Translation error: {np.linalg.norm(T[:3, 3] - true_transform[:3, 3]):.6f}")
    print(f"Fitness: {stats['fitness']:.4f}")
    print(f"RMSE: {stats['inlier_rmse']:.6f}")
    print(f"Converged: {stats['converged']}\n")


def compare_icp_types():
    """Compare different ICP types on the same data."""

    print("=== Comparing ICP Types ===\n")

    # Create test data
    src_pcd = create_test_point_cloud(400, "cube")
    true_transform = np.eye(4)
    true_transform[:3, 3] = [0.3, 0.2, 0.1]
    tgt_pcd = apply_transformation(src_pcd, true_transform)
    tgt_pcd += np.random.normal(0, 0.01, tgt_pcd.shape)

    icp_types = ["point_to_point", "point_to_plane"]

    for icp_type in icp_types:
        print(f"{icp_type.upper()} ICP:")
        icp = create_icp_register(
            icp_type, max_iterations=30, max_correspondence_distance=0.1
        )
        T, stats = icp.register(src_pcd, tgt_pcd)

        print(
            f"  Translation error: {np.linalg.norm(T[:3, 3] - true_transform[:3, 3]):.6f}"
        )
        print(f"  Fitness: {stats['fitness']:.4f}")
        print(f"  RMSE: {stats['inlier_rmse']:.6f}")
        print(f"  Converged: {stats['converged']}")
        print()


if __name__ == "__main__":
    try:
        test_icp_registration()
        compare_icp_types()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
