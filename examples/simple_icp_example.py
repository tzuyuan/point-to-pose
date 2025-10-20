#!/usr/bin/env python3
"""
Simple example of using Open3D ICP registration.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append("/home/justin/code/point-to-pose")

from point2pose.modules.register.open3d_icp_register import create_icp_register


def main():
    """Simple ICP registration example."""

    # Create two point clouds
    print("Creating sample point clouds...")

    # Source point cloud - a simple cube
    n_points = 1000
    source_pcd = np.random.rand(n_points, 3) * 2 - 1  # Points in [-1, 1]^3

    # Target point cloud - same cube but translated and rotated
    target_pcd = source_pcd.copy()

    # Apply a known transformation
    # Translation
    target_pcd += np.array([0.5, 0.3, 0.2])

    # Add some noise
    target_pcd += np.random.normal(0, 0.01, target_pcd.shape)

    print(f"Source point cloud shape: {source_pcd.shape}")
    print(f"Target point cloud shape: {target_pcd.shape}")

    # Create ICP register
    print("\nPerforming ICP registration...")
    icp = create_icp_register(
        icp_type="point_to_point", max_iterations=50, max_correspondence_distance=0.1
    )

    # Perform registration
    transformation_matrix, stats = icp.register(source_pcd, target_pcd)

    # Print results
    print(f"\nRegistration Results:")
    print(f"Transformation Matrix:")
    print(transformation_matrix)
    print(f"\nStatistics:")
    print(f"  Fitness: {stats['fitness']:.4f}")
    print(f"  RMSE: {stats['inlier_rmse']:.6f}")
    print(f"  Converged: {stats['converged']}")
    print(f"  Number of residuals: {len(stats['residuals'])}")

    if len(stats["residuals"]) > 0:
        print(f"  Mean residual: {np.mean(stats['residuals']):.6f}")
        print(f"  Std residual: {np.std(stats['residuals']):.6f}")

    # Test with initial pose
    print(f"\nTesting with initial pose...")
    init_pose = np.eye(4)
    init_pose[:3, 3] = [0.4, 0.2, 0.1]  # Close to true transformation

    transformation_matrix_init, stats_init = icp.register(
        source_pcd, target_pcd, init_pose
    )

    print(f"Results with initial pose:")
    print(f"  Fitness: {stats_init['fitness']:.4f}")
    print(f"  RMSE: {stats_init['inlier_rmse']:.6f}")
    print(f"  Converged: {stats_init['converged']}")


if __name__ == "__main__":
    main()
