import numpy as np
from typing import Tuple, Dict, Optional

try:
    import open3d as o3d  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Open3D is required for ICP registration. Install with: pip install open3d"
    ) from exc

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER


@REGISTER.register_module("open3d_icp")
class Open3DICPRegister(Register):
    """
    Open3D ICP (Iterative Closest Point) registration implementation.

    This class provides various ICP algorithms from Open3D:
    - Point-to-point ICP
    - Point-to-plane ICP
    - Generalized ICP
    """

    def __init__(self, config=None):
        super().__init__(config)

        # Default ICP parameters
        self.icp_type = config.get(
            "icp_type", "point_to_point"
        )  # point_to_point, point_to_plane, generalized
        self.max_iterations = config.get("max_iterations", 30)
        self.max_correspondence_distance = config.get(
            "max_correspondence_distance", 0.02
        )
        self.relative_fitness = config.get("relative_fitness", 1e-6)
        self.relative_rmse = config.get("relative_rmse", 1e-6)
        self.estimation_method = config.get(
            "estimation_method",
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        # For point-to-plane ICP
        self.radius_normal = config.get("radius_normal", 0.01)
        self.max_nn_normal = config.get("max_nn_normal", 30)

        # For generalized ICP
        self.robust_kernel = config.get(
            "robust_kernel", o3d.pipelines.registration.robust_kernel.RobustKernel()
        )

    def register(
        self,
        source_pcd: np.ndarray,
        target_pcd: np.ndarray,
        init_pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform ICP registration between source and target point clouds.

        Args:
            source_pcd: Source point cloud (N, 3)
            target_pcd: Target point cloud (M, 3)
            init_pose: Initial transformation matrix (4, 4), optional

        Returns:
            transformation_matrix: 4x4 transformation matrix
            stats: Dictionary containing registration statistics
        """
        stats = {}

        # Convert numpy arrays to Open3D point clouds
        source = self._numpy_to_open3d(source_pcd)
        target = self._numpy_to_open3d(target_pcd)

        # Apply initial transformation if provided
        if init_pose is not None:
            source.transform(init_pose)

        # Prepare point clouds based on ICP type
        if self.icp_type == "point_to_plane":
            source, target = self._prepare_point_to_plane(source, target)
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        elif self.icp_type == "generalized":
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
                    robust_kernel=self.robust_kernel
                )
            )
        else:  # point_to_point
            estimation_method = self.estimation_method

        # Perform ICP registration
        try:
            result = o3d.pipelines.registration.registration_icp(
                source=source,
                target=target,
                max_correspondence_distance=self.max_correspondence_distance,
                estimation_method=estimation_method,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=self.relative_fitness,
                    relative_rmse=self.relative_rmse,
                    max_iteration=self.max_iterations,
                ),
            )

            # Extract results
            transformation_matrix = result.transformation
            fitness = result.fitness
            inlier_rmse = result.inlier_rmse

            # Compute residuals
            source_transformed = source.transform(transformation_matrix)
            residuals = self._compute_residuals(source_transformed, target)

            # Store statistics
            stats.update(
                {
                    "fitness": fitness,
                    "inlier_rmse": inlier_rmse,
                    "residuals": residuals,
                    "converged": result.fitness > 0,
                    "num_iterations": (
                        len(result.correspondence_set)
                        if hasattr(result, "correspondence_set")
                        else 0
                    ),
                }
            )

            # Recover initial pose if used
            if init_pose is not None:
                transformation_matrix = transformation_matrix @ init_pose

        except (RuntimeError, ValueError) as e:
            print(f"ICP registration failed: {e}")
            # Return identity transformation on failure
            transformation_matrix = np.eye(4)
            stats.update(
                {
                    "fitness": 0.0,
                    "inlier_rmse": float("inf"),
                    "residuals": np.array([]),
                    "converged": False,
                    "error": str(e),
                }
            )

        return transformation_matrix, stats

    def _numpy_to_open3d(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """Convert numpy array to Open3D point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _prepare_point_to_plane(
        self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Prepare point clouds for point-to-plane ICP by computing normals."""
        # Compute normals for both point clouds
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius_normal, max_nn=self.max_nn_normal
            )
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius_normal, max_nn=self.max_nn_normal
            )
        )

        return source, target

    def _compute_residuals(
        self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """Compute point-to-point residuals between source and target."""
        # Build KDTree for target
        target_tree = o3d.geometry.KDTreeFlann(target)

        residuals = []
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)

        for point in source_points:
            # Find closest point in target
            [k, idx, _] = target_tree.search_knn_vector_3d(point, 1)
            if k > 0:
                closest_point = target_points[idx[0]]
                residual = np.linalg.norm(point - closest_point)
                residuals.append(residual)

        return np.array(residuals)


def create_icp_register(
    icp_type: str = "point_to_point", **kwargs
) -> Open3DICPRegister:
    """
    Factory function to create ICP register with specified parameters.

    Args:
        icp_type: Type of ICP ("point_to_point", "point_to_plane", "generalized")
        **kwargs: Additional configuration parameters

    Returns:
        Open3DICPRegister instance
    """
    config = {"icp_type": icp_type, **kwargs}
    return Open3DICPRegister(config)


def test_icp_registration():
    """Test the ICP registration functionality."""

    # Create sample point clouds
    def create_sample_point_cloud(n_points: int = 1000) -> np.ndarray:
        """Create a sample point cloud."""
        # Create a simple cube
        points = np.random.rand(n_points, 3) * 2 - 1  # Points in [-1, 1]^3
        return points

    # Test different ICP types
    source_pcd = create_sample_point_cloud(500)
    target_pcd = create_sample_point_cloud(500)

    # Add some noise to target
    target_pcd += np.random.normal(0, 0.01, target_pcd.shape)

    # Test point-to-point ICP
    print("Testing Point-to-Point ICP:")
    icp_pp = create_icp_register("point_to_point", max_iterations=50)
    _, stats_pp = icp_pp.register(source_pcd, target_pcd)
    print(f"Fitness: {stats_pp['fitness']:.4f}")
    print(f"RMSE: {stats_pp['inlier_rmse']:.6f}")
    print(f"Converged: {stats_pp['converged']}")

    # Test point-to-plane ICP
    print("\nTesting Point-to-Plane ICP:")
    icp_ptp = create_icp_register("point_to_plane", max_iterations=50)
    _, stats_ptp = icp_ptp.register(source_pcd, target_pcd)
    print(f"Fitness: {stats_ptp['fitness']:.4f}")
    print(f"RMSE: {stats_ptp['inlier_rmse']:.6f}")
    print(f"Converged: {stats_ptp['converged']}")

    # Test with initial pose
    print("\nTesting with Initial Pose:")
    initial_pose = np.eye(4)
    initial_pose[:3, 3] = [0.1, 0.1, 0.1]  # Small translation
    _, stats_init = icp_pp.register(source_pcd, target_pcd, initial_pose)
    print(f"Fitness: {stats_init['fitness']:.4f}")
    print(f"RMSE: {stats_init['inlier_rmse']:.6f}")
    print(f"Converged: {stats_init['converged']}")


# Example usage and testing
if __name__ == "__main__":
    test_icp_registration()
