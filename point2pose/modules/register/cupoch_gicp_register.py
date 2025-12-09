import numpy as np
from typing import Tuple, Dict, Optional

try:
    import cupoch as cph  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Cupoch is required for Generalized ICP registration. Install with: pip install cupoch"
    ) from exc

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER


@REGISTER.register_module("cupoch_gicp")
class CupochGICPRegister(Register):
    """
    Cupoch Generalized ICP (GICP) registration implementation.

    This class provides dense point cloud registration using Cupoch's GPU-accelerated
    Generalized ICP algorithm. GICP is a probabilistic extension of ICP that accounts
    for local surface geometry by using covariance matrices.

    It is designed to work with dense point clouds (e.g., masked point clouds from depth images)
    rather than sparse keypoints.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # Default GICP parameters
        self.max_correspondence_distance = config.get(
            "max_correspondence_distance", 0.015
        )

        self.min_num_points = config.get("min_num_points", 5)

        # For normal estimation (required for GICP)
        self.radius_normal = config.get("radius_normal", 0.01)
        self.max_nn_normal = config.get("max_nn_normal", 30)
        self.debug_level = config.get("debug_level", 0)

    def register(
        self,
        source_pcd: np.ndarray,
        target_pcd: np.ndarray,
        init_pose: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform Generalized ICP registration between source and target point clouds.

        Args:
            source_pcd: Source point cloud (N, 3) - typically keypoints or model points
            target_pcd: Target point cloud (M, 3) - typically dense masked points from depth image
            init_pose: Initial transformation matrix (4, 4), optional

        Returns:
            transformation_matrix: 4x4 transformation matrix
            stats: Dictionary containing registration statistics
        """
        stats = {}

        if (
            source_pcd.shape[0] < self.min_num_points
            or target_pcd.shape[0] < self.min_num_points
        ):
            return None, stats

        # Convert numpy arrays to Cupoch point clouds
        source = self._numpy_to_cupoch(source_pcd)
        target = self._numpy_to_cupoch(target_pcd)

        # GICP requires normals, so compute them for both point clouds
        # source, target = self._prepare_for_gicp(source, target)

        # Prepare initial pose
        pose_init = (
            init_pose.astype(np.float32)
            if init_pose is not None
            else np.eye(4, dtype=np.float32)
        )

        # Perform Generalized ICP registration
        try:
            result = cph.registration.registration_generalized_icp(
                source=source,
                target=target,
                max_correspondence_distance=self.max_correspondence_distance,
                init=pose_init,
                estimation=cph.registration.TransformationEstimationForGeneralizedICP(),
            )

            # Extract results
            transformation_matrix = result.transformation.astype(np.float64)

            if self.debug_level > 0:
                self._debug_visualize(source, target, transformation_matrix)

            # fitness = result.fitness
            # inlier_rmse = result.inlier_rmse

            # Compute residuals
            # source_transformed = source.transform(transformation_matrix)
            # residuals = self._compute_residuals(source_transformed, target)

            # Store statistics
            # stats.update(
            #     {
            #         "fitness": fitness,
            #         "inlier_rmse": inlier_rmse,
            #         "residuals": residuals,
            #         "converged": result.fitness > 0,
            #     }
            # )

        except (RuntimeError, ValueError) as e:
            print(f"Cupoch Generalized ICP registration failed: {e}")
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

        if result.fitness == 0 or np.isnan(transformation_matrix).any():
            return None, stats

        return transformation_matrix, stats

    def _numpy_to_cupoch(self, points: np.ndarray):
        """Convert numpy array to Cupoch point cloud."""
        pcd = cph.geometry.PointCloud()
        pcd.points = cph.utility.Vector3fVector(points.astype(np.float32))
        return pcd

    # def _prepare_for_gicp(
    #     self, source: cph.geometry.PointCloud, target: cph.geometry.PointCloud
    # ) -> Tuple[cph.geometry.PointCloud, cph.geometry.PointCloud]:
    #     """
    #     Prepare point clouds for Generalized ICP by computing normals.
    #     GICP requires normals to compute local surface geometry (covariance matrices).
    #     """
    #     # Compute normals for both point clouds
    #     source.estimate_normals(
    #         search_param=cph.geometry.KDTreeSearchParamHybrid(
    #             radius=self.radius_normal, max_nn=self.max_nn_normal
    #         )
    #     )
    #     target.estimate_normals(
    #         search_param=cph.geometry.KDTreeSearchParamHybrid(
    #             radius=self.radius_normal, max_nn=self.max_nn_normal
    #         )
    #     )

    #     return source, target

    def _compute_residuals(
        self, source: cph.geometry.PointCloud, target: cph.geometry.PointCloud
    ) -> np.ndarray:
        """Compute point-to-point residuals between source and target."""
        # Build KDTree for target
        target_tree = cph.geometry.KDTreeFlann(target)

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

    def _debug_visualize(
        self,
        source: cph.geometry.PointCloud,
        target: cph.geometry.PointCloud,
        transformation_matrix: np.ndarray,
    ):
        """Debug visualize the registration results."""
        source.transform(transformation_matrix)

        # paint source yellow
        num_src = len(source.points)
        num_tgt = len(target.points)
        source_colors = np.tile(
            np.array([[1.0, 1.0, 0.0]], dtype=np.float32), (num_src, 1)
        )
        source.colors = cph.utility.Vector3fVector(source_colors)

        # paint target blue
        target_colors = np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (num_tgt, 1)
        )
        target.colors = cph.utility.Vector3fVector(target_colors)

        cph.visualization.draw_geometries([source, target])
