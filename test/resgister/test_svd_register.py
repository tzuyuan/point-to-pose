import numpy as np
import pytest
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from point2pose.modules.register.svd_register import SVDRegister
from point2pose.utils.transform import transform_pts


class TestSVDRegister:
    """Test suite for SVDRegister class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.register = SVDRegister()

    def test_init(self):
        """Test SVDRegister initialization."""
        assert isinstance(self.register, SVDRegister)

    def test_basic_registration(self):
        """Test basic SVD registration with known transformation."""
        # Create source points (unit cube corners)
        src_pcd = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )

        # Apply a known transformation: 90-degree rotation around Z + translation
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t_true = np.array([2, 3, 4])

        T_true = np.eye(4)
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true

        # Transform source points to get target points
        tgt_pcd = transform_pts(T_true, src_pcd)

        # Register
        T_estimated = self.register.register_once(src_pcd, tgt_pcd)

        # Check that the transformation is close to the true one
        np.testing.assert_allclose(T_estimated, T_true, atol=1e-10)

    def test_registration_with_init_pose(self):
        """Test SVD registration with initial pose."""
        # Create source points
        src_pcd = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Apply initial transformation
        init_pose = np.eye(4)
        init_pose[:3, 3] = [1, 1, 1]  # Translation by [1,1,1]

        # Apply final transformation
        R_final = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t_final = np.array([0, 0, 0])

        T_final = np.eye(4)
        T_final[:3, :3] = R_final
        T_final[:3, 3] = t_final

        # Total transformation
        T_total = T_final @ init_pose

        # Transform source points
        tgt_pcd = transform_pts(T_total, src_pcd)

        # Register with init_pose
        T_estimated = self.register.register_once(src_pcd, tgt_pcd, init_pose)

        # Should recover the final transformation
        np.testing.assert_allclose(T_estimated, T_total, atol=1e-10)

    def test_rotation_matrix_properties(self):
        """Test that the estimated rotation matrix has proper properties."""
        # Create test points
        src_pcd = np.random.rand(10, 3)
        tgt_pcd = np.random.rand(10, 3)

        T = self.register.register_once(src_pcd, tgt_pcd)
        R = T[:3, :3]

        # Check orthogonality: R @ R.T should be identity
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

        # Check determinant should be +1 (proper rotation)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_identical_point_clouds(self):
        """Test registration with identical point clouds."""
        # Create source points
        src_pcd = np.random.rand(5, 3)
        tgt_pcd = src_pcd.copy()

        T = self.register.register_once(src_pcd, tgt_pcd)

        # Should return identity transformation
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_single_point(self):
        """Test registration with single point (should return identity)."""
        src_pcd = np.array([[1, 2, 3]])
        tgt_pcd = np.array([[1, 2, 3]])

        T = self.register.register_once(src_pcd, tgt_pcd)

        # Should return identity transformation
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_translation_only(self):
        """Test registration with translation-only transformation."""
        # Create source points
        src_pcd = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Apply translation only
        translation = np.array([5, -3, 2])
        T_true = np.eye(4)
        T_true[:3, 3] = translation

        tgt_pcd = transform_pts(T_true, src_pcd)

        T_estimated = self.register.register_once(src_pcd, tgt_pcd)

        # Check translation
        np.testing.assert_allclose(T_estimated[:3, 3], translation, atol=1e-10)

        # Check rotation is identity
        np.testing.assert_allclose(T_estimated[:3, :3], np.eye(3), atol=1e-10)

    def test_rotation_only(self):
        """Test registration with rotation-only transformation."""
        # Create source points
        src_pcd = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Apply 180-degree rotation around Z-axis
        R_true = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        T_true = np.eye(4)
        T_true[:3, :3] = R_true

        tgt_pcd = transform_pts(T_true, src_pcd)

        T_estimated = self.register.register_once(src_pcd, tgt_pcd)

        # Check rotation
        np.testing.assert_allclose(T_estimated[:3, :3], R_true, atol=1e-10)

        # Check translation is zero
        np.testing.assert_allclose(T_estimated[:3, 3], np.zeros(3), atol=1e-10)

    def test_reflection_case_handling(self):
        """Test that the method handles reflection cases correctly."""
        # Create points that would result in reflection
        src_pcd = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        # Create target points that are a reflection
        tgt_pcd = np.array(
            [[0, 0, 0], [1, 0, 0], [0, -1, 0]]  # This creates a reflection
        )

        T = self.register.register_once(src_pcd, tgt_pcd)
        R = T[:3, :3]

        # Even with reflection input, should return proper rotation (det = +1)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_large_point_cloud(self):
        """Test registration with a large number of points."""
        # Generate random point cloud
        np.random.seed(42)  # For reproducible results
        src_pcd = np.random.rand(100, 3)

        # Apply random transformation
        R_true = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]])
        t_true = np.array([1.5, -2.3, 0.7])

        T_true = np.eye(4)
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true

        tgt_pcd = transform_pts(T_true, src_pcd)

        T_estimated = self.register.register_once(src_pcd, tgt_pcd)

        # Should still be accurate
        np.testing.assert_allclose(T_estimated, T_true, atol=1e-10)

    def test_noisy_points(self):
        """Test registration with noisy point clouds."""
        # Create source points
        src_pcd = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])

        # Apply transformation
        R_true = np.array([[0.707, -0.707, 0], [0.707, 0.707, 0], [0, 0, 1]])
        t_true = np.array([1, 1, 1])

        T_true = np.eye(4)
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true

        tgt_pcd = transform_pts(T_true, src_pcd)

        # Add noise
        noise = np.random.normal(0, 0.01, tgt_pcd.shape)
        tgt_pcd_noisy = tgt_pcd + noise

        T_estimated = self.register.register_once(src_pcd, tgt_pcd_noisy)

        # Should still be reasonably accurate (within noise level)
        np.testing.assert_allclose(T_estimated, T_true, atol=0.1)

    def test_svd_fit_internal_method(self):
        """Test the internal _svd_fit method directly."""
        # Create test points
        pa = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        qa = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        T = self.register._svd_fit(pa, qa)

        # Should be a valid SE(3) matrix
        assert T.shape == (4, 4)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10

        # Test that it actually transforms the points correctly
        pa_transformed = transform_pts(T, pa)
        np.testing.assert_allclose(pa_transformed, qa, atol=1e-10)

    def test_edge_case_empty_points(self):
        """Test edge case with empty point clouds."""
        src_pcd = np.empty((0, 3))
        tgt_pcd = np.empty((0, 3))

        # This should raise an error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            self.register.register_once(src_pcd, tgt_pcd)


if __name__ == "__main__":
    pytest.main([__file__])
