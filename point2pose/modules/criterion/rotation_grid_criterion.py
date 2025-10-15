import numpy as np

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("rotation_grid")
class RotationGridCriterion(SampleCriterion):
    """
    Rotation grid criterion checks for every N iterations and returns true.
    ## TODO: implementation unfinished
    """

    def __init__(self, config):
        super().__init__()
        self._discretization_angle = config.get("discretization_angle", 60)

        self.num_dirs = self.n_dirs_for_degree(self._discretization_angle)

        # vector of unit directions uniformly on a sphere
        self.v = self.fibonacci_sphere(self.num_dirs)  # shape (num_dirs, 3)
        # we assume the first vector to be the initial direction
        self.c = self.v[0, :]

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        obj = context.objects[obj_id]
        init_R = obj.init_pose[:3, :3]
        R = obj.pose[:3, :3]

        R_to_fist = R @ init_R.T

        u = R_to_fist @ self.c

        # compute inner product of u and v
        inner_product = np.dot(u, self.v)

    def fibonacci_sphere(self, n: int):
        k = np.arange(n) + 0.5
        z = 1 - 2 * k / n
        r = np.sqrt(np.clip(1 - z * z, 0.0, 1.0))
        phi = (np.pi * (1 + 5**0.5)) * k  # golden angle progression
        x, y = r * np.cos(phi), r * np.sin(phi)
        return np.stack([x, y, z], axis=1)

    def n_dirs_for_degree(self, dir_deg):
        # Target geodesic spacing (radians)
        alpha = np.deg2rad(max(0.5, float(dir_deg)))  # guard against too-small/zero
        # For small alpha, spherical cap area π α^2 ≈ 4π/N  =>  N ≈ (2/α)^2
        n = int(np.ceil((2.0 / alpha) ** 2))
        return max(n, 6)  # tiny floor to avoid degeneracy
