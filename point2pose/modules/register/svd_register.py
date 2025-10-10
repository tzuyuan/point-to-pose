import numpy as np


from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("svd")
class SVDRegister(Register):
    def __init__(self, config=None):
        super().__init__(config)

    def register(self, src_pcd, tgt_pcd, init_pose=None):
        """
        Perform a single registration step using SVD.
        """
        # transform the points if the initial pose is given
        if init_pose is not None:
            p0 = transform_pts(init_pose, src_pcd)
        else:
            p0 = src_pcd.copy()

        # fit transformation using svd
        T = self._svd_fit(p0, tgt_pcd)

        # recover init pose if used
        if init_pose is not None:
            T = T @ init_pose

        return T

    def _svd_fit(self, pa, qa):
        """
        Rigid SVD fit (Procrustes) of Pa->Qa with known correspondence.
        """
        # Pa, Qa: (M,3) centered fit
        # compute the centroids
        cp = pa.mean(axis=0)
        cq = qa.mean(axis=0)
        # center the points
        P = pa - cp
        Q = qa - cq
        # compute the covariance matrix
        H = P.T @ Q
        # compute the SVD
        U, _S, Vt = np.linalg.svd(H)
        # rotation
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # translation
        t = cq - R @ cp
        # return SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
