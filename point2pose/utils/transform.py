import numpy as np


def transform_pts(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a rigid transform to a set of 3D points.

    Args:
        T: (4,4) homogeneous transform matrix
        pts: (N,3) points

    Returns:
        (N,3) transformed points
    """
    assert T.shape == (4, 4)
    assert pts.ndim == 2 and pts.shape[1] == 3
    N = pts.shape[0]
    pts_h = np.c_[pts, np.ones((N, 1))]  # (N,4)
    pts_transformed_h = (T @ pts_h.T).T  # (N,4)
    return pts_transformed_h[:, :3]  # (N,3)


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo
