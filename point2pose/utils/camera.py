import numpy as np


def convert_pixel_to_world(
    pixel,
    depth_image,
    cam_intrinsic,
    cam2world=np.eye(4),
    depth_factor=1.0,
    remove_invalid=False,
    min_depth=0.05,
):
    """
    Convert pixel coordinates to world coordinates with optional neighbor-based depth filling.
    We assume the depth image is aligned with the rgb image.

    Args:
        pixel (tuple or np.ndarray): Either a single (x, y) or an array of shape (N, 2).
        depth_image (np.ndarray): (H, W) raw depth image.
        cam_intrinsic (np.ndarray): (3, 3) intrinsics.
        cam2world (np.ndarray): (4, 4) camera-to-world transform.
        depth_factor (float): depth scale (e.g., 1000 for mm->m).remove_invalid (bool): if True, drop invalid rows; else keep NaNs, preserving shape.

    Returns:
        world_pts: (N, 3) if remove_invalid=False; else (M, 3) with M <= N.
        valid_mask: (N,) boolean validity mask (post-fill).
    """
    # Normalize input to (N, 2)
    pixels = np.asarray(pixel)
    if pixels.ndim == 1:
        pixels = pixels[None, :]

    H, W = depth_image.shape[:2]
    N = pixels.shape[0]

    fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
    cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]

    # Integer indices
    xs = pixels[:, 0].astype(int)
    ys = pixels[:, 1].astype(int)

    # Bounds
    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    # Sample depths
    z_raw = np.full(N, np.nan, dtype=float)
    z_raw[in_bounds] = depth_image[ys[in_bounds], xs[in_bounds]].astype(float)

    # Scale
    z = z_raw / float(depth_factor)

    valid_depth = np.isfinite(z) & (z > min_depth)

    # Final validity mask
    valid = in_bounds & valid_depth & np.isfinite(z)

    # Prepare output
    world_pts = np.full((N, 3), np.nan, dtype=float)

    if np.any(valid):
        u = xs[valid].astype(float)
        v = ys[valid].astype(float)
        zv = z[valid]

        Xc = (u - cx) * zv / fx
        Yc = (v - cy) * zv / fy
        Zc = zv

        Pc = np.stack([Xc, Yc, Zc], axis=1)  # (M, 3)
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]
        world_pts[valid] = (R @ Pc.T).T + t

    if remove_invalid:
        return world_pts[valid], valid
    else:
        return world_pts, valid
