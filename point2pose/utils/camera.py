import numpy as np


def convert_pixel_to_world(
    pixel,
    depth_image,
    cam_intrinsics,
    depth_factor=1.0,
    cam2world=np.eye(4),
    remove_invalid=False,
    min_depth=0.05,
    max_depth=1.0,
):
    """
    Convert pixel coordinates to world coordinates with optional neighbor-based depth filling.
    We assume the depth image is aligned with the rgb image.

    Args:
        pixel (tuple or np.ndarray): Either a single (x, y) or an array of shape (N, 2).
        depth_image (np.ndarray): (H, W) raw depth image.
        cam_intrinsics (np.ndarray): (3, 3) intrinsics.
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

    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

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

    valid_depth = np.isfinite(z) & (z > min_depth) & (z < max_depth)

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


def convert_pixel_within_mask_to_world(
    mask,
    depth_image,
    cam_intrinsics,
    depth_factor=1.0,
    cam2world=np.eye(4),
    remove_invalid=False,
    min_depth=0.05,
):
    """
    Convert all pixels inside a segmentation mask to 3D world coordinates.

    This is a convenience wrapper around convert_pixel_to_world that extracts
    pixel coordinates from the provided mask and performs the same projection.

    Args:
        mask (np.ndarray): Binary/boolean mask. Accepts (H, W), (C, H, W), or (N, 1, H, W).
                           Any non-zero value is treated as inside the mask.
        depth_image (np.ndarray): (H, W) raw depth image.
        cam_intrinsics (np.ndarray): (3, 3) camera intrinsics.
        depth_factor (float): Depth scale (e.g., 1000 for mm->m).
        cam2world (np.ndarray): (4, 4) camera-to-world transform.
        remove_invalid (bool): If True, drop invalid rows; else keep NaNs, preserving shape.
        min_depth (float): Minimum valid depth in meters after scaling.

    Returns:
        world_pts (np.ndarray): (K, 3) 3D points corresponding to mask pixels. If
                                remove_invalid=False, K equals the number of mask
                                pixels and invalid rows contain NaNs. If True, K is
                                the number of valid mask pixels.
        valid_mask (np.ndarray): (K,) boolean array indicating which mask pixels are valid.
    """
    m = np.asarray(mask)
    if m.ndim == 2:
        mask2d = m > 0
    elif m.ndim >= 3:
        # Collapse leading dimensions and treat any non-zero as inside the mask
        collapsed = m.reshape(-1, m.shape[-2], m.shape[-1])
        mask2d = np.any(collapsed > 0, axis=0)
    else:
        raise ValueError("mask must be 2D or have leading dims ending with (H, W)")

    ys, xs = np.where(mask2d)
    if xs.size == 0:
        if remove_invalid:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=bool)
        else:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=bool)

    pixels = np.stack([xs, ys], axis=1)

    world_pts, valid = convert_pixel_to_world(
        pixel=pixels,
        depth_image=depth_image,
        cam_intrinsics=cam_intrinsics,
        depth_factor=depth_factor,
        cam2world=cam2world,
        remove_invalid=remove_invalid,
        min_depth=min_depth,
    )

    return world_pts, valid
