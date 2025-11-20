import numpy as np

from typing import Tuple


# def convert_pixel_to_world(
#     pixel,
#     depth_image,
#     cam_intrinsics,
#     depth_factor=1.0,
#     cam2world=np.eye(4),
#     remove_invalid=False,
#     min_depth=0.05,
#     max_depth=1.0,
# ):
#     """
#     Convert pixel coordinates to world coordinates with optional neighbor-based depth filling.
#     We assume the depth image is aligned with the rgb image.

#     Args:
#         pixel (tuple or np.ndarray): Either a single (x, y) or an array of shape (N, 2).
#         depth_image (np.ndarray): (H, W) raw depth image.
#         cam_intrinsics (np.ndarray): (3, 3) intrinsics.
#         cam2world (np.ndarray): (4, 4) camera-to-world transform.
#         depth_factor (float): depth scale (e.g., 1000 for mm->m).remove_invalid (bool): if True, drop invalid rows; else keep NaNs, preserving shape.

#     Returns:
#         world_pts: (N, 3) if remove_invalid=False; else (M, 3) with M <= N.
#         valid_mask: (N,) boolean validity mask (post-fill).
#     """
#     # Normalize input to (N, 2)
#     pixels = np.asarray(pixel)
#     if pixels.ndim == 1:
#         pixels = pixels[None, :]

#     H, W = depth_image.shape[:2]
#     N = pixels.shape[0]

#     fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
#     cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

#     # Integer indices
#     xs = pixels[:, 0].astype(int)
#     ys = pixels[:, 1].astype(int)

#     # Bounds
#     in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

#     # Sample depths
#     z_raw = np.full(N, np.nan, dtype=float)
#     z_raw[in_bounds] = depth_image[ys[in_bounds], xs[in_bounds]].astype(float)

#     # Scale
#     z = z_raw / float(depth_factor)

#     valid_depth = np.isfinite(z) & (z > min_depth) & (z < max_depth)

#     # Final validity mask
#     valid = in_bounds & valid_depth & np.isfinite(z)

#     # Prepare output
#     world_pts = np.full((N, 3), np.nan, dtype=float)

#     if np.any(valid):
#         u = xs[valid].astype(float)
#         v = ys[valid].astype(float)
#         zv = z[valid]

#         Xc = (u - cx) * zv / fx
#         Yc = (v - cy) * zv / fy
#         Zc = zv

#         Pc = np.stack([Xc, Yc, Zc], axis=1)  # (M, 3)
#         R = cam2world[:3, :3]
#         t = cam2world[:3, 3]
#         world_pts[valid] = (R @ Pc.T).T + t

#     if remove_invalid:
#         return world_pts[valid], valid
#     else:
#         return world_pts, valid


def convert_pixel_to_world(
    pixel,
    depth_image,
    cam_intrinsics,
    depth_factor=1.0,
    cam2world=np.eye(4),
    remove_invalid=False,
    min_depth=0.05,
    max_depth=1.0,
    fill_missing_depth=False,  # NEW: if True, fill NaN depth with n×n neighbor mean
    window_size=3,  # odd integer window size (3,5,7,...)
    min_neighbors=1,  # require at least this many valid neighbors to fill
):
    """
    Convert pixel coordinates to world coordinates with optional neighbor-based depth filling.
    We assume the depth image is aligned with the RGB image.

    Args:
        pixel (tuple or np.ndarray): Either a single (x, y) or an array of shape (N, 2).
        depth_image (np.ndarray): (H, W) raw depth image.
        cam_intrinsics (np.ndarray): (3, 3) intrinsics.
        cam2world (np.ndarray): (4, 4) camera-to-world transform.
        depth_factor (float): depth scale (e.g., 1000 for mm->m).
        remove_invalid (bool): if True, drop invalid rows; else keep NaNs, preserving shape.
        min_depth (float): minimum valid depth in meters (applied after scaling).
        max_depth (float): maximum valid depth in meters (applied after scaling).
        fill_missing_depth (bool): if True, fill NaN depth by averaging an n×n neighborhood.
        window_size (int): odd integer window size for filling (e.g., 3, 5).
        min_neighbors (int): minimum number of valid neighbors required to perform fill.

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

    # Sample depths (raw units)
    z_raw = np.full(N, np.nan, dtype=float)
    z_raw[in_bounds] = depth_image[ys[in_bounds], xs[in_bounds]].astype(float)

    # Optional: fill missing (only when the sampled value is NaN)
    if fill_missing_depth and window_size >= 3 and (window_size % 2 == 1):
        half = window_size // 2
        # Work in *scaled* meters for thresholding neighbors
        z_scaled = z_raw / float(depth_factor)

        # Indices that are in-bounds and NaN (missing)
        need_fill = in_bounds & ~np.isfinite(z_scaled)
        idxs = np.where(need_fill)[0]

        for i in idxs:
            x, y = xs[i], ys[i]
            x0 = max(0, x - half)
            x1 = min(W, x + half + 1)
            y0 = max(0, y - half)
            y1 = min(H, y + half + 1)

            # Extract neighborhood in raw units then scale
            neigh_raw = depth_image[y0:y1, x0:x1].astype(float)
            neigh_scaled = neigh_raw / float(depth_factor)

            # Valid neighbors: finite and within depth range
            neigh_valid = (
                np.isfinite(neigh_scaled)
                & (neigh_scaled > min_depth)
                & (neigh_scaled < max_depth)
            )

            if np.count_nonzero(neigh_valid) >= min_neighbors:
                # Average of valid neighbors in *scaled* meters
                z_fill = float(np.nanmean(neigh_scaled[neigh_valid]))
                z_scaled[i] = z_fill
                z_raw[i] = z_fill * float(depth_factor)  # keep z_raw consistent
            # else: leave as NaN (will remain invalid)

        # Continue using z_scaled below
        z = z_scaled
    else:
        # Scale (meters)
        z = z_raw / float(depth_factor)

    # Depth validity after (possible) fill
    valid_depth = np.isfinite(z) & (z > min_depth) & (z < max_depth)

    # Final validity mask
    valid = in_bounds & valid_depth

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


def compute_normals_from_depth(
    depth,
    depth_factor=1.0,
    cam_intrinsics=np.eye(3),
    min_depth=0.05,
    max_depth=1.0,
    fill_missing_depth=False,  # NEW: if True, fill NaN depth with n×n neighbor mean
    window_size=3,  # NEW: odd integer window size (3,5,7,...)
    min_neighbors=1,  # NEW: require at least this many valid neighbors to fill
):
    """
    depth: (H, W) depth map in meters
    K: (3, 3) intrinsics

    returns:
        normals: (H, W, 3) unit normals in camera frame
        points:  (H, W, 3) 3D points in camera frame
    """
    H, W = depth.shape
    # Create all pixel coordinates using meshgrid
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)
    pts, _ = convert_pixel_to_world(
        pixel=pixels,
        depth_image=depth,
        cam_intrinsics=cam_intrinsics,
        depth_factor=depth_factor,
        remove_invalid=False,
        min_depth=min_depth,
        max_depth=max_depth,
        fill_missing_depth=fill_missing_depth,  # if True, fill NaN depth with n×n neighbor mean
        window_size=window_size,  # odd integer window size (3,5,7,...)
        min_neighbors=min_neighbors,  # require at least this many valid neighbors to fill
    )  # (N, 3)

    pts = pts.reshape(H, W, 3)

    # mask invalid depths
    invalid = (depth == 0.0) | ~np.isfinite(depth)

    # we compute normals for interior pixels [1:-1, 1:-1]
    P = pts[1:-1, 1:-1]  # (H-2, W-2, 3)
    Px = pts[1:-1, 2:]  # (H-2, W-2, 3) right neighbor
    Py = pts[2:, 1:-1]  # (H-2, W-2, 3) down neighbor

    Tu = Px - P  # tangent along +u
    Tv = Py - P  # tangent along +v

    # cross product Tu x Tv
    N = np.cross(Tu, Tv)  # (H-2, W-2, 3)

    # normalize
    norm = np.linalg.norm(N, axis=-1, keepdims=True)
    eps = 1e-8
    N = N / (norm + eps)

    # full-sized normal image, initialize with zeros
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[1:-1, 1:-1] = N

    # invalidate normals where any of the 3 depth samples are invalid
    inv_center = invalid[1:-1, 1:-1]
    inv_right = invalid[1:-1, 2:]
    inv_down = invalid[2:, 1:-1]
    bad = inv_center | inv_right | inv_down

    normals[1:-1, 1:-1][bad] = 0.0  # or np.nan if you prefer

    # (Optional) orient normals to face the camera (assuming camera at origin, looking +Z)
    # If dot(n, p) > 0, flip it so it roughly points towards the camera.
    P_full = pts
    dots = (normals * P_full).sum(axis=-1)  # (H, W)
    flip_mask = dots > 0
    normals[flip_mask] *= -1.0

    return normals, pts


def extract_masked_region(
    img: np.ndarray, mask: np.ndarray, padding: int = 10
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Extract the bounding box region containing the mask from an image.

    Args:
        img: Image as numpy array (H, W, 3) RGB
        mask: Binary mask (H, W)
        padding: Additional padding around the bounding box

    Returns:
        cropped_image: Cropped image containing the masked region
        cropped_mask: Cropped mask aligned with the image
        bbox: Bounding box (x_min, y_min, x_max, y_max) in original image coordinates
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask_binary = (mask > 128).astype(np.uint8)
    else:
        mask_binary = mask.astype(np.uint8)

    # Find bounding box
    coords = np.where(mask_binary > 0)
    if len(coords[0]) == 0:
        # No mask found, return original image
        return img, mask, (0, 0, img.shape[1], img.shape[0])

    y_min, y_max = max(0, coords[0].min() - padding), min(
        img.shape[0], coords[0].max() + padding + 1
    )
    x_min, x_max = max(0, coords[1].min() - padding), min(
        img.shape[1], coords[1].max() + padding + 1
    )

    # Crop image and mask
    cropped_img = img[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask_binary[y_min:y_max, x_min:x_max].copy()

    return cropped_img, cropped_mask, (x_min, y_min, x_max, y_max)
