import os

import numpy as np
import cv2
import torch

from point2pose.utils.transform import to_homo


def _normalize_object_masks(mask_data):
    """
    Normalize different mask formats into a list of 2D boolean object masks.
    Supports list/tuple, torch.Tensor, and np.ndarray with shapes:
      [N,1,H,W], [N,H,W], [1,H,W], [H,W], [H,W,1]
    """
    if mask_data is None:
        return []

    if isinstance(mask_data, (list, tuple)):
        object_masks = []
        for m in mask_data:
            arr = (
                m.detach().cpu().numpy()
                if isinstance(m, torch.Tensor)
                else np.asarray(m)
            )
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim == 2:
                object_masks.append(arr.astype(bool))
        return object_masks

    arr = (
        mask_data.detach().cpu().numpy()
        if isinstance(mask_data, torch.Tensor)
        else np.asarray(mask_data)
    )

    if arr.ndim == 4:
        if arr.shape[1] == 1:
            return [arr[i, 0].astype(bool) for i in range(arr.shape[0])]
        collapsed = np.any(arr > 0, axis=1)
        return [collapsed[i].astype(bool) for i in range(collapsed.shape[0])]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return [arr[0].astype(bool)]
        if arr.shape[-1] == 1:
            return [arr[..., 0].astype(bool)]
        return [arr[i].astype(bool) for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr.astype(bool)]
    return []


def draw_points_on_image(image, points, colors):
    # if points are tensor
    # print(points.shape)
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if isinstance(colors, np.ndarray):
        colors = colors.tolist()

    for i in range(points.shape[0]):
        cv2.circle(
            image,
            points[i, :].astype(int).reshape(2),
            radius=5,
            color=colors[i],
            thickness=-1,
        )


def _map_track_ids_to_obj_rows(obj, track_ids):
    """Map global track IDs to object keypoint rows. Returns -1 for missing IDs."""
    tids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    rows = np.full(tids.shape[0], -1, dtype=np.int64)

    t2o = getattr(obj, "track_idx_2_obj_idx", None)
    if t2o is None:
        return rows

    t2o = np.asarray(t2o).reshape(-1)
    if t2o.size == 0:
        return rows

    ok = (tids >= 0) & (tids < t2o.shape[0])
    if np.any(ok):
        rows[ok] = np.asarray(t2o[tids[ok]], dtype=np.int64)
    return rows


def get_n_uncertainty_colors(uncertainties, u_min=0.0, u_max=1.0, inverse=False):
    """
    Map uncertainties to BGR colors using OpenCV colormap (e.g., COLORMAP_JET).

    Args:
        uncertainties (np.ndarray): (N,) array of uncertainty values.
        u_min (float): minimum value for normalization.
        u_max (float): maximum value for normalization.
        inverse (bool): if True, invert the normalized scale.

    Returns:
        np.ndarray: (N, 3) array of colors in BGR (uint8).
    """
    # Normalize to [0,1]
    norm = np.clip((uncertainties - u_min) / (u_max - u_min + 1e-8), 0, 1)
    if inverse:
        norm = 1 - norm

    # Convert to 0–255 uint8 for colormap lookup
    norm_uint8 = (norm * 255).astype(np.uint8)

    # Apply OpenCV colormap (e.g., COLORMAP_JET)
    colors_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

    # Convert to plain array (N, 3)
    return colors_bgr.reshape(-1, 3)


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """
    Copied from FoundationPose.
    Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(
            int
        )  # (2,2)
        img = cv2.line(
            img,
            uv[0].tolist(),
            uv[1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def _box_corners_from_min_max(min_xyz, max_xyz):
    """Return 8 box corners with a stable edge-compatible ordering."""
    xmin, ymin, zmin = np.asarray(min_xyz, dtype=float).reshape(3)
    xmax, ymax, zmax = np.asarray(max_xyz, dtype=float).reshape(3)
    return np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )


def _box_corners_from_center_extent(center, extent, rotation=None):
    """Return 8 oriented box corners from center, extent, and optional 3x3 rotation."""
    c = np.asarray(center, dtype=float).reshape(3)
    e = np.asarray(extent, dtype=float).reshape(3)
    half = 0.5 * e
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=float,
    )
    if rotation is None:
        return local + c.reshape(1, 3)
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    return (R @ local.T).T + c.reshape(1, 3)


def draw_oriented_3d_box(
    K, img, ob_in_cam, box_corners, line_color=(0, 255, 0), linewidth=2
):
    """
    Draw a 3D box from 8 corners expressed in the local object frame.
    `ob_in_cam` maps those local corners to the camera frame.
    """
    corners = np.asarray(box_corners, dtype=float).reshape(-1, 3)
    if corners.shape[0] != 8:
        raise ValueError(f"box_corners must have shape (8,3), got {corners.shape}")

    pts_cam = (ob_in_cam @ to_homo(corners).T).T[:, :3]
    if not np.all(np.isfinite(pts_cam)):
        return img

    proj = (K @ pts_cam.T).T
    denom = np.clip(proj[:, 2], 1e-6, None)
    uv = np.round(proj[:, :2] / denom.reshape(-1, 1)).astype(int)

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for i0, i1 in edges:
        img = cv2.line(
            img,
            uv[i0].tolist(),
            uv[i1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
    return img


def _resolve_per_object_payload(payload, obj_idx):
    """Resolve per-object payload from dict/list/singleton containers."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(obj_idx, None)
    try:
        arr = np.asarray(payload, dtype=object)
        if arr.ndim >= 1 and obj_idx < len(payload):
            return payload[obj_idx]
    except Exception:
        pass
    return payload


def _resolve_mesh_geometry_for_object(
    mesh_vertices_by_object, mesh_faces_by_object, obj_idx
):
    """
    Return (vertices, faces) for one object index.
    Supports:
      - separate per-object vertices/faces containers (dict/list/singleton)
      - mesh-like payload with `.vertices` and `.faces`
    """
    verts_src = _resolve_per_object_payload(mesh_vertices_by_object, obj_idx)
    faces_src = _resolve_per_object_payload(mesh_faces_by_object, obj_idx)

    if (
        verts_src is not None
        and hasattr(verts_src, "vertices")
        and hasattr(verts_src, "faces")
    ):
        mesh_obj = verts_src
        verts_src = getattr(mesh_obj, "vertices", None)
        if faces_src is None:
            faces_src = getattr(mesh_obj, "faces", None)

    if verts_src is None or faces_src is None:
        return None, None

    try:
        verts = np.asarray(verts_src, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces_src, dtype=np.int64).reshape(-1, 3)
    except Exception:
        return None, None

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return None, None
    return verts, faces


def _rasterize_mesh_silhouette(vertices_obj, faces, ob_in_cam, K, image_h, image_w):
    """Rasterize projected mesh silhouette mask for one object pose."""
    mask = np.zeros((image_h, image_w), dtype=np.uint8)

    verts_h = np.hstack(
        [
            vertices_obj.astype(np.float64),
            np.ones((vertices_obj.shape[0], 1), dtype=np.float64),
        ]
    )
    verts_cam = (ob_in_cam @ verts_h.T).T[:, :3]
    z = verts_cam[:, 2]

    proj = (K @ verts_cam.T).T
    uv = proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)

    face_z = z[faces]
    valid_faces = np.all(face_z > 1e-6, axis=1)
    if not np.any(valid_faces):
        return mask

    tris = uv[faces[valid_faces]]
    finite = np.isfinite(tris).all(axis=(1, 2))
    tris = tris[finite]
    if tris.shape[0] == 0:
        return mask

    tri_min = tris.min(axis=1)
    tri_max = tris.max(axis=1)
    intersects = ~(
        (tri_max[:, 0] < 0)
        | (tri_max[:, 1] < 0)
        | (tri_min[:, 0] >= image_w)
        | (tri_min[:, 1] >= image_h)
    )
    tris = tris[intersects]
    if tris.shape[0] == 0:
        return mask

    tris_i32 = np.round(tris).astype(np.int32)
    cv2.fillPoly(mask, tris_i32, 255)
    return mask


def draw_projected_mesh_contour(
    K,
    img,
    ob_in_cam,
    vertices_obj,
    faces,
    line_color=(0, 255, 255),
    linewidth=2,
):
    """
    Draw projected contour of a mesh at pose `ob_in_cam` onto `img`.
    """
    h, w = img.shape[:2]
    silhouette = _rasterize_mesh_silhouette(
        vertices_obj=vertices_obj,
        faces=faces,
        ob_in_cam=ob_in_cam,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        image_h=h,
        image_w=w,
    )
    contours, _ = cv2.findContours(
        silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        cv2.drawContours(
            img,
            contours,
            -1,
            color=tuple(int(x) for x in line_color),
            thickness=int(linewidth),
            lineType=cv2.LINE_AA,
        )
    return img


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(
    image,
    ob_in_cam,
    scale=0.1,
    K=np.eye(3),
    thickness=3,
    transparency=0,
    is_input_rgb=False,
):
    """
    Copied from FoundationPose.
    @color: BGR
    """
    if is_input_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = image.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp


def _make_translation(t_xyz):
    """Create a 4x4 translation matrix."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(t_xyz, dtype=float).reshape(3)
    return T


def _extract_bbox_info(bbox_like):
    """
    Extract bbox attributes from a bbox-like object or a (2,3) array.
    Returns keys: mn, mx, center, extent, rot. Any entry may be None if unavailable.
    """
    if bbox_like is None:
        return {"mn": None, "mx": None, "center": None, "extent": None, "rot": None}

    # Case 0: dictionary bbox metadata (e.g., {"center":..., "extent":..., "rot":...})
    if isinstance(bbox_like, dict):
        mn = mx = center = extent = rot = None

        # Optional packed min/max array under common keys.
        mm = bbox_like.get("bbox", None)
        if mm is None:
            mm = bbox_like.get("min_max", None)
        if mm is None:
            mm = bbox_like.get("bbox_min_max", None)
        if mm is not None:
            mm_arr = np.asarray(mm)
            if mm_arr.ndim == 2 and mm_arr.shape == (2, 3):
                mn = mm_arr.min(axis=0).astype(float)
                mx = mm_arr.max(axis=0).astype(float)

        # Direct min/max fields.
        if mn is None:
            for key in ("mn", "min_bound", "min_xyz", "min"):
                if key in bbox_like:
                    try:
                        mn = np.asarray(bbox_like[key], dtype=float).reshape(3)
                        break
                    except Exception:
                        mn = None
        if mx is None:
            for key in ("mx", "max_bound", "max_xyz", "max"):
                if key in bbox_like:
                    try:
                        mx = np.asarray(bbox_like[key], dtype=float).reshape(3)
                        break
                    except Exception:
                        mx = None

        for key in ("center", "c"):
            if key in bbox_like:
                try:
                    center = np.asarray(bbox_like[key], dtype=float).reshape(3)
                    break
                except Exception:
                    center = None
        for key in ("extent", "size", "dims"):
            if key in bbox_like:
                try:
                    extent = np.asarray(bbox_like[key], dtype=float).reshape(3)
                    break
                except Exception:
                    extent = None
        for key in ("rot", "R", "rotation"):
            if key in bbox_like:
                try:
                    rot = np.asarray(bbox_like[key], dtype=float).reshape(3, 3)
                    break
                except Exception:
                    rot = None

        if extent is None and (mn is not None) and (mx is not None):
            extent = mx - mn
        if center is None and (mn is not None) and (mx is not None):
            center = 0.5 * (mn + mx)
        if (mn is None or mx is None) and (center is not None) and (extent is not None):
            half = 0.5 * extent
            mn = center - half
            mx = center + half

        return {"mn": mn, "mx": mx, "center": center, "extent": extent, "rot": rot}

    # Case 1: numeric (2,3) min/max
    arr = np.asarray(bbox_like)
    if arr.ndim == 2 and arr.shape == (2, 3):
        mn = arr.min(axis=0).astype(float)
        mx = arr.max(axis=0).astype(float)
        center = 0.5 * (mn + mx)
        extent = mx - mn
        return {"mn": mn, "mx": mx, "center": center, "extent": extent, "rot": None}

    # Case 2: Open3D-like bbox object
    mn = mx = center = extent = rot = None

    if hasattr(bbox_like, "get_min_bound") and hasattr(bbox_like, "get_max_bound"):
        try:
            mn = np.asarray(bbox_like.get_min_bound(), dtype=float).reshape(3)
            mx = np.asarray(bbox_like.get_max_bound(), dtype=float).reshape(3)
        except Exception:
            mn = mx = None

    if mn is None and hasattr(bbox_like, "min_bound"):
        try:
            mn = np.asarray(bbox_like.min_bound, dtype=float).reshape(3)
        except Exception:
            mn = None
    if mx is None and hasattr(bbox_like, "max_bound"):
        try:
            mx = np.asarray(bbox_like.max_bound, dtype=float).reshape(3)
        except Exception:
            mx = None

    # Center
    if hasattr(bbox_like, "get_center"):
        try:
            center = np.asarray(bbox_like.get_center(), dtype=float).reshape(3)
        except Exception:
            center = None
    if center is None and hasattr(bbox_like, "center"):
        try:
            center = np.asarray(bbox_like.center, dtype=float).reshape(3)
        except Exception:
            center = None

    # Extent
    if hasattr(bbox_like, "get_extent"):
        try:
            extent = np.asarray(bbox_like.get_extent(), dtype=float).reshape(3)
        except Exception:
            extent = None
    if extent is None and hasattr(bbox_like, "extent"):
        try:
            extent = np.asarray(bbox_like.extent, dtype=float).reshape(3)
        except Exception:
            extent = None

    # OBB orientation (if available)
    if hasattr(bbox_like, "R"):
        try:
            rot = np.asarray(bbox_like.R, dtype=float).reshape(3, 3)
        except Exception:
            rot = None
    if rot is None and hasattr(bbox_like, "rotation"):
        try:
            rot = np.asarray(bbox_like.rotation, dtype=float).reshape(3, 3)
        except Exception:
            rot = None

    # Fill missing pieces from others when possible
    if extent is None and (mn is not None) and (mx is not None):
        extent = mx - mn
    if center is None and (mn is not None) and (mx is not None):
        center = 0.5 * (mn + mx)

    # If bounds missing but have center+extent, synthesize bounds
    if (mn is None or mx is None) and (center is not None) and (extent is not None):
        half = 0.5 * extent
        mn = center - half
        mx = center + half

    return {"mn": mn, "mx": mx, "center": center, "extent": extent, "rot": rot}


def _resolve_pose_and_bbox(
    pose_in_cam,
    bbox_source,
    bbox_frame="center",
    assume_pose_is_bbox_center=False,
):
    """
    Interprets pose_in_cam as either:
      - 'mesh'  : T_cam_mesh, bbox corners in mesh coords [mn,mx]
      - 'center': T_cam_center, bbox corners in centered coords [-half,+half]
    This makes the toggle visibly different (useful for debugging frame mismatch).
    """
    info = _extract_bbox_info(bbox_source)
    mn = info["mn"]
    mx = info["mx"]
    center = info["center"]
    extent = info["extent"]
    rot = info["rot"]

    if bbox_frame not in ("mesh", "center"):
        raise ValueError(f"bbox_frame must be 'mesh' or 'center', got {bbox_frame}")

    if bbox_frame == "mesh":
        if assume_pose_is_bbox_center:
            if extent is None and (mn is not None) and (mx is not None):
                extent = mx - mn
            if extent is None:
                return pose_in_cam, None, None
            half = 0.5 * extent
            return pose_in_cam, np.vstack([-half, +half]), None

        if (rot is not None) and (center is not None) and (extent is not None):
            corners = _box_corners_from_center_extent(
                center=center, extent=extent, rotation=rot
            )
            return pose_in_cam, None, corners

        if mn is None or mx is None:
            # fallback to centered if bounds unavailable
            if extent is None:
                return pose_in_cam, None, None
            half = 0.5 * extent
            return pose_in_cam, np.vstack([-half, +half]), None
        return pose_in_cam, np.vstack([mn, mx]), None

    # bbox_frame == "center"
    if extent is None and (mn is not None) and (mx is not None):
        extent = mx - mn
    if extent is None:
        return pose_in_cam, None, None
    half = 0.5 * extent
    return pose_in_cam, np.vstack([-half, +half]), None


def visualize_and_save_tracking_results(
    frame,
    objects,
    track_table,
    frame_id=None,
    visualize_points=True,
    points_vis_method="visible_uncertainty",
    save_images=False,
    output_image_dir=None,
    camera_intrinsics=np.eye(3),
    bbox_min_max=None,
    mesh_vertices_by_object=None,
    mesh_faces_by_object=None,
    project_mesh_contour=False,
    mesh_contour_line_color=(0, 255, 255),
    mesh_contour_linewidth=2,
    vis_mask=False,
    bbox_frame="mesh",
):
    """
    Visualize tracking results on the frame.
    points_vis_method options:
      - "uncertainty"
      - "visible"
      - "visible_valid"
      - "visible_uncertainty"
      - "registration_used_valid" (points used by registration this frame;
        green=confirmed/valid map points, red=tentative/invalid map points)
      - "registration_correspondence" (draw source->target reprojection lines;
        green=inlier, red=outlier)
    """
    display_frame = frame.rgb.copy()
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

    height, width = display_frame.shape[:2]

    # Draw segmentation masks if available
    # if vis_mask and hasattr(frame, "mask") and frame.mask is not None:
    #     object_masks = _normalize_object_masks(frame.mask)
    #     mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    #     mask_overlay[..., 1] = 255  # Green base

    #     for i, obj_mask in enumerate(object_masks):
    #         if np.any(obj_mask):
    #             # Color each object differently
    #             hue = (i + 3) / (len(object_masks) + 3) * 255
    #             mask_overlay[obj_mask, 0] = hue
    #             mask_overlay[obj_mask, 2] = 255

    #     mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
    #     display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)

    if visualize_points:
        if points_vis_method == "uncertainty":

            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                uncertainty_color = get_n_uncertainty_colors(
                    track_table.uncertainty[track_table.obj2track_map[i]]
                )
                draw_points_on_image(
                    display_frame,
                    track_table.track_2d[track_table.obj2track_map[i]],
                    uncertainty_color,
                )
        elif points_vis_method == "visible":
            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                # Generate N by 3 array with (0,255,0) for each row
                track_2d_points = track_table.track_2d[track_table.obj2track_map[i]]
                N = len(track_2d_points)
                visible_color = np.full((N, 3), (0, 0, 255), dtype=np.uint8)
                visible_color[track_table.visible[track_table.obj2track_map[i]]] = (
                    0,
                    255,
                    0,
                )
                draw_points_on_image(
                    display_frame,
                    track_2d_points,
                    visible_color,
                )
        elif points_vis_method == "visible_valid":
            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                # Generate N by 3 array with (0,0,255) (Red) for each row
                track_idx = track_table.obj2track_map[i]
                track_2d_points = track_table.track_2d[track_idx]
                N = len(track_2d_points)
                colors = np.full((N, 3), (0, 0, 255), dtype=np.uint8)

                # Set Green (0, 255, 0) for points that are both visible and valid
                visible_mask = track_table.visible[track_idx]
                valid_mask = track_table.valid[track_idx]
                colors[visible_mask & valid_mask] = (0, 255, 0)

                draw_points_on_image(
                    display_frame,
                    track_2d_points,
                    colors,
                )
        elif points_vis_method == "visible_uncertainty":
            # Plot only visible points, colored by their uncertainty colors
            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                track_idx = track_table.obj2track_map[i]
                track_2d_points = track_table.track_2d[track_idx]
                visible_mask = track_table.visible[track_idx]

                if np.any(visible_mask):
                    uncertainty_color = get_n_uncertainty_colors(
                        track_table.uncertainty[track_idx]
                    )

                    draw_points_on_image(
                        display_frame,
                        track_2d_points[visible_mask],
                        uncertainty_color[visible_mask],
                    )
        elif points_vis_method in ("registration_used_valid", "reg_used_valid"):
            # Plot only points used by registration for each object in this frame.
            # Color by object-map validity:
            #   green -> confirmed (obj.valid=True)
            #   red   -> tentative / invalid (obj.valid=False or unmapped)
            n_tracks = int(len(track_table.track_2d))
            for obj in objects:
                _idx = getattr(obj, "curr_frame_indices", None)
                if _idx is None:
                    continue
                reg_idx = np.asarray(_idx, dtype=np.int64).reshape(-1)
                if reg_idx.size == 0:
                    continue
                reg_idx = reg_idx[(reg_idx >= 0) & (reg_idx < n_tracks)]
                if reg_idx.size == 0:
                    continue

                pts = track_table.track_2d[reg_idx]
                colors = np.full((reg_idx.shape[0], 3), (255, 0, 0), dtype=np.uint8)

                # Prefer object validity for coloring.
                obj_valid = np.asarray(getattr(obj, "valid", np.array([])), dtype=bool)
                obj_rows = _map_track_ids_to_obj_rows(obj, reg_idx)
                row_ok = (obj_rows >= 0) & (obj_rows < obj_valid.shape[0])
                valid_mask = np.zeros(reg_idx.shape[0], dtype=bool)
                if np.any(row_ok):
                    valid_mask[row_ok] = obj_valid[obj_rows[row_ok]]
                else:
                    # Fallback: use track table depth-validity if object mapping is unavailable.
                    valid_mask = np.asarray(track_table.valid[reg_idx], dtype=bool)

                colors[valid_mask] = (0, 255, 0)
                draw_points_on_image(display_frame, pts, colors)
        elif points_vis_method in (
            "registration_correspondence",
            "reg_correspondence",
        ):
            # Draw source->target correspondence vectors for points used by registration.
            # Source: projected map keypoint in current camera frame.
            # Target: tracked 2D point in current frame.
            # Color: green=inlier, red=outlier, yellow=unknown.
            n_tracks = int(len(track_table.track_2d))
            K = np.asarray(camera_intrinsics, dtype=float).reshape(3, 3)
            for obj in objects:
                _idx = getattr(obj, "curr_frame_indices", None)
                if _idx is None:
                    continue
                reg_idx = np.asarray(_idx, dtype=np.int64).reshape(-1)
                if reg_idx.size == 0:
                    continue

                reg_idx = reg_idx[(reg_idx >= 0) & (reg_idx < n_tracks)]
                if reg_idx.size == 0:
                    continue

                rows = _map_track_ids_to_obj_rows(obj, reg_idx)
                rows_ok = (rows >= 0) & (rows < len(getattr(obj, "key_points", [])))
                if not np.any(rows_ok):
                    continue

                reg_idx_ok = reg_idx[rows_ok]
                rows_ok_idx = rows[rows_ok]
                obs_uv = np.asarray(track_table.track_2d[reg_idx_ok], dtype=float)

                src_obj = np.asarray(obj.key_points[rows_ok_idx], dtype=float)
                src_h = np.hstack(
                    [src_obj, np.ones((src_obj.shape[0], 1), dtype=float)]
                )
                src_cam = (np.asarray(obj.pose, dtype=float) @ src_h.T).T[:, :3]
                z = src_cam[:, 2]
                z_ok = np.isfinite(z) & (z > 1e-6)
                if not np.any(z_ok):
                    continue

                src_cam = src_cam[z_ok]
                obs_uv = obs_uv[z_ok]

                proj = (K @ src_cam.T).T
                pred_uv = proj[:, :2] / proj[:, 2:3]
                finite = np.isfinite(pred_uv).all(axis=1) & np.isfinite(obs_uv).all(
                    axis=1
                )
                if not np.any(finite):
                    continue

                pred_uv = pred_uv[finite]
                obs_uv = obs_uv[finite]

                inliers_full = np.asarray(
                    getattr(obj, "inliers", np.array([], dtype=bool)),
                    dtype=bool,
                ).reshape(-1)
                # Start unknown (yellow). Then override with inlier/outlier when available.
                colors = np.full((pred_uv.shape[0], 3), (0, 255, 255), dtype=np.uint8)
                if inliers_full.size == reg_idx.size:
                    inl_rows_ok = inliers_full[rows_ok]
                    inl = inl_rows_ok[z_ok][finite]
                    colors[inl] = (0, 255, 0)
                    colors[~inl] = (0, 0, 255)

                # Draw vectors and endpoints.
                for uv_pred, uv_obs, col in zip(pred_uv, obs_uv, colors):
                    p0 = tuple(np.round(uv_pred).astype(int).tolist())
                    p1 = tuple(np.round(uv_obs).astype(int).tolist())
                    cv2.line(
                        display_frame,
                        p0,
                        p1,
                        color=tuple(int(x) for x in col),
                        thickness=1,
                    )
                    cv2.circle(
                        display_frame, p0, radius=2, color=(255, 255, 255), thickness=-1
                    )
                    cv2.circle(
                        display_frame,
                        p1,
                        radius=3,
                        color=tuple(int(x) for x in col),
                        thickness=-1,
                    )
        # elif points_vis_method == "frame_id":
        #     # Color each point based on the frame id it was first seen (object.key_point_frames)
        #     for i, obj in enumerate(objects):
        #         if i not in track_table.obj2track_map:
        #             continue

        #         track_idx = track_table.obj2track_map[i]
        #         track_2d_points = track_table.track_2d[track_idx]
        #         visible_mask = track_table.visible[track_idx]

        #         # Only proceed if there are visible points
        #         if not np.any(visible_mask):
        #             continue
        #         if obj.key_point_frames.shape[0] == 0:
        #             continue
        #         # Align per-object track order with object's key point order
        #         # Assume key_point_frames order corresponds to obj2track_map order
        #         num_tracks_for_obj = len(track_idx)
        #         kp_frames_for_obj = obj.key_point_frames[:num_tracks_for_obj].astype(
        #             np.int32
        #         )

        #         # Frame ids for visible points; replace unknown -1 with current frame id if available
        #         frame_ids = kp_frames_for_obj[visible_mask]
        #         if frame_id is not None:
        #             frame_ids = frame_ids.copy()
        #             frame_ids[frame_ids == -1] = int(frame_id)

        #         # Use cached, distinctive colors per frame id
        #         colors_bgr = self._colors_for_frame_ids(frame_ids)

        #         # Draw only visible points for this object, using aligned colors
        #         draw_points_on_image(
        #             display_frame,
        #             track_2d_points[visible_mask],
        #             colors_bgr,
        #         )

    # Draw pose information
    for i, obj in enumerate(objects):
        if obj.pose is not None:
            pose_mesh_in_cam = obj.pose @ obj.init_pose

            if project_mesh_contour:
                verts_obj, faces = _resolve_mesh_geometry_for_object(
                    mesh_vertices_by_object=mesh_vertices_by_object,
                    mesh_faces_by_object=mesh_faces_by_object,
                    obj_idx=i,
                )
                if verts_obj is not None and faces is not None:
                    display_frame = draw_projected_mesh_contour(
                        K=camera_intrinsics,
                        img=display_frame,
                        ob_in_cam=pose_mesh_in_cam,
                        vertices_obj=verts_obj,
                        faces=faces,
                        line_color=mesh_contour_line_color,
                        linewidth=mesh_contour_linewidth,
                    )

            # Resolve per-object bbox override (if provided).
            bbox_src = None
            if bbox_min_max is not None:
                if isinstance(bbox_min_max, dict):
                    bbox_src = bbox_min_max.get(i, None)
                else:
                    bbox_arr = np.asarray(bbox_min_max)
                    if bbox_arr.ndim == 3 and i < len(bbox_min_max):
                        bbox_src = bbox_min_max[i]
                    elif bbox_arr.ndim == 2:
                        bbox_src = bbox_min_max

            # Otherwise, fall back to obj.bbox (Open3D-like AABB/OBB) if available.
            bbox_from_object = False
            if bbox_src is None:
                bbox_src = getattr(obj, "bbox", None)
                bbox_from_object = True

            pose_use, bbox_min_max_use, bbox_corners_use = _resolve_pose_and_bbox(
                pose_in_cam=pose_mesh_in_cam,
                bbox_source=bbox_src,
                bbox_frame=bbox_frame,
                assume_pose_is_bbox_center=bbox_from_object,
            )

            if bbox_corners_use is not None:
                display_frame = draw_oriented_3d_box(
                    camera_intrinsics,
                    display_frame,
                    pose_use,
                    bbox_corners_use,
                )
            elif bbox_min_max_use is not None:
                display_frame = draw_posed_3d_box(
                    camera_intrinsics,
                    display_frame,
                    pose_use,
                    bbox_min_max_use,
                )

            # Draw axis (in the same frame as the bbox choice).
            display_frame = draw_xyz_axis(
                image=display_frame, ob_in_cam=pose_use, K=camera_intrinsics
            )

    # Save image if flag is enabled and frame_id is provided
    if save_images and frame_id is not None:
        image_filename = os.path.join(output_image_dir, f"frame_{frame_id:06d}.png")
        cv2.imwrite(str(image_filename), display_frame)

    return display_frame


def visualize_and_save_tracking_results_with_gt(
    frame,
    objects,
    track_table,
    est_result_frame=None,
    gt_pose=None,
    gt_poses=None,
    gt_object_indices=None,
    frame_id=None,
    visualize_points=True,
    points_vis_method="visible_uncertainty",
    save_images=False,
    output_image_dir=None,
    camera_intrinsics=np.eye(3),
    bbox_min_max=None,
    mesh_vertices_by_object=None,
    mesh_faces_by_object=None,
    project_mesh_contour=False,
    mesh_contour_line_color=(0, 255, 255),
    mesh_contour_linewidth=2,
    gt_bbox_min_max=None,
    gt_bbox_min_max_by_object=None,
    bbox_frame="mesh",
    pred_pose_color=(0, 255, 0),  # Green for predicted pose
    gt_pose_color=(0, 0, 255),  # Red for GT pose
):
    """
    Visualize tracking results on the frame with ground truth pose and bounding box overlay.

    This function extends visualize_and_save_tracking_results by also drawing the GT pose
    and bounding box in a different color to allow comparison with the predicted pose.

    Args:
        frame: Frame object containing RGB image
        objects: List of tracked objects
        track_table: Track table containing point tracks
        gt_pose: Ground truth pose (4x4 transformation matrix) in camera frame. If None, GT won't be drawn.
        frame_id: Frame identifier for saving images
        visualize_points: Whether to visualize tracked points
        points_vis_method: Method for visualizing points
            ("uncertainty", "visible", "visible_valid", "visible_uncertainty",
             "registration_used_valid")
        save_images: Whether to save visualization images
        output_image_dir: Directory to save images
        camera_intrinsics: Camera intrinsic matrix (3x3)
        bbox_min_max: Bounding box min/max for predicted pose (2x3 array)
        gt_bbox_min_max: Bounding box min/max for GT pose (2x3 array). If None, uses bbox_min_max.
        pred_pose_color: BGR color for predicted pose visualization (default: green)
        gt_pose_color: BGR color for GT pose visualization (default: red)

    Returns:
        display_frame: Image with visualizations overlaid
    """
    # First, call the original function to get the base visualization
    if est_result_frame is not None:
        display_frame = est_result_frame
    else:
        display_frame = visualize_and_save_tracking_results(
            frame=frame,
            objects=objects,
            track_table=track_table,
            frame_id=None,  # Don't save yet, we'll save after adding GT
            visualize_points=visualize_points,
            points_vis_method=points_vis_method,
            save_images=False,  # Don't save yet
            output_image_dir=None,
            camera_intrinsics=camera_intrinsics,
            bbox_min_max=bbox_min_max,
            mesh_vertices_by_object=mesh_vertices_by_object,
            mesh_faces_by_object=mesh_faces_by_object,
            project_mesh_contour=project_mesh_contour,
            mesh_contour_line_color=mesh_contour_line_color,
            mesh_contour_linewidth=mesh_contour_linewidth,
            bbox_frame=bbox_frame,
        )

    # Build a unified list of GT overlays: (obj_idx, pose)
    gt_items = []
    if gt_poses is not None:
        if isinstance(gt_poses, dict):
            for obj_idx, pose in gt_poses.items():
                if pose is not None:
                    gt_items.append((int(obj_idx), pose))
        else:
            if gt_object_indices is None:
                gt_object_indices = range(len(gt_poses))
            for obj_idx, pose in zip(gt_object_indices, gt_poses):
                if pose is not None:
                    gt_items.append((int(obj_idx), pose))
    elif gt_pose is not None:
        gt_items.append((0, gt_pose))

    # Overlay GT pose and bounding box for each available object
    for obj_idx, obj_gt_pose in gt_items:
        if project_mesh_contour:
            verts_obj, faces = _resolve_mesh_geometry_for_object(
                mesh_vertices_by_object=mesh_vertices_by_object,
                mesh_faces_by_object=mesh_faces_by_object,
                obj_idx=obj_idx,
            )
            if verts_obj is not None and faces is not None:
                display_frame = draw_projected_mesh_contour(
                    K=camera_intrinsics,
                    img=display_frame,
                    ob_in_cam=obj_gt_pose,
                    vertices_obj=verts_obj,
                    faces=faces,
                    line_color=gt_pose_color,
                    linewidth=mesh_contour_linewidth,
                )

        # Select bbox for current GT object.
        gt_bbox_min_max_local = None

        if gt_bbox_min_max_by_object is not None:
            if isinstance(gt_bbox_min_max_by_object, dict):
                gt_bbox_min_max_local = gt_bbox_min_max_by_object.get(obj_idx, None)
            else:
                if 0 <= obj_idx < len(gt_bbox_min_max_by_object):
                    gt_bbox_min_max_local = gt_bbox_min_max_by_object[obj_idx]

        if gt_bbox_min_max_local is None and gt_bbox_min_max is not None:
            gt_bbox_min_max_local = gt_bbox_min_max

        if gt_bbox_min_max_local is None and bbox_min_max is not None:
            if (
                isinstance(bbox_min_max, (list, tuple, np.ndarray))
                and np.asarray(bbox_min_max).ndim == 3
            ):
                if 0 <= obj_idx < len(bbox_min_max):
                    gt_bbox_min_max_local = bbox_min_max[obj_idx]
            else:
                gt_bbox_min_max_local = bbox_min_max

        if gt_bbox_min_max_local is None and 0 <= obj_idx < len(objects):
            gt_bbox_min_max_local = getattr(objects[obj_idx], "bbox", None)

        pose_use, bbox_min_max_use, bbox_corners_use = _resolve_pose_and_bbox(
            pose_in_cam=obj_gt_pose,
            bbox_source=gt_bbox_min_max_local,
            bbox_frame=bbox_frame,
        )

        if bbox_corners_use is not None:
            # Draw GT oriented bounding box.
            display_frame = draw_oriented_3d_box(
                camera_intrinsics,
                display_frame,
                pose_use,
                bbox_corners_use,
                line_color=gt_pose_color,
                linewidth=2,
            )
        elif bbox_min_max_use is not None:
            # Draw GT bounding box
            display_frame = draw_posed_3d_box(
                camera_intrinsics,
                display_frame,
                pose_use,
                bbox_min_max_use,
                line_color=gt_pose_color,
                linewidth=2,
            )
        # Draw GT coordinate axes (in the same frame as the bbox choice).
        display_frame = draw_xyz_axis(
            image=display_frame,
            ob_in_cam=pose_use,
            K=camera_intrinsics,
            thickness=3,
        )

    # # Re-draw predicted pose with specified color to ensure it's visible
    # # (in case GT was drawn on top)
    # for obj in objects:
    #     if obj.pose is not None:
    #         pose = obj.pose @ obj.init_pose
    #         if bbox_min_max is not None:
    #             bbox_min_max_local = bbox_min_max
    #         else:
    #             half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
    #             bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

    #         # Re-draw predicted bounding box with specified color
    #         display_frame = draw_posed_3d_box(
    #             camera_intrinsics,
    #             display_frame,
    #             pose,
    #             bbox_min_max_local,
    #             line_color=pred_pose_color,
    #             linewidth=2,
    #         )
    #         # Re-draw predicted coordinate axes
    #         display_frame = draw_xyz_axis(
    #             image=display_frame,
    #             ob_in_cam=pose,
    #             K=camera_intrinsics,
    #             thickness=3,
    #         )

    # Save image if flag is enabled and frame_id is provided
    if save_images and frame_id is not None:
        image_filename = os.path.join(output_image_dir, f"frame_{frame_id:06d}.png")
        cv2.imwrite(str(image_filename), display_frame)

    return display_frame


# def colors_for_frame_ids(frame_ids: np.ndarray) -> np.ndarray:
#     """Return consistent BGR colors for the provided frame ids."""
#     if frame_ids.size == 0:
#         return np.empty((0, 3), dtype=np.uint8)

#     colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
#     unique_ids = np.unique(frame_ids.astype(np.int64))

#     for fid in unique_ids:
#         fid_int = int(fid)
#         if fid_int not in self._frame_color_lookup:
#             self._frame_color_lookup[fid_int] = generate_next_frame_color()

#         colors[frame_ids == fid] = self._frame_color_lookup[fid_int]

#     return colors


# def generate_next_frame_color():
#     """Generate a new distinctive HSV-based color and convert it to BGR."""
#     golden_ratio_conjugate = 0.6180339887498949
#     base_index = len(self._frame_color_lookup)
#     saturation_cycle = (255, 230, 200, 180)
#     value_cycle = (255, 235, 215)

#     attempt = 0
#     while True:
#         idx = base_index + attempt
#         hue = int(round(((idx * golden_ratio_conjugate) % 1.0) * 179)) % 180
#         saturation = saturation_cycle[idx % len(saturation_cycle)]
#         value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]

#         hsv_tuple = (hue, saturation, value)
#         if hsv_tuple not in self._frame_color_used_hsv:
#             self._frame_color_used_hsv.add(hsv_tuple)
#             hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
#             return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

#         attempt += 1
