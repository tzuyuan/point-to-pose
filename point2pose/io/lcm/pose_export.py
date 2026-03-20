from __future__ import annotations

import numpy as np

from point2pose.utils.transform import inverse_SE3, transform_pts


def _extract_bbox_info(bbox_like):
    if bbox_like is None:
        return {"mn": None, "mx": None, "center": None, "extent": None, "rot": None}

    if isinstance(bbox_like, dict):
        mn = mx = center = extent = rot = None

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

    arr = np.asarray(bbox_like)
    if arr.ndim == 2 and arr.shape == (2, 3):
        mn = arr.min(axis=0).astype(float)
        mx = arr.max(axis=0).astype(float)
        center = 0.5 * (mn + mx)
        extent = mx - mn
        return {"mn": mn, "mx": mx, "center": center, "extent": extent, "rot": None}

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

    if extent is None and (mn is not None) and (mx is not None):
        extent = mx - mn
    if center is None and (mn is not None) and (mx is not None):
        center = 0.5 * (mn + mx)
    if (mn is None or mx is None) and (center is not None) and (extent is not None):
        half = 0.5 * extent
        mn = center - half
        mx = center + half

    return {"mn": mn, "mx": mx, "center": center, "extent": extent, "rot": rot}


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def _collect_object_frame_dense_points(
    obj, max_keyframes: int = 8, max_points_per_keyframe: int = 1200
) -> np.ndarray:
    keyframes = list(getattr(obj, "keyframes", []) or [])
    if not keyframes:
        return np.empty((0, 3), dtype=float)

    pts_obj_all = []
    for keyframe in keyframes[-max_keyframes:]:
        dense_pts = np.asarray(getattr(keyframe, "dense_pts", np.empty((0, 3))), dtype=float)
        if dense_pts.ndim != 2 or dense_pts.shape[1] != 3 or dense_pts.shape[0] == 0:
            continue
        finite_mask = np.all(np.isfinite(dense_pts), axis=1)
        dense_pts = dense_pts[finite_mask]
        if dense_pts.shape[0] == 0:
            continue
        dense_pts = _subsample_points(dense_pts, max_points_per_keyframe)
        pts_obj = transform_pts(inverse_SE3(np.asarray(keyframe.pose, dtype=float)), dense_pts)
        pts_obj_all.append(pts_obj)

    if not pts_obj_all:
        return np.empty((0, 3), dtype=float)

    pts_obj = np.concatenate(pts_obj_all, axis=0)
    finite_mask = np.all(np.isfinite(pts_obj), axis=1)
    pts_obj = pts_obj[finite_mask]
    if pts_obj.shape[0] == 0:
        return pts_obj

    center = np.median(pts_obj, axis=0)
    dist = np.linalg.norm(pts_obj - center[None, :], axis=1)
    if dist.shape[0] >= 50:
        keep = dist <= np.percentile(dist, 97.5)
        if np.count_nonzero(keep) >= 20:
            pts_obj = pts_obj[keep]
    return pts_obj


def _bbox_from_dense_keyframes(obj):
    pts_obj = _collect_object_frame_dense_points(obj)
    if pts_obj.shape[0] < 20:
        return None

    pts_obj = _subsample_points(pts_obj, 6000)
    try:
        import open3d as o3d

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts_obj.astype(np.float64))

        obb = None
        if hasattr(point_cloud, "get_minimal_oriented_bounding_box"):
            try:
                obb = point_cloud.get_minimal_oriented_bounding_box(robust=True)
            except TypeError:
                obb = point_cloud.get_minimal_oriented_bounding_box()
            except RuntimeError:
                obb = None

        if obb is None:
            try:
                obb = point_cloud.get_oriented_bounding_box(robust=True)
            except TypeError:
                obb = point_cloud.get_oriented_bounding_box()

        center = np.asarray(obb.center, dtype=float).reshape(3)
        extent = np.asarray(obb.extent, dtype=float).reshape(3)
        rot = np.asarray(obb.R, dtype=float).reshape(3, 3)
        if np.all(np.isfinite(center)) and np.all(np.isfinite(extent)) and np.all(
            extent > 1e-4
        ):
            return {"center": center, "extent": extent, "rot": rot, "frame": "object"}
    except Exception:
        pass

    mn = pts_obj.min(axis=0)
    mx = pts_obj.max(axis=0)
    extent = np.maximum(mx - mn, 1e-3)
    pad = np.maximum(0.005, 0.05 * extent)
    return {"bbox": np.vstack([mn - pad, mx + pad]), "frame": "object"}


def _reject_bbox_outliers_from_key_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 6:
        return pts

    bbox_points = pts
    center = np.median(bbox_points, axis=0)
    abs_dev = np.abs(bbox_points - center[None, :])
    mad = np.median(abs_dev, axis=0)
    spread = np.ptp(bbox_points, axis=0)
    axis_scale = np.maximum(1.4826 * mad, 0.10 * spread + 0.003)
    axis_keep = np.all(abs_dev <= (3.5 * axis_scale)[None, :], axis=1)
    if np.count_nonzero(axis_keep) >= 4:
        bbox_points = bbox_points[axis_keep]

    if bbox_points.shape[0] >= 6:
        center = np.median(bbox_points, axis=0)
        dist = np.linalg.norm(bbox_points - center[None, :], axis=1)
        dist_med = np.median(dist)
        dist_mad = np.median(np.abs(dist - dist_med))
        dist_scale = max(1.4826 * dist_mad, 0.02)
        radial_keep = dist <= (dist_med + 3.5 * dist_scale)
        if np.count_nonzero(radial_keep) >= 4:
            bbox_points = bbox_points[radial_keep]

    if bbox_points.shape[0] >= 12:
        lo = np.percentile(bbox_points, 2.5, axis=0)
        hi = np.percentile(bbox_points, 97.5, axis=0)
        soft_keep = np.all((bbox_points >= lo) & (bbox_points <= hi), axis=1)
        if np.count_nonzero(soft_keep) >= 4:
            bbox_points = bbox_points[soft_keep]

    return bbox_points


def _bbox_from_key_points(obj):
    key_points = np.asarray(getattr(obj, "key_points", np.empty((0, 3))), dtype=float)
    if key_points.ndim != 2 or key_points.shape[1] != 3:
        return None

    finite_mask = np.all(np.isfinite(key_points), axis=1)
    key_points = key_points[finite_mask]
    if key_points.shape[0] < 3:
        return None

    bbox_points = _reject_bbox_outliers_from_key_points(key_points)
    if bbox_points.shape[0] < 3:
        bbox_points = key_points

    mn = bbox_points.min(axis=0)
    mx = bbox_points.max(axis=0)
    extent = np.maximum(mx - mn, 1e-3)
    pad = np.maximum(0.01, 0.1 * extent)
    return {"bbox": np.vstack([mn - pad, mx + pad]), "frame": "object"}


def _bbox_from_sdf_volume(obj):
    sdf_meta = getattr(obj, "sdf", None)
    if isinstance(sdf_meta, dict):
        vol_bnds = sdf_meta.get("vol_bnds", None)
        if vol_bnds is not None:
            vol_bnds = np.asarray(vol_bnds, dtype=float)
            if vol_bnds.shape == (3, 2):
                return {"bbox": vol_bnds.T.copy(), "frame": "object"}

    sdf_volume = getattr(obj, "sdf_volume", None)
    vol_bnds = getattr(sdf_volume, "_vol_bnds", None)
    if vol_bnds is None:
        return None

    vol_bnds = np.asarray(vol_bnds, dtype=float)
    if vol_bnds.shape != (3, 2):
        return None
    return {"bbox": vol_bnds.T.copy(), "frame": "object"}


def resolve_visualization_box(obj):
    pose_rel = np.asarray(getattr(obj, "pose", np.eye(4)), dtype=float)
    init_pose = np.asarray(getattr(obj, "init_pose", np.eye(4)), dtype=float)

    bbox_source = getattr(obj, "bbox", None)
    if bbox_source is None:
        bbox_source = getattr(obj, "init_bbox", None)
    if bbox_source is not None:
        if isinstance(bbox_source, dict):
            # Dict-based bbox metadata in this repo is typically expressed in the
            # object/mesh frame unless a centered frame is explicitly requested.
            bbox_frame = str(bbox_source.get("frame", "object")).lower()
            if bbox_frame in {"object", "object_local", "mesh"}:
                return pose_rel, bbox_source, False
        return pose_rel @ init_pose, bbox_source, True

    dense_bbox = _bbox_from_dense_keyframes(obj)
    if dense_bbox is not None:
        return pose_rel, dense_bbox, False

    keypoint_bbox = _bbox_from_key_points(obj)
    if keypoint_bbox is not None:
        return pose_rel, keypoint_bbox, False

    sdf_bbox = _bbox_from_sdf_volume(obj)
    if sdf_bbox is not None:
        return pose_rel, sdf_bbox, False

    return pose_rel @ init_pose, None, True


def resolve_bbox_center_pose(
    pose_in_cam: np.ndarray,
    bbox_source,
    assume_pose_is_bbox_center: bool,
) -> np.ndarray:
    pose_in_cam = np.asarray(pose_in_cam, dtype=float)
    if bbox_source is None or assume_pose_is_bbox_center:
        return pose_in_cam

    info = _extract_bbox_info(bbox_source)
    center = info["center"]
    rot = info["rot"]
    mn = info["mn"]
    mx = info["mx"]

    if center is None and (mn is not None) and (mx is not None):
        center = 0.5 * (np.asarray(mn, dtype=float) + np.asarray(mx, dtype=float))

    if center is None and rot is None:
        return pose_in_cam

    local_transform = np.eye(4, dtype=float)
    if rot is not None:
        local_transform[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
    if center is not None:
        local_transform[:3, 3] = np.asarray(center, dtype=float).reshape(3)
    return pose_in_cam @ local_transform


def _sorted_rotation_and_extent(rotation_matrix: np.ndarray, extent: np.ndarray):
    extent = np.asarray(extent, dtype=float).reshape(3)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    order = np.argsort(extent)[::-1]
    extent_sorted = extent[order]
    rotation_sorted = rotation_matrix[:, order]
    if np.linalg.det(rotation_sorted) < 0.0:
        rotation_sorted[:, 2] *= -1.0
    return rotation_sorted, extent_sorted


def _rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    trace = np.trace(rotation_matrix)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def object_name_from_index(obj_idx: int) -> str:
    return f"obj_{int(obj_idx)}"


def _transform_to_pose_vector(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    quat_xyzw = _rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    qx, qy, qz, qw = quat_xyzw
    position = transform[:3, 3].astype(np.float32)
    return np.asarray(
        [position[0], position[1], position[2], qw, qx, qy, qz],
        dtype=np.float32,
    )


def build_mesh_pose_vector(
    obj,
    camera_to_world: np.ndarray | None = None,
) -> np.ndarray:
    pose_rel = np.asarray(getattr(obj, "pose", np.eye(4)), dtype=float)
    init_pose = np.asarray(getattr(obj, "init_pose", np.eye(4)), dtype=float)
    pose_in_cam = pose_rel @ init_pose
    pose_world = pose_in_cam
    if camera_to_world is not None:
        pose_world = np.asarray(camera_to_world, dtype=float).reshape(4, 4) @ pose_in_cam
    return _transform_to_pose_vector(pose_world)


def build_bbox_pose_vector(
    obj,
    camera_to_world: np.ndarray | None = None,
    min_extent: float = 1e-3,
) -> np.ndarray:
    pose_in_cam, bbox_source, assume_pose_is_bbox_center = resolve_visualization_box(obj)
    bbox_center_in_cam = resolve_bbox_center_pose(
        pose_in_cam=pose_in_cam,
        bbox_source=bbox_source,
        assume_pose_is_bbox_center=assume_pose_is_bbox_center,
    )

    info = _extract_bbox_info(bbox_source)
    extent = info["extent"]
    if extent is None and (info["mn"] is not None) and (info["mx"] is not None):
        extent = np.asarray(info["mx"], dtype=float) - np.asarray(info["mn"], dtype=float)
    if extent is None:
        extent = np.full(3, float(min_extent), dtype=float)
    extent = np.maximum(np.asarray(extent, dtype=float).reshape(3), float(min_extent))

    rotation_cam, extent_sorted = _sorted_rotation_and_extent(
        bbox_center_in_cam[:3, :3], extent
    )
    bbox_center_sorted = np.asarray(bbox_center_in_cam, dtype=float).copy()
    bbox_center_sorted[:3, :3] = rotation_cam

    pose_world = bbox_center_sorted
    if camera_to_world is not None:
        pose_world = np.asarray(camera_to_world, dtype=float).reshape(4, 4) @ pose_world

    pose_vector = _transform_to_pose_vector(pose_world)
    return np.asarray(
        [
            pose_vector[0],
            pose_vector[1],
            pose_vector[2],
            pose_vector[3],
            pose_vector[4],
            pose_vector[5],
            pose_vector[6],
            extent_sorted[0],
            extent_sorted[1],
            extent_sorted[2],
        ],
        dtype=np.float32,
    )
