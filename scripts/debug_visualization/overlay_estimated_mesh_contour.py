#!/usr/bin/env python3
"""
Overlay projected object-mesh contour on dataset images using estimated poses
saved in a finished point-to-pose sequence, and export an additional video
with projected 3D bounding boxes.

Supported datasets:
  - ho3d
  - ycbineoat
  - ycbinisaac (auto-renders all objects in multi-object sequences)

Example:
  python3 scripts/debug_visualization/overlay_estimated_mesh_contour.py \
      --dataset ho3d \
      --run_dir /home/justin/code/point-to-pose/results/ho3d_single/AP11 \
      --data_root /home/justin/data/HO3D_V3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.sources.dataset.datareader import (  # noqa: E402
    Ho3dReader,
    YCBInIsaacReader,
    YcbineoatReader,
)


def _to_pose_matrix(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()

    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _extract_pose_from_entry(entry, object_idx: int) -> Optional[np.ndarray]:
    if entry is None:
        return None

    if isinstance(entry, dict):
        keys_to_try: Iterable[object] = (
            object_idx,
            str(object_idx),
            f"obj_{object_idx}",
        )
        for key in keys_to_try:
            if key in entry:
                return _extract_pose_from_entry(entry[key], 0)
        return None

    if isinstance(entry, (list, tuple)):
        if object_idx < len(entry):
            return _extract_pose_from_entry(entry[object_idx], 0)
        return None

    arr = np.asarray(entry)
    if arr.shape == (4, 4):
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        if object_idx < arr.shape[0]:
            return arr[object_idx].astype(np.float64)
        return None
    if arr.ndim == 3 and arr.shape[:2] == (4, 4):
        if object_idx < arr.shape[2]:
            return arr[:, :, object_idx].astype(np.float64)
        return None
    return None


def _extract_object_pose_series(
    npz_data, key: str, object_idx: int, frame_ids_raw: Optional[np.ndarray]
) -> List[Optional[np.ndarray]]:
    if key not in npz_data:
        raise KeyError(f"Pose key '{key}' not found in metadata.")

    arr = npz_data[key]

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 2 and arr.shape == (4, 4):
            if object_idx != 0:
                return [None]
            return [_to_pose_matrix(arr)]

        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            if object_idx != 0:
                return [None for _ in range(arr.shape[0])]
            return [_to_pose_matrix(arr[i]) for i in range(arr.shape[0])]

        if arr.ndim == 3 and arr.shape[:2] == (4, 4):
            if object_idx != 0:
                return [None for _ in range(arr.shape[2])]
            return [_to_pose_matrix(arr[:, :, i]) for i in range(arr.shape[2])]

        if arr.ndim == 4 and arr.shape[-2:] == (4, 4):
            n_frame_ids = (
                np.asarray(frame_ids_raw).reshape(-1).shape[0]
                if frame_ids_raw is not None
                else None
            )

            if n_frame_ids is not None and arr.shape[0] == n_frame_ids:
                if object_idx >= arr.shape[1]:
                    return [None for _ in range(arr.shape[0])]
                return [_to_pose_matrix(arr[i, object_idx]) for i in range(arr.shape[0])]

            if n_frame_ids is not None and arr.shape[1] == n_frame_ids:
                if object_idx >= arr.shape[0]:
                    return [None for _ in range(arr.shape[1])]
                return [_to_pose_matrix(arr[object_idx, i]) for i in range(arr.shape[1])]

            # Fallback: choose orientation with more frame-like axis.
            if arr.shape[0] >= arr.shape[1]:
                if object_idx >= arr.shape[1]:
                    return [None for _ in range(arr.shape[0])]
                return [_to_pose_matrix(arr[i, object_idx]) for i in range(arr.shape[0])]
            if object_idx >= arr.shape[0]:
                return [None for _ in range(arr.shape[1])]
            return [_to_pose_matrix(arr[object_idx, i]) for i in range(arr.shape[1])]

        raise ValueError(
            f"Unsupported pose array format for key '{key}': shape={arr.shape}, dtype={arr.dtype}"
        )

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr_obj = np.asarray(arr, dtype=object)
        if arr_obj.ndim == 0:
            return [_extract_pose_from_entry(arr_obj.item(), object_idx)]
        if arr_obj.ndim == 2:
            n_frame_ids = (
                np.asarray(frame_ids_raw).reshape(-1).shape[0]
                if frame_ids_raw is not None
                else None
            )
            if (
                n_frame_ids is not None
                and arr_obj.shape[0] == n_frame_ids
                and object_idx < arr_obj.shape[1]
            ):
                return [_to_pose_matrix(arr_obj[i, object_idx]) for i in range(arr_obj.shape[0])]
            if (
                n_frame_ids is not None
                and arr_obj.shape[1] == n_frame_ids
                and object_idx < arr_obj.shape[0]
            ):
                return [_to_pose_matrix(arr_obj[object_idx, i]) for i in range(arr_obj.shape[1])]

            if arr_obj.shape[0] >= arr_obj.shape[1]:
                if object_idx >= arr_obj.shape[1]:
                    return [None for _ in range(arr_obj.shape[0])]
                return [_to_pose_matrix(arr_obj[i, object_idx]) for i in range(arr_obj.shape[0])]
            if object_idx >= arr_obj.shape[0]:
                return [None for _ in range(arr_obj.shape[1])]
            return [_to_pose_matrix(arr_obj[object_idx, i]) for i in range(arr_obj.shape[1])]

        raw = arr_obj.reshape(-1)
        poses: List[Optional[np.ndarray]] = []
        for item in raw:
            poses.append(_extract_pose_from_entry(item, object_idx))
        return poses

    raise ValueError(
        f"Unsupported pose container format for key '{key}': "
        f"type={type(arr)}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}"
    )


def _extract_frame_ids(npz_data, n_rows: int) -> np.ndarray:
    if "frame_id" not in npz_data:
        return np.arange(n_rows, dtype=int)

    raw = np.asarray(npz_data["frame_id"]).reshape(-1)
    frame_ids = np.zeros((n_rows,), dtype=int)
    for i in range(n_rows):
        if i < raw.shape[0]:
            try:
                frame_ids[i] = int(np.asarray(raw[i]).reshape(-1)[0])
            except Exception:
                frame_ids[i] = i
        else:
            frame_ids[i] = i
    return frame_ids


def _select_pose_key(
    npz_data,
    dataset: str,
    pose_key: str,
    object_idx: int,
) -> Tuple[str, List[Optional[np.ndarray]]]:
    frame_ids_raw = npz_data["frame_id"] if "frame_id" in npz_data else None

    if pose_key != "auto":
        pose_series = _extract_object_pose_series(
            npz_data, pose_key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        return pose_key, pose_series

    if dataset == "ycbinisaac":
        candidates = ("obj_pose_all", "obj_pose", "pose_local", "pose_frontend")
    else:
        candidates = ("obj_pose", "obj_pose_all", "pose_local", "pose_frontend")

    last_nonempty_key = None
    last_nonempty_series = None
    for key in candidates:
        if key not in npz_data:
            continue
        series = _extract_object_pose_series(
            npz_data, key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        num_valid = sum(p is not None for p in series)
        if num_valid > 0:
            return key, series
        if len(series) > 0:
            last_nonempty_key = key
            last_nonempty_series = series

    if last_nonempty_key is not None and last_nonempty_series is not None:
        return last_nonempty_key, last_nonempty_series

    raise KeyError(
        "No supported pose key found. Tried: obj_pose_all, obj_pose, pose_local, pose_frontend"
    )


def _extract_init_pose_series(
    npz_data,
    n_rows: int,
    object_idx: int,
    prefer_multi_object_key: bool,
) -> Tuple[List[np.ndarray], Optional[str]]:
    key_candidates = ["obj_init_pose_all", "obj_init_pose"]
    if not prefer_multi_object_key:
        key_candidates = ["obj_init_pose", "obj_init_pose_all"]

    frame_ids_raw = npz_data["frame_id"] if "frame_id" in npz_data else None
    init_poses: Optional[List[Optional[np.ndarray]]] = None
    used_key = None
    for key in key_candidates:
        if key not in npz_data:
            continue
        series = _extract_object_pose_series(
            npz_data, key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        if len(series) == 0:
            continue
        init_poses = series
        used_key = key
        break

    if init_poses is None:
        return [np.eye(4, dtype=float) for _ in range(n_rows)], None

    if len(init_poses) == 1 and n_rows > 1 and init_poses[0] is not None:
        return [init_poses[0].copy() for _ in range(n_rows)], used_key

    out: List[np.ndarray] = []
    for i in range(n_rows):
        if i < len(init_poses) and init_poses[i] is not None:
            out.append(init_poses[i])
        else:
            out.append(np.eye(4, dtype=float))
    return out, used_key


def _as_trimesh(mesh_or_scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    elif isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            raise ValueError("Loaded scene has no geometry.")
        mesh = mesh_or_scene.dump(concatenate=True)
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Could not convert to trimesh.Trimesh, got {type(mesh)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces.")
    return mesh


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    mesh = _as_trimesh(loaded)
    mesh.remove_infinite_values()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh became empty after cleanup: {mesh_path}")
    return mesh


def _resolve_run_mesh_path(run_dir: str, object_idx: int) -> Optional[str]:
    mesh_dir = Path(run_dir) / "mesh"
    if not mesh_dir.is_dir():
        return None

    candidates = [
        mesh_dir / f"pred_mesh_obj_{object_idx}_textured.glb",
        mesh_dir / f"pred_mesh_obj_{object_idx}.ply",
        mesh_dir / f"pred_mesh_obj_{object_idx}.obj",
        mesh_dir / f"pred_mesh_obj_{object_idx}.stl",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    wildcard = sorted(mesh_dir.glob(f"pred_mesh_obj_{object_idx}*"))
    for path in wildcard:
        if path.is_file():
            return str(path)
    return None


def _resolve_ycb_mesh_path(model_root: str, object_name: str) -> str:
    candidates = [
        os.path.join(model_root, object_name, "textured_simple.obj"),
        os.path.join(model_root, "models", object_name, "textured_simple.obj"),
        os.path.join(model_root, object_name, "google_16k", "textured.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find mesh for '{object_name}' under model_root={model_root}. "
        f"Checked: {candidates}"
    )


def _load_reader(dataset: str, data_root: str, video_name: str):
    if dataset == "ho3d":
        video_dir = os.path.join(data_root, "evaluation", video_name)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"HO3D sequence directory not found: {video_dir}")
        return Ho3dReader(video_dir, data_root)

    video_dir = os.path.join(data_root, video_name)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Sequence directory not found: {video_dir}")

    if dataset == "ycbineoat":
        return YcbineoatReader(video_dir)
    if dataset == "ycbinisaac":
        return YCBInIsaacReader(video_dir)

    raise ValueError(f"Unsupported dataset: {dataset}")


def _resolve_dataset_mesh_path(
    dataset: str,
    reader,
    model_root: Optional[str],
    object_idx: int,
    object_name: Optional[str],
) -> str:
    if dataset == "ho3d":
        # Match Ho3dReader.get_gt_mesh() mapping but return an explicit path.
        video_to_name = {
            "AP": "019_pitcher_base",
            "MPM": "010_potted_meat_can",
            "SB": "021_bleach_cleanser",
            "SM": "006_mustard_bottle",
        }
        video_name = reader.get_video_name()
        object_name_ho3d = None
        for prefix, mapped_name in video_to_name.items():
            if video_name.startswith(prefix):
                object_name_ho3d = mapped_name
                break
        if object_name_ho3d is None:
            raise KeyError(
                f"Could not map HO3D sequence '{video_name}' to an object name. "
                "Pass --mesh_path explicitly."
            )
        mesh_path = os.path.join(
            reader.ho3d_root, "models", object_name_ho3d, "textured_simple.obj"
        )
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"HO3D mesh file not found: {mesh_path}")
        return mesh_path

    if model_root is None:
        raise ValueError(
            "For YCB datasets, pass --model_root (unless --mesh_path or run mesh exists)."
        )

    if dataset == "ycbineoat":
        video_name = reader.get_video_name()
        canonical = reader.videoname_to_object.get(video_name, None)
        if canonical is None:
            raise KeyError(
                f"Could not map video '{video_name}' to YCB object name. "
                "Pass --mesh_path explicitly."
            )
        return _resolve_ycb_mesh_path(model_root, canonical)

    # ycbinisaac
    all_names = reader.get_object_names()
    if object_name is None:
        if object_idx < 0 or object_idx >= len(all_names):
            raise IndexError(
                f"object_idx={object_idx} out of range for object names {all_names}"
            )
        object_name = all_names[object_idx]

    canonical = reader.videoname_to_object.get(object_name, object_name)
    return _resolve_ycb_mesh_path(model_root, canonical)


def _load_frame_bgr(reader, dataset: str, frame_idx: int) -> np.ndarray:
    if dataset == "ho3d":
        bgr = cv2.imread(reader.color_files[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read HO3D frame: {reader.color_files[frame_idx]}")
        return bgr

    rgb = reader.get_color(frame_idx)
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _rasterize_silhouette(
    vertices_obj: np.ndarray,
    faces: np.ndarray,
    T_obj_in_cam: np.ndarray,
    K: np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    mask = np.zeros((image_h, image_w), dtype=np.uint8)

    verts_h = np.hstack(
        [vertices_obj.astype(np.float64), np.ones((vertices_obj.shape[0], 1), dtype=np.float64)]
    )
    verts_cam = (T_obj_in_cam @ verts_h.T).T[:, :3]
    z = verts_cam[:, 2]

    proj = (K @ verts_cam.T).T
    uv = proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)

    face_z = z[faces]
    valid_faces = np.all(face_z > 1e-6, axis=1)
    if not np.any(valid_faces):
        return mask

    tris = uv[faces[valid_faces]]  # (Nf, 3, 2)
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


def _compute_bbox_corners_from_vertices(vertices_obj: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices_obj, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
        raise ValueError(
            "Expected non-empty vertices array with shape (N, 3) for 3D bbox corners."
        )

    xyz_min = verts.min(axis=0)
    xyz_max = verts.max(axis=0)
    x0, y0, z0 = xyz_min
    x1, y1, z1 = xyz_max
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


def _draw_projected_3d_bbox(
    image: np.ndarray,
    bbox_corners_obj: np.ndarray,
    T_obj_in_cam: np.ndarray,
    K: np.ndarray,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> bool:
    if bbox_corners_obj.shape != (8, 3):
        return False

    corners_h = np.hstack(
        [bbox_corners_obj.astype(np.float64), np.ones((8, 1), dtype=np.float64)]
    )
    corners_cam = (T_obj_in_cam @ corners_h.T).T[:, :3]
    z = corners_cam[:, 2]

    proj = (K @ corners_cam.T).T
    valid = z > 1e-6
    uv = np.full((8, 2), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv[valid] = proj[valid, :2] / proj[valid, 2:3]

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
    drew_any = False
    for idx0, idx1 in edges:
        if not valid[idx0] or not valid[idx1]:
            continue
        p0 = uv[idx0]
        p1 = uv[idx1]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue

        p0_i = tuple(np.round(np.clip(p0, -1e7, 1e7)).astype(np.int32))
        p1_i = tuple(np.round(np.clip(p1, -1e7, 1e7)).astype(np.int32))
        cv2.line(
            image,
            p0_i,
            p1_i,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        drew_any = True
    return drew_any


def _draw_projected_xyz_axis(
    image: np.ndarray,
    T_obj_in_cam: np.ndarray,
    K: np.ndarray,
    axis_scale: float,
    thickness: int,
) -> bool:
    if axis_scale <= 0:
        return False

    axis_points_obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_scale, 0.0, 0.0],  # +X
            [0.0, axis_scale, 0.0],  # +Y
            [0.0, 0.0, axis_scale],  # +Z
        ],
        dtype=np.float64,
    )
    points_h = np.hstack([axis_points_obj, np.ones((4, 1), dtype=np.float64)])
    points_cam = (T_obj_in_cam @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    proj = (K @ points_cam.T).T

    valid = z > 1e-6
    uv = np.full((4, 2), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv[valid] = proj[valid, :2] / proj[valid, 2:3]

    if not valid[0] or not np.isfinite(uv[0]).all():
        return False

    origin = tuple(np.round(np.clip(uv[0], -1e7, 1e7)).astype(np.int32))
    axis_colors_bgr = (
        (0, 0, 255),  # X axis: red
        (0, 255, 0),  # Y axis: green
        (255, 0, 0),  # Z axis: blue
    )
    drew_any = False
    for axis_idx, color_bgr in enumerate(axis_colors_bgr, start=1):
        if not valid[axis_idx] or not np.isfinite(uv[axis_idx]).all():
            continue
        end_pt = tuple(np.round(np.clip(uv[axis_idx], -1e7, 1e7)).astype(np.int32))
        cv2.line(
            image,
            origin,
            end_pt,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        drew_any = True
    return drew_any


def _try_parse_int(value) -> Optional[int]:
    try:
        return int(str(value))
    except Exception:
        return None


def _build_frame_id_to_reader_index(reader) -> Dict[int, int]:
    mapping: Dict[int, int] = {}

    id_strs = getattr(reader, "id_strs", None)
    if id_strs is not None:
        for idx, raw in enumerate(id_strs):
            fid = _try_parse_int(raw)
            if fid is not None:
                mapping.setdefault(fid, idx)

    color_files = getattr(reader, "color_files", None)
    if color_files is not None:
        for idx, path in enumerate(color_files):
            stem = os.path.splitext(os.path.basename(str(path)))[0]
            fid = _try_parse_int(stem)
            if fid is not None:
                mapping.setdefault(fid, idx)

    for idx in range(len(reader)):
        mapping.setdefault(int(idx), idx)

    return mapping


def _resolve_object_selection(dataset: str, reader, object_idx: int, object_name: Optional[str]):
    if dataset != "ycbinisaac":
        return object_idx, object_name

    available_names = reader.get_object_names()
    if len(available_names) == 0:
        raise RuntimeError("No objects found in ycbinisaac reader.")

    resolved_idx = object_idx
    resolved_name = None

    if object_name is not None:
        if object_name in available_names:
            name_idx = available_names.index(object_name)
        else:
            canonical_to_names = {}
            for name in available_names:
                canonical = reader.videoname_to_object.get(name, name)
                canonical_to_names.setdefault(canonical, []).append(name)
            matches = canonical_to_names.get(object_name, [])
            if len(matches) == 1:
                name_idx = available_names.index(matches[0])
            elif len(matches) > 1:
                raise ValueError(
                    f"Object name '{object_name}' maps to multiple sequence objects: {matches}. "
                    f"Pass --object_idx explicitly."
                )
            else:
                raise ValueError(
                    f"Object '{object_name}' not found in sequence objects {available_names}."
                )

        if object_idx != 0 and object_idx != name_idx:
            raise ValueError(
                f"--object_idx ({object_idx}) conflicts with --object_name '{object_name}' "
                f"(index {name_idx})."
            )
        resolved_idx = name_idx
        resolved_name = available_names[name_idx]
    else:
        if resolved_idx < 0 or resolved_idx >= len(available_names):
            raise IndexError(
                f"--object_idx={resolved_idx} out of range for objects {available_names}"
            )
        resolved_name = available_names[resolved_idx]

    return resolved_idx, resolved_name


def _compose_object_pose(pose_obj: np.ndarray, init_pose: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pose_obj
    if mode == "pre_multiply_init":
        return init_pose @ pose_obj
    # Default + backward-compatible behavior.
    return pose_obj @ init_pose


def _normalize_pose_series_length(
    pose_series: List[Optional[np.ndarray]], n_rows: int
) -> List[Optional[np.ndarray]]:
    if len(pose_series) == n_rows:
        return pose_series

    out: List[Optional[np.ndarray]] = []
    for i in range(n_rows):
        if i < len(pose_series):
            out.append(pose_series[i])
        else:
            out.append(None)
    return out


def _create_video_writer(
    video_path: str,
    frame_size: Tuple[int, int],
    fps: float,
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    parent = os.path.dirname(video_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    width, height = int(frame_size[0]), int(frame_size[1])
    if width <= 0 or height <= 0:
        return None, None

    # Try common codecs in order. mp4v is broadly supported in OpenCV builds.
    codec_candidates = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if writer is not None and writer.isOpened():
            return writer, codec
        if writer is not None:
            writer.release()
    return None, None


def _lookup_gt_pose(
    reader,
    dataset: str,
    frame_id: int,
    reader_frame_idx: int,
    object_name: Optional[str],
    strict_frame_id_match: bool = False,
) -> Optional[np.ndarray]:
    if strict_frame_id_match:
        if dataset == "ycbinisaac":
            if object_name is None:
                return None
            id_strs = getattr(reader, "id_strs", None)
            if id_strs is None or reader_frame_idx < 0 or reader_frame_idx >= len(id_strs):
                return None
            target_id = str(id_strs[reader_frame_idx])
            pose_by_id = getattr(reader, "gt_pose_file_by_id_by_object", {}).get(
                object_name, {}
            )
            pose_path = pose_by_id.get(target_id, None)
            if pose_path is None or not os.path.exists(pose_path):
                return None
            try:
                return _to_pose_matrix(np.loadtxt(pose_path).reshape(4, 4))
            except Exception:
                return None

        if dataset == "ycbineoat":
            id_strs = getattr(reader, "id_strs", None)
            if id_strs is None or reader_frame_idx < 0 or reader_frame_idx >= len(id_strs):
                return None
            target_id = str(id_strs[reader_frame_idx])
            if not hasattr(reader, "_gt_pose_file_by_id"):
                pose_files = getattr(reader, "gt_pose_files", [])
                pose_map = {}
                for p in pose_files:
                    stem = os.path.splitext(os.path.basename(str(p)))[0]
                    pose_map[stem] = p
                reader._gt_pose_file_by_id = pose_map
            pose_path = reader._gt_pose_file_by_id.get(target_id, None)
            if pose_path is None or not os.path.exists(pose_path):
                return None
            try:
                return _to_pose_matrix(np.loadtxt(pose_path).reshape(4, 4))
            except Exception:
                return None

        if dataset == "ho3d":
            id_strs = getattr(reader, "id_strs", None)
            if id_strs is not None and 0 <= reader_frame_idx < len(id_strs):
                fid = _try_parse_int(id_strs[reader_frame_idx])
                if fid is not None and fid != int(frame_id):
                    return None
            try:
                return _to_pose_matrix(reader.get_gt_pose(reader_frame_idx))
            except Exception:
                return None

    try:
        if dataset == "ycbinisaac":
            pose = reader.get_gt_pose(reader_frame_idx, obj_name=object_name)
        else:
            pose = reader.get_gt_pose(reader_frame_idx)
    except Exception:
        return None
    return _to_pose_matrix(pose)


def run(args):
    run_dir = os.path.abspath(args.run_dir)
    video_name = args.video_name if args.video_name else os.path.basename(run_dir.rstrip("/"))

    if args.meta_data_path:
        meta_data_path = os.path.abspath(args.meta_data_path)
    else:
        meta_data_path = os.path.join(run_dir, "meta_data", "meta_data.npz")
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")

    reader = _load_reader(args.dataset, os.path.abspath(args.data_root), video_name)
    render_all_ycbinisaac_objects = False
    if args.dataset == "ycbinisaac":
        available_names = reader.get_object_names()
        if len(available_names) > 1 and args.object_name is None and int(args.object_idx) == 0:
            render_all_ycbinisaac_objects = True
            selected_objects = [(idx, name) for idx, name in enumerate(available_names)]
        else:
            resolved_object_idx, resolved_object_name = _resolve_object_selection(
                dataset=args.dataset,
                reader=reader,
                object_idx=int(args.object_idx),
                object_name=args.object_name,
            )
            selected_objects = [(resolved_object_idx, resolved_object_name)]
    else:
        resolved_object_idx, resolved_object_name = _resolve_object_selection(
            dataset=args.dataset,
            reader=reader,
            object_idx=int(args.object_idx),
            object_name=args.object_name,
        )
        selected_objects = [(resolved_object_idx, resolved_object_name)]

    with np.load(meta_data_path, allow_pickle=True) as npz_data:
        if "frame_id" in npz_data:
            n_rows = int(np.asarray(npz_data["frame_id"]).reshape(-1).shape[0])
        elif args.gt_pose_only:
            n_rows = int(len(reader))
        else:
            # Need estimated pose series to infer row count when frame_id is missing.
            max_pose_rows = 0
            for object_idx, _ in selected_objects:
                _, pose_series = _select_pose_key(
                    npz_data=npz_data,
                    dataset=args.dataset,
                    pose_key=args.pose_key,
                    object_idx=object_idx,
                )
                max_pose_rows = max(max_pose_rows, len(pose_series))
            n_rows = int(max_pose_rows)

        if n_rows <= 0:
            object_indices = [item[0] for item in selected_objects]
            raise RuntimeError(
                f"No usable rows found for selected objects {object_indices}."
            )

        if "frame_id" in npz_data:
            frame_ids = _extract_frame_ids(npz_data, n_rows)
        else:
            frame_ids = np.arange(n_rows, dtype=int)

        object_states = []
        for object_idx, object_name in selected_objects:
            if args.gt_pose_only:
                pose_key = "gt_pose"
                pose_series = [None for _ in range(n_rows)]
                init_pose_series = [np.eye(4, dtype=float) for _ in range(n_rows)]
                init_pose_key = None
                valid_pose_count = n_rows
            else:
                pose_key, pose_series_raw = _select_pose_key(
                    npz_data=npz_data,
                    dataset=args.dataset,
                    pose_key=args.pose_key,
                    object_idx=object_idx,
                )
                pose_series = _normalize_pose_series_length(
                    pose_series_raw, n_rows=n_rows
                )
                init_pose_series, init_pose_key = _extract_init_pose_series(
                    npz_data=npz_data,
                    n_rows=n_rows,
                    object_idx=object_idx,
                    prefer_multi_object_key=pose_key.endswith("_all"),
                )
                valid_pose_count = int(sum(p is not None for p in pose_series))
                if valid_pose_count == 0:
                    print(
                        f"[Warn] Skipping object_idx={object_idx} name={object_name}: "
                        f"no valid poses in key '{pose_key}'."
                    )
                    continue

            object_states.append(
                {
                    "object_idx": object_idx,
                    "object_name": object_name,
                    "label": (
                        f"{object_idx}:{object_name}"
                        if object_name is not None
                        else str(object_idx)
                    ),
                    "pose_key": pose_key,
                    "pose_series": pose_series,
                    "init_pose_key": init_pose_key,
                    "init_pose_series": init_pose_series,
                    "valid_pose_count": valid_pose_count,
                }
            )

    if len(object_states) == 0:
        raise RuntimeError("No selected objects contain valid poses.")

    K = np.asarray(reader.K, dtype=np.float64).reshape(3, 3)

    mesh_source = "dataset" if args.use_gt_mesh else args.mesh_source
    model_root_abs = os.path.abspath(args.model_root) if args.model_root else None
    explicit_mesh_path = os.path.abspath(args.mesh_path) if args.mesh_path else None
    if explicit_mesh_path is not None and not os.path.exists(explicit_mesh_path):
        raise FileNotFoundError(f"--mesh_path does not exist: {explicit_mesh_path}")
    if explicit_mesh_path is not None and len(object_states) > 1:
        raise ValueError(
            "--mesh_path supports single-object overlay only. "
            "For multi-object ycbinisaac overlays, omit --mesh_path so per-object meshes are resolved automatically."
        )

    for state in object_states:
        object_idx = int(state["object_idx"])
        object_name = state["object_name"]
        mesh_source_used = "explicit"
        mesh_path = explicit_mesh_path
        if mesh_path is None:
            if mesh_source in ("auto", "run"):
                mesh_path = _resolve_run_mesh_path(run_dir, object_idx)
                if mesh_path is not None:
                    mesh_source_used = "run"

            if mesh_source == "run" and mesh_path is None:
                raise FileNotFoundError(
                    f"Run mesh for object_idx={object_idx} not found under "
                    f"{os.path.join(run_dir, 'mesh')}"
                )

            if mesh_path is None:
                mesh_path = _resolve_dataset_mesh_path(
                    dataset=args.dataset,
                    reader=reader,
                    model_root=model_root_abs,
                    object_idx=object_idx,
                    object_name=object_name,
                )
                mesh_source_used = "dataset"

        mesh = _load_mesh(mesh_path)
        state["vertices_obj"] = np.asarray(mesh.vertices, dtype=np.float64)
        state["faces"] = np.asarray(mesh.faces, dtype=np.int64)
        state["bbox_corners_obj"] = _compute_bbox_corners_from_vertices(state["vertices_obj"])
        state["mesh_path"] = mesh_path
        state["mesh_source_used"] = mesh_source_used

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(run_dir, "contour_overlay")
    os.makedirs(output_dir, exist_ok=True)

    color_bgr: Tuple[int, int, int] = (
        int(args.line_color[0]),
        int(args.line_color[1]),
        int(args.line_color[2]),
    )
    thickness = int(args.line_thickness)
    save_contour_video = not bool(args.no_video)
    save_bbox_video = not bool(args.no_video)
    video_fps = float(args.video_fps)
    if video_fps <= 0:
        raise ValueError("--video_fps must be > 0")
    axis_scale = float(args.axis_scale)
    if axis_scale <= 0:
        raise ValueError("--axis_scale must be > 0")
    axis_thickness = int(args.axis_thickness)
    if axis_thickness <= 0:
        raise ValueError("--axis_thickness must be > 0")
    if args.video_path:
        video_path = os.path.abspath(args.video_path)
    else:
        video_path = os.path.join(output_dir, f"{video_name}_contour_overlay.mp4")
    if args.bbox_video_path:
        bbox_video_path = os.path.abspath(args.bbox_video_path)
    else:
        bbox_video_path = os.path.join(output_dir, f"{video_name}_bbox_overlay.mp4")

    total_rows = n_rows
    num_written = 0
    num_contour_video_written = 0
    num_bbox_video_written = 0
    num_skipped = 0
    num_no_contour = 0
    num_no_bbox = 0
    num_missing_pose = 0
    num_invalid_frame_id = 0
    num_duplicate_frame = 0
    missing_pose_by_object = {state["label"]: 0 for state in object_states}
    no_contour_by_object = {state["label"]: 0 for state in object_states}
    no_bbox_by_object = {state["label"]: 0 for state in object_states}

    frame_id_to_reader_idx = _build_frame_id_to_reader_index(reader)
    seen_frame_ids = set()
    contour_video_writer: Optional[cv2.VideoWriter] = None
    contour_video_codec: Optional[str] = None
    contour_video_frame_size: Optional[Tuple[int, int]] = None
    bbox_video_writer: Optional[cv2.VideoWriter] = None
    bbox_video_codec: Optional[str] = None
    bbox_video_frame_size: Optional[Tuple[int, int]] = None

    print(f"[Info] Dataset: {args.dataset}")
    print(f"[Info] Sequence: {video_name}")
    print(f"[Info] Metadata: {meta_data_path}")
    if args.gt_pose_only and args.strict_gt_frame_id_match:
        print("[Info] GT frame-ID mode: strict (no index fallback)")
    if render_all_ycbinisaac_objects:
        print(
            f"[Info] ycbinisaac multi-object mode: rendering all {len(object_states)} objects"
        )
    for state in object_states:
        print(
            f"[Info] Object: idx={state['object_idx']}, name={state['object_name']}, "
            f"pose_key={state['pose_key']}, "
            f"init_pose_key={state['init_pose_key'] if state['init_pose_key'] is not None else 'identity'}, "
            f"mesh_source={state['mesh_source_used']}, mesh={state['mesh_path']}"
        )
    if args.gt_pose_only:
        print("[Info] Pose source: dataset GT pose (--gt_pose_only)")
    else:
        print(f"[Info] Pose source: metadata key + compose mode ({args.pose_compose_mode})")
    print(f"[Info] Output: {output_dir}")
    print(f"[Info] Frames in reader: {len(reader)}, metadata rows: {total_rows}")
    if args.show_axis:
        print(
            f"[Info] Axis overlay: enabled (scale={axis_scale}, thickness={axis_thickness})"
        )
    else:
        print("[Info] Axis overlay: disabled")
    if save_contour_video:
        print(f"[Info] Video output: {video_path} @ {video_fps:.3f} fps")
    if save_bbox_video:
        print(f"[Info] 3D bbox video output: {bbox_video_path} @ {video_fps:.3f} fps")
    if not save_contour_video and not save_bbox_video:
        print("[Info] Video output: disabled (--no_video)")

    for row_idx in range(0, total_rows, max(1, int(args.stride))):
        frame_id = int(frame_ids[row_idx])
        if frame_id < args.start_frame:
            continue
        if args.end_frame >= 0 and frame_id > args.end_frame:
            continue

        reader_frame_idx = frame_id_to_reader_idx.get(frame_id)
        if reader_frame_idx is None and 0 <= frame_id < len(reader):
            reader_frame_idx = frame_id

        if (
            reader_frame_idx is None
            or reader_frame_idx < 0
            or reader_frame_idx >= len(reader)
        ):
            num_invalid_frame_id += 1
            num_skipped += 1
            continue

        if not args.allow_duplicate_frame_ids and frame_id in seen_frame_ids:
            num_duplicate_frame += 1
            num_skipped += 1
            continue

        frame_bgr = _load_frame_bgr(reader, args.dataset, reader_frame_idx)
        overlay = frame_bgr.copy()
        bbox_overlay = frame_bgr.copy()
        image_h, image_w = frame_bgr.shape[:2]
        row_has_any_pose = False
        row_has_any_contour = False
        row_has_any_bbox = False

        for state in object_states:
            if args.gt_pose_only:
                T_obj_in_cam = _lookup_gt_pose(
                    reader=reader,
                    dataset=args.dataset,
                    frame_id=frame_id,
                    reader_frame_idx=reader_frame_idx,
                    object_name=state["object_name"],
                    strict_frame_id_match=bool(args.strict_gt_frame_id_match),
                )
                if T_obj_in_cam is None:
                    missing_pose_by_object[state["label"]] += 1
                    continue
            else:
                pose_obj = state["pose_series"][row_idx]
                if pose_obj is None:
                    missing_pose_by_object[state["label"]] += 1
                    continue
                init_pose = state["init_pose_series"][row_idx]
                T_obj_in_cam = _compose_object_pose(
                    pose_obj=pose_obj,
                    init_pose=init_pose,
                    mode=args.pose_compose_mode,
                )

            row_has_any_pose = True
            silhouette = _rasterize_silhouette(
                vertices_obj=state["vertices_obj"],
                faces=state["faces"],
                T_obj_in_cam=T_obj_in_cam,
                K=K,
                image_h=image_h,
                image_w=image_w,
            )

            contours, _ = cv2.findContours(
                silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                no_contour_by_object[state["label"]] += 1
            else:
                row_has_any_contour = True
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    color=color_bgr,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            drew_bbox = _draw_projected_3d_bbox(
                image=bbox_overlay,
                bbox_corners_obj=state["bbox_corners_obj"],
                T_obj_in_cam=T_obj_in_cam,
                K=K,
                color_bgr=color_bgr,
                thickness=thickness,
            )
            if drew_bbox:
                row_has_any_bbox = True
            else:
                no_bbox_by_object[state["label"]] += 1

            if args.show_axis:
                _draw_projected_xyz_axis(
                    image=overlay,
                    T_obj_in_cam=T_obj_in_cam,
                    K=K,
                    axis_scale=axis_scale,
                    thickness=axis_thickness,
                )
                _draw_projected_xyz_axis(
                    image=bbox_overlay,
                    T_obj_in_cam=T_obj_in_cam,
                    K=K,
                    axis_scale=axis_scale,
                    thickness=axis_thickness,
                )

        if not row_has_any_pose:
            num_missing_pose += 1
            num_skipped += 1
            continue
        seen_frame_ids.add(frame_id)
        if not row_has_any_contour:
            num_no_contour += 1
        if not row_has_any_bbox:
            num_no_bbox += 1

        if args.allow_duplicate_frame_ids:
            out_name = f"frame_{frame_id:06d}_row_{row_idx:06d}.png"
        else:
            out_name = f"frame_{frame_id:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        ok = cv2.imwrite(out_path, overlay)
        if ok:
            num_written += 1
        else:
            num_skipped += 1

        if save_contour_video:
            if contour_video_writer is None:
                h, w = overlay.shape[:2]
                contour_video_frame_size = (int(w), int(h))
                contour_video_writer, contour_video_codec = _create_video_writer(
                    video_path=video_path,
                    frame_size=contour_video_frame_size,
                    fps=video_fps,
                )
                if contour_video_writer is None:
                    print(
                        f"[Warn] Failed to initialize video writer for {video_path}. "
                        "Skipping mp4 export."
                    )
                    save_contour_video = False
                else:
                    print(f"[Info] Video codec: {contour_video_codec}")

            if save_contour_video and contour_video_writer is not None:
                if (
                    contour_video_frame_size is not None
                    and (overlay.shape[1], overlay.shape[0]) != contour_video_frame_size
                ):
                    overlay_to_write = cv2.resize(
                        overlay,
                        contour_video_frame_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    overlay_to_write = overlay
                contour_video_writer.write(overlay_to_write)
                num_contour_video_written += 1

        if save_bbox_video:
            if bbox_video_writer is None:
                h, w = bbox_overlay.shape[:2]
                bbox_video_frame_size = (int(w), int(h))
                bbox_video_writer, bbox_video_codec = _create_video_writer(
                    video_path=bbox_video_path,
                    frame_size=bbox_video_frame_size,
                    fps=video_fps,
                )
                if bbox_video_writer is None:
                    print(
                        f"[Warn] Failed to initialize 3D bbox video writer for {bbox_video_path}. "
                        "Skipping 3D bbox mp4 export."
                    )
                    save_bbox_video = False
                else:
                    print(f"[Info] 3D bbox video codec: {bbox_video_codec}")

            if save_bbox_video and bbox_video_writer is not None:
                if (
                    bbox_video_frame_size is not None
                    and (bbox_overlay.shape[1], bbox_overlay.shape[0]) != bbox_video_frame_size
                ):
                    bbox_to_write = cv2.resize(
                        bbox_overlay,
                        bbox_video_frame_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    bbox_to_write = bbox_overlay
                bbox_video_writer.write(bbox_to_write)
                num_bbox_video_written += 1

    if contour_video_writer is not None:
        contour_video_writer.release()
    if bbox_video_writer is not None:
        bbox_video_writer.release()

    print(
        "[Done] wrote={} skipped={} no_contour={} missing_pose={} "
        "no_bbox={} invalid_frame_id={} duplicate_frame={} output_dir={} contour_video_frames={} "
        "bbox_video_frames={}".format(
            num_written,
            num_skipped,
            num_no_contour,
            num_missing_pose,
            num_no_bbox,
            num_invalid_frame_id,
            num_duplicate_frame,
            output_dir,
            num_contour_video_written,
            num_bbox_video_written,
        )
    )
    if save_contour_video and num_contour_video_written > 0:
        print(f"[Info] Saved video: {video_path}")
    elif not args.no_video:
        print("[Warn] Contour video was requested but no frames were written to mp4.")

    if save_bbox_video and num_bbox_video_written > 0:
        print(f"[Info] Saved 3D bbox video: {bbox_video_path}")
    elif not args.no_video:
        print("[Warn] 3D bbox video was requested but no frames were written to mp4.")
    for state in object_states:
        label = state["label"]
        print(
            f"[Info] Object stats [{label}]: missing_pose_rows={missing_pose_by_object[label]}, "
            f"no_contour_rows={no_contour_by_object[label]}, "
            f"no_bbox_rows={no_bbox_by_object[label]}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a finished sequence (meta_data + mesh) and overlay mesh contour on "
            "dataset images using estimated poses. Also exports a 3D bbox visualization mp4."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ho3d", "ycbineoat", "ycbinisaac"],
        help="Dataset/algorithm mode.",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Finished sequence directory (contains meta_data/ and optionally mesh/).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Sequence name. If omitted, inferred from run_dir basename.",
    )
    parser.add_argument(
        "--meta_data_path",
        type=str,
        default=None,
        help="Path to meta_data.npz. Default: <run_dir>/meta_data/meta_data.npz",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help=(
            "Pose key in metadata "
            "(obj_pose_all, obj_pose, pose_local, pose_frontend, or auto)."
        ),
    )
    parser.add_argument(
        "--gt_pose_only",
        action="store_true",
        help=(
            "Use dataset GT pose only for overlay (ignores metadata estimated pose "
            "selection/composition)."
        ),
    )
    parser.add_argument(
        "--strict_gt_frame_id_match",
        action="store_true",
        help=(
            "When used with --gt_pose_only, require exact frame-id match for GT pose files "
            "(skip frame on mismatch; no index fallback)."
        ),
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help=(
            "Explicit mesh path. If omitted, use run mesh pred_mesh_obj_<object_idx>* "
            "then dataset mesh fallback. "
            "Only supported for single-object overlay."
        ),
    )
    parser.add_argument(
        "--mesh_source",
        type=str,
        default="auto",
        choices=["auto", "run", "dataset"],
        help=(
            "Mesh source when --mesh_path is not provided. "
            "'dataset' uses dataset GT CAD mesh."
        ),
    )
    parser.add_argument(
        "--use_gt_mesh",
        action="store_true",
        help="Alias for --mesh_source=dataset.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="YCB model root directory (needed for dataset-mesh fallback).",
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=0,
        help=(
            "Object index for run mesh and ycbinisaac object selection. "
            "For ycbinisaac multi-object sequences, default object_idx=0 with no "
            "--object_name renders all sequence objects."
        ),
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="ycbinisaac object folder name override (optional, forces single-object mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for contour overlays. Default: <run_dir>/contour_overlay",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--allow_duplicate_frame_ids",
        action="store_true",
        help=(
            "If set, writes one file per metadata row even if frame_id repeats "
            "(filename includes row index)."
        ),
    )
    parser.add_argument(
        "--pose_compose_mode",
        type=str,
        default="post_multiply_init",
        choices=["post_multiply_init", "pre_multiply_init", "none"],
        help=(
            "How to compose pose with init pose: "
            "post_multiply_init => pose @ init (default/backward-compatible), "
            "pre_multiply_init => init @ pose, none => pose only."
        ),
    )
    parser.add_argument(
        "--line_color",
        nargs=3,
        type=int,
        default=[0, 255, 0],
        metavar=("B", "G", "R"),
        help="Contour/3D-bbox color in BGR.",
    )
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="Overlay projected XYZ object axis on both contour and 3D-bbox outputs.",
    )
    parser.add_argument(
        "--axis_scale",
        type=float,
        default=0.08,
        help="Length of each axis in object coordinates (same unit as mesh).",
    )
    parser.add_argument(
        "--axis_thickness",
        type=int,
        default=2,
        help="Projected XYZ-axis line thickness in pixels.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable mp4 export (contour and 3D bbox videos).",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="FPS for output mp4 video.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Explicit output mp4 path. Default: <output_dir>/<video_name>_contour_overlay.mp4",
    )
    parser.add_argument(
        "--bbox_video_path",
        type=str,
        default=None,
        help="Explicit output mp4 path for 3D bbox visualization. Default: <output_dir>/<video_name>_bbox_overlay.mp4",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
