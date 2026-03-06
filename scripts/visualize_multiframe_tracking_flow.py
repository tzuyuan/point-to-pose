#!/usr/bin/env python3
"""
Visualize multi-frame object point tracking on RGB frames.

Default behavior reads tracked object points from `meta_data.npz` and renders:
  - tracking_sequence: per-frame overlays with tracked points and short trajectories
  - overlap_flow_on_anchor.png: one anchor image overlaid with flows to multiple frames
  - pairwise_anchor_to_target: side-by-side anchor/target flow visualizations

Fallback behavior (`--point_source lk`) uses LK optical flow from sampled anchor points.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Add project root for local imports.
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point2pose.io.sources.dataset.datareader import Ho3dReader, YCBInIsaacReader


def _is_valid_sequence_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "rgb").is_dir():
        return False
    # YCBInIsaac/YCBInEOAT-style sequence.
    if (path / "cam_K.txt").is_file():
        return True
    # HO3D evaluation sequence.
    if (path / "meta").is_dir():
        return True
    return False


def resolve_video_path(dataset_root: str, video_name: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    candidates = [root, root / video_name, root / video_name / video_name]
    for cand in candidates:
        if _is_valid_sequence_dir(cand):
            return cand
    raise FileNotFoundError(
        "Could not resolve sequence path. Checked: "
        f"{[str(c) for c in candidates]}"
    )


def _infer_ho3d_root(dataset_root: Path, video_path: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root.parent,
        video_path.parent,
        video_path.parent.parent,
    ]
    checked = []
    for cand in candidates:
        cand = cand.resolve()
        if str(cand) in checked:
            continue
        checked.append(str(cand))
        if (cand / "masks").is_dir() and (cand / "models").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not infer HO3D root directory containing masks/ and models/. "
        f"Checked: {checked}"
    )


def build_reader(video_path: Path, dataset_root: str) -> Tuple[object, str]:
    if (video_path / "cam_K.txt").is_file():
        return YCBInIsaacReader(str(video_path)), "ycb"

    if (video_path / "meta").is_dir():
        ho3d_root = _infer_ho3d_root(Path(dataset_root).expanduser().resolve(), video_path)
        return Ho3dReader(str(video_path), str(ho3d_root)), "ho3d"

    raise ValueError(
        f"Unsupported sequence layout at {video_path}. Expected cam_K.txt (YCB) or meta/ (HO3D)."
    )


def reader_get_color_rgb(reader, frame_idx: int) -> np.ndarray:
    if hasattr(reader, "get_color"):
        rgb = np.asarray(reader.get_color(frame_idx))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    color_files = getattr(reader, "color_files", None)
    if color_files is None:
        raise AttributeError("Reader does not expose get_color() or color_files.")
    if frame_idx < 0 or frame_idx >= len(color_files):
        raise IndexError(f"Frame index out of range: {frame_idx}")

    bgr = cv2.imread(str(color_files[frame_idx]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read color frame: {color_files[frame_idx]}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_frame_list(frame_list: str) -> List[int]:
    if frame_list is None:
        return []
    s = frame_list.strip()
    if s == "":
        return []
    out = []
    for token in s.split(","):
        token = token.strip()
        if token == "":
            continue
        out.append(int(token))
    return out


def build_frame_id_to_reader_index(reader) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    id_strs = getattr(reader, "id_strs", None)
    if id_strs is not None:
        for idx, sid in enumerate(id_strs):
            try:
                mapping[int(str(sid))] = idx
            except Exception:
                continue

    if len(mapping) == 0:
        for idx in range(len(reader)):
            mapping[idx] = idx
    return mapping


def resolve_reader_frame_index(
    reader, frame_id: int, frame_id_to_reader_idx: Dict[int, int]
) -> int:
    idx = frame_id_to_reader_idx.get(int(frame_id))
    if idx is None and 0 <= int(frame_id) < len(reader):
        idx = int(frame_id)
    if idx is None or idx < 0 or idx >= len(reader):
        raise IndexError(f"Could not map frame_id={frame_id} to reader frame index.")
    return int(idx)


def reader_get_color_rgb_by_frame_id(
    reader, frame_id: int, frame_id_to_reader_idx: Dict[int, int]
) -> np.ndarray:
    idx = resolve_reader_frame_index(reader, frame_id, frame_id_to_reader_idx)
    return reader_get_color_rgb(reader, idx)


def resolve_meta_data_path(
    meta_data_path: str,
    results_dir: str,
    video_name: str,
) -> Path:
    if meta_data_path:
        p = Path(meta_data_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Metadata file not found: {p}")
        return p

    candidates: List[Path] = []
    if results_dir:
        candidates.append(
            Path(results_dir).expanduser().resolve()
            / video_name
            / "meta_data"
            / "meta_data.npz"
        )

    pattern = str(
        PROJECT_ROOT / "results" / "*" / video_name / "meta_data" / "meta_data.npz"
    )
    for p in sorted(glob.glob(pattern)):
        candidates.append(Path(p).resolve())

    seen = set()
    for cand in candidates:
        s = str(cand)
        if s in seen:
            continue
        seen.add(s)
        if cand.is_file():
            return cand

    checked = [str(x) for x in candidates]
    raise FileNotFoundError(
        "Could not resolve metadata path. "
        "Provide --meta_data_path or --results_dir. "
        f"Checked: {checked}"
    )


def _extract_frame_ids(npz_data) -> np.ndarray:
    if "frame_id" in npz_data.files:
        return np.asarray(npz_data["frame_id"]).reshape(-1).astype(int)

    for base in ("obj_key_points", "track2d"):
        offsets_key = f"{base}_offsets"
        if offsets_key in npz_data.files:
            n = int(np.asarray(npz_data[offsets_key]).reshape(-1).shape[0])
            return np.arange(n, dtype=int)

    return np.asarray([], dtype=int)


def _ragged_slice(npz_data, key: str, row_idx: int) -> np.ndarray:
    data_key = f"{key}_data"
    offsets_key = f"{key}_offsets"
    lengths_key = f"{key}_lengths"
    if (
        data_key not in npz_data.files
        or offsets_key not in npz_data.files
        or lengths_key not in npz_data.files
    ):
        return np.asarray([], dtype=float)

    data = np.asarray(npz_data[data_key])
    offsets = np.asarray(npz_data[offsets_key]).reshape(-1)
    lengths = np.asarray(npz_data[lengths_key]).reshape(-1)
    if row_idx < 0 or row_idx >= len(offsets) or row_idx >= len(lengths):
        return np.asarray([], dtype=data.dtype)
    off = int(offsets[row_idx])
    length = int(lengths[row_idx])
    if length <= 0:
        return np.asarray([], dtype=data.dtype)
    return np.asarray(data[off : off + length])


def _ragged_slice_2d(npz_data, key: str, row_idx: int, dim: int) -> np.ndarray:
    flat = _ragged_slice(npz_data, key, row_idx)
    if flat.size == 0:
        return np.zeros((0, dim), dtype=np.float64)
    n = int(flat.size // dim)
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    if flat.size != n * dim:
        flat = flat[: n * dim]
    return flat.reshape(n, dim)


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
        keys_to_try: Iterable[object] = (object_idx, str(object_idx), f"obj_{object_idx}")
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
    if key not in npz_data.files:
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

            if arr.shape[0] >= arr.shape[1]:
                if object_idx >= arr.shape[1]:
                    return [None for _ in range(arr.shape[0])]
                return [_to_pose_matrix(arr[i, object_idx]) for i in range(arr.shape[0])]
            if object_idx >= arr.shape[0]:
                return [None for _ in range(arr.shape[1])]
            return [_to_pose_matrix(arr[object_idx, i]) for i in range(arr.shape[1])]

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
        return [_extract_pose_from_entry(item, object_idx) for item in raw]

    raise ValueError(
        f"Unsupported pose container format for key '{key}': "
        f"type={type(arr)}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}"
    )


def _select_pose_key(
    npz_data,
    pose_key: str,
    object_idx: int,
) -> Tuple[str, List[Optional[np.ndarray]]]:
    frame_ids_raw = npz_data["frame_id"] if "frame_id" in npz_data.files else None

    if pose_key != "auto":
        poses = _extract_object_pose_series(
            npz_data, pose_key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        return pose_key, poses

    candidates = ("obj_pose_all", "obj_pose", "pose_local", "pose_frontend")
    last_nonempty_key = None
    last_nonempty_series = None
    for key in candidates:
        if key not in npz_data.files:
            continue
        series = _extract_object_pose_series(
            npz_data, key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        valid_count = int(sum(p is not None for p in series))
        if valid_count > 0:
            return key, series
        if len(series) > 0:
            last_nonempty_key = key
            last_nonempty_series = series

    if last_nonempty_key is not None and last_nonempty_series is not None:
        return last_nonempty_key, last_nonempty_series

    raise KeyError(
        "No supported pose key found. Tried: obj_pose_all, obj_pose, pose_local, pose_frontend"
    )


def _project_points_cam_to_image(
    points_cam: np.ndarray,
    K: np.ndarray,
    image_w: int,
    image_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if points_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    finite_xyz = np.isfinite(points_cam).all(axis=1)
    z = points_cam[:, 2]
    valid_z = z > 1e-6
    base_mask = finite_xyz & valid_z
    base_idx = np.where(base_mask)[0]
    if base_idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    pts = points_cam[base_idx]
    proj = (K @ pts.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    in_bounds = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u < float(image_w))
        & (v < float(image_h))
    )
    final_idx = base_idx[in_bounds]
    uv = np.stack([u[in_bounds], v[in_bounds]], axis=1).astype(np.float32)
    return uv, final_idx.astype(np.int64)


def _choose_point_ids(
    anchor_ids: np.ndarray,
    all_ids: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    all_unique = np.unique(all_ids.astype(np.int64)) if all_ids.size > 0 else np.asarray([], dtype=np.int64)
    anchor_unique = (
        np.unique(anchor_ids.astype(np.int64))
        if anchor_ids.size > 0
        else np.asarray([], dtype=np.int64)
    )
    if all_unique.size == 0:
        return all_unique

    if max_points <= 0:
        if anchor_unique.size > 0:
            return np.sort(anchor_unique)
        return np.sort(all_unique)

    rng = np.random.default_rng(seed)
    selected = anchor_unique.copy()
    if selected.size > max_points:
        pick = rng.choice(selected.size, size=max_points, replace=False)
        return np.sort(selected[pick])

    if selected.size < max_points:
        extras = np.setdiff1d(all_unique, selected, assume_unique=False)
        if extras.size > 0:
            need = min(max_points - selected.size, extras.size)
            pick = rng.choice(extras.size, size=need, replace=False)
            selected = np.concatenate([selected, extras[pick]])

    if selected.size == 0:
        need = min(max_points, all_unique.size)
        pick = rng.choice(all_unique.size, size=need, replace=False)
        selected = all_unique[pick]
    return np.sort(selected.astype(np.int64))


def build_trajectories_from_metadata(
    npz_data,
    reader,
    frame_id_to_reader_idx: Dict[int, int],
    anchor_frame: int,
    end_frame: int,
    max_points: int,
    object_idx: int,
    pose_key: str,
    meta_point_source: str,
    use_obj_valid: bool,
    seed: int = 0,
) -> Tuple[np.ndarray, str, Optional[str]]:
    frame_ids = _extract_frame_ids(npz_data)
    if frame_ids.size == 0:
        raise RuntimeError("Metadata does not contain frame IDs or ragged offsets.")

    # Prefer object-map points (3D map projected with pose) unless disabled/unavailable.
    source_used = None
    pose_key_used: Optional[str] = None
    pose_series: List[Optional[np.ndarray]] = []
    if meta_point_source in ("auto", "obj_key_points"):
        if "obj_key_points_data" in npz_data.files:
            try:
                pose_key_used, pose_series = _select_pose_key(
                    npz_data=npz_data,
                    pose_key=pose_key,
                    object_idx=object_idx,
                )
                source_used = "obj_key_points"
            except Exception:
                if meta_point_source == "obj_key_points":
                    raise
    if source_used is None:
        if "track2d_data" not in npz_data.files:
            raise RuntimeError(
                "Metadata does not contain usable tracked points. "
                "Expected obj_key_points (+pose) or track2d."
            )
        source_used = "track2d"

    n_rows = int(frame_ids.shape[0])
    if source_used == "obj_key_points" and len(pose_series) < n_rows:
        pose_series = pose_series + [None] * (n_rows - len(pose_series))

    try:
        anchor_rgb = reader_get_color_rgb_by_frame_id(
            reader, anchor_frame, frame_id_to_reader_idx
        )
    except Exception:
        anchor_rgb = reader_get_color_rgb(reader, max(0, min(anchor_frame, len(reader) - 1)))
    image_h, image_w = anchor_rgb.shape[:2]
    K = np.asarray(reader.K, dtype=np.float64).reshape(3, 3)

    row_by_frame: Dict[int, int] = {}
    for row_idx, fid in enumerate(frame_ids.tolist()):
        if int(fid) not in row_by_frame:
            row_by_frame[int(fid)] = int(row_idx)

    t = int(end_frame - anchor_frame + 1)
    if t <= 0:
        raise RuntimeError("Invalid frame range for metadata trajectory export.")

    ids_per_offset: List[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(t)]
    uv_per_offset: List[np.ndarray] = [np.zeros((0, 2), dtype=np.float32) for _ in range(t)]
    all_ids_accum: List[np.ndarray] = []

    for frame_id in range(anchor_frame, end_frame + 1):
        offset = frame_id - anchor_frame
        row_idx = row_by_frame.get(int(frame_id))
        if row_idx is None:
            continue

        if source_used == "track2d":
            pts2d = _ragged_slice_2d(npz_data, "track2d", row_idx, 2)
            if pts2d.shape[0] == 0:
                continue
            ids = np.arange(pts2d.shape[0], dtype=np.int64)
            mask = np.isfinite(pts2d).all(axis=1)
            vis = _ragged_slice(npz_data, "visibles", row_idx).reshape(-1).astype(bool, copy=False)
            if vis.size == ids.size:
                mask &= vis
            mask &= (
                (pts2d[:, 0] >= 0.0)
                & (pts2d[:, 1] >= 0.0)
                & (pts2d[:, 0] < float(image_w))
                & (pts2d[:, 1] < float(image_h))
            )
            if not np.any(mask):
                continue
            ids = ids[mask]
            uv = pts2d[mask].astype(np.float32)
        else:
            if row_idx >= len(pose_series):
                continue
            pose = pose_series[row_idx]
            if pose is None:
                continue
            pts3d = _ragged_slice_2d(npz_data, "obj_key_points", row_idx, 3)
            if pts3d.shape[0] == 0:
                continue
            ids = np.arange(pts3d.shape[0], dtype=np.int64)
            base_mask = np.isfinite(pts3d).all(axis=1)
            if use_obj_valid:
                obj_valid = _ragged_slice(npz_data, "obj_valid", row_idx).reshape(-1).astype(
                    bool, copy=False
                )
                if obj_valid.size == ids.size:
                    base_mask &= obj_valid
            if not np.any(base_mask):
                continue
            ids_sel = ids[base_mask]
            pts3d_sel = pts3d[base_mask]
            pts_h = np.concatenate(
                [pts3d_sel.astype(np.float64), np.ones((pts3d_sel.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            pts_cam = (np.asarray(pose, dtype=np.float64) @ pts_h.T).T[:, :3]
            uv, keep_local_idx = _project_points_cam_to_image(
                points_cam=pts_cam,
                K=K,
                image_w=image_w,
                image_h=image_h,
            )
            if uv.shape[0] == 0:
                continue
            ids = ids_sel[keep_local_idx]

        ids_per_offset[offset] = ids.astype(np.int64, copy=False)
        uv_per_offset[offset] = uv.astype(np.float32, copy=False)
        all_ids_accum.append(ids)

    if len(all_ids_accum) == 0:
        raise RuntimeError(
            "No valid metadata points available in the selected frame range."
        )

    all_ids = np.concatenate(all_ids_accum, axis=0)
    anchor_ids = ids_per_offset[0] if len(ids_per_offset) > 0 else np.asarray([], dtype=np.int64)
    chosen_ids = _choose_point_ids(
        anchor_ids=anchor_ids,
        all_ids=all_ids,
        max_points=int(max_points),
        seed=seed,
    )
    if chosen_ids.size == 0:
        raise RuntimeError("No point IDs selected from metadata.")

    id_to_local = {int(pid): i for i, pid in enumerate(chosen_ids.tolist())}
    trajectories = np.full((chosen_ids.size, t, 2), np.nan, dtype=np.float32)
    for offset in range(t):
        ids = ids_per_offset[offset]
        if ids.size == 0:
            continue
        uv = uv_per_offset[offset]
        local_idx = np.fromiter((id_to_local.get(int(i), -1) for i in ids), dtype=np.int64)
        keep = local_idx >= 0
        if not np.any(keep):
            continue
        trajectories[local_idx[keep], offset, :] = uv[keep]

    return trajectories, source_used, pose_key_used


def get_tracking_mask(
    reader,
    reader_kind: str,
    anchor_frame: int,
    object_name: str,
    use_union_mask_when_no_object: bool,
) -> np.ndarray:
    if reader_kind == "ho3d":
        if object_name:
            print("[Warn] --object_name is ignored for HO3D (single-object sequence).")
        h, w = reader_get_color_rgb(reader, anchor_frame).shape[:2]
        if not use_union_mask_when_no_object:
            return np.full((h, w), 255, dtype=np.uint8)
        mask = reader.get_mask(anchor_frame)
        if mask is None:
            return np.full((h, w), 255, dtype=np.uint8)
        return (np.asarray(mask) > 0).astype(np.uint8) * 255

    if object_name:
        names = reader.get_object_names()
        if object_name not in names:
            raise ValueError(f"Object '{object_name}' not found. Available: {names}")
        mask = reader.get_mask(anchor_frame, obj_name=object_name)
        return (np.asarray(mask) > 0).astype(np.uint8) * 255

    if not use_union_mask_when_no_object:
        h, w = reader_get_color_rgb(reader, anchor_frame).shape[:2]
        return np.full((h, w), 255, dtype=np.uint8)

    masks = reader.get_masks(anchor_frame)
    if len(masks) == 0:
        h, w = reader_get_color_rgb(reader, anchor_frame).shape[:2]
        return np.full((h, w), 255, dtype=np.uint8)
    union_mask = np.zeros_like(np.asarray(masks[0], dtype=np.uint8))
    for m in masks:
        union_mask = np.maximum(union_mask, (np.asarray(m) > 0).astype(np.uint8))
    return union_mask * 255


def sample_anchor_points(
    gray: np.ndarray,
    mask_u8: np.ndarray,
    max_points: int,
    quality_level: float,
    min_distance: float,
    seed: int = 0,
) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_points),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        mask=mask_u8,
        blockSize=7,
        useHarrisDetector=False,
    )
    if points is not None and points.shape[0] > 0:
        tracked = points.reshape(-1, 2).astype(np.float32)
    else:
        tracked = np.zeros((0, 2), dtype=np.float32)

    need = max(0, int(max_points) - int(tracked.shape[0]))
    if need == 0:
        return tracked

    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        ys, xs = np.where(np.ones_like(gray, dtype=bool))
    if ys.size == 0:
        return tracked

    used = set()
    if tracked.shape[0] > 0:
        rounded = np.round(tracked).astype(np.int32)
        h, w = gray.shape[:2]
        for x, y in rounded:
            if 0 <= x < w and 0 <= y < h:
                used.add((int(x), int(y)))

    coords = np.stack([xs, ys], axis=1)
    if len(used) > 0:
        keep = np.array([(int(x), int(y)) not in used for x, y in coords], dtype=bool)
        coords = coords[keep]
    if coords.shape[0] == 0:
        return tracked

    n = min(need, int(coords.shape[0]))
    rng = np.random.default_rng(seed + 1)
    pick = rng.choice(coords.shape[0], size=n, replace=False)
    random_pts = coords[pick].astype(np.float32)

    if tracked.shape[0] == 0:
        return random_pts
    return np.concatenate([tracked, random_pts], axis=0)


def track_points_forward(
    reader,
    anchor_frame: int,
    end_frame: int,
    init_points: np.ndarray,
) -> np.ndarray:
    """
    Returns:
        trajectories: (N, T, 2), where T = end_frame - anchor_frame + 1.
                      Invalid/lost points are NaN.
    """
    n = int(init_points.shape[0])
    t = int(end_frame - anchor_frame + 1)
    trajectories = np.full((n, t, 2), np.nan, dtype=np.float32)
    if n == 0 or t <= 0:
        return trajectories

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    current = init_points.astype(np.float32).copy()
    active = np.ones((n,), dtype=bool)
    trajectories[:, 0, :] = current

    prev_gray = cv2.cvtColor(reader_get_color_rgb(reader, anchor_frame), cv2.COLOR_RGB2GRAY)
    for frame_id in range(anchor_frame + 1, end_frame + 1):
        offset = frame_id - anchor_frame
        next_gray = cv2.cvtColor(reader_get_color_rgb(reader, frame_id), cv2.COLOR_RGB2GRAY)

        active_idx = np.where(active)[0]
        next_current = np.full_like(current, np.nan, dtype=np.float32)
        if active_idx.size > 0:
            prev_pts = current[active_idx].reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_pts, None, **lk_params
            )

            if next_pts is not None and status is not None:
                status = status.reshape(-1).astype(bool)
                good_idx = active_idx[status]
                bad_idx = active_idx[~status]
                next_flat = next_pts.reshape(-1, 2)
                next_current[good_idx] = next_flat[status]
                active[bad_idx] = False
            else:
                active[active_idx] = False

        current = next_current
        trajectories[:, offset, :] = current
        prev_gray = next_gray

    return trajectories


def point_colors_bgr(n: int, seed: int = 0) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(30, 255, size=(n, 3), endpoint=True, dtype=np.uint8)


def draw_tracking_sequence(
    reader,
    frame_id_to_reader_idx: Dict[int, int],
    trajectories: np.ndarray,
    anchor_frame: int,
    end_frame: int,
    out_dir: Path,
    trail_len: int = 25,
    trail_thickness: int = 2,
    point_radius: int = 2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = trajectories.shape[0]
    colors = point_colors_bgr(n, seed=42)

    for frame_id in range(anchor_frame, end_frame + 1):
        offset = frame_id - anchor_frame
        try:
            frame_rgb = reader_get_color_rgb_by_frame_id(reader, frame_id, frame_id_to_reader_idx)
        except Exception:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for p_idx in range(n):
            pt = trajectories[p_idx, offset]
            if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                continue
            color = tuple(int(x) for x in colors[p_idx])

            start = max(0, offset - int(trail_len))
            trail = trajectories[p_idx, start : offset + 1]
            valid = np.isfinite(trail[:, 0]) & np.isfinite(trail[:, 1])
            trail = trail[valid]
            if trail.shape[0] >= 2:
                poly = np.round(trail).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    frame_bgr,
                    [poly],
                    isClosed=False,
                    color=color,
                    thickness=max(1, int(trail_thickness)),
                )

            center = tuple(np.round(pt).astype(int).tolist())
            cv2.circle(
                frame_bgr,
                center,
                radius=max(1, int(point_radius)),
                color=color,
                thickness=-1,
            )

        cv2.imwrite(str(out_dir / f"frame_{frame_id:06d}.png"), frame_bgr)


def distinct_frame_colors(n: int) -> List[tuple]:
    colors = []
    for i in range(max(1, n)):
        hue = int(round((179.0 * i) / max(1, n)))
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def draw_overlap_flow_on_anchor(
    anchor_bgr: np.ndarray,
    trajectories: np.ndarray,
    anchor_frame: int,
    target_frames: List[int],
    out_file: Path,
    line_thickness: int = 1,
    point_radius: int = 2,
) -> None:
    canvas = anchor_bgr.copy()
    n_targets = len(target_frames)
    frame_colors = distinct_frame_colors(n_targets)
    anchor_pts = trajectories[:, 0, :]

    for i, target_frame in enumerate(target_frames):
        offset = target_frame - anchor_frame
        if offset < 0 or offset >= trajectories.shape[1]:
            continue
        color = frame_colors[i]
        target_pts = trajectories[:, offset, :]

        valid = (
            np.isfinite(anchor_pts[:, 0])
            & np.isfinite(anchor_pts[:, 1])
            & np.isfinite(target_pts[:, 0])
            & np.isfinite(target_pts[:, 1])
        )
        for p0, p1 in zip(anchor_pts[valid], target_pts[valid]):
            pt0 = tuple(np.round(p0).astype(int).tolist())
            pt1 = tuple(np.round(p1).astype(int).tolist())
            cv2.line(
                canvas,
                pt0,
                pt1,
                color=color,
                thickness=max(1, int(line_thickness)),
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                pt1,
                radius=max(1, int(point_radius)),
                color=color,
                thickness=-1,
            )

        y = 28 + 20 * i
        cv2.rectangle(canvas, (10, y - 10), (24, y + 4), color, thickness=-1)
        cv2.putText(
            canvas,
            f"target frame {target_frame}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"Anchor frame {anchor_frame}: overlap flow from multiple frames",
        (10, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)


def draw_continuous_flow_on_anchor(
    anchor_bgr: np.ndarray,
    trajectories: np.ndarray,
    anchor_frame: int,
    target_frame: int,
    out_file: Path,
    line_thickness: int = 1,
    point_radius: int = 2,
) -> None:
    """
    Draw continuous per-point trajectories from anchor_frame to target_frame
    on top of the anchor image.
    """
    offset = target_frame - anchor_frame
    if offset < 1 or offset >= trajectories.shape[1]:
        return

    canvas = anchor_bgr.copy()
    n = trajectories.shape[0]
    colors = point_colors_bgr(n, seed=123)

    drawn = 0
    for p_idx in range(n):
        path = trajectories[p_idx, : offset + 1]
        valid = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
        path = path[valid]
        if path.shape[0] < 2:
            continue

        poly = np.round(path).astype(np.int32).reshape(-1, 1, 2)
        color = tuple(int(x) for x in colors[p_idx])
        cv2.polylines(
            canvas,
            [poly],
            isClosed=False,
            color=color,
            thickness=max(1, int(line_thickness)),
        )
        cv2.circle(
            canvas,
            tuple(poly[-1, 0].tolist()),
            radius=max(1, int(point_radius)),
            color=color,
            thickness=-1,
        )
        drawn += 1

    cv2.putText(
        canvas,
        f"Continuous flow: {anchor_frame} -> {target_frame} | tracks: {drawn}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)


def draw_pairwise_anchor_to_target(
    reader,
    frame_id_to_reader_idx: Dict[int, int],
    trajectories: np.ndarray,
    anchor_frame: int,
    target_frames: List[int],
    out_dir: Path,
    line_thickness: int = 1,
    point_radius: int = 2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_bgr = cv2.cvtColor(
        reader_get_color_rgb_by_frame_id(reader, anchor_frame, frame_id_to_reader_idx),
        cv2.COLOR_RGB2BGR,
    )
    ah, aw = anchor_bgr.shape[:2]
    gap = 40

    anchor_pts = trajectories[:, 0, :]
    frame_colors = distinct_frame_colors(len(target_frames))

    for i, target_frame in enumerate(target_frames):
        try:
            target_bgr = cv2.cvtColor(
                reader_get_color_rgb_by_frame_id(reader, target_frame, frame_id_to_reader_idx),
                cv2.COLOR_RGB2BGR,
            )
        except Exception:
            continue
        th, tw = target_bgr.shape[:2]

        h = max(ah, th)
        w = aw + gap + tw
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:ah, :aw] = anchor_bgr
        canvas[:th, aw + gap : aw + gap + tw] = target_bgr

        target_pts = trajectories[:, target_frame - anchor_frame, :]
        valid = (
            np.isfinite(anchor_pts[:, 0])
            & np.isfinite(anchor_pts[:, 1])
            & np.isfinite(target_pts[:, 0])
            & np.isfinite(target_pts[:, 1])
        )
        color = frame_colors[i]
        for p0, p1 in zip(anchor_pts[valid], target_pts[valid]):
            pt0 = tuple(np.round(p0).astype(int).tolist())
            pt1_local = np.round(p1).astype(int)
            pt1 = (int(pt1_local[0] + aw + gap), int(pt1_local[1]))
            cv2.line(
                canvas,
                pt0,
                pt1,
                color=color,
                thickness=max(1, int(line_thickness)),
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                pt0,
                radius=max(1, int(point_radius)),
                color=(255, 255, 255),
                thickness=-1,
            )
            cv2.circle(
                canvas,
                pt1,
                radius=max(1, int(point_radius)),
                color=color,
                thickness=-1,
            )

        cv2.putText(
            canvas,
            f"anchor {anchor_frame}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"target {target_frame}",
            (aw + gap + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / f"anchor_{anchor_frame:06d}_to_{target_frame:06d}.png"
        cv2.imwrite(str(out_path), canvas)


def pick_target_frames(
    anchor_frame: int, end_frame: int, overlap_samples: int, custom_frames: List[int]
) -> List[int]:
    if len(custom_frames) > 0:
        out = []
        for f in custom_frames:
            if f <= anchor_frame or f > end_frame:
                continue
            out.append(int(f))
        return sorted(list(set(out)))

    if end_frame <= anchor_frame:
        return []

    count = max(1, int(overlap_samples))
    sampled = np.linspace(anchor_frame + 1, end_frame, num=count)
    sampled = np.round(sampled).astype(int)
    sampled = np.unique(sampled)
    return sampled.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-frame object tracking flow overlays."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/justin/data/YCBMultiTrack_new",
        help=(
            "Dataset root path. Supports YCB-style roots/sequences and "
            "HO3D roots (e.g. /path/to/HO3D_V3 or /path/to/HO3D_V3/evaluation)."
        ),
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="006_mustard_bottle_010_potted_meat_can_005_tomato_soup_can",
        help="Sequence folder name.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/justin/code/point-to-pose/results/visualization_exports",
        help="Root output directory.",
    )
    parser.add_argument("--anchor_frame", type=int, default=0)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=350,
        help="Track forward for at most this many frames from anchor.",
    )

    parser.add_argument(
        "--point_source",
        type=str,
        default="metadata",
        choices=["metadata", "lk"],
        help="Point source: metadata-driven (default) or LK optical flow fallback.",
    )
    parser.add_argument(
        "--meta_data_path",
        type=str,
        default="",
        help="Path to meta_data.npz. If empty, auto-resolve from --results_dir / results/*.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Results root containing <video_name>/meta_data/meta_data.npz.",
    )
    parser.add_argument(
        "--meta_point_source",
        type=str,
        default="auto",
        choices=["auto", "obj_key_points", "track2d"],
        help=(
            "Metadata field to visualize. "
            "'obj_key_points' projects map points with pose; "
            "'track2d' draws tracked 2D points directly."
        ),
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key for obj_key_points projection (obj_pose/obj_pose_all/pose_local/pose_frontend/auto).",
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=0,
        help="Object index used when selecting per-object pose arrays from metadata.",
    )
    parser.add_argument(
        "--disable_obj_valid_filter",
        action="store_true",
        help="Do not apply obj_valid mask when using obj_key_points.",
    )

    # LK fallback args.
    parser.add_argument(
        "--object_name",
        type=str,
        default="",
        help=(
            "LK mode only: object name for mask-restricted sampling. "
            "If empty, full-frame sampling is used by default."
        ),
    )
    parser.add_argument(
        "--use_union_mask_when_no_object",
        action="store_true",
        help="LK mode only: when --object_name is empty, sample from union object masks.",
    )
    parser.add_argument("--max_points", type=int, default=1200)
    parser.add_argument("--quality_level", type=float, default=0.001)
    parser.add_argument("--min_distance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--trail_len", type=int, default=25)
    parser.add_argument(
        "--trail_thickness",
        type=int,
        default=3,
        help="Line thickness for trajectory trails in per-frame overlays.",
    )
    parser.add_argument(
        "--flow_line_thickness",
        type=int,
        default=2,
        help="Line thickness for overlap/pairwise/continuous flow drawings.",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=4,
        help="Point radius for all flow/tracking visualizations.",
    )
    parser.add_argument(
        "--overlap_samples",
        type=int,
        default=6,
        help="How many target frames to overlap on anchor image.",
    )
    parser.add_argument(
        "--overlap_frames",
        type=str,
        default="",
        help="Comma-separated explicit target frames, e.g. '100,250,400'.",
    )
    parser.add_argument("--clean_output", action="store_true")
    args = parser.parse_args()

    video_path = resolve_video_path(args.dataset_root, args.video_name)
    reader, reader_kind = build_reader(video_path, args.dataset_root)
    total_frames = len(reader)
    if total_frames <= 0:
        raise RuntimeError("Sequence has no frames.")

    anchor = max(0, min(int(args.anchor_frame), total_frames - 1))
    if args.max_frames <= 0:
        end_frame = total_frames - 1
    else:
        end_frame = min(total_frames - 1, anchor + int(args.max_frames) - 1)
    if end_frame < anchor:
        end_frame = anchor

    out_dir = (
        Path(args.output_root).expanduser().resolve()
        / args.video_name
        / "multiframe_tracking_flow"
    )
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id_to_reader_idx = build_frame_id_to_reader_index(reader)

    print(f"[Info] Sequence path: {video_path}")
    print(f"[Info] Reader type:   {reader_kind}")
    print(f"[Info] Output path:   {out_dir}")
    print(f"[Info] Anchor frame:  {anchor}")
    print(f"[Info] End frame:     {end_frame}")
    print(f"[Info] Point source:  {args.point_source}")

    if args.point_source == "metadata":
        meta_path = resolve_meta_data_path(
            meta_data_path=args.meta_data_path,
            results_dir=args.results_dir,
            video_name=args.video_name,
        )
        print(f"[Info] Metadata:     {meta_path}")
        with np.load(str(meta_path), allow_pickle=True) as npz_data:
            trajectories, meta_source_used, pose_key_used = build_trajectories_from_metadata(
                npz_data=npz_data,
                reader=reader,
                frame_id_to_reader_idx=frame_id_to_reader_idx,
                anchor_frame=anchor,
                end_frame=end_frame,
                max_points=args.max_points,
                object_idx=int(args.object_idx),
                pose_key=str(args.pose_key),
                meta_point_source=str(args.meta_point_source),
                use_obj_valid=not bool(args.disable_obj_valid_filter),
                seed=int(args.seed),
            )
        print(f"[Info] Metadata points: {meta_source_used}")
        if pose_key_used is not None:
            print(f"[Info] Pose key used:  {pose_key_used}")
        print(f"[Info] Selected tracks: {trajectories.shape[0]}")
    else:
        anchor_rgb_for_lk = reader_get_color_rgb(reader, anchor)
        anchor_gray = cv2.cvtColor(anchor_rgb_for_lk, cv2.COLOR_RGB2GRAY)
        track_mask = get_tracking_mask(
            reader=reader,
            reader_kind=reader_kind,
            anchor_frame=anchor,
            object_name=args.object_name.strip(),
            use_union_mask_when_no_object=args.use_union_mask_when_no_object,
        )
        init_points = sample_anchor_points(
            gray=anchor_gray,
            mask_u8=track_mask,
            max_points=args.max_points,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
            seed=args.seed,
        )
        if init_points.shape[0] == 0:
            raise RuntimeError("No anchor points sampled; cannot run LK tracking.")
        print(f"[Info] Initial sampled points (LK): {init_points.shape[0]}")
        trajectories = track_points_forward(
            reader=reader,
            anchor_frame=anchor,
            end_frame=end_frame,
            init_points=init_points,
        )
        print("[Info] Point tracking complete (LK).")

    if trajectories.shape[0] == 0:
        raise RuntimeError("No trajectories to visualize.")

    preview_offset = 0
    preview_valid = np.isfinite(trajectories[:, 0, 0]) & np.isfinite(trajectories[:, 0, 1])
    if not np.any(preview_valid):
        for off in range(1, trajectories.shape[1]):
            valid = np.isfinite(trajectories[:, off, 0]) & np.isfinite(trajectories[:, off, 1])
            if np.any(valid):
                preview_offset = off
                preview_valid = valid
                break
    preview_frame = anchor + preview_offset
    preview_rgb = reader_get_color_rgb_by_frame_id(reader, preview_frame, frame_id_to_reader_idx)
    anchor_preview = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    for p in trajectories[preview_valid, preview_offset, :]:
        cv2.circle(
            anchor_preview,
            tuple(np.round(p).astype(int).tolist()),
            radius=2,
            color=(0, 255, 255),
            thickness=-1,
        )
    cv2.imwrite(
        str(out_dir / f"anchor_points_frame_{preview_frame:06d}.png"),
        anchor_preview,
    )

    sequence_dir = out_dir / "tracking_sequence"
    draw_tracking_sequence(
        reader=reader,
        frame_id_to_reader_idx=frame_id_to_reader_idx,
        trajectories=trajectories,
        anchor_frame=anchor,
        end_frame=end_frame,
        out_dir=sequence_dir,
        trail_len=args.trail_len,
        trail_thickness=args.trail_thickness,
        point_radius=args.point_radius,
    )
    print(f"[Info] Saved per-frame tracking overlays: {sequence_dir}")

    custom_overlap_frames = parse_frame_list(args.overlap_frames)
    target_frames = pick_target_frames(
        anchor_frame=anchor,
        end_frame=end_frame,
        overlap_samples=args.overlap_samples,
        custom_frames=custom_overlap_frames,
    )
    print(f"[Info] Overlap target frames: {target_frames}")

    if len(target_frames) > 0:
        anchor_rgb = reader_get_color_rgb_by_frame_id(reader, anchor, frame_id_to_reader_idx)
        anchor_bgr = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
        draw_overlap_flow_on_anchor(
            anchor_bgr=anchor_bgr,
            trajectories=trajectories,
            anchor_frame=anchor,
            target_frames=target_frames,
            out_file=out_dir / "overlap_flow_on_anchor.png",
            line_thickness=args.flow_line_thickness,
            point_radius=args.point_radius,
        )
        pairwise_dir = out_dir / "pairwise_anchor_to_target"
        draw_pairwise_anchor_to_target(
            reader=reader,
            frame_id_to_reader_idx=frame_id_to_reader_idx,
            trajectories=trajectories,
            anchor_frame=anchor,
            target_frames=target_frames,
            out_dir=pairwise_dir,
            line_thickness=args.flow_line_thickness,
            point_radius=args.point_radius,
        )
        continuous_dir = out_dir / "continuous_anchor_to_target"
        for target_frame in target_frames:
            draw_continuous_flow_on_anchor(
                anchor_bgr=anchor_bgr,
                trajectories=trajectories,
                anchor_frame=anchor,
                target_frame=target_frame,
                out_file=continuous_dir
                / f"continuous_anchor_{anchor:06d}_to_{target_frame:06d}.png",
                line_thickness=args.flow_line_thickness,
                point_radius=args.point_radius,
            )
        print(f"[Info] Saved overlap and pairwise flow visualizations to: {out_dir}")
    else:
        print("[Warn] No target frames selected for overlap/pairwise outputs.")

    print("[Done] Multi-frame tracking flow export completed.")


if __name__ == "__main__":
    main()
