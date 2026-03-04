#!/usr/bin/env python3
"""
Evaluate HO3D reconstructed mesh Chamfer distance with disconnected-part filtering.

This script:
1) loads predicted mesh from a finished run folder,
2) applies connected-component filtering (same logic as visualize_textured_mesh.py),
3) aligns using first valid predicted/GT pose pair from meta_data + dataset,
4) computes mesh Chamfer distance via point2pose.utils.mesh_eval.evaluate_reconstructed_mesh.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.sources.dataset.datareader import Ho3dReader  # noqa: E402
from point2pose.utils.mesh_eval import (  # noqa: E402
    evaluate_reconstructed_mesh,
    find_gt_visible_mesh_path,
)


def _to_pose_matrix(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _extract_frame_ids(npz_data, n_rows: int) -> np.ndarray:
    if "frame_id" not in npz_data:
        return np.arange(n_rows, dtype=np.int64)
    raw = np.asarray(npz_data["frame_id"]).reshape(-1)
    out = np.arange(n_rows, dtype=np.int64)
    n = min(n_rows, raw.shape[0])
    for i in range(n):
        try:
            out[i] = int(np.asarray(raw[i]).reshape(-1)[0])
        except Exception:
            out[i] = int(i)
    return out


def _extract_pose_series(
    npz_data, key: str, object_idx: int, frame_ids_raw: Optional[np.ndarray]
) -> List[Optional[np.ndarray]]:
    if key not in npz_data:
        raise KeyError(f"Missing key: {key}")
    arr = npz_data[key]

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 2 and arr.shape == (4, 4):
            return [_to_pose_matrix(arr)] if object_idx == 0 else [None]
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            if object_idx != 0:
                return [None for _ in range(arr.shape[0])]
            return [_to_pose_matrix(arr[i]) for i in range(arr.shape[0])]
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
        raise ValueError(
            f"Unsupported pose shape for key={key}: shape={arr.shape}, dtype={arr.dtype}"
        )

    arr_obj = np.asarray(arr, dtype=object).reshape(-1)
    out: List[Optional[np.ndarray]] = []
    for entry in arr_obj:
        if isinstance(entry, (list, tuple)):
            if object_idx < len(entry):
                out.append(_to_pose_matrix(entry[object_idx]))
            else:
                out.append(None)
            continue
        out.append(_to_pose_matrix(entry))
    return out


def _select_pose_key(npz_data, pose_key: str, object_idx: int) -> Tuple[str, List[Optional[np.ndarray]]]:
    frame_ids_raw = npz_data["frame_id"] if "frame_id" in npz_data else None
    if pose_key != "auto":
        return pose_key, _extract_pose_series(
            npz_data=npz_data, key=pose_key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )

    for key in ("obj_pose", "obj_pose_all", "pose_local", "pose_frontend"):
        if key not in npz_data:
            continue
        series = _extract_pose_series(
            npz_data=npz_data, key=key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        if any(p is not None for p in series):
            return key, series
    raise KeyError("No usable pose key found (tried obj_pose, obj_pose_all, pose_local, pose_frontend).")


def _extract_init_pose_series(
    npz_data, n_rows: int, object_idx: int, prefer_all: bool
) -> Tuple[List[np.ndarray], Optional[str]]:
    key_candidates = ["obj_init_pose_all", "obj_init_pose"] if prefer_all else ["obj_init_pose", "obj_init_pose_all"]
    frame_ids_raw = npz_data["frame_id"] if "frame_id" in npz_data else None

    init_series: Optional[List[Optional[np.ndarray]]] = None
    used_key = None
    for key in key_candidates:
        if key not in npz_data:
            continue
        series = _extract_pose_series(
            npz_data=npz_data, key=key, object_idx=object_idx, frame_ids_raw=frame_ids_raw
        )
        if len(series) > 0:
            init_series = series
            used_key = key
            break

    if init_series is None:
        return [np.eye(4, dtype=np.float64) for _ in range(n_rows)], None

    if len(init_series) == 1 and n_rows > 1 and init_series[0] is not None:
        return [init_series[0].copy() for _ in range(n_rows)], used_key

    out: List[np.ndarray] = []
    for i in range(n_rows):
        if i < len(init_series) and init_series[i] is not None:
            out.append(init_series[i])
        else:
            out.append(np.eye(4, dtype=np.float64))
    return out, used_key


def _compose_pose(pose_obj: np.ndarray, init_pose: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pose_obj
    if mode == "pre_multiply_init":
        return init_pose @ pose_obj
    return pose_obj @ init_pose


def _resolve_mesh_path(run_dir: str, object_idx: int, mesh_path: Optional[str]) -> str:
    if mesh_path:
        path = os.path.abspath(mesh_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--mesh_path does not exist: {path}")
        return path

    mesh_dir = os.path.join(run_dir, "mesh")
    candidates = [
        os.path.join(mesh_dir, f"pred_mesh_obj_{object_idx}_textured.glb"),
        os.path.join(mesh_dir, f"pred_mesh_obj_{object_idx}.ply"),
        os.path.join(mesh_dir, f"pred_mesh_obj_{object_idx}.obj"),
        os.path.join(mesh_dir, f"pred_mesh_obj_{object_idx}.stl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    wildcard = sorted(Path(mesh_dir).glob(f"pred_mesh_obj_{object_idx}*"))
    for path in wildcard:
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        f"Could not find predicted mesh for object_idx={object_idx} under {mesh_dir}"
    )


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()


def filter_disconnected_components(
    mesh: o3d.geometry.TriangleMesh,
    keep_components: int = 1,
    min_component_triangles: int = 0,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, int]]:
    if keep_components < 1:
        keep_components = 1
    if min_component_triangles < 0:
        min_component_triangles = 0

    labels, triangle_counts, _ = mesh.cluster_connected_triangles()
    labels_np = np.asarray(labels, dtype=np.int64)
    counts_np = np.asarray(triangle_counts, dtype=np.int64)
    if counts_np.size == 0 or labels_np.size == 0:
        return mesh, {
            "num_components": int(counts_np.size),
            "kept_components": int(counts_np.size),
            "removed_triangles": 0,
            "kept_triangles": int(len(mesh.triangles)),
        }

    order = np.argsort(-counts_np)
    keep_ids = order[: int(keep_components)]
    if min_component_triangles > 0:
        keep_ids = np.asarray(
            [
                cid
                for cid in keep_ids.tolist()
                if int(counts_np[cid]) >= int(min_component_triangles)
            ],
            dtype=np.int64,
        )
        if keep_ids.size == 0:
            keep_ids = np.asarray([int(order[0])], dtype=np.int64)

    keep_mask = np.isin(labels_np, keep_ids)
    remove_idx = np.flatnonzero(~keep_mask).astype(np.int64)
    if remove_idx.size == 0:
        return mesh, {
            "num_components": int(counts_np.size),
            "kept_components": int(keep_ids.size),
            "removed_triangles": 0,
            "kept_triangles": int(np.count_nonzero(keep_mask)),
        }

    mesh_filtered = copy.deepcopy(mesh)
    mesh_filtered.remove_triangles_by_index(remove_idx.tolist())
    _cleanup_mesh(mesh_filtered)
    return mesh_filtered, {
        "num_components": int(counts_np.size),
        "kept_components": int(keep_ids.size),
        "removed_triangles": int(remove_idx.size),
        "kept_triangles": int(len(mesh_filtered.triangles)),
    }


def _load_mesh_o3d(mesh_path: str) -> o3d.geometry.TriangleMesh:
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    except TypeError:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Failed to load mesh or empty mesh: {mesh_path}")
    return mesh


def _resolve_first_alignment_pair(
    meta_data_path: str,
    reader: Ho3dReader,
    pose_key: str,
    object_idx: int,
    pose_compose_mode: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    with np.load(meta_data_path, allow_pickle=True) as npz_data:
        used_pose_key, pose_series = _select_pose_key(
            npz_data=npz_data, pose_key=pose_key, object_idx=object_idx
        )
        n_rows = len(pose_series)
        frame_ids = _extract_frame_ids(npz_data, n_rows)
        init_series, used_init_key = _extract_init_pose_series(
            npz_data=npz_data,
            n_rows=n_rows,
            object_idx=object_idx,
            prefer_all=used_pose_key.endswith("_all"),
        )

    for row_idx in range(n_rows):
        frame_id = int(frame_ids[row_idx])
        if frame_id < 0 or frame_id >= len(reader):
            continue
        pose_obj = pose_series[row_idx]
        if pose_obj is None:
            continue
        pred_pose_first = _compose_pose(
            pose_obj=pose_obj, init_pose=init_series[row_idx], mode=pose_compose_mode
        )
        gt_pose_first = reader.get_gt_pose(frame_id)
        if gt_pose_first is None:
            continue
        print(
            f"[Info] First alignment frame={frame_id}, row={row_idx}, "
            f"pose_key={used_pose_key}, init_key={used_init_key if used_init_key else 'identity'}"
        )
        return (
            np.asarray(pred_pose_first, dtype=np.float64),
            np.asarray(gt_pose_first, dtype=np.float64),
            frame_id,
        )

    raise RuntimeError("Could not find a valid predicted/GT first-pose alignment pair.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HO3D mesh Chamfer distance with disconnected-component filtering."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="HO3D root (contains evaluation/<video_name>/...).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Result run folder (contains mesh/ and meta_data/meta_data.npz).",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="HO3D sequence name (default: basename of --run_dir).",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Optional explicit predicted mesh path.",
    )
    parser.add_argument(
        "--meta_data_path",
        type=str,
        default=None,
        help="Optional explicit meta_data.npz path (default: <run_dir>/meta_data/meta_data.npz).",
    )
    parser.add_argument("--object_idx", type=int, default=0)
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key: auto/obj_pose/obj_pose_all/pose_local/pose_frontend.",
    )
    parser.add_argument(
        "--pose_compose_mode",
        type=str,
        default="post_multiply_init",
        choices=["post_multiply_init", "pre_multiply_init", "none"],
        help="How to compose predicted pose with init pose.",
    )
    parser.add_argument(
        "--keep_components",
        type=int,
        default=1,
        help="Number of largest connected components to keep.",
    )
    parser.add_argument(
        "--min_component_triangles",
        type=int,
        default=0,
        help="Drop kept components below this triangle count.",
    )
    parser.add_argument(
        "--skip_filter_disconnected",
        action="store_true",
        help="Disable disconnected-component filtering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for intermediate files and logs (default: <run_dir>/mesh_eval).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    video_name = args.video_name if args.video_name else os.path.basename(run_dir.rstrip("/"))
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(run_dir, "mesh_eval")
    )
    os.makedirs(output_dir, exist_ok=True)

    meta_data_path = (
        os.path.abspath(args.meta_data_path)
        if args.meta_data_path
        else os.path.join(run_dir, "meta_data", "meta_data.npz")
    )
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"meta_data.npz not found: {meta_data_path}")

    video_dir = os.path.join(os.path.abspath(args.data_root), "evaluation", video_name)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"HO3D sequence directory not found: {video_dir}")
    reader = Ho3dReader(video_dir, os.path.abspath(args.data_root))

    pred_mesh_path = _resolve_mesh_path(
        run_dir=run_dir, object_idx=int(args.object_idx), mesh_path=args.mesh_path
    )
    print(f"[Info] Sequence: {video_name}")
    print(f"[Info] Meta data: {meta_data_path}")
    print(f"[Info] Pred mesh: {pred_mesh_path}")

    gt_visible_mesh_path = find_gt_visible_mesh_path(reader.video_dir)
    if gt_visible_mesh_path is None:
        raise FileNotFoundError(f"GT visible mesh not found under {reader.video_dir}")
    print(f"[Info] GT visible mesh: {gt_visible_mesh_path}")

    pred_pose_first, gt_pose_first, align_frame_id = _resolve_first_alignment_pair(
        meta_data_path=meta_data_path,
        reader=reader,
        pose_key=args.pose_key,
        object_idx=int(args.object_idx),
        pose_compose_mode=args.pose_compose_mode,
    )
    print(f"[Info] Alignment frame: {align_frame_id}")

    mesh_for_eval_path = pred_mesh_path
    if not args.skip_filter_disconnected:
        mesh_o3d = _load_mesh_o3d(pred_mesh_path)
        mesh_filtered, stats = filter_disconnected_components(
            mesh_o3d,
            keep_components=int(args.keep_components),
            min_component_triangles=int(args.min_component_triangles),
        )
        filtered_path = os.path.join(
            output_dir, f"pred_mesh_obj_{int(args.object_idx)}_filtered.obj"
        )
        ok = o3d.io.write_triangle_mesh(filtered_path, mesh_filtered)
        if not ok:
            raise RuntimeError(f"Failed to save filtered mesh: {filtered_path}")
        mesh_for_eval_path = filtered_path
        print(
            "[Info] Disconnected filter: "
            f"components={stats['num_components']} kept={stats['kept_components']} "
            f"removed_triangles={stats['removed_triangles']} kept_triangles={stats['kept_triangles']}"
        )
        print(f"[Info] Filtered mesh: {filtered_path}")

    mesh_cd_cm = evaluate_reconstructed_mesh(
        pred_mesh_path=mesh_for_eval_path,
        gt_visible_mesh_path=gt_visible_mesh_path,
        output_dir=output_dir,
        mesh_prefix=video_name,
        pred_pose_first=pred_pose_first,
        gt_pose_first=gt_pose_first,
    )

    if np.isfinite(mesh_cd_cm):
        print(f"[Result] HO3D mesh Chamfer distance: {mesh_cd_cm:.4f} cm")
    else:
        print("[Result] HO3D mesh Chamfer distance: skipped/invalid")

    txt_path = os.path.join(output_dir, "mesh_cd_result.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"sequence: {video_name}\n")
        f.write(f"mesh_path_input: {pred_mesh_path}\n")
        f.write(f"mesh_path_eval: {mesh_for_eval_path}\n")
        f.write(f"gt_visible_mesh_path: {gt_visible_mesh_path}\n")
        f.write(f"alignment_frame_id: {align_frame_id}\n")
        f.write(f"mesh_cd_cm: {mesh_cd_cm}\n")
    print(f"[Info] Saved result: {txt_path}")


if __name__ == "__main__":
    main()
