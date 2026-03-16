#!/usr/bin/env python3
"""
Overlay estimated mesh contour and visible-point trajectory trails, then export MP4.

This script combines:
  - mesh projection/contour overlay behavior from overlay_estimated_mesh_contour.py
  - metadata point projection/trail behavior from visualize_multiframe_tracking_flow.py

For each rendered frame, it draws trajectories for points that are visible in the
current frame over the last K frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.debug_visualization.overlay_estimated_mesh_contour import (  # noqa: E402
    _build_frame_id_to_reader_index,
    _compose_object_pose,
    _create_video_writer,
    _extract_frame_ids,
    _extract_init_pose_series,
    _load_frame_bgr,
    _load_mesh,
    _load_reader,
    _normalize_pose_series_length,
    _rasterize_silhouette,
    _resolve_dataset_mesh_path,
    _resolve_object_selection,
    _resolve_run_mesh_path,
    _select_pose_key,
)
from scripts.visualize_multiframe_tracking_flow import (  # noqa: E402
    _project_points_cam_to_image,
    _ragged_slice,
    _ragged_slice_2d,
    point_colors_bgr,
)


def _resolve_metadata_path(run_dir: str, explicit_meta_data_path: Optional[str]) -> str:
    if explicit_meta_data_path:
        path = os.path.abspath(explicit_meta_data_path)
    else:
        path = os.path.join(os.path.abspath(run_dir), "meta_data", "meta_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return path


def _resolve_point_source(npz_data, requested: str) -> str:
    if requested not in ("auto", "obj_key_points", "track2d"):
        raise ValueError(f"Unsupported point source: {requested}")

    if requested == "obj_key_points":
        if "obj_key_points_data" not in npz_data.files:
            raise KeyError("Requested --point_source=obj_key_points but obj_key_points is missing.")
        return "obj_key_points"

    if requested == "track2d":
        if "track2d_data" not in npz_data.files:
            raise KeyError("Requested --point_source=track2d but track2d is missing.")
        return "track2d"

    if "obj_key_points_data" in npz_data.files:
        return "obj_key_points"
    if "track2d_data" in npz_data.files:
        return "track2d"
    raise KeyError("No usable point source found in metadata (obj_key_points or track2d).")


def _max_point_capacity(npz_data, point_source: str, n_rows: int) -> int:
    if point_source == "obj_key_points":
        lengths_key = "obj_key_points_lengths"
        dim = 3
    else:
        lengths_key = "track2d_lengths"
        dim = 2

    if lengths_key in npz_data.files:
        lengths = np.asarray(npz_data[lengths_key]).reshape(-1)
        if lengths.size > 0:
            return int(max(0, np.max(lengths) // dim))

    max_points = 0
    base = "obj_key_points" if point_source == "obj_key_points" else "track2d"
    for row_idx in range(n_rows):
        arr = _ragged_slice_2d(npz_data, base, row_idx, dim=dim)
        if arr.shape[0] > max_points:
            max_points = int(arr.shape[0])
    return int(max_points)


def _ensure_capacity(
    tracks: np.ndarray,
    min_rows: int,
    n_cols: int,
) -> np.ndarray:
    if min_rows < tracks.shape[0]:
        return tracks
    new_rows = max(min_rows + 1, int(1.5 * max(1, tracks.shape[0])))
    out = np.full((new_rows, n_cols, 2), np.nan, dtype=np.float32)
    if tracks.shape[0] > 0:
        out[: tracks.shape[0], :, :] = tracks
    return out


def _build_visible_point_tracks(
    npz_data,
    frame_ids: np.ndarray,
    image_w: int,
    image_h: int,
    point_source: str,
    pose_series_composed: List[Optional[np.ndarray]],
    K: np.ndarray,
    use_obj_valid: bool,
    use_visibles: bool,
) -> np.ndarray:
    n_rows = int(frame_ids.shape[0])
    capacity = _max_point_capacity(npz_data=npz_data, point_source=point_source, n_rows=n_rows)
    tracks = np.full((capacity, n_rows, 2), np.nan, dtype=np.float32)

    for row_idx in range(n_rows):
        if point_source == "track2d":
            pts2d = _ragged_slice_2d(npz_data, "track2d", row_idx, dim=2)
            if pts2d.shape[0] == 0:
                continue

            ids = np.arange(pts2d.shape[0], dtype=np.int64)
            mask = np.isfinite(pts2d).all(axis=1)
            if use_visibles and "visibles_data" in npz_data.files:
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
            pose = pose_series_composed[row_idx] if row_idx < len(pose_series_composed) else None
            if pose is None:
                continue

            pts3d = _ragged_slice_2d(npz_data, "obj_key_points", row_idx, dim=3)
            if pts3d.shape[0] == 0:
                continue

            ids = np.arange(pts3d.shape[0], dtype=np.int64)
            mask = np.isfinite(pts3d).all(axis=1)
            if use_obj_valid and "obj_valid_data" in npz_data.files:
                obj_valid = _ragged_slice(npz_data, "obj_valid", row_idx).reshape(-1).astype(
                    bool, copy=False
                )
                if obj_valid.size == ids.size:
                    mask &= obj_valid

            if not np.any(mask):
                continue

            ids_sel = ids[mask]
            pts_sel = pts3d[mask]
            pts_h = np.concatenate(
                [pts_sel.astype(np.float64), np.ones((pts_sel.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            pts_cam = (np.asarray(pose, dtype=np.float64) @ pts_h.T).T[:, :3]
            uv, keep_local = _project_points_cam_to_image(
                points_cam=pts_cam,
                K=K,
                image_w=image_w,
                image_h=image_h,
            )
            if uv.shape[0] == 0:
                continue
            ids = ids_sel[keep_local]

        if ids.size == 0:
            continue
        max_id = int(np.max(ids))
        tracks = _ensure_capacity(tracks, min_rows=max_id, n_cols=n_rows)
        tracks[ids, row_idx, :] = uv.astype(np.float32, copy=False)

    return tracks


def _draw_visible_point_trajectories(
    image_bgr: np.ndarray,
    tracks: np.ndarray,
    row_idx: int,
    trail_len: int,
    colors_bgr: np.ndarray,
    point_radius: int,
    trail_thickness: int,
    allowed_point_mask: Optional[np.ndarray],
) -> int:
    if tracks.shape[0] == 0:
        return 0

    curr = tracks[:, row_idx, :]
    visible_mask = np.isfinite(curr[:, 0]) & np.isfinite(curr[:, 1])
    if allowed_point_mask is not None:
        visible_mask &= allowed_point_mask
    visible_ids = np.where(visible_mask)[0]
    if visible_ids.size == 0:
        return 0

    start = max(0, row_idx - int(max(1, trail_len)) + 1)
    n_drawn = 0
    for point_id in visible_ids.tolist():
        point_now = curr[point_id]
        color = tuple(int(x) for x in colors_bgr[point_id])

        trail = tracks[point_id, start : row_idx + 1, :]
        valid_trail = np.isfinite(trail[:, 0]) & np.isfinite(trail[:, 1])
        trail = trail[valid_trail]
        if trail.shape[0] >= 2:
            poly = np.round(trail).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                image_bgr,
                [poly],
                isClosed=False,
                color=color,
                thickness=max(1, int(trail_thickness)),
                lineType=cv2.LINE_AA,
            )

        cv2.circle(
            image_bgr,
            tuple(np.round(point_now).astype(int).tolist()),
            radius=max(1, int(point_radius)),
            color=color,
            thickness=-1,
        )
        n_drawn += 1

    return int(n_drawn)


def _select_consistent_point_subset(
    tracks: np.ndarray,
    downsample_ratio: float,
    max_visible_points: int,
    seed: int,
) -> np.ndarray:
    if tracks.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    if downsample_ratio <= 0.0 or downsample_ratio > 1.0:
        raise ValueError("--point_downsample_ratio must be in (0, 1].")

    valid_any = np.isfinite(tracks[:, :, 0]) & np.isfinite(tracks[:, :, 1])
    candidate_ids = np.where(np.any(valid_any, axis=1))[0]
    keep_mask = np.zeros((tracks.shape[0],), dtype=bool)
    if candidate_ids.size == 0:
        return keep_mask

    n_keep = int(np.floor(candidate_ids.size * float(downsample_ratio)))
    if n_keep <= 0:
        n_keep = 1
    if max_visible_points > 0:
        n_keep = min(n_keep, int(max_visible_points))
    n_keep = max(1, min(n_keep, candidate_ids.size))

    if n_keep == candidate_ids.size:
        chosen = candidate_ids
    else:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(candidate_ids.size, size=n_keep, replace=False)
        chosen = np.sort(candidate_ids[pick])

    keep_mask[chosen] = True
    return keep_mask


def run(args: argparse.Namespace) -> None:
    run_dir = os.path.abspath(args.run_dir)
    video_name = args.video_name if args.video_name else os.path.basename(run_dir.rstrip("/"))
    meta_data_path = _resolve_metadata_path(
        run_dir=run_dir, explicit_meta_data_path=args.meta_data_path
    )

    reader = _load_reader(args.dataset, os.path.abspath(args.data_root), video_name)
    object_idx, object_name = _resolve_object_selection(
        dataset=args.dataset,
        reader=reader,
        object_idx=int(args.object_idx),
        object_name=args.object_name,
    )

    with np.load(meta_data_path, allow_pickle=True) as npz_data:
        pose_key, pose_series_raw = _select_pose_key(
            npz_data=npz_data,
            dataset=args.dataset,
            pose_key=args.pose_key,
            object_idx=object_idx,
        )
        n_rows = (
            int(np.asarray(npz_data["frame_id"]).reshape(-1).shape[0])
            if "frame_id" in npz_data.files
            else len(pose_series_raw)
        )
        if n_rows <= 0:
            raise RuntimeError("No metadata rows found.")
        frame_ids = _extract_frame_ids(npz_data=npz_data, n_rows=n_rows)
        pose_series = _normalize_pose_series_length(pose_series_raw, n_rows=n_rows)
        init_pose_series, init_pose_key = _extract_init_pose_series(
            npz_data=npz_data,
            n_rows=n_rows,
            object_idx=object_idx,
            prefer_multi_object_key=pose_key.endswith("_all"),
        )
        point_source = _resolve_point_source(npz_data=npz_data, requested=args.point_source)

        pose_series_composed: List[Optional[np.ndarray]] = []
        for row_idx in range(n_rows):
            pose_obj = pose_series[row_idx] if row_idx < len(pose_series) else None
            if pose_obj is None:
                pose_series_composed.append(None)
                continue
            init_pose = init_pose_series[row_idx]
            pose_series_composed.append(
                _compose_object_pose(
                    pose_obj=pose_obj,
                    init_pose=init_pose,
                    mode=args.pose_compose_mode,
                )
            )

        if args.mesh_pose_mode == "pose_only":
            pose_series_mesh = pose_series
        else:
            pose_series_mesh = pose_series_composed

        if args.point_pose_mode == "pose_only":
            point_pose_series = pose_series
        else:
            point_pose_series = pose_series_mesh

        sample_bgr = _load_frame_bgr(reader=reader, dataset=args.dataset, frame_idx=0)
        image_h, image_w = sample_bgr.shape[:2]
        K = np.asarray(reader.K, dtype=np.float64).reshape(3, 3)
        tracks = _build_visible_point_tracks(
            npz_data=npz_data,
            frame_ids=frame_ids,
            image_w=image_w,
            image_h=image_h,
            point_source=point_source,
            pose_series_composed=point_pose_series,
            K=K,
            use_obj_valid=not bool(args.disable_obj_valid_filter),
            use_visibles=not bool(args.disable_visibles_filter),
        )

    if tracks.shape[0] == 0:
        raise RuntimeError("No point tracks were produced from metadata.")

    mesh_source = "dataset" if args.use_gt_mesh else args.mesh_source
    model_root_abs = os.path.abspath(args.model_root) if args.model_root else None
    explicit_mesh_path = os.path.abspath(args.mesh_path) if args.mesh_path else None
    if explicit_mesh_path is not None and not os.path.exists(explicit_mesh_path):
        raise FileNotFoundError(f"--mesh_path does not exist: {explicit_mesh_path}")

    mesh_path = explicit_mesh_path
    mesh_source_used = "explicit" if explicit_mesh_path else None
    if mesh_path is None and mesh_source in ("auto", "run"):
        mesh_path = _resolve_run_mesh_path(run_dir=run_dir, object_idx=object_idx)
        if mesh_path is not None:
            mesh_source_used = "run"
    if mesh_source == "run" and mesh_path is None:
        raise FileNotFoundError(
            f"Run mesh for object_idx={object_idx} not found under {os.path.join(run_dir, 'mesh')}"
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
    vertices_obj = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(run_dir, "mesh_visible_point_trajectory")
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.video_path:
        video_path = os.path.abspath(args.video_path)
    else:
        video_path = os.path.join(output_dir, f"{video_name}_mesh_visible_point_traj.mp4")

    line_color = (int(args.line_color[0]), int(args.line_color[1]), int(args.line_color[2]))
    frame_id_to_reader_idx = _build_frame_id_to_reader_index(reader)
    point_colors = point_colors_bgr(tracks.shape[0], seed=int(args.seed))
    allowed_point_mask = _select_consistent_point_subset(
        tracks=tracks,
        downsample_ratio=float(args.point_downsample_ratio),
        max_visible_points=int(args.max_visible_points),
        seed=int(args.seed),
    )
    selected_point_count = int(np.sum(allowed_point_mask))
    save_images = bool(args.save_frames) or bool(args.save_images_dir)
    frames_dir = (
        os.path.abspath(args.save_images_dir)
        if args.save_images_dir
        else os.path.join(output_dir, "frames")
    )
    if save_images:
        os.makedirs(frames_dir, exist_ok=True)

    writer: Optional[cv2.VideoWriter] = None
    writer_codec: Optional[str] = None
    writer_size: Optional[Tuple[int, int]] = None

    num_written = 0
    num_skipped = 0
    num_invalid_frame_id = 0
    num_duplicate = 0
    num_missing_pose = 0
    seen_frame_ids = set()

    print(f"[Info] Dataset: {args.dataset}")
    print(f"[Info] Sequence: {video_name}")
    print(f"[Info] Metadata: {meta_data_path}")
    print(f"[Info] Pose key used: {pose_key}")
    print(f"[Info] Init pose key used: {init_pose_key if init_pose_key else 'identity'}")
    print(f"[Info] Point source: {point_source}")
    print(f"[Info] Point pose mode: {args.point_pose_mode}")
    print(f"[Info] Mesh pose mode: {args.mesh_pose_mode}")
    print(f"[Info] Object: idx={object_idx}, name={object_name}")
    print(f"[Info] Pose compose mode: {args.pose_compose_mode}")
    print(f"[Info] Mesh source: {mesh_source_used}, mesh={mesh_path}")
    print(f"[Info] Trail length: {int(args.trail_len)}")
    print(
        f"[Info] Point subset: selected={selected_point_count} / total={tracks.shape[0]} "
        f"(ratio={float(args.point_downsample_ratio):.4f}, max_visible_points={int(args.max_visible_points)})"
    )
    print(f"[Info] Output dir: {output_dir}")
    print(f"[Info] Video path: {video_path} @ {float(args.video_fps):.3f} fps")
    print(f"[Info] Metadata rows: {len(frame_ids)}, point capacity: {tracks.shape[0]}")

    if save_images:
        print(f"[Info] Frame image output: {frames_dir}")

    for row_idx in range(0, len(frame_ids), max(1, int(args.stride))):
        frame_id = int(frame_ids[row_idx])
        if frame_id < int(args.start_frame):
            continue
        if int(args.end_frame) >= 0 and frame_id > int(args.end_frame):
            continue
        if (not bool(args.allow_duplicate_frame_ids)) and frame_id in seen_frame_ids:
            num_duplicate += 1
            num_skipped += 1
            continue

        reader_idx = frame_id_to_reader_idx.get(frame_id)
        if reader_idx is None and 0 <= frame_id < len(reader):
            reader_idx = frame_id
        if reader_idx is None or reader_idx < 0 or reader_idx >= len(reader):
            num_invalid_frame_id += 1
            num_skipped += 1
            continue

        pose = pose_series_mesh[row_idx] if row_idx < len(pose_series_mesh) else None
        if pose is None:
            num_missing_pose += 1
            num_skipped += 1
            continue

        frame_bgr = _load_frame_bgr(reader=reader, dataset=args.dataset, frame_idx=reader_idx)
        overlay = frame_bgr.copy()

        silhouette = _rasterize_silhouette(
            vertices_obj=vertices_obj,
            faces=faces,
            T_obj_in_cam=pose,
            K=K,
            image_h=overlay.shape[0],
            image_w=overlay.shape[1],
        )
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cv2.drawContours(
                overlay,
                contours,
                -1,
                color=line_color,
                thickness=max(1, int(args.line_thickness)),
                lineType=cv2.LINE_AA,
            )

        num_visible_now = _draw_visible_point_trajectories(
            image_bgr=overlay,
            tracks=tracks,
            row_idx=row_idx,
            trail_len=int(args.trail_len),
            colors_bgr=point_colors,
            point_radius=int(args.point_radius),
            trail_thickness=int(args.trail_thickness),
            allowed_point_mask=allowed_point_mask,
        )
        cv2.putText(
            overlay,
            (
                f"frame={frame_id} row={row_idx} "
                f"visible_now={num_visible_now} trail_len={int(args.trail_len)}"
            ),
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is None:
            h, w = overlay.shape[:2]
            writer_size = (int(w), int(h))
            writer, writer_codec = _create_video_writer(
                video_path=video_path,
                frame_size=writer_size,
                fps=float(args.video_fps),
            )
            if writer is None:
                raise RuntimeError(f"Failed to initialize video writer: {video_path}")
            print(f"[Info] Video codec: {writer_codec}")

        frame_to_write = overlay
        if writer_size is not None and (overlay.shape[1], overlay.shape[0]) != writer_size:
            frame_to_write = cv2.resize(overlay, writer_size, interpolation=cv2.INTER_LINEAR)
        writer.write(frame_to_write)

        if save_images:
            if bool(args.allow_duplicate_frame_ids):
                out_name = f"frame_{frame_id:06d}_row_{row_idx:06d}.png"
            else:
                out_name = f"frame_{frame_id:06d}.png"
            cv2.imwrite(os.path.join(frames_dir, out_name), overlay)

        seen_frame_ids.add(frame_id)
        num_written += 1

    if writer is not None:
        writer.release()

    if num_written == 0:
        raise RuntimeError("No frames were written. Check frame range and metadata alignment.")

    print(
        "[Done] wrote={} skipped={} missing_pose={} invalid_frame_id={} duplicate_frame={} "
        "video={}".format(
            num_written,
            num_skipped,
            num_missing_pose,
            num_invalid_frame_id,
            num_duplicate,
            video_path,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay projected mesh contour plus trajectory trails of currently visible points, "
            "then save as MP4."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ho3d", "ycbineoat", "ycbinisaac"],
        help="Dataset/reader mode.",
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
        help="Pose key (obj_pose_all, obj_pose, pose_local, pose_frontend, or auto).",
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=0,
        help="Object index for selecting per-object metadata entries.",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Optional object name (primarily for ycbinisaac).",
    )
    parser.add_argument(
        "--pose_compose_mode",
        type=str,
        default="post_multiply_init",
        choices=["post_multiply_init", "pre_multiply_init", "none"],
        help=(
            "How to combine pose and init pose: "
            "post_multiply_init => pose @ init (default), "
            "pre_multiply_init => init @ pose, none => pose."
        ),
    )
    parser.add_argument(
        "--mesh_pose_mode",
        type=str,
        default="compose",
        choices=["compose", "pose_only"],
        help=(
            "Pose used for mesh rendering. "
            "compose uses --pose_compose_mode with init pose (default), "
            "pose_only uses raw pose key directly."
        ),
    )

    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Explicit mesh path. If omitted, resolve from run mesh or dataset CAD mesh.",
    )
    parser.add_argument(
        "--mesh_source",
        type=str,
        default="auto",
        choices=["auto", "run", "dataset"],
        help="Mesh source when --mesh_path is not provided.",
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
        help="YCB model root (needed when dataset mesh fallback is used for YCB datasets).",
    )

    parser.add_argument(
        "--point_source",
        type=str,
        default="auto",
        choices=["auto", "obj_key_points", "track2d"],
        help=(
            "Point source for trails. obj_key_points projects 3D map points; "
            "track2d uses stored 2D tracks."
        ),
    )
    parser.add_argument(
        "--point_pose_mode",
        type=str,
        default="pose_only",
        choices=["pose_only", "match_mesh"],
        help=(
            "Pose used for obj_key_points projection. "
            "pose_only uses raw pose key directly (recommended), "
            "match_mesh uses mesh-composed pose (pose compose mode)."
        ),
    )
    parser.add_argument(
        "--disable_obj_valid_filter",
        action="store_true",
        help="Do not apply obj_valid filtering for obj_key_points.",
    )
    parser.add_argument(
        "--disable_visibles_filter",
        action="store_true",
        help="Do not apply visibles filtering for track2d.",
    )
    parser.add_argument(
        "--trail_len",
        type=int,
        default=20,
        help="Past K frames to draw for currently visible points.",
    )
    parser.add_argument(
        "--max_visible_points",
        type=int,
        default=600,
        help=(
            "Global max number of point IDs to render consistently across all frames. "
            "<=0 means no max cap."
        ),
    )
    parser.add_argument(
        "--point_downsample_ratio",
        type=float,
        default=1.0,
        help=(
            "Global downsample ratio for visualization point IDs in (0,1]. "
            "Selection is done once and reused across all frames."
        ),
    )
    parser.add_argument("--trail_thickness", type=int, default=2)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--allow_duplicate_frame_ids",
        action="store_true",
        help="Render duplicate frame IDs if they appear in metadata.",
    )

    parser.add_argument(
        "--line_color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=[0, 255, 0],
        help="Mesh contour color in BGR.",
    )
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save per-frame overlay PNGs.",
    )
    parser.add_argument(
        "--save_images_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-frame PNGs. If set, image saving is automatically enabled. "
            "Default when --save_frames is set: <output_dir>/frames"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
