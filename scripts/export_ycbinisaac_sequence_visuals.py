#!/usr/bin/env python3
"""
Export YCBInIsaac sequence data and tracking visualizations.

This script saves:
  1) RGB frame sequence
  2) Segmentation masks and segmentation overlays
  3) Depth images (raw millimeters + colored)
  4) Point tracking / pose tracking result images copied from an existing run
  5) A sampled pose graph visualization (3 sampled frames by default) with sparse lines
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root for local imports.
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point2pose.io.sources.dataset.datareader import YCBInIsaacReader
from point2pose.utils.visualization import draw_xyz_axis


TrajectoryDict = Dict[int, Tuple[np.ndarray, np.ndarray]]


def _is_valid_sequence_dir(path: Path) -> bool:
    return path.is_dir() and (path / "rgb").is_dir() and (path / "cam_K.txt").is_file()


def resolve_video_path(dataset_root: str, video_name: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    direct = root
    nested = root / video_name
    doubly_nested = root / video_name / video_name

    candidates = [direct, nested, doubly_nested]
    for cand in candidates:
        if _is_valid_sequence_dir(cand):
            return cand

    raise FileNotFoundError(
        "Could not resolve sequence path. Checked: " f"{[str(c) for c in candidates]}"
    )


def distinct_bgr_colors(n: int) -> List[Tuple[int, int, int]]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        hue = int(round((179.0 * i) / max(1, n)))
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 1e-6)
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    depth_vals = depth_m[valid]
    d_min = float(np.percentile(depth_vals, 2))
    d_max = float(np.percentile(depth_vals, 98))
    if d_max <= d_min + 1e-6:
        d_max = d_min + 1e-6

    norm = np.clip((depth_m - d_min) / (d_max - d_min), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    return depth_color


def build_segmentation_overlay(
    rgb: np.ndarray, masks: List[np.ndarray], object_names: List[str]
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    colors = distinct_bgr_colors(len(masks))

    for i, mask in enumerate(masks):
        if i >= len(colors):
            break
        mask_bool = np.asarray(mask) > 0
        if not np.any(mask_bool):
            continue
        color_vec = np.array(colors[i], dtype=np.float32)
        src = overlay[mask_bool].astype(np.float32)
        blended = 0.55 * src + 0.45 * color_vec
        overlay[mask_bool] = blended.astype(np.uint8)

    for i, obj_name in enumerate(object_names):
        if i >= len(colors):
            break
        y = 30 + i * 22
        cv2.rectangle(overlay, (10, y - 12), (26, y + 4), colors[i], thickness=-1)
        cv2.putText(
            overlay,
            obj_name,
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def build_segmentation_overlay_pro(
    rgb: np.ndarray,
    masks: List[np.ndarray],
    object_names: List[str],
    fill_alpha: float = 0.22,
    contour_thickness: int = 2,
    draw_fill: bool = True,
    draw_labels: bool = True,
    draw_legend: bool = False,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    colors = distinct_bgr_colors(len(masks))  # you already have this

    # --- 1) light fills ---
    if draw_fill:
        for i, mask in enumerate(masks):
            if i >= len(colors):
                break
            m = np.asarray(mask) > 0
            if not np.any(m):
                continue
            color = np.array(colors[i], dtype=np.float32)
            src = out[m].astype(np.float32)
            out[m] = ((1.0 - fill_alpha) * src + fill_alpha * color).astype(np.uint8)

    # --- 2) crisp contours (halo + colored edge) ---
    for i, mask in enumerate(masks):
        if i >= len(colors):
            break
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.sum() == 0:
            continue

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # halo for contrast on any background
        cv2.drawContours(
            out, contours, -1, (0, 0, 0), contour_thickness + 2, cv2.LINE_AA
        )
        # colored contour
        cv2.drawContours(out, contours, -1, colors[i], contour_thickness, cv2.LINE_AA)

        # --- 3) on-object labels ---
        if draw_labels and i < len(object_names):
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                # place near top of object for fewer occlusions
                k = np.argmin(ys)
                x0, y0 = int(xs[k]), int(ys[k])

                text = object_names[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = 0.5
                th = 1
                (tw, th_text), baseline = cv2.getTextSize(text, font, fs, th)

                pad = 4
                x1 = max(0, x0)
                y1 = max(0, y0 - th_text - baseline - 2 * pad)
                x2 = min(out.shape[1] - 1, x1 + tw + 2 * pad)
                y2 = min(out.shape[0] - 1, y1 + th_text + baseline + 2 * pad)

                # label background
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                cv2.rectangle(out, (x1, y1), (x2, y2), colors[i], thickness=1)

                # text with outline
                tx, ty = x1 + pad, y2 - baseline - pad
                cv2.putText(out, text, (tx, ty), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    out, text, (tx, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA
                )

    # --- 4) optional compact legend ---
    if draw_legend:
        y = 24
        for i, name in enumerate(object_names[: len(colors)]):
            cv2.rectangle(out, (10, y - 10), (24, y + 4), (0, 0, 0), -1)
            cv2.rectangle(out, (10, y - 10), (24, y + 4), colors[i], 1)
            cv2.rectangle(out, (28, y - 13), (28 + 8 * len(name), y + 6), (0, 0, 0), -1)
            cv2.putText(
                out,
                name,
                (32, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22

    return out


def save_sequence_data(
    reader: YCBInIsaacReader, frame_ids: List[int], output_root: Path
) -> None:
    rgb_dir = output_root / "rgb_sequence"
    seg_overlay_dir = output_root / "segmentation_overlay"
    seg_mask_root = output_root / "segmentation_masks"
    depth_color_dir = output_root / "depth_colormap"

    for d in [rgb_dir, seg_overlay_dir, seg_mask_root, depth_color_dir]:
        d.mkdir(parents=True, exist_ok=True)

    object_names = reader.get_object_names()
    for obj_name in object_names:
        (seg_mask_root / obj_name).mkdir(parents=True, exist_ok=True)

    for frame_id in frame_ids:
        rgb = reader.get_color(frame_id)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_dir / f"frame_{frame_id:06d}.png"), bgr)

        masks = reader.get_masks(frame_id)
        seg_overlay = build_segmentation_overlay_pro(rgb, masks, object_names)
        cv2.imwrite(str(seg_overlay_dir / f"frame_{frame_id:06d}.png"), seg_overlay)

        for obj_name, mask in zip(object_names, masks):
            mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
            cv2.imwrite(
                str(seg_mask_root / obj_name / f"frame_{frame_id:06d}.png"), mask_u8
            )

        depth_m = reader.get_depth(frame_id)
        depth_color = colorize_depth(depth_m)
        cv2.imwrite(str(depth_color_dir / f"frame_{frame_id:06d}.png"), depth_color)


def copy_result_renders(
    result_root: Path, frame_ids: List[int], output_root: Path
) -> None:
    mappings = {
        "registration_correspondence": "point_tracking_result",
        "with_gt_registration_correspondence": "point_tracking_with_gt",
    }

    for source_name, target_name in mappings.items():
        src_dir = result_root / source_name
        if not src_dir.is_dir():
            continue

        dst_dir = output_root / target_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for frame_id in frame_ids:
            src_file = src_dir / f"frame_{frame_id:06d}.png"
            if not src_file.is_file():
                continue
            shutil.copy2(src_file, dst_dir / src_file.name)
            copied += 1

        # Fallback: copy all if frame ids do not match the result naming.
        if copied == 0:
            for src_file in sorted(src_dir.glob("*.png")):
                shutil.copy2(src_file, dst_dir / src_file.name)


def quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    x, y, z, w = q / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def load_pose_matrices_from_pose_logs(
    pose_log_dir: Path, num_objects: int
) -> Dict[int, np.ndarray]:
    """
    Read TUM pose logs (tx ty tz qx qy qz qw) into per-object (N,4,4) matrices.
    """
    out: Dict[int, np.ndarray] = {}
    if not pose_log_dir.is_dir():
        return out

    for obj_idx in range(num_objects):
        pose_file = pose_log_dir / f"obj_{obj_idx}_pose.txt"
        if not pose_file.is_file():
            continue

        poses = []
        with pose_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )

                pose = np.eye(4, dtype=float)
                pose[:3, :3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                pose[:3, 3] = np.asarray([tx, ty, tz], dtype=float)
                poses.append(pose)

        if len(poses) == 0:
            continue
        out[obj_idx] = np.asarray(poses, dtype=float)

    return out


def render_pose_tracking_axis_only(
    reader: YCBInIsaacReader,
    frame_ids: List[int],
    output_dir: Path,
    pose_source: str = "auto",
    pose_log_dir: Path | None = None,
    axis_scale: float = 0.08,
    axis_thickness: int = 2,
) -> str:
    """
    Save axis-only pose overlays (no bbox, no points).
    Returns the actually used pose source: "pose_log" or "gt".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    object_names = reader.get_object_names()
    pose_source = str(pose_source).strip().lower()

    logs_by_object: Dict[int, np.ndarray] = {}
    source_used = "gt"

    if pose_source in ("auto", "pose_log"):
        logs_by_object = load_pose_matrices_from_pose_logs(
            pose_log_dir=Path(pose_log_dir) if pose_log_dir is not None else Path(""),
            num_objects=len(object_names),
        )
        if len(logs_by_object) > 0:
            # Logged pose is relative to init object frame in this pipeline setup.
            # Compose with first-frame GT pose to place axes in camera coordinates.
            for obj_idx, obj_name in enumerate(object_names):
                mats = logs_by_object.get(obj_idx, None)
                if mats is None or mats.shape[0] == 0:
                    continue
                init_gt = reader.get_gt_pose(0, obj_name=obj_name)
                if init_gt is not None:
                    logs_by_object[obj_idx] = mats @ np.asarray(init_gt, dtype=float)
            source_used = "pose_log"
        elif pose_source == "pose_log":
            source_used = "pose_log"

    K = np.asarray(reader.K, dtype=float).reshape(3, 3)
    for frame_id in frame_ids:
        bgr = cv2.cvtColor(reader.get_color(frame_id), cv2.COLOR_RGB2BGR)
        for obj_idx, obj_name in enumerate(object_names):
            pose = None
            if source_used == "pose_log":
                mats = logs_by_object.get(obj_idx, None)
                if mats is not None and frame_id < mats.shape[0]:
                    pose = mats[frame_id]
            else:
                pose = reader.get_gt_pose(frame_id, obj_name=obj_name)

            if pose is None:
                continue

            bgr = draw_xyz_axis(
                image=bgr,
                ob_in_cam=np.asarray(pose, dtype=float).reshape(4, 4),
                scale=float(axis_scale),
                K=K,
                thickness=int(max(1, axis_thickness)),
                is_input_rgb=False,
            )

        cv2.imwrite(str(output_dir / f"frame_{frame_id:06d}.png"), bgr)

    return source_used


def load_translations_from_pose_logs(
    pose_log_dir: Path, num_objects: int
) -> TrajectoryDict:
    traj: TrajectoryDict = {}
    if not pose_log_dir.is_dir():
        return traj

    for obj_idx in range(num_objects):
        pose_file = pose_log_dir / f"obj_{obj_idx}_pose.txt"
        if not pose_file.is_file():
            continue

        xyz_list = []
        with pose_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                xyz_list.append([tx, ty, tz])

        if len(xyz_list) == 0:
            continue

        xyz = np.asarray(xyz_list, dtype=float)
        frames = np.arange(xyz.shape[0], dtype=int)
        traj[obj_idx] = (frames, xyz)
    return traj


def load_translations_from_gt(reader: YCBInIsaacReader) -> TrajectoryDict:
    traj: TrajectoryDict = {}
    object_names = reader.get_object_names()
    num_frames = len(reader)

    for obj_idx, obj_name in enumerate(object_names):
        frame_ids = []
        xyz = []
        for frame_id in range(num_frames):
            pose = reader.get_gt_pose(frame_id, obj_name=obj_name)
            if pose is None:
                continue
            frame_ids.append(frame_id)
            xyz.append(np.asarray(pose[:3, 3], dtype=float))

        if len(xyz) == 0:
            continue

        traj[obj_idx] = (np.asarray(frame_ids, dtype=int), np.asarray(xyz, dtype=float))
    return traj


def choose_pose_source(
    reader: YCBInIsaacReader, pose_source: str, pose_log_dir: Path
) -> Tuple[TrajectoryDict, str]:
    pose_source = pose_source.strip().lower()

    if pose_source in ("auto", "pose_log"):
        traj_logs = load_translations_from_pose_logs(
            pose_log_dir=pose_log_dir, num_objects=reader.num_objects
        )
        if len(traj_logs) > 0:
            return traj_logs, "pose_log"
        if pose_source == "pose_log":
            return {}, "pose_log"

    traj_gt = load_translations_from_gt(reader)
    return traj_gt, "gt"


def sample_evenly(n: int, sample_count: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    k = int(max(1, sample_count))
    if n <= k:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, num=k)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)
    return idx


def append_last_if_needed(arr_xyz: np.ndarray, full_xyz: np.ndarray) -> np.ndarray:
    if arr_xyz.shape[0] == 0:
        return arr_xyz
    if np.linalg.norm(arr_xyz[-1] - full_xyz[-1]) > 1e-12:
        return np.vstack([arr_xyz, full_xyz[-1]])
    return arr_xyz


def plot_sampled_pose_graph(
    trajectories: TrajectoryDict,
    object_names: List[str],
    output_path: Path,
    sample_count: int = 3,
    sparse_stride: int = 40,
    title: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    cmap = plt.get_cmap("tab10", max(1, len(object_names)))
    any_plotted = False

    for obj_idx, obj_name in enumerate(object_names):
        if obj_idx not in trajectories:
            continue

        frame_ids, xyz = trajectories[obj_idx]
        if xyz.shape[0] == 0:
            continue

        color = cmap(obj_idx)
        stride = max(1, int(sparse_stride))
        sparse_xyz = append_last_if_needed(xyz[::stride], xyz)

        # Similar to clustered-registration debug plots: show all keypoints/nodes
        # as a light background point cloud, then overlay graph edges/nodes.
        ax3d.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c="0.55",
            s=10,
            alpha=0.25,
            marker=".",
        )
        ax_top.scatter(
            xyz[:, 0],
            xyz[:, 2],
            c="0.55",
            s=10,
            alpha=0.25,
            marker=".",
        )

        ax3d.plot(
            sparse_xyz[:, 0],
            sparse_xyz[:, 1],
            sparse_xyz[:, 2],
            color=color,
            linewidth=1.6,
            alpha=0.35,
        )
        ax_top.plot(
            sparse_xyz[:, 0],
            sparse_xyz[:, 2],
            color=color,
            linewidth=1.6,
            alpha=0.35,
            label=obj_name,
        )

        sampled_local_idx = sample_evenly(xyz.shape[0], sample_count)
        sampled_xyz = xyz[sampled_local_idx]

        ax3d.plot(
            sampled_xyz[:, 0],
            sampled_xyz[:, 1],
            sampled_xyz[:, 2],
            color=color,
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=5,
        )
        ax_top.plot(
            sampled_xyz[:, 0],
            sampled_xyz[:, 2],
            color=color,
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=5,
        )

        any_plotted = True

    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D Pose Graph (all nodes + sampled graph)")
    ax3d.grid(True)

    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("z (m)")
    ax_top.set_title("Top View (all nodes + sampled graph)")
    ax_top.grid(True)
    if any_plotted:
        ax_top.legend(loc="best")

    fig.suptitle(title if title else "Pose Graph", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_frame_ids(
    total_frames: int, start: int, max_frames: int, step: int
) -> List[int]:
    if total_frames <= 0:
        return []
    start = max(0, min(int(start), total_frames - 1))
    step = max(1, int(step))
    if max_frames <= 0:
        end = total_frames
    else:
        end = min(total_frames, start + int(max_frames))
    return list(range(start, end, step))


def build_explicit_frame_ids(total_frames: int, frame_ids: List[int]) -> List[int]:
    if total_frames <= 0:
        return []
    out: List[int] = []
    seen = set()
    for fid in frame_ids:
        idx = int(fid)
        if idx < 0 or idx >= total_frames:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YCBInIsaac sequence visuals and sampled pose graph."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/justin/data/YCBMultiTrack_new",
        help="Dataset root path.",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="006_mustard_bottle_010_potted_meat_can_005_tomato_soup_can",
        help="Sequence folder name.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbinisaac_all",
        help="Root containing ycbinisaac result folders.",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default="",
        help="Result folder name. Empty uses --video_name.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/justin/code/point-to-pose/results/visualization_exports",
        help="Root output directory.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Number of frames to export. <=0 exports all frames from start.",
    )
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument(
        "--frame_ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit frame IDs to export (overrides --start_frame/--max_frames/--frame_step).",
    )
    parser.add_argument(
        "--pose_source",
        type=str,
        default="auto",
        choices=["auto", "pose_log", "gt"],
        help="Source for pose graph trajectories.",
    )
    parser.add_argument(
        "--pose_log_dir",
        type=str,
        default="/home/justin/code/point-to-pose/debug/poses",
        help="Directory containing obj_i_pose.txt files.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=3,
        help="Number of sampled nodes per object for pose graph.",
    )
    parser.add_argument(
        "--sparse_stride",
        type=int,
        default=40,
        help="Stride for sparse trajectory lines in pose graph.",
    )
    parser.add_argument(
        "--skip_copy_result_renders",
        action="store_true",
        help="Do not copy point-tracking renderings from existing result folders.",
    )
    parser.add_argument(
        "--axis_scale",
        type=float,
        default=0.08,
        help="Axis length (meters) for axis-only pose overlays.",
    )
    parser.add_argument(
        "--axis_thickness",
        type=int,
        default=2,
        help="Axis line thickness for axis-only pose overlays.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Delete output folder before writing new files.",
    )
    args = parser.parse_args()

    video_path = resolve_video_path(args.dataset_root, args.video_name)
    reader = YCBInIsaacReader(str(video_path))

    if args.frame_ids is not None and len(args.frame_ids) > 0:
        frame_ids = build_explicit_frame_ids(
            total_frames=len(reader), frame_ids=args.frame_ids
        )
    else:
        frame_ids = build_frame_ids(
            total_frames=len(reader),
            start=args.start_frame,
            max_frames=args.max_frames,
            step=args.frame_step,
        )
    if len(frame_ids) == 0:
        raise RuntimeError(
            "No frames selected. Check --frame_ids or --start_frame/--max_frames/--frame_step."
        )

    output_dir = Path(args.output_root).expanduser().resolve() / args.video_name
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Sequence path: {video_path}")
    print(f"[Info] Output path:   {output_dir}")
    print(
        f"[Info] Frames:        {len(frame_ids)} (first={frame_ids[0]}, last={frame_ids[-1]})"
    )
    print(f"[Info] Objects:       {reader.get_object_names()}")

    save_sequence_data(reader=reader, frame_ids=frame_ids, output_root=output_dir)
    print("[Info] Saved RGB / segmentation / depth exports.")

    pose_axis_source_used = render_pose_tracking_axis_only(
        reader=reader,
        frame_ids=frame_ids,
        output_dir=output_dir / "pose_tracking_result",
        pose_source=args.pose_source,
        pose_log_dir=Path(args.pose_log_dir).expanduser().resolve(),
        axis_scale=args.axis_scale,
        axis_thickness=args.axis_thickness,
    )
    print(
        f"[Info] Saved axis-only pose tracking result ({pose_axis_source_used}): "
        f"{output_dir / 'pose_tracking_result'}"
    )

    # Keep a GT-only axis overlay for quick reference.
    render_pose_tracking_axis_only(
        reader=reader,
        frame_ids=frame_ids,
        output_dir=output_dir / "pose_tracking_with_gt",
        pose_source="gt",
        pose_log_dir=Path(args.pose_log_dir).expanduser().resolve(),
        axis_scale=args.axis_scale,
        axis_thickness=args.axis_thickness,
    )
    print(
        f"[Info] Saved axis-only GT pose overlay: {output_dir / 'pose_tracking_with_gt'}"
    )

    if not args.skip_copy_result_renders:
        result_name = (
            args.result_name.strip() if args.result_name.strip() else args.video_name
        )
        result_dir = Path(args.results_root).expanduser().resolve() / result_name
        if result_dir.is_dir():
            copy_result_renders(
                result_root=result_dir, frame_ids=frame_ids, output_root=output_dir
            )
            print(f"[Info] Copied tracking renderings from: {result_dir}")
        else:
            print(f"[Warn] Result folder not found, skipping copy: {result_dir}")

    if args.pose_source == "gt":
        trajectories = load_translations_from_gt(reader)
        source_used = "gt"
    else:
        trajectories, source_used = choose_pose_source(
            reader=reader,
            pose_source=args.pose_source,
            pose_log_dir=Path(args.pose_log_dir).expanduser().resolve(),
        )

    if len(trajectories) == 0 and source_used == "pose_log":
        print("[Warn] No pose logs found. Falling back to GT poses for pose graph.")
        trajectories = load_translations_from_gt(reader)
        source_used = "gt"

    if len(trajectories) > 0:
        pose_graph_path = output_dir / "pose_graph" / "pose_graph_sampled.png"
        plot_sampled_pose_graph(
            trajectories=trajectories,
            object_names=reader.get_object_names(),
            output_path=pose_graph_path,
            sample_count=args.sample_count,
            sparse_stride=args.sparse_stride,
            title=f"{args.video_name} | pose source: {source_used}",
        )
        print(f"[Info] Saved pose graph: {pose_graph_path}")
    else:
        print("[Warn] Could not build pose graph (no trajectories found).")

    print("[Done] Export completed.")


if __name__ == "__main__":
    main()
