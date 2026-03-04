#!/usr/bin/env python3
"""
Visualize how object points are tracked from one anchor image across multiple frames.

Outputs:
  - tracking_sequence: per-frame overlays with tracked points and short trajectories
  - overlap_flow_on_anchor.png: one anchor image overlaid with flows to multiple frames
  - pairwise_anchor_to_target: side-by-side anchor/target flow visualizations
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Add project root for local imports.
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point2pose.io.sources.dataset.datareader import YCBInIsaacReader


def _is_valid_sequence_dir(path: Path) -> bool:
    return path.is_dir() and (path / "rgb").is_dir() and (path / "cam_K.txt").is_file()


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


def get_tracking_mask(
    reader: YCBInIsaacReader, anchor_frame: int, object_name: str
) -> np.ndarray:
    if object_name:
        names = reader.get_object_names()
        if object_name not in names:
            raise ValueError(f"Object '{object_name}' not found. Available: {names}")
        mask = reader.get_mask(anchor_frame, obj_name=object_name)
        return (np.asarray(mask) > 0).astype(np.uint8) * 255

    masks = reader.get_masks(anchor_frame)
    if len(masks) == 0:
        h, w = reader.get_color(anchor_frame).shape[:2]
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
        return points.reshape(-1, 2).astype(np.float32)

    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        ys, xs = np.where(np.ones_like(gray, dtype=bool))
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    n = min(int(max_points), int(ys.size))
    rng = np.random.default_rng(seed)
    pick = rng.choice(ys.size, size=n, replace=False)
    pts = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)
    return pts


def track_points_forward(
    reader: YCBInIsaacReader,
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

    prev_gray = cv2.cvtColor(reader.get_color(anchor_frame), cv2.COLOR_RGB2GRAY)
    for frame_id in range(anchor_frame + 1, end_frame + 1):
        offset = frame_id - anchor_frame
        next_gray = cv2.cvtColor(reader.get_color(frame_id), cv2.COLOR_RGB2GRAY)

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
    reader: YCBInIsaacReader,
    trajectories: np.ndarray,
    anchor_frame: int,
    end_frame: int,
    out_dir: Path,
    trail_len: int = 25,
    trail_thickness: int = 2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = trajectories.shape[0]
    colors = point_colors_bgr(n, seed=42)

    for frame_id in range(anchor_frame, end_frame + 1):
        offset = frame_id - anchor_frame
        frame_bgr = cv2.cvtColor(reader.get_color(frame_id), cv2.COLOR_RGB2BGR)

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
            cv2.circle(frame_bgr, center, radius=2, color=color, thickness=-1)

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
            cv2.line(canvas, pt0, pt1, color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pt1, radius=2, color=color, thickness=-1)

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
        cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=1)
        cv2.circle(canvas, tuple(poly[-1, 0].tolist()), radius=2, color=color, thickness=-1)
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
    reader: YCBInIsaacReader,
    trajectories: np.ndarray,
    anchor_frame: int,
    target_frames: List[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_bgr = cv2.cvtColor(reader.get_color(anchor_frame), cv2.COLOR_RGB2BGR)
    ah, aw = anchor_bgr.shape[:2]
    gap = 40

    anchor_pts = trajectories[:, 0, :]
    frame_colors = distinct_frame_colors(len(target_frames))

    for i, target_frame in enumerate(target_frames):
        target_bgr = cv2.cvtColor(reader.get_color(target_frame), cv2.COLOR_RGB2BGR)
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
            cv2.line(canvas, pt0, pt1, color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pt0, radius=2, color=(255, 255, 255), thickness=-1)
            cv2.circle(canvas, pt1, radius=2, color=color, thickness=-1)

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
        description="Generate multi-frame point tracking flow from one anchor image."
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
        "--object_name",
        type=str,
        default="",
        help="Object name for mask-restricted sampling. Empty means union mask.",
    )
    parser.add_argument("--max_points", type=int, default=300)
    parser.add_argument("--quality_level", type=float, default=0.01)
    parser.add_argument("--min_distance", type=float, default=7.0)
    parser.add_argument("--trail_len", type=int, default=25)
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
    reader = YCBInIsaacReader(str(video_path))
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

    print(f"[Info] Sequence path: {video_path}")
    print(f"[Info] Output path:   {out_dir}")
    print(f"[Info] Anchor frame:  {anchor}")
    print(f"[Info] End frame:     {end_frame}")

    anchor_rgb = reader.get_color(anchor)
    anchor_gray = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2GRAY)
    track_mask = get_tracking_mask(reader, anchor, args.object_name.strip())
    init_points = sample_anchor_points(
        gray=anchor_gray,
        mask_u8=track_mask,
        max_points=args.max_points,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        seed=0,
    )
    if init_points.shape[0] == 0:
        raise RuntimeError("No anchor points sampled; cannot run tracking flow.")
    print(f"[Info] Initial sampled points: {init_points.shape[0]}")

    # Save sampled anchor points preview.
    anchor_preview = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
    for p in init_points:
        cv2.circle(
            anchor_preview,
            tuple(np.round(p).astype(int).tolist()),
            radius=2,
            color=(0, 255, 255),
            thickness=-1,
        )
    cv2.imwrite(str(out_dir / "anchor_sampled_points.png"), anchor_preview)

    trajectories = track_points_forward(
        reader=reader,
        anchor_frame=anchor,
        end_frame=end_frame,
        init_points=init_points,
    )
    print("[Info] Point tracking complete.")

    sequence_dir = out_dir / "tracking_sequence"
    draw_tracking_sequence(
        reader=reader,
        trajectories=trajectories,
        anchor_frame=anchor,
        end_frame=end_frame,
        out_dir=sequence_dir,
        trail_len=args.trail_len,
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
        anchor_bgr = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
        draw_overlap_flow_on_anchor(
            anchor_bgr=anchor_bgr,
            trajectories=trajectories,
            anchor_frame=anchor,
            target_frames=target_frames,
            out_file=out_dir / "overlap_flow_on_anchor.png",
        )
        pairwise_dir = out_dir / "pairwise_anchor_to_target"
        draw_pairwise_anchor_to_target(
            reader=reader,
            trajectories=trajectories,
            anchor_frame=anchor,
            target_frames=target_frames,
            out_dir=pairwise_dir,
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
            )
        print(f"[Info] Saved overlap and pairwise flow visualizations to: {out_dir}")
    else:
        print("[Warn] No target frames selected for overlap/pairwise outputs.")

    print("[Done] Multi-frame tracking flow export completed.")


if __name__ == "__main__":
    main()
