#!/usr/bin/env python3
"""
Run overlay_estimated_mesh_contour.py across all finished sequence folders.

Examples
--------
# ycbinisaac (also accepts alias ycb_inisaac)
python3 scripts/debug_visualization/run_overlay_estimated_mesh_contour_all.py \
    --dataset ycb_inisaac \
    --run_root /home/justin/results/eccv_point2pose/final_results/ycb_multi_track_final \
    --data_root /home/justin/data/YCBMultiTrack_new \
    --model_root /home/justin/data/HO3D_V3/models \
    --mesh_source dataset \
    --show_axis --axis_scale 0.08 --axis_thickness 5 --line_thickness 5

# ycbineoat
python3 scripts/debug_visualization/run_overlay_estimated_mesh_contour_all.py \
    --dataset ycbineoat \
    --run_root /home/justin/results/eccv_point2pose/final_results/ycbineoat_all_final \
    --data_root /home/justin/data/YCBInEOAT \
    --model_root /home/justin/data/HO3D_V3/models \
    --mesh_source dataset \
    --show_axis --axis_scale 0.08 --axis_thickness 5 --line_thickness 5
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _normalize_dataset(dataset: str) -> str:
    key = dataset.strip().lower().replace("-", "").replace("_", "")
    if key == "ycbinisaac":
        return "ycbinisaac"
    if key == "ycbineoat":
        return "ycbineoat"
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Use one of: ycbinisaac, ycb_inisaac, ycbineoat."
    )


def _discover_run_dirs(run_root: Path) -> List[Path]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    run_dirs = []
    for child in sorted(run_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        meta_path = child / "meta_data" / "meta_data.npz"
        if meta_path.is_file():
            run_dirs.append(child)
    return run_dirs


def _build_command(args: argparse.Namespace, overlay_script: Path, run_dir: Path) -> List[str]:
    cmd = [
        args.python_exe,
        str(overlay_script),
        "--dataset",
        args.dataset,
        "--run_dir",
        str(run_dir),
        "--data_root",
        str(args.data_root),
        "--mesh_source",
        args.mesh_source,
        "--model_root",
        str(args.model_root),
        "--pose_key",
        args.pose_key,
        "--line_thickness",
        str(args.line_thickness),
        "--axis_scale",
        str(args.axis_scale),
        "--axis_thickness",
        str(args.axis_thickness),
        "--video_fps",
        str(args.video_fps),
        "--start_frame",
        str(args.start_frame),
        "--end_frame",
        str(args.end_frame),
        "--stride",
        str(args.stride),
        "--pose_compose_mode",
        args.pose_compose_mode,
    ]

    if args.show_axis:
        cmd.append("--show_axis")
    if args.no_video:
        cmd.append("--no_video")
    if args.allow_duplicate_frame_ids:
        cmd.append("--allow_duplicate_frame_ids")
    if args.gt_pose_only:
        cmd.append("--gt_pose_only")
    if args.strict_gt_frame_id_match:
        cmd.append("--strict_gt_frame_id_match")

    if args.object_name:
        cmd.extend(["--object_name", args.object_name])
    else:
        cmd.extend(["--object_idx", str(args.object_idx)])

    if args.line_color is not None:
        cmd.extend(["--line_color", *[str(v) for v in args.line_color]])

    if args.output_root is not None:
        seq_out_dir = args.output_root / run_dir.name
        cmd.extend(["--output_dir", str(seq_out_dir)])

    return cmd


def run_all(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    overlay_script = script_dir / "overlay_estimated_mesh_contour.py"
    if not overlay_script.is_file():
        raise FileNotFoundError(f"Overlay script not found: {overlay_script}")

    run_dirs = _discover_run_dirs(args.run_root)
    if len(run_dirs) == 0:
        raise RuntimeError(
            f"No sequence run dirs with meta_data/meta_data.npz found under: {args.run_root}"
        )
    if args.max_sequences > 0:
        run_dirs = run_dirs[: args.max_sequences]

    print(f"[Info] Dataset: {args.dataset}")
    print(f"[Info] Run root: {args.run_root}")
    print(f"[Info] Data root: {args.data_root}")
    print(f"[Info] Model root: {args.model_root}")
    if args.output_root is not None:
        print(f"[Info] Output root: {args.output_root}")
    print(f"[Info] Num sequences: {len(run_dirs)}")
    if args.dry_run:
        print("[Info] Dry run enabled: commands will be printed but not executed.")

    num_ok = 0
    num_fail = 0
    failed = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        seq_name = run_dir.name
        print()
        print(f"[{idx}/{len(run_dirs)}] {seq_name}")
        cmd = _build_command(args, overlay_script, run_dir)
        if args.dry_run:
            print("[Command]", " ".join(cmd))
            num_ok += 1
            continue

        ret = subprocess.run(cmd, check=False)
        if ret.returncode == 0:
            num_ok += 1
            print(f"[OK] {seq_name}")
            continue

        num_fail += 1
        failed.append(seq_name)
        print(f"[Fail] {seq_name} (rc={ret.returncode})")
        if args.stop_on_error:
            break

    print()
    print(f"[Done] success={num_ok} failed={num_fail}")
    if len(failed) > 0:
        print("[Failed Sequences]")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run contour/3D-bbox overlay visualization over all sequence run directories."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name: ycbinisaac (or ycb_inisaac), ycbineoat.",
    )
    parser.add_argument(
        "--run_root",
        required=True,
        type=Path,
        help="Root dir containing sequence run folders.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        type=Path,
        help="Dataset root directory for reader input.",
    )
    parser.add_argument(
        "--model_root",
        required=True,
        type=Path,
        help="YCB model root directory (dataset-mesh lookup).",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable used to call overlay_estimated_mesh_contour.py.",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key passed through to overlay script.",
    )
    parser.add_argument(
        "--mesh_source",
        type=str,
        default="dataset",
        choices=["auto", "run", "dataset"],
        help="Mesh source passed through to overlay script.",
    )
    parser.add_argument("--line_thickness", type=int, default=5)
    parser.add_argument(
        "--line_color",
        nargs=3,
        type=int,
        default=[0, 255, 0],
        metavar=("B", "G", "R"),
        help="Contour and bbox color in BGR.",
    )
    parser.add_argument("--show_axis", action="store_true")
    parser.add_argument("--axis_scale", type=float, default=0.08)
    parser.add_argument("--axis_thickness", type=int, default=5)
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--pose_compose_mode",
        type=str,
        default="post_multiply_init",
        choices=["post_multiply_init", "pre_multiply_init", "none"],
    )
    parser.add_argument(
        "--gt_pose_only",
        action="store_true",
        help=(
            "Forwarded to overlay script: render using dataset GT pose only "
            "(ignores estimated metadata pose)."
        ),
    )
    parser.add_argument(
        "--strict_gt_frame_id_match",
        action="store_true",
        help=(
            "Forwarded to overlay script: strict GT frame-id matching "
            "(no index fallback) when --gt_pose_only is enabled."
        ),
    )
    parser.add_argument(
        "--allow_duplicate_frame_ids",
        action="store_true",
        help="Forwarded to overlay script.",
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=0,
        help=(
            "Object index for single-object overlays. For ycbinisaac with multi-object "
            "sequences, default 0 and no object_name renders all objects."
        ),
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Optional object name override (single-object mode).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Optional output root for overlays. "
            "If set, each sequence writes to <output_root>/<sequence_name>."
        ),
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately on first failed sequence.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=-1,
        help="If > 0, run only the first N discovered sequences.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print discovered sequences/commands without running overlays.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    args.dataset = _normalize_dataset(args.dataset)
    args.run_root = Path(args.run_root).expanduser().resolve()
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.model_root = Path(args.model_root).expanduser().resolve()
    if args.output_root is not None:
        args.output_root = Path(args.output_root).expanduser().resolve()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    if not args.model_root.is_dir():
        raise FileNotFoundError(f"model_root does not exist: {args.model_root}")
    if args.output_root is not None and args.output_root.exists() and not args.output_root.is_dir():
        raise ValueError(f"output_root exists but is not a directory: {args.output_root}")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    exit_code = run_all(args)
    sys.exit(exit_code)
