"""
Capture live tracking, then train a 2DGS/3DGS reconstruction -- one command.

Runs capture.py and train_gaussians.py as separate subprocesses (not in-process)
so the live-tracking GPU models (TAPIR/SAM2/nvdiffrast) and gsplat's CUDA training
never share a process/CUDA context.

Run:
    python examples/realsense_tracking/reconstruction/run.py \
        --tracking-config configs/pipeline/reconstruction_capture.yaml \
        --train-config configs/reconstruct/export_test.yaml \
        --name my_capture
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def should_enable_live_viewer(requested: bool, env_var: str = "POINT2POSE_ENABLE_LIVE_VIEWER") -> bool:
    if not requested:
        return False
    value = os.environ.get(env_var, "").strip().lower()
    return requested or value in {"1", "true", "yes", "on", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="export dir name; frames/poses go to debug/<name>/")
    ap.add_argument("--tracking-config", default=str(REPO / "configs/pipeline/reconstruction_capture.yaml"))
    ap.add_argument("--train-config", default=str(REPO / "configs/reconstruct/export_test.yaml"))
    ap.add_argument("--viewer", action="store_true",
                    help="live viewer during training via the point2pose render-loop pattern")
    args = ap.parse_args()

    use_viewer = should_enable_live_viewer(args.viewer)
    if args.viewer and use_viewer:
        print("[run] Launching the live training viewer.")

    export_dir = REPO / "debug" / args.name

    capture_cmd = [
        sys.executable, str(HERE / "capture.py"),
        "--config", args.tracking_config,
        "--export-dir", str(export_dir),
    ]

    print(f"[run] capturing to {export_dir} -- quit the tracking window ('q') when done")
    subprocess.run(capture_cmd, cwd=str(REPO), check=True)

    train_cmd = [
        sys.executable, str(HERE / "train_gaussians.py"),
        "--config", args.train_config,
        "--results_path", str(export_dir),
    ]
    if use_viewer:
        train_cmd.append("--viewer")
    print(f"[run] training gaussians from {export_dir}")
    subprocess.run(train_cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
