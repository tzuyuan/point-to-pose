"""Sweep driver: runs run_ho3d_runtime over (num_points x video) combinations."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import os
import traceback

import torch

from experiments.ho3d.run_ho3d_runtime import run_ho3d_runtime


DEFAULT_NUM_POINTS = [10, 30, 50, 100, 200, 400]
DEFAULT_VIDEOS = ["AP10", "MPM12"]


def run_sweep(
    data_path: str,
    out_root: str,
    config_path: str,
    num_points_list,
    videos,
    max_frames=None,
    run_mesh_eval: bool = True,
):
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, "sweep_run.log")

    summary = []
    with open(log_path, "a", encoding="utf-8") as logf:
        for n in num_points_list:
            n_dir = os.path.join(out_root, f"n{n:03d}")
            os.makedirs(n_dir, exist_ok=True)
            for video in videos:
                tag = f"n={n} video={video}"
                print(f"\n========== {tag} ==========")
                logf.write(f"\n========== {tag} ==========\n")
                logf.flush()
                try:
                    csv_path = run_ho3d_runtime(
                        data_path=data_path,
                        video_name=video,
                        out_dir=n_dir,
                        config_path=config_path,
                        num_points=n,
                        max_frames=max_frames,
                        run_mesh_eval=run_mesh_eval,
                    )
                    summary.append((n, video, "ok", csv_path))
                    logf.write(f"OK  {tag} -> {csv_path}\n")
                except Exception as e:
                    summary.append((n, video, "fail", str(e)))
                    err_msg = (
                        f"FAIL {tag}: {e}\n{traceback.format_exc()}\n"
                    )
                    print(err_msg)
                    logf.write(err_msg)
                finally:
                    logf.flush()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    summary_path = os.path.join(out_root, "sweep_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("num_points\tvideo\tstatus\tdetail\n")
        for n, v, s, d in summary:
            f.write(f"{n}\t{v}\t{s}\t{d}\n")
    print(f"\nSweep summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/justin/code/point-to-pose/results/runtime_analysis_20260505",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d_exp/eccv_final.yaml",
    )
    parser.add_argument(
        "--num_points",
        type=str,
        default=",".join(str(n) for n in DEFAULT_NUM_POINTS),
        help="Comma-separated list, e.g. 10,30,50,100,200,400",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=",".join(DEFAULT_VIDEOS),
        help="Comma-separated HO3D video names",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_mesh_eval", action="store_true")

    args = parser.parse_args()
    num_points_list = [int(x.strip()) for x in args.num_points.split(",") if x.strip()]
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    run_sweep(
        data_path=args.data_path,
        out_root=args.out_root,
        config_path=args.config_path,
        num_points_list=num_points_list,
        videos=videos,
        max_frames=args.max_frames,
        run_mesh_eval=not args.no_mesh_eval,
    )
