"""Resolution sweep driver for the 2D point tracker.

Runs TAPIR at three input resolutions on three sequences (two HO3D + one YCBMultiTrack
single-object). Captures per-frame timings + ADD/ADD-S quality at each setting.

Output layout:
  results/runtime_analysis_20260505/resolution_sweep/tapir/<HxW>/<video>/timings.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import os
import traceback

import torch

from experiments.ho3d.run_ho3d_runtime import run_ho3d_runtime
from experiments.ycbinisaac.run_ycbinisaac_runtime import run_ycbinisaac_runtime


# (height, width) — TAPIR is square per its config style
DEFAULT_RESOLUTIONS = [(256, 256), (384, 384), (480, 480)]

# (kind, video_name) — kind chooses runner/dataset
DEFAULT_SEQUENCES = [
    ("ho3d", "AP10"),
    ("ho3d", "MPM12"),
    ("ycb",  "006_mustard_bottle"),
]


def run_sweep(
    out_root: str,
    config_path: str,
    resolutions,
    sequences,
    ho3d_data_path: str,
    ycb_data_path: str,
    ycb_model_path: str,
    num_points: int = 30,
    max_frames=None,
):
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, "resolution_sweep_run.log")

    summary = []
    with open(log_path, "a", encoding="utf-8") as logf:
        for (h, w) in resolutions:
            res_dir = os.path.join(out_root, f"{h}x{w}")
            os.makedirs(res_dir, exist_ok=True)
            for kind, video in sequences:
                tag = f"res={h}x{w} {kind}/{video}"
                print(f"\n========== {tag} ==========")
                logf.write(f"\n========== {tag} ==========\n")
                logf.flush()
                try:
                    if kind == "ho3d":
                        csv_path = run_ho3d_runtime(
                            data_path=ho3d_data_path,
                            video_name=video,
                            out_dir=res_dir,
                            config_path=config_path,
                            num_points=num_points,
                            max_frames=max_frames,
                            run_mesh_eval=False,
                            resize_height=h,
                            resize_width=w,
                        )
                    elif kind == "ycb":
                        csv_path = run_ycbinisaac_runtime(
                            data_path=ycb_data_path,
                            video_name=video,
                            out_dir=res_dir,
                            config_path=config_path,
                            model_path=ycb_model_path,
                            max_frames=max_frames,
                            run_quality_eval=True,
                            resize_height=h,
                            resize_width=w,
                        )
                    else:
                        raise ValueError(f"Unknown sequence kind: {kind}")
                    summary.append((h, w, kind, video, "ok", csv_path))
                    logf.write(f"OK  {tag} -> {csv_path}\n")
                except Exception as e:
                    summary.append((h, w, kind, video, "fail", str(e)))
                    err_msg = f"FAIL {tag}: {e}\n{traceback.format_exc()}\n"
                    print(err_msg)
                    logf.write(err_msg)
                finally:
                    logf.flush()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    summary_path = os.path.join(out_root, "resolution_sweep_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("h\tw\tkind\tvideo\tstatus\tdetail\n")
        for h, w, k, v, s, d in summary:
            f.write(f"{h}\t{w}\t{k}\t{v}\t{s}\t{d}\n")
    print(f"\nResolution sweep summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/justin/code/point-to-pose/results/runtime_analysis_20260505/resolution_sweep/tapir",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d_exp/eccv_final.yaml",
    )
    parser.add_argument(
        "--ho3d_data_path", type=str, default="/home/justin/data/HO3D_V3/"
    )
    parser.add_argument(
        "--ycb_data_path", type=str, default="/home/justin/data/YCBMultiTrack_new"
    )
    parser.add_argument(
        "--ycb_model_path", type=str, default="/home/justin/data/HO3D_V3/models"
    )
    parser.add_argument("--num_points", type=int, default=30)
    parser.add_argument(
        "--resolutions",
        type=str,
        default=",".join(f"{h}x{w}" for h, w in DEFAULT_RESOLUTIONS),
        help="Comma-separated HxW pairs, e.g. 256x256,384x384,480x480",
    )
    parser.add_argument("--max_frames", type=int, default=None)

    args = parser.parse_args()
    res_list = []
    for token in args.resolutions.split(","):
        token = token.strip()
        if not token:
            continue
        h, w = token.lower().split("x")
        res_list.append((int(h), int(w)))

    run_sweep(
        out_root=args.out_root,
        config_path=args.config_path,
        resolutions=res_list,
        sequences=DEFAULT_SEQUENCES,
        ho3d_data_path=args.ho3d_data_path,
        ycb_data_path=args.ycb_data_path,
        ycb_model_path=args.ycb_model_path,
        num_points=args.num_points,
        max_frames=args.max_frames,
    )
