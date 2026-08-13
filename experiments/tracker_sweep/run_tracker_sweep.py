"""Sweep all trackers over the pose-tracking benchmarks (object pose metrics).

For each tracker, the dataset's ECCV base config is used unchanged except for
the `tracker:` block (and image saving disabled for speed), and the existing
per-dataset runners are invoked as subprocesses:

  - ycbmultitrack_real -> experiments/ycbinisaac/run_ycbinisaac_all.py
                          (data: YCBMultiTrack_recalib, GT masks, no segmenter)
  - ho3d               -> experiments/ho3d/run_ho3d_all.py
                          (data: HO3D_V3 evaluation sequences, SAM2 segmenter)

Outputs (never touching the ECCV results directories):
  /home/justin/results/tracker_pose_benchmark/<dataset>/<tracker>/...
  /home/justin/results/tracker_pose_benchmark/configs/<dataset>_<tracker>.yaml
  /home/justin/results/tracker_pose_benchmark/logs/<dataset>_<tracker>.log

Sequences already present in a tracker's output directory are skipped, so the
sweep is resumable.

Example:
    python experiments/tracker_sweep/run_tracker_sweep.py \
        --datasets ycbmultitrack_real,ho3d \
        --trackers tapir,tapnext,trackon,litetracker,cotracker3_online
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/home/justin/results/tracker_pose_benchmark")

CKPT = REPO / "checkpoints"

TRACKER_BLOCKS = {
    # baseline: keep the dataset config's own tapir block untouched
    "tapir": None,
    "cotracker3_online": {
        "type": "cotracker3_online",
        "params": {
            "img_height": 480,
            "img_width": 640,
            "resize_height": 480,
            "resize_width": 640,
            "device": "cuda",
            "window_len": 16,
            "v2": False,
            "checkpoint_path": "/home/justin/code/co-tracker-realtime/checkpoints/scaled_online.pth",
        },
    },
    "tapnext": {
        "type": "tapnext",
        "params": {
            "device": "cuda",
            "input_resolution": 256,
            "visible_threshold": 0.5,
            "half_precision": True,
            "checkpoint_path": str(CKPT / "tapnext" / "tapnextpp_ckpt.pt"),
        },
    },
    "trackon": {
        "type": "trackon",
        "params": {
            "device": "cuda",
            "vit_backbone": "dinov2_s",
            "visible_threshold": 0.8,
            "inference_memory_size": 72,
            "checkpoint_path": str(
                CKPT / "trackon" / "trackon2_dinov2_checkpoint.pt"
            ),
        },
    },
    "litetracker": {
        "type": "litetracker",
        "params": {
            "device": "cuda",
            "window_len": 16,
            "bf16": True,
            "checkpoint_path": "/home/justin/code/co-tracker-realtime/checkpoints/scaled_online.pth",
        },
    },
}

# Base configs are the exact `eccv final` (6af191a) versions, extracted into
# configs_eccv/ — the working-tree configs drifted post-ECCV (ablation commit).
_ECCV_CFG = REPO / "experiments" / "tracker_sweep" / "configs_eccv"

DATASETS = {
    "ycbmultitrack_real": {
        # configs/ycbinisaac/eccv_final.yaml from the `eccv` branch — per the
        # author, the actual config behind the paper's YCBMultiTrack table.
        # Differs from the 6af191a-committed ycbinisaac_single.yaml only in
        # sdf_bounds_padding (0.05 vs 0.2) and sdf_max_voxels (20k vs 20M).
        "base_config": _ECCV_CFG / "ycbinisaac_eccv_final.yaml",
        "runner": REPO / "experiments" / "ycbinisaac" / "run_ycbinisaac_all.py",
        "args": lambda out_dir, cfg_path: [
            # ECCV-submission GT (identical images/depth/masks to recalib;
            # only annotated poses differ) — matches the paper's evaluation.
            "--data_path", "/home/justin/data/YCBMultiTrack_new_eccv_submission",
            "--out_dir", str(out_dir),
            "--config_path", str(cfg_path),
            "--model_path", "/home/justin/data/HO3D_V3/models",
            "--summary_dir", str(out_dir / "summary"),
        ],
    },
    "ho3d": {
        # configs/ho3d_exp/eccv_final.yaml from the `eccv` branch — the config
        # that bit-exactly reproduces the paper's HO3D table (tapir at 480).
        "base_config": _ECCV_CFG / "eccv_final.yaml",
        "runner": REPO / "experiments" / "ho3d" / "run_ho3d_all.py",
        "args": lambda out_dir, cfg_path: [
            "--data_path", "/home/justin/data/HO3D_V3/",
            "--out_dir", str(out_dir),
            "--config_path", str(cfg_path),
            "--skip_existing",
        ],
    },
}


def make_config(dataset, tracker, cfg_dir):
    cfg = OmegaConf.load(DATASETS[dataset]["base_config"])
    block = TRACKER_BLOCKS[tracker]
    if block is not None:
        cfg.tracker = OmegaConf.create(block)
    # visualization images are the dominant disk/time cost; metrics don't need them
    cfg.visualization.params.save_images = False

    # The post-ECCV "clean up code" commit (5a4b56f) introduced a
    # sample-stabilization gate defaulting to 5 frames, which blocks re-sampling
    # after pose-jump rejections and costs ~18 ADD AUC on occlusion-heavy HO3D
    # sequences. Disable it to reproduce the ECCV-paper behavior.
    cfg.pipeline.params.sample_stabilize_frames = 0

    cfg_path = cfg_dir / f"{dataset}_{tracker}.yaml"
    OmegaConf.save(cfg, cfg_path)
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ycbmultitrack_real,ho3d")
    ap.add_argument(
        "--trackers", default="tapir,tapnext,trackon,litetracker,cotracker3_online"
    )
    args = ap.parse_args()

    cfg_dir = OUT_ROOT / "configs"
    log_dir = OUT_ROOT / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MPLBACKEND", "Agg")

    failures = []
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        for tracker in args.trackers.split(","):
            tracker = tracker.strip()
            out_dir = OUT_ROOT / dataset / tracker
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = make_config(dataset, tracker, cfg_dir)
            log_path = log_dir / f"{dataset}_{tracker}.log"

            cmd = [
                sys.executable,
                str(DATASETS[dataset]["runner"]),
                *DATASETS[dataset]["args"](out_dir, cfg_path),
            ]
            print(f"=== [{dataset} / {tracker}] -> {log_path}", flush=True)
            with open(log_path, "a") as log_f:
                ret = subprocess.call(
                    cmd, cwd=str(REPO), env=env, stdout=log_f, stderr=log_f
                )
            status = "OK" if ret == 0 else f"FAILED (exit {ret})"
            print(f"=== [{dataset} / {tracker}] {status}", flush=True)
            if ret != 0:
                failures.append((dataset, tracker, ret))

    if failures:
        print(f"SWEEP DONE with {len(failures)} failures: {failures}", flush=True)
    else:
        print("SWEEP DONE, all runs OK", flush=True)


if __name__ == "__main__":
    main()
