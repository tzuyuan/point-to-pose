"""Runtime profiling runner for the modular pipeline on a single HO3D video.

Captures per-frame module timings to CSV. Skips heavy visualization to keep
timings clean. Predicted poses are still saved and an optional final mesh-CD
evaluation is run so result quality stays verifiable.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import os
import shutil

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np

from point2pose.io.sources.dataset.datareader import Ho3dReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.evaluation import (
    adi_err,
    add_err,
    compute_auc,
)
from point2pose.utils.mesh_eval import (
    export_final_meshes_from_pipeline,
    evaluate_reconstructed_mesh,
    find_gt_visible_mesh_path,
)


CSV_COLUMNS = [
    "frame_id",
    "num_active_tracks",
    "configured_num_points",
    # frontend sub-modules (CUDA-synced)
    "segmenter_ms",
    "tracker_ms",
    "two_d_to_three_d_ms",
    "register_ms",
    "dense_recovery_ms",
    "extract_valid_ms",
    # pipeline-level modules
    "frontend_total_ms",
    "track_table_ms",
    "track_compact_ms",
    "recovery_ms",
    "local_opt_ms",
    "keyframe_ms",
    "global_opt_ms",
    "tsdf_ms",
    "logging_ms",
    "total_ms",
]


def _gather_row(pipeline, frame_id: int, configured_num_points: int) -> dict:
    fe = getattr(pipeline, "last_frontend_timings", {}) or {}
    mt = getattr(pipeline, "last_step_module_times", {}) or {}

    try:
        active_tracks = int(len(pipeline.frontend.tracker._active_global_ids))
    except AttributeError:
        active_tracks = -1

    return {
        "frame_id": frame_id,
        "num_active_tracks": active_tracks,
        "configured_num_points": configured_num_points,
        "segmenter_ms": float(fe.get("segmenter", 0.0)) * 1000.0,
        "tracker_ms": float(fe.get("tracker", 0.0)) * 1000.0,
        "two_d_to_three_d_ms": float(fe.get("2d_to_3d", 0.0)) * 1000.0,
        "register_ms": float(fe.get("registration", 0.0)) * 1000.0,
        "dense_recovery_ms": float(fe.get("dense_recovery", 0.0)) * 1000.0,
        "extract_valid_ms": float(fe.get("extract_valid", 0.0)) * 1000.0,
        "frontend_total_ms": float(mt.get("frontend", 0.0)) * 1000.0,
        "track_table_ms": float(mt.get("track_table", 0.0)) * 1000.0,
        "track_compact_ms": float(mt.get("track_compact", 0.0)) * 1000.0,
        "recovery_ms": float(mt.get("recovery", 0.0)) * 1000.0,
        "local_opt_ms": float(mt.get("local_opt", 0.0)) * 1000.0,
        "keyframe_ms": float(mt.get("keyframe", 0.0)) * 1000.0,
        "global_opt_ms": float(mt.get("global_opt", 0.0)) * 1000.0,
        "tsdf_ms": float(mt.get("tsdf", 0.0)) * 1000.0,
        "logging_ms": float(mt.get("logging", 0.0)) * 1000.0,
        "total_ms": float(mt.get("total", 0.0)) * 1000.0,
    }


def _gt_bbox_minmax(reader):
    mesh = reader.get_gt_mesh()
    bmin, bmax = mesh.bounds
    return np.vstack([bmin.astype(float), bmax.astype(float)])


def run_ho3d_runtime(
    data_path: str,
    video_name: str,
    out_dir: str,
    config_path: str,
    num_points: int,
    max_frames: int | None = None,
    run_mesh_eval: bool = True,
    resize_height: int | None = None,
    resize_width: int | None = None,
):
    video_path = os.path.join(data_path, "evaluation", video_name)
    reader = Ho3dReader(video_path, data_path)
    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name)
    mesh_folder = os.path.join(out_folder, "mesh")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(mesh_folder, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    # Override sampler num_points so the sweep can vary it cleanly.
    OmegaConf.update(
        cfg, "sampler.params.num_points", int(num_points), force_add=True
    )
    if resize_height is not None:
        OmegaConf.update(
            cfg, "tracker.params.resize_height", int(resize_height), force_add=True
        )
    if resize_width is not None:
        OmegaConf.update(
            cfg, "tracker.params.resize_width", int(resize_width), force_add=True
        )
    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    # Suppress periodic mesh export during timing — only the final mesh matters.
    cfg.pipeline.params.sdf_mesh_save_every = 0
    if cfg.pipeline.params.get("save_meta_data", True):
        cfg.pipeline.params.meta_data_save_path = os.path.join(out_folder, "meta_data")

    if cfg.pipeline.type == "single_process":
        pipeline = PipelineSingleProcess(cfg)
    elif cfg.pipeline.type == "modular":
        pipeline = ModularPipeline(cfg)
    else:
        raise ValueError(f"Invalid pipeline type: {cfg.pipeline.type}")

    csv_path = os.path.join(out_folder, "timings.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    out_poses = []
    gt_poses = []
    gt_ids = []
    gt_bbox_minmax = _gt_bbox_minmax(reader)

    wall_start = time.time()
    for i, color_file in enumerate(reader.color_files):
        if max_frames is not None and i >= max_frames:
            break
        color = cv2.imread(color_file)
        H, W = color.shape[:2]
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = reader.get_depth(i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        frame = Frame(
            id=i,
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=reader.K,
            depth_factor=1.0,
            timestamp=time.time(),
        )

        out_pose = pipeline.step(frame)
        out_poses.append(out_pose.reshape(4, 4))

        gt_pose = reader.get_gt_pose(i)
        if gt_pose is not None:
            gt_ids.append(i)
            gt_poses.append(gt_pose)

        if i == 0 and gt_pose is not None:
            pipeline.objects[0].init_pose = gt_pose

        writer.writerow(_gather_row(pipeline, i, int(num_points)))
        csv_file.flush()

    csv_file.close()
    wall_total = time.time() - wall_start

    np.save(os.path.join(out_folder, "pred_poses.npy"), np.asarray(out_poses))

    # run_meta
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    meta = {
        "video_name": video_name,
        "config_path": os.path.abspath(config_path),
        "num_points": int(num_points),
        "resize_height": int(resize_height) if resize_height is not None else None,
        "resize_width": int(resize_width) if resize_width is not None else None,
        "num_frames": len(out_poses),
        "max_frames": max_frames,
        "wall_total_s": wall_total,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bbox_minmax": gt_bbox_minmax.tolist(),
    }
    with open(os.path.join(out_folder, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Quick quality eval (so we know the run wasn't garbage)
    try:
        mesh_paths = export_final_meshes_from_pipeline(pipeline, mesh_folder)
        if len(gt_poses) > 0:
            gt_arr = np.array(gt_poses)
            pred_arr = np.array(out_poses)[gt_ids]
            pred_raw = pred_arr.copy()
            pred_arr = pred_arr @ inverse_SE3(pred_arr[0]) @ gt_arr[0]
            mesh = reader.get_gt_mesh()
            adi_e = np.array(
                [adi_err(pred_arr[i], gt_arr[i], mesh.vertices.copy()) for i in range(len(pred_arr))]
            )
            add_e = np.array(
                [add_err(pred_arr[i], gt_arr[i], mesh.vertices.copy()) for i in range(len(pred_arr))]
            )
            adds_auc = compute_auc(adi_e) * 100.0
            add_auc = compute_auc(add_e) * 100.0
            mesh_cd_cm = np.nan
            if run_mesh_eval:
                gt_visible_mesh_path = find_gt_visible_mesh_path(reader.video_dir)
                mesh_cd_cm = float(
                    evaluate_reconstructed_mesh(
                        pred_mesh_path=mesh_paths.get(0, None),
                        gt_visible_mesh_path=gt_visible_mesh_path,
                        output_dir=mesh_folder,
                        mesh_prefix=video_name,
                        pred_pose_first=pred_raw[0],
                        gt_pose_first=gt_arr[0],
                    )
                )
            quality = {
                "add_s_err_mean_cm": float(adi_e.mean() * 100.0),
                "add_err_mean_cm": float(add_e.mean() * 100.0),
                "add_s_auc": float(adds_auc),
                "add_auc": float(add_auc),
                "mesh_cd_cm": mesh_cd_cm,
            }
            with open(os.path.join(out_folder, "quality.json"), "w", encoding="utf-8") as f:
                json.dump(quality, f, indent=2)
            print(
                f"[runtime] {video_name} np={num_points}: "
                f"frames={len(out_poses)} wall={wall_total:.1f}s "
                f"ADD-S_err={quality['add_s_err_mean_cm']:.2f}cm "
                f"ADD-S_AUC={quality['add_s_auc']:.2f}"
            )
    except Exception as e:
        print(f"[runtime] {video_name} np={num_points}: quality eval failed: {e}")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument("--video_name", "-v", type=str, default="AP10")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/runtime_analysis_20260505",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d_exp/eccv_final.yaml",
    )
    parser.add_argument("--num_points", "-n", type=int, default=30)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame cap for quicker debugging or small sweeps.",
    )
    parser.add_argument(
        "--no_mesh_eval",
        action="store_true",
        help="Skip the final mesh CD evaluation (still saves predicted poses).",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Override tracker.params.resize_height. If unset, uses config default.",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Override tracker.params.resize_width. If unset, uses config default.",
    )

    args = parser.parse_args()
    run_ho3d_runtime(
        data_path=args.data_path,
        video_name=args.video_name,
        out_dir=args.out_dir,
        config_path=args.config_path,
        num_points=args.num_points,
        max_frames=args.max_frames,
        run_mesh_eval=not args.no_mesh_eval,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
    )
