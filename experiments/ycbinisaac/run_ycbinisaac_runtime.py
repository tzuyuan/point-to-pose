"""Runtime profiling runner for YCBMultiTrack (YCBInIsaac dataset format).

Captures per-frame module timings to CSV. Skips heavy visualization.
Intended for single-object sequences but supports multi-object (uses obj 0 quality).
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

from omegaconf import OmegaConf
import torch
import numpy as np

from point2pose.io.sources.dataset.datareader import YCBInIsaacReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.evaluation import adi_err, add_err, compute_auc

# Reuse the row schema from the HO3D runtime runner.
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "ho3d")
)
from run_ho3d_runtime import CSV_COLUMNS, _gather_row  # noqa: E402


def _try_load_ycb_mesh(model_path: str, obj_name: str):
    candidates = [
        os.path.join(model_path, obj_name, "textured.obj"),
        os.path.join(model_path, "models", obj_name, "textured_simple.obj"),
        os.path.join(model_path, obj_name, "google_16k", "textured.obj"),
        os.path.join(model_path, obj_name, "textured_simple.obj"),
    ]
    import trimesh
    for p in candidates:
        if os.path.exists(p):
            return trimesh.load(p, force="mesh")
    return None


def run_ycbinisaac_runtime(
    data_path: str,
    video_name: str,
    out_dir: str,
    config_path: str,
    model_path: str,
    max_frames: int | None = None,
    run_quality_eval: bool = True,
):
    video_path = os.path.join(data_path, video_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    reader = YCBInIsaacReader(video_path)
    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name)
    mesh_folder = os.path.join(out_folder, "mesh")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(mesh_folder, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    cfg.pipeline.params.sdf_mesh_save_every = 0
    object_names = reader.get_object_names()
    print(f"Found objects in video {video_name}: {object_names}")
    cfg.pipeline.params.max_num_obj = max(1, len(object_names))
    cfg.pipeline.params.use_segmenter = False
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

    out_poses_by_obj = {n: [] for n in object_names}
    gt_poses_by_obj = {n: [] for n in object_names}
    eval_ids_by_obj = {n: [] for n in object_names}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wall_start = time.time()
    n_frames = len(reader) if max_frames is None else min(len(reader), max_frames)
    for i in range(n_frames):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)

        masks = reader.get_masks(i, use_init_mask=(i == 0))
        mask = np.stack(masks, axis=0)
        mask = torch.from_numpy(mask).unsqueeze(1).to(device)

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
        if out_pose.ndim == 2:
            out_pose = out_pose.reshape(1, 4, 4)

        gt_pose_map = reader.get_gt_poses(i)
        for obj_idx, obj_name in enumerate(object_names[: out_pose.shape[0]]):
            out_poses_by_obj[obj_name].append(out_pose[obj_idx].reshape(4, 4))
            gp = gt_pose_map.get(obj_name, None)
            gt_poses_by_obj[obj_name].append(gp)
            if gp is not None:
                eval_ids_by_obj[obj_name].append(i)
            if i == 0 and gp is not None and obj_idx < len(pipeline.objects):
                pipeline.objects[obj_idx].init_pose = gp

        # We don't have per-frame "configured num points" here (no override),
        # so write -1 to indicate "config default".
        row = _gather_row(pipeline, i, configured_num_points=-1)
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    wall_total = time.time() - wall_start

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    meta = {
        "video_name": video_name,
        "config_path": os.path.abspath(config_path),
        "object_names": list(object_names),
        "num_frames": n_frames,
        "max_frames": max_frames,
        "wall_total_s": wall_total,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_folder, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    quality = {}
    if run_quality_eval:
        try:
            for obj_name in object_names:
                eval_ids = np.asarray(eval_ids_by_obj[obj_name], dtype=int)
                if eval_ids.size == 0:
                    continue
                gt_arr = np.array(
                    [gt_poses_by_obj[obj_name][i] for i in range(len(eval_ids))]
                )
                pred_arr = np.array(out_poses_by_obj[obj_name])[eval_ids]
                # Align first frame
                pred_arr = pred_arr @ inverse_SE3(pred_arr[0]) @ gt_arr[0]

                mesh = _try_load_ycb_mesh(model_path, obj_name)
                if mesh is None:
                    print(f"[ycb runtime] {obj_name}: mesh not found at {model_path}, skip quality")
                    continue

                adi_e = np.array(
                    [adi_err(pred_arr[k], gt_arr[k], mesh.vertices.copy())
                     for k in range(len(pred_arr))]
                )
                add_e = np.array(
                    [add_err(pred_arr[k], gt_arr[k], mesh.vertices.copy())
                     for k in range(len(pred_arr))]
                )
                quality[obj_name] = {
                    "add_s_err_mean_cm": float(adi_e.mean() * 100.0),
                    "add_err_mean_cm": float(add_e.mean() * 100.0),
                    "add_s_auc": float(compute_auc(adi_e) * 100.0),
                    "add_auc": float(compute_auc(add_e) * 100.0),
                    "n_eval_frames": int(len(pred_arr)),
                }
                print(
                    f"[ycb runtime] {video_name}/{obj_name}: "
                    f"frames={n_frames} wall={wall_total:.1f}s "
                    f"ADD-S_err={quality[obj_name]['add_s_err_mean_cm']:.2f}cm "
                    f"ADD-S_AUC={quality[obj_name]['add_s_auc']:.2f}"
                )
        except Exception as e:
            print(f"[ycb runtime] quality eval failed: {e}")

    with open(os.path.join(out_folder, "quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", "-d", type=str, default="/home/justin/data/YCBMultiTrack_new"
    )
    parser.add_argument("--video_name", "-v", type=str, default="006_mustard_bottle")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/runtime_analysis_20260505/ycb",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d_exp/eccv_final.yaml",
        help="Pipeline config to use. Defaults to HO3D eccv_final.yaml since the "
             "modular pipeline params are dataset-agnostic.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/home/justin/data/HO3D_V3/models",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_quality_eval", action="store_true")

    args = parser.parse_args()
    run_ycbinisaac_runtime(
        data_path=args.data_path,
        video_name=args.video_name,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
        max_frames=args.max_frames,
        run_quality_eval=not args.no_quality_eval,
    )
