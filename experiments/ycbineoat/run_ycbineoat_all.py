import sys
from pathlib import Path
import time
import os
import argparse
import shutil

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np
import trimesh

from point2pose.io.sources.dataset.datareader import YcbineoatReader
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.visualization import (
    visualize_and_save_tracking_results,
    visualize_and_save_tracking_results_with_gt,
)
from point2pose.utils.evaluation import (
    adi_err,
    add_err,
    compute_auc,
    plot_evaluation_results,
    plot_error_over_time,
    plot_pose_errors,
    plot_pose_error_comparison,
    plot_recall_vs_threshold,
)
from point2pose.utils.mesh_eval import (
    export_final_meshes_from_pipeline,
    evaluate_reconstructed_mesh,
    find_gt_visible_mesh_path,
)


def load_ycb_mesh(reader, model_root):
    """
    Load the YCB object mesh using the video->object mapping from YcbineoatReader.
    """
    video_name = reader.get_video_name()
    ob_name = reader.videoname_to_object.get(video_name)

    if not ob_name:
        raise ValueError(f"No object mapping found for video: {video_name}")

    candidates = [
        os.path.join(model_root, ob_name, "textured_simple.obj"),
        os.path.join(model_root, "models", ob_name, "textured_simple.obj"),
        os.path.join(model_root, ob_name, "google_16k", "textured.obj"),
    ]

    mesh_path = None
    for path in candidates:
        if os.path.exists(path):
            mesh_path = path
            break

    if mesh_path is None:
        raise FileNotFoundError(
            f"Could not find mesh for {ob_name} in {model_root}. Checked: {candidates}"
        )

    print(f"Loading mesh from {mesh_path}")
    return trimesh.load(mesh_path)


def gt_bbox_minmax_from_mesh(mesh):
    bmin, bmax = mesh.bounds
    return np.vstack([bmin.astype(float), bmax.astype(float)])


def get_all_video_names(data_path):
    """Get all valid YCBInEOAT sequence folder names from data_path."""
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset root directory not found: {data_path}")

    video_names = []
    for item in sorted(os.listdir(data_path)):
        item_path = os.path.join(data_path, item)
        if not os.path.isdir(item_path):
            continue

        rgb_path = os.path.join(item_path, "rgb")
        if not os.path.isdir(rgb_path):
            continue

        if len(os.listdir(rgb_path)) == 0:
            continue

        cam_k_path = os.path.join(item_path, "cam_K.txt")
        if not os.path.isfile(cam_k_path):
            continue

        video_names.append(item)

    return video_names


def run_ycbineoat_single(
    data_path: str, video_name: str, out_dir: str, config_path: str, model_path: str
):
    """Process a single YCBInEOAT sequence and return evaluation metrics."""
    video_path = os.path.join(data_path, video_name)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    reader = YcbineoatReader(video_path)
    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    vis_folder = os.path.join(out_folder, "output_images")
    with_gt_folder = os.path.join(out_folder, "with_gt")
    mesh_folder = os.path.join(out_folder, "mesh")

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(with_gt_folder, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params
    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    cfg.pipeline.params.sdf_mesh_save_every = 1

    if cfg.pipeline.params.get("save_meta_data", True):
        meta_data_folder = os.path.join(out_folder, "meta_data")
        cfg.pipeline.params.meta_data_save_path = meta_data_folder

    if cfg.pipeline.type != "modular":
        raise ValueError(
            f"Only 'modular' pipeline is supported, got: {cfg.pipeline.type}"
        )
    pipeline = ModularPipeline(cfg)

    mesh = load_ycb_mesh(reader, model_path)
    gt_bbox_minmax = gt_bbox_minmax_from_mesh(mesh)

    out_poses = []
    gt_poses = []
    gt_ids = []

    for i, _ in enumerate(reader.color_files):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask = reader.get_mask(i)
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

        display_frame = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method=vis_cfg.points_vis_method,
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
        )

        visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame,
            gt_pose=gt_pose,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method=vis_cfg.points_vis_method,
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
            gt_bbox_min_max=gt_bbox_minmax,
            pred_pose_color=(0, 255, 0),
        )

    pipeline.data_logger.save_now()
    mesh_paths = export_final_meshes_from_pipeline(pipeline, mesh_folder)

    if len(gt_poses) == 0:
        print(
            f"Warning: No GT poses found for video {video_name}, skipping evaluation."
        )
        return None

    gt_poses = np.array(gt_poses)
    pred_poses = np.array(out_poses)[gt_ids]
    pred_poses_raw = pred_poses.copy()

    pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]

    adi_errs = []
    add_errs = []

    for i in range(len(pred_poses)):
        adi = adi_err(pred_poses[i], gt_poses[i], mesh.vertices.copy())
        add = add_err(pred_poses[i], gt_poses[i], mesh.vertices.copy())
        adi_errs.append(adi)
        add_errs.append(add)

    adi_errs = np.array(adi_errs)
    add_errs = np.array(add_errs)
    adds_auc = compute_auc(adi_errs) * 100
    add_auc = compute_auc(add_errs) * 100

    print(
        f"video {video_name}, ADD-S_err: {adi_errs.mean()*100:.2f}[cm], "
        f"ADD_errs: {add_errs.mean()*100:.2f}[cm], "
        f"ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}"
    )

    plot_evaluation_results(
        add_s_errs=adi_errs,
        add_errs=add_errs,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
    )

    plot_error_over_time(
        add_s_errs=adi_errs,
        add_errs=add_errs,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
    )

    plot_pose_errors(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
    )

    plot_pose_error_comparison(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
    )

    plot_recall_vs_threshold(
        add_s_errs=adi_errs,
        add_errs=add_errs,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
        max_threshold=10.0,
    )

    gt_visible_mesh_path = find_gt_visible_mesh_path(reader.video_dir)
    mesh_cd_cm = evaluate_reconstructed_mesh(
        pred_mesh_path=mesh_paths.get(0, None),
        gt_visible_mesh_path=gt_visible_mesh_path,
        output_dir=mesh_folder,
        mesh_prefix=video_name,
        pred_pose_first=pred_poses_raw[0],
        gt_pose_first=gt_poses[0],
    )
    if np.isfinite(mesh_cd_cm):
        print(f"video {video_name}, mesh_CD: {mesh_cd_cm:.3f}[cm]")
    else:
        print(f"video {video_name}, mesh_CD: skipped")

    return {
        "video_name": video_name,
        "add_s_err_mean": adi_errs.mean() * 100,
        "add_err_mean": add_errs.mean() * 100,
        "add_s_auc": adds_auc,
        "add_auc": add_auc,
        "mesh_cd_cm": mesh_cd_cm,
    }


def run_ycbineoat_all(data_path: str, out_dir: str, config_path: str, model_path: str):
    """Process all YCBInEOAT sequences and print a summary."""
    video_names = get_all_video_names(data_path)
    print(f"Found {len(video_names)} videos to process: {video_names}")
    print("-" * 80)

    results = []
    for idx, video_name in enumerate(video_names, 1):
        print(f"\n[{idx}/{len(video_names)}] Processing video: {video_name}")
        try:
            result = run_ycbineoat_single(
                data_path=data_path,
                video_name=video_name,
                out_dir=out_dir,
                config_path=config_path,
                model_path=model_path,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:
            print(f"Error processing video {video_name}: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    if len(results) == 0:
        print("No results to display.")
        return

    results.sort(key=lambda x: x["video_name"])

    for result in results:
        print(
            f"{result['video_name']}, ADD-S_err: {result['add_s_err_mean']:.2f}[cm], "
            f"ADD_errs: {result['add_err_mean']:.2f}[cm], "
            f"ADD-S_AUC: {result['add_s_auc']:.2f}, ADD_AUC: {result['add_auc']:.2f}, "
            f"mesh_CD: {result['mesh_cd_cm']:.3f}[cm]"
        )

    avg_add_s_err = np.mean([r["add_s_err_mean"] for r in results])
    avg_add_err = np.mean([r["add_err_mean"] for r in results])
    avg_add_s_auc = np.mean([r["add_s_auc"] for r in results])
    avg_add_auc = np.mean([r["add_auc"] for r in results])
    valid_mesh = [r["mesh_cd_cm"] for r in results if np.isfinite(r["mesh_cd_cm"])]
    avg_mesh_cd = float(np.mean(valid_mesh)) if len(valid_mesh) > 0 else np.inf

    print("-" * 80)
    print(
        f"Average, ADD-S_err: {avg_add_s_err:.2f}[cm], "
        f"ADD_errs: {avg_add_err:.2f}[cm], "
        f"ADD-S_AUC: {avg_add_s_auc:.2f}, ADD_AUC: {avg_add_auc:.2f}, "
        f"mesh_CD: {avg_mesh_cd:.3f}[cm]"
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/YCBInEOAT")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbineoat_all",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ycbineoat/ycbineoat_all.yaml",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/home/justin/data/YCBInEOAT/YCB_models_with_ply",
        help="Root directory of YCB models.",
    )
    args = parser.parse_args()

    run_ycbineoat_all(
        data_path=args.data_path,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
    )
