import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))


import os
import argparse
import shutil
import gc
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np

from point2pose.io.sources.dataset.datareader import Ho3dReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
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


def _cleanup_cuda_memory():
    """Best-effort cleanup between videos and after failures."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Build a GT bbox (min/max in object frame) once per sequence
def gt_bbox_minmax_from_mesh(reader):
    # reader.get_gt_mesh() should return the HO3D object mesh in meters, object frame
    mesh = reader.get_gt_mesh()
    bmin, bmax = mesh.bounds  # shape (3,), (3,)
    return np.vstack([bmin.astype(float), bmax.astype(float)])  # (2,3)


def get_all_video_names(data_path):
    """Get all video names from the HO3D evaluation directory."""
    evaluation_dir = os.path.join(data_path, "evaluation")
    if not os.path.exists(evaluation_dir):
        raise ValueError(f"Evaluation directory not found: {evaluation_dir}")

    video_names = []
    for item in os.listdir(evaluation_dir):
        item_path = os.path.join(evaluation_dir, item)
        if os.path.isdir(item_path):
            # Check if it has a rgb subdirectory (to confirm it's a valid video)
            rgb_path = os.path.join(item_path, "rgb")
            if os.path.exists(rgb_path) and len(os.listdir(rgb_path)) > 0:
                video_names.append(item)

    return sorted(video_names)


def run_ho3d_single(data_path: str, video_name: str, out_dir: str, config_path: str):
    """Process a single HO3D video and return evaluation metrics."""
    video_path = os.path.join(data_path, os.path.join("evaluation/", video_name))

    reader = Ho3dReader(video_path, data_path)

    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    vis_folder = os.path.join(out_folder, "output_images")
    with_gt_folder = os.path.join(out_folder, "with_gt")
    mesh_folder = os.path.join(out_folder, "mesh")
    vis_folder_reg_corr = os.path.join(out_folder, "registration_correspondence")
    with_gt_folder_reg_corr = os.path.join(
        out_folder, "with_gt_registration_correspondence"
    )

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(with_gt_folder, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(vis_folder_reg_corr, exist_ok=True)
    os.makedirs(with_gt_folder_reg_corr, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params
    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    cfg.pipeline.params.sdf_mesh_save_every = 1

    # Set meta_data_save_path to save in the results folder for this dataset
    if cfg.pipeline.params.get("save_meta_data", True):
        meta_data_folder = os.path.join(out_folder, "meta_data")
        cfg.pipeline.params.meta_data_save_path = meta_data_folder

    pipeline = None
    try:
        if cfg.pipeline.type == "single_process":
            pipeline = PipelineSingleProcess(cfg)
        elif cfg.pipeline.type == "modular":
            pipeline = ModularPipeline(cfg)
        else:
            raise ValueError(f"Invalid pipeline type: {cfg.pipeline.type}")

        gt_bbox_minmax = gt_bbox_minmax_from_mesh(reader)
        out_poses = []
        gt_poses = []
        gt_ids = []

        last_frame_idx = -1
        try:
            for i, color_file in enumerate(reader.color_files):
                last_frame_idx = i
                color = cv2.imread(color_file)
                H, W = color.shape[:2]
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                depth = reader.get_depth(i)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                mask = reader.get_mask(i)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
                # Create Frame object
                frame = Frame(
                    id=i,
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    intrinsics=reader.K,
                    depth_factor=1.0,
                    timestamp=time.time(),
                )

                # get out pose from the pipeline
                out_pose = pipeline.step(frame)
                out_poses.append(out_pose.reshape(4, 4))

                # get gt pose from the reader
                gt_pose = reader.get_gt_pose(i)
                if gt_pose is not None:
                    gt_ids.append(i)
                    gt_poses.append(gt_pose)

                if i == 0:
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

                display_frame_reg_corr = visualize_and_save_tracking_results(
                    frame=frame,
                    objects=pipeline.objects,
                    track_table=pipeline.track_table,
                    frame_id=i,
                    visualize_points=vis_cfg.visualize_points,
                    points_vis_method="registration_correspondence",
                    save_images=vis_cfg.save_images,
                    output_image_dir=vis_folder_reg_corr,
                    camera_intrinsics=reader.K,
                    bbox_min_max=gt_bbox_minmax,
                )

                visualize_and_save_tracking_results_with_gt(
                    frame=frame,
                    objects=pipeline.objects,
                    track_table=pipeline.track_table,
                    est_result_frame=display_frame_reg_corr,
                    gt_pose=gt_pose,
                    frame_id=i,
                    visualize_points=vis_cfg.visualize_points,
                    points_vis_method="registration_correspondence",
                    save_images=vis_cfg.save_images,
                    output_image_dir=with_gt_folder_reg_corr,
                    camera_intrinsics=reader.K,
                    bbox_min_max=gt_bbox_minmax,
                    gt_bbox_min_max=gt_bbox_minmax,
                    pred_pose_color=(0, 255, 0),
                )
        except Exception as exc:
            raise RuntimeError(
                f"Video {video_name} failed at frame {last_frame_idx} "
                f"(file={reader.color_files[last_frame_idx] if last_frame_idx >= 0 else 'N/A'})"
            ) from exc

        mesh_paths = export_final_meshes_from_pipeline(pipeline, mesh_folder)

        if len(gt_poses) == 0:
            print(
                f"Warning: No GT poses found for video {video_name}, skipping evaluation."
            )
            return None

        gt_poses = np.array(gt_poses)
        pred_poses = np.array(out_poses)[gt_ids]
        pred_poses_raw = pred_poses.copy()

        ######### Align first frame
        pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]

        adi_errs = []
        add_errs = []
        mesh = reader.get_gt_mesh()

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
            f"video {video_name}, ADD-S_err: {adi_errs.mean()*100:.2f}[cm], ADD_errs: {add_errs.mean()*100:.2f}[cm], ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}"
        )

        if getattr(pipeline, "save_meta_data", False):
            pipeline.data_logger.save_now()

        # Generate evaluation plots
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

        # Generate pose error plots
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

        # Generate recall vs threshold plot
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

        # Return metrics for summary
        return {
            "video_name": video_name,
            "add_s_err_mean": adi_errs.mean() * 100,
            "add_err_mean": add_errs.mean() * 100,
            "add_s_auc": adds_auc,
            "add_auc": add_auc,
            "mesh_cd_cm": mesh_cd_cm,
        }
    finally:
        # Ensure CUDA memory is released before next sequence, even after exceptions.
        try:
            if pipeline is not None and getattr(pipeline, "save_meta_data", False):
                pipeline.data_logger.save_now()
        except Exception:
            pass
        del pipeline
        _cleanup_cuda_memory()


def run_ho3d_all(
    data_path: str, out_dir: str, config_path: str, rerun_existing: bool = True
):
    """Process all HO3D videos and generate a summary."""
    os.makedirs(out_dir, exist_ok=True)

    # Get all video names
    video_names = get_all_video_names(data_path)
    print(f"Found {len(video_names)} videos to process: {video_names}")
    print("-" * 80)

    # Process each video
    results = []
    skipped_count = 0
    for idx, video_name in enumerate(video_names, 1):
        video_out_dir = os.path.join(out_dir, video_name)
        if os.path.isdir(video_out_dir) and not rerun_existing:
            print(
                f"\n[{idx}/{len(video_names)}] Skipping video: {video_name} "
                f"(existing output found at {video_out_dir})"
            )
            skipped_count += 1
            continue

        print(f"\n[{idx}/{len(video_names)}] Processing video: {video_name}")
        try:
            result = run_ho3d_single(data_path, video_name, out_dir, config_path)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            err_dir = os.path.join(out_dir, video_name)
            os.makedirs(err_dir, exist_ok=True)
            err_log = os.path.join(err_dir, "run_error.log")
            with open(err_log, "w", encoding="utf-8") as f:
                f.write(f"Video: {video_name}\n")
                f.write(f"Config: {config_path}\n")
                f.write(f"Data: {data_path}\n")
                f.write(f"Error: {repr(e)}\n\n")
                traceback.print_exc(file=f)
            traceback.print_exc()
            print(f"Saved traceback to: {err_log}")
            _cleanup_cuda_memory()
            continue

    summary_lines = ["", "=" * 80, "SUMMARY OF RESULTS", "=" * 80]
    summary_lines.append(f"Processed videos: {len(results)}")
    summary_lines.append(f"Skipped videos: {skipped_count}")

    if len(results) == 0:
        summary_lines.append("No results to display.")
    else:
        # Sort results by video name for consistent output
        results.sort(key=lambda x: x["video_name"])

        for result in results:
            summary_lines.append(
                f"{result['video_name']}, ADD-S_err: {result['add_s_err_mean']:.2f}[cm], "
                f"ADD_errs: {result['add_err_mean']:.2f}[cm], "
                f"ADD-S_AUC: {result['add_s_auc']:.2f}, ADD_AUC: {result['add_auc']:.2f}, "
                f"mesh_CD: {result['mesh_cd_cm']:.3f}[cm]"
            )

        # Compute averages
        avg_add_s_err = np.mean([r["add_s_err_mean"] for r in results])
        avg_add_err = np.mean([r["add_err_mean"] for r in results])
        avg_add_s_auc = np.mean([r["add_s_auc"] for r in results])
        avg_add_auc = np.mean([r["add_auc"] for r in results])
        valid_mesh = [r["mesh_cd_cm"] for r in results if np.isfinite(r["mesh_cd_cm"])]
        avg_mesh_cd = float(np.mean(valid_mesh)) if len(valid_mesh) > 0 else np.inf

        summary_lines.append("-" * 80)
        summary_lines.append(
            f"Average, ADD-S_err: {avg_add_s_err:.2f}[cm], "
            f"ADD_errs: {avg_add_err:.2f}[cm], "
            f"ADD-S_AUC: {avg_add_s_auc:.2f}, ADD_AUC: {avg_add_auc:.2f}, "
            f"mesh_CD: {avg_mesh_cd:.3f}[cm]"
        )
    summary_lines.append("=" * 80)

    for line in summary_lines:
        print(line)

    summary_path = os.path.join(out_dir, "summary_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ho3d_all",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d/ho3d_single.yaml",
    )
    parser.add_argument(
        "--rerun_existing",
        dest="rerun_existing",
        action="store_true",
        help="Rerun a sequence even if its output directory already exists.",
    )
    parser.add_argument(
        "--skip_existing",
        dest="rerun_existing",
        action="store_false",
        help="Skip a sequence if its output directory already exists.",
    )
    parser.set_defaults(rerun_existing=True)

    args = parser.parse_args()

    run_ho3d_all(
        args.data_path,
        args.out_dir,
        args.config_path,
        rerun_existing=bool(args.rerun_existing),
    )
