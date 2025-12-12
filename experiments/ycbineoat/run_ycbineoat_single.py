import sys
from pathlib import Path
import time
import os
import argparse
import glob

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np
import trimesh

from point2pose.io.sources.dataset.datareader import YcbineoatReader
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


def load_ycb_mesh(reader, model_root):
    """
    Load the YCB object mesh.
    Overrides the hardcoded path in YcbineoatReader.get_gt_mesh
    """
    video_name = reader.get_video_name()
    ob_name = reader.videoname_to_object.get(video_name)

    if not ob_name:
        raise ValueError(f"No object mapping found for video: {video_name}")

    # Common YCB model structure: {model_root}/{ob_name}/textured_simple.obj
    # or {model_root}/models/{ob_name}/textured_simple.obj

    candidates = [
        os.path.join(model_root, ob_name, "textured_simple.obj"),
        os.path.join(model_root, "models", ob_name, "textured_simple.obj"),
        os.path.join(
            model_root, ob_name, "google_16k", "textured.obj"
        ),  # Some versions
    ]

    mesh_path = None
    for p in candidates:
        if os.path.exists(p):
            mesh_path = p
            break

    if mesh_path is None:
        raise FileNotFoundError(
            f"Could not find mesh for {ob_name} in {model_root}. Checked: {candidates}"
        )

    print(f"Loading mesh from {mesh_path}")
    return trimesh.load(mesh_path)


def gt_bbox_minmax_from_mesh(mesh):
    bmin, bmax = mesh.bounds  # shape (3,), (3,)
    return np.vstack([bmin.astype(float), bmax.astype(float)])  # (2,3)


def run_ycbineoat_single(
    data_path: str, video_name: str, out_dir: str, config_path: str, model_path: str
):
    # data_path should be the root containing the video folders
    # video_name is the folder name (e.g. bleach0)
    video_path = os.path.join(data_path, video_name)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    reader = YcbineoatReader(video_path)

    # We use the video name from the reader which is derived from the path
    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    vis_folder = os.path.join(out_folder, "output_images")
    with_gt_folder = os.path.join(out_folder, "with_gt")

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")
    os.system(f"rm -rf {vis_folder} && mkdir -p {vis_folder}")
    os.system(f"rm -rf {with_gt_folder} && mkdir -p {with_gt_folder}")

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params

    # Set meta_data_save_path to save in the results folder for this dataset
    if cfg.pipeline.params.get("save_meta_data", True):
        meta_data_folder = os.path.join(out_folder, "meta_data")
        cfg.pipeline.params.meta_data_save_path = meta_data_folder

    if cfg.pipeline.type == "single_process":
        pipeline = PipelineSingleProcess(cfg)
    elif cfg.pipeline.type == "modular":
        pipeline = ModularPipeline(cfg)
    else:
        raise ValueError(f"Invalid pipeline type: {cfg.pipeline.type}")

    # Load mesh using helper instead of reader.get_gt_mesh()
    mesh = load_ycb_mesh(reader, model_path)
    gt_bbox_minmax = gt_bbox_minmax_from_mesh(mesh)

    out_poses = []
    gt_poses = []
    gt_ids = []

    for i, color_file in enumerate(reader.color_files):
        # YcbineoatReader.get_color returns RGB resized, but we can also just read the file directly
        # consistent with run_ho3d_single.py which reads the file and converts.
        # However, Reader might handle resizing if downscale != 1.
        # Let's rely on the reader to get dimensions right if downscale is used,
        # but Ho3dReader in run_ho3d_single reads directly.
        # YcbineoatReader stores W, H scaled.

        # Let's use cv2.imread and resize manually to match reader.W/H if needed,
        # or use reader.get_color(i) which returns RGB.
        # pipeline expects RGB usually.

        # reader.get_color(i) returns RGB image resized to self.W, self.H
        rgb = reader.get_color(i)

        # reader.get_depth(i) returns depth in meters, resized
        depth = reader.get_depth(i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask = reader.get_mask(
            i
        )  # returns uint8 mask 0 or 1 (or 255?) - verify datareader
        # YcbineoatReader.get_mask: returns (H,W) 0 or 1 (boolean cast to uint8)

        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        # Create Frame object
        frame = Frame(
            id=i,
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=reader.K,
            depth_factor=1.0,  # Reader already converts to meters
            timestamp=time.time(),
        )

        # visualize color, depth, and mask
        # cv2.imshow("color", frame.rgb)
        # cv2.imshow("depth", frame.depth)
        # cv2.imshow("mask", frame.mask)
        # cv2.waitKey(0)

        # get out pose from the pipeline
        out_pose = pipeline.step(frame)
        out_poses.append(out_pose.reshape(4, 4))

        # get gt pose from the reader
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is not None:
            gt_ids.append(i)
            gt_poses.append(gt_pose)

        if i == 0 and gt_pose is not None:
            pipeline.objects[0].init_pose = gt_pose
        elif i == 0:
            print("Warning: No GT pose for first frame!")

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

    gt_poses = np.array(gt_poses)
    # Filter pred_poses to those that have GT
    if len(gt_ids) > 0:
        pred_poses = np.array(out_poses)[gt_ids]
    else:
        print("No GT poses found, skipping evaluation.")
        return

    ######### Align first frame
    if len(pred_poses) > 0 and len(gt_poses) > 0:
        pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]

        adi_errs = []
        add_errs = []

        # mesh already loaded

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
    else:
        print("Not enough poses for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults adapted for typical YCB structure or left generic
    parser.add_argument(
        "--data_path",
        "-d",
        default="/home/justin/data/YCBInEOAT",
        type=str,
        help="Root directory containing video folders (e.g. /path/to/YCB_Video/data)",
    )
    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        default="bleach0",
        help="Name of the video folder (e.g. 0048 or bleach0)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/ycbineoat_single",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="./configs/ycbineoat/ycbineoat_single.yaml",  # Default to HO3D config as base, user might want to change
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default="/home/justin/data/YCBInEOAT/YCB_models_with_ply",
        type=str,
        help="Root directory for YCB models (e.g. /path/to/YCB_Video/models)",
    )

    args = parser.parse_args()

    # Convert relative paths to absolute if needed, or rely on user providing valid paths

    run_ycbineoat_single(
        args.data_path, args.video_name, args.out_dir, args.config_path, args.model_path
    )
