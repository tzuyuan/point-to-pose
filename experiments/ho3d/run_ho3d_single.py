import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))


import os
import argparse
import shutil

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np

from point2pose.io.sources.dataset.datareader import Ho3dReader
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


# Build a GT bbox (min/max in object frame) once per sequence
def gt_bbox_minmax_from_mesh(reader):
    # reader.get_gt_mesh() should return the HO3D object mesh in meters, object frame
    mesh = reader.get_gt_mesh()
    bmin, bmax = mesh.bounds  # shape (3,), (3,)
    return np.vstack([bmin.astype(float), bmax.astype(float)])  # (2,3)


def run_ho3d_single(data_path: str, video_name: str, out_dir: str, config_path: str):
    video_path = os.path.join(data_path, os.path.join("evaluation/", video_name))

    reader = Ho3dReader(video_path, data_path)

    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    vis_folder = os.path.join(out_folder, "output_images")
    with_gt_folder = os.path.join(out_folder, "with_gt")
    mesh_folder = os.path.join(out_folder, "mesh")
    vis_folder_visible_unc = os.path.join(out_folder, "visible_uncertainty")
    vis_folder_visible_valid = os.path.join(out_folder, "visible_valid")
    vis_folder_reg_used_valid = os.path.join(out_folder, "registration_used_valid")
    vis_folder_reg_corr = os.path.join(out_folder, "registration_correspondence")

    with_gt_folder_visible_unc = os.path.join(out_folder, "with_gt_visible_uncertainty")
    with_gt_folder_visible_valid = os.path.join(out_folder, "with_gt_visible_valid")
    with_gt_folder_reg_used_valid = os.path.join(
        out_folder, "with_gt_registration_used_valid"
    )
    with_gt_folder_reg_corr = os.path.join(
        out_folder, "with_gt_registration_correspondence"
    )

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(with_gt_folder, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(vis_folder_visible_unc, exist_ok=True)
    os.makedirs(vis_folder_visible_valid, exist_ok=True)
    os.makedirs(vis_folder_reg_used_valid, exist_ok=True)
    os.makedirs(vis_folder_reg_corr, exist_ok=True)
    os.makedirs(with_gt_folder_visible_unc, exist_ok=True)
    os.makedirs(with_gt_folder_visible_valid, exist_ok=True)
    os.makedirs(with_gt_folder_reg_used_valid, exist_ok=True)
    os.makedirs(with_gt_folder_reg_corr, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params
    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    cfg.pipeline.params.sdf_mesh_save_every = 1

    # Set meta_data_save_path to save in the results folder for this dataset
    if cfg.pipeline.params.get("save_meta_data", True):
        meta_data_folder = os.path.join(out_folder, "meta_data")
        cfg.pipeline.params.meta_data_save_path = meta_data_folder

    if cfg.pipeline.type != "modular":
        raise ValueError(
            f"Only 'modular' pipeline is supported, got: {cfg.pipeline.type}"
        )
    pipeline = ModularPipeline(cfg)

    gt_bbox_minmax = gt_bbox_minmax_from_mesh(reader)
    out_poses = []
    gt_poses = []
    gt_ids = []

    for i, color_file in enumerate(reader.color_files):
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

        # display_frame = visualize_and_save_tracking_results(
        #     frame=frame,
        #     objects=pipeline.objects,
        #     track_table=pipeline.track_table,
        #     frame_id=i,
        #     visualize_points=vis_cfg.visualize_points,
        #     points_vis_method=vis_cfg.points_vis_method,
        #     save_images=vis_cfg.save_images,
        #     output_image_dir=vis_folder,
        #     camera_intrinsics=reader.K,
        #     bbox_min_max=gt_bbox_minmax,
        # )

        # visualize_and_save_tracking_results_with_gt(
        #     frame=frame,
        #     objects=pipeline.objects,
        #     track_table=pipeline.track_table,
        #     est_result_frame=display_frame,
        #     gt_pose=gt_pose,
        #     frame_id=i,
        #     visualize_points=vis_cfg.visualize_points,
        #     points_vis_method=vis_cfg.points_vis_method,
        #     save_images=vis_cfg.save_images,
        #     output_image_dir=with_gt_folder,
        #     camera_intrinsics=reader.K,
        #     bbox_min_max=gt_bbox_minmax,
        #     gt_bbox_min_max=gt_bbox_minmax,
        #     pred_pose_color=(0, 255, 0),
        # )

        # --- A) visible_uncertainty ---
        display_frame_visible_unc = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="visible_uncertainty",  # force this mode
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder_visible_unc,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
        )

        visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame_visible_unc,
            gt_pose=gt_pose,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="visible_uncertainty",
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder_visible_unc,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
            gt_bbox_min_max=gt_bbox_minmax,
            pred_pose_color=(0, 255, 0),
        )

        # --- D) registration_correspondence ---
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

        # --- C) registration_used_valid ---
        display_frame_reg_used_valid = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="registration_used_valid",
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder_reg_used_valid,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
        )

        visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame_reg_used_valid,
            gt_pose=gt_pose,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="registration_used_valid",
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder_reg_used_valid,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
            gt_bbox_min_max=gt_bbox_minmax,
            pred_pose_color=(0, 255, 0),
        )

        # --- B) visible_valid ---
        display_frame_visible_valid = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="visible_valid",  # force this mode
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder_visible_valid,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
        )

        visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame_visible_valid,
            gt_pose=gt_pose,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="visible_valid",
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder_visible_valid,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_minmax,
            gt_bbox_min_max=gt_bbox_minmax,
            pred_pose_color=(0, 255, 0),
        )

        # if i == 100:
        #     break

    mesh_paths = export_final_meshes_from_pipeline(pipeline, mesh_folder)
    if len(gt_poses) == 0:
        print(
            f"Warning: No GT poses found for video {video_name}, skipping evaluation."
        )
        return

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument("--video_name", "-v", type=str, default="AP12")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ho3d_single",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d/ho3d_single.yaml",
    )

    args = parser.parse_args()

    run_ho3d_single(args.data_path, args.video_name, args.out_dir, args.config_path)
