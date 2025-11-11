import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))


import os
import glob
import argparse

import cv2
import time
from omegaconf import OmegaConf
import torch
import numpy as np

from point2pose.io.sources.dataset.datareader import Ho3dReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.data_types.frame import Frame
from point2pose.utils.visualization import visualize_and_save_tracking_results


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

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")
    os.system(f"rm -rf {vis_folder} && mkdir -p {vis_folder}")

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params
    pipeline = PipelineSingleProcess(cfg)

    gt_bbox_minmax = gt_bbox_minmax_from_mesh(reader)
    out_poses = []
    gt_poses = []

    for i, color_file in enumerate(reader.color_files):
        color = cv2.imread(color_file)
        H, W = color.shape[:2]
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = reader.get_depth(i)

        id_str = reader.id_strs[i]
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
        out_poses.append(out_pose)

        # get gt pose from the reader
        gt_pose = reader.get_gt_pose(i)
        gt_poses.append(gt_pose)

        if i == 0:
            pipeline.objects[0].init_pose = gt_pose

        visualize_and_save_tracking_results(
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

    print(f"Done {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument("--video_name", type=str, default="MPM10")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ho3d_single",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d/ho3d_single.yaml",
    )

    args = parser.parse_args()

    run_ho3d_single(args.data_path, args.video_name, args.out_dir, args.config_path)
