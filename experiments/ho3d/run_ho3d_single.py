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

from point2pose.io.sources.dataset.datareader import Ho3dReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.data_types.frame import Frame


def run_ho3d_single(data_path: str, video_name: str, out_dir: str, config_path: str):
    video_path = os.path.join(data_path, os.path.join("evaluation/", video_name))

    reader = Ho3dReader(video_path, data_path)

    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    # if os.path.exists(os.path.join(out_folder, "ob_in_cam")):
    #     pose_files = sorted(glob.glob(os.path.join(out_folder, "ob_in_cam", "*.txt")))
    #     if len(pose_files) == len(reader.color_files):
    #         print(f"{out_folder} done before, skip")
    #     return

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")

    cfg = OmegaConf.load(config_path)

    pipeline = PipelineSingleProcess(cfg)

    for i, color_file in enumerate(reader.color_files):
        color = cv2.imread(color_file)
        H, W = color.shape[:2]
        depth = reader.get_depth(i)

        frame = Frame(color, depth, reader.K)

        id_str = reader.id_strs[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        # Create Frame object
        frame = Frame(
            id=i,
            rgb=color,
            depth=depth,
            mask=mask,
            intrinsics=reader.K,
            depth_factor=1.0,
            timestamp=time.time(),
        )

        pipeline.step(frame)

    print(f"Done {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/justin/data/HO3D_V3/")
    parser.add_argument("--video_name", type=str, default="MPM10")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/debug/ho3d_single",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ho3d/ho3d_single.yaml",
    )

    args = parser.parse_args()

    run_ho3d_single(args.data_path, args.video_name, args.out_dir, args.config_path)
