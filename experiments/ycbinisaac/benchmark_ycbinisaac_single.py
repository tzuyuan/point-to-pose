import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.ycbinisaac.benchmark_ycbinisaac_all import benchmark_ycbinisaac_single


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/justin/data/YCBMultiTrack_new"
    )
    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        required=True,
        help="Sequence folder name under --data_path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbmultitrackreal",
        help="Root results directory containing <video_name>/meta_data/meta_data.npz.",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ycbinisaac/ycbinisaac_single.yaml",
        help="Unused in metadata-only benchmarking; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/home/justin/data/HO3D_V3/models",
        help="Root directory of YCB models.",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Directory for benchmark tables/plots (default: <out_dir>/<video_name>/benchmark_summary).",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key in metadata to use: auto, obj_pose_all, obj_pose, pose_local, pose_frontend.",
    )
    parser.add_argument(
        "--skip_mesh_cd",
        action="store_true",
        help="Skip mesh Chamfer-distance evaluation for faster benchmarking.",
    )
    args = parser.parse_args()

    benchmark_ycbinisaac_single(
        data_path=args.data_path,
        video_name=args.video_name,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
        summary_dir=args.summary_dir,
        pose_key=args.pose_key,
        skip_mesh_cd=args.skip_mesh_cd,
    )
