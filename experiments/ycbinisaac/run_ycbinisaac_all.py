import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.ycbinisaac.benchmark_ycbinisaac_all import (
    benchmark_ycbinisaac_all,
    get_all_video_names,
)
from experiments.ycbinisaac.run_ycbinisaac_single import run_ycbinisaac_single

__all__ = ["benchmark_ycbinisaac_all", "get_all_video_names", "run_ycbinisaac_all"]


def run_ycbinisaac_all(
    data_path: str,
    out_dir: str,
    config_path: str,
    model_path: str,
    summary_dir: str | None = None,
    pose_key: str = "auto",
):
    """
    Run the full pipeline for every valid sequence and then benchmark from metadata.
    """
    video_names = get_all_video_names(data_path)
    print(f"Found {len(video_names)} videos to run: {video_names}")
    print("-" * 80)

    run_failures = []
    for idx, video_name in enumerate(video_names, 1):
        print(f"\n[{idx}/{len(video_names)}] Running sequence: {video_name}")
        try:
            run_ycbinisaac_single(
                data_path=data_path,
                video_name=video_name,
                out_dir=out_dir,
                config_path=config_path,
                model_path=model_path,
            )
        except Exception as exc:
            print(f"Error running sequence {video_name}: {exc}")
            run_failures.append({"video_name": video_name, "error": str(exc)})
            traceback.print_exc()

    if len(run_failures) > 0:
        print("\n" + "=" * 80)
        print(
            f"Completed run stage with {len(run_failures)} failed sequence(s). "
            "Proceeding to benchmark stage."
        )
        print("=" * 80)

    benchmark_result = benchmark_ycbinisaac_all(
        data_path=data_path,
        out_dir=out_dir,
        config_path=config_path,
        model_path=model_path,
        summary_dir=summary_dir,
        pose_key=pose_key,
    )
    benchmark_result["run_failures"] = run_failures
    return benchmark_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/justin/data/YCBMultiTrack_new"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbinisaac_all",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ycbinisaac/ycbinisaac_single.yaml",
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
        help="Directory for benchmark tables/plots (default: <out_dir>/benchmark_summary).",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key in metadata to use: auto, obj_pose_all, obj_pose, pose_local, pose_frontend.",
    )
    args = parser.parse_args()

    run_ycbinisaac_all(
        data_path=args.data_path,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
        summary_dir=args.summary_dir,
        pose_key=args.pose_key,
    )
