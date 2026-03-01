import sys
from pathlib import Path
import os
import argparse
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.ycbinisaac.run_ycbinisaac_single import run_ycbinisaac_single


def get_all_video_names(data_path):
    """Get all valid YCBInIsaac sequence folder names from data_path."""
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


def run_ycbinisaac_all(data_path: str, out_dir: str, config_path: str, model_path: str):
    """Process all YCBInIsaac sequences and print a summary."""
    video_names = get_all_video_names(data_path)
    print(f"Found {len(video_names)} videos to process: {video_names}")
    print("-" * 80)

    results = []
    for idx, video_name in enumerate(video_names, 1):
        print(f"\n[{idx}/{len(video_names)}] Processing video: {video_name}")
        try:
            result = run_ycbinisaac_single(
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
    parser.add_argument("--data_path", type=str, default="/home/justin/data/test")
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
    args = parser.parse_args()

    run_ycbinisaac_all(
        data_path=args.data_path,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
    )
