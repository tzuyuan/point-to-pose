import sys
from pathlib import Path
import os
import argparse

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from point2pose.io.sources.dataset.datareader import Ho3dReader
from point2pose.utils.transform import inverse_SE3
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


def load_poses_from_metadata(metadata_path: str):
    """
    Load poses from metadata NPZ file.

    Args:
        metadata_path: Path to metadata NPZ file

    Returns:
        poses: (N, 4, 4) array of pose matrices
        frame_ids: (N,) array of frame IDs
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    data = np.load(metadata_path, allow_pickle=True)

    if "obj_pose" not in data:
        raise KeyError(f"obj_pose not found in metadata file: {metadata_path}")

    poses = data["obj_pose"]
    frame_ids = data.get("frame_id", np.arange(len(poses)))

    # Handle different pose shapes: (N, 4, 4) or (4, 4, N) or object array
    if poses.ndim == 3:
        if poses.shape[0] == 4 and poses.shape[1] == 4:
            # Shape (4, 4, N) -> transpose to (N, 4, 4)
            poses = np.transpose(poses, (2, 0, 1))
        elif poses.shape[1] == 4 and poses.shape[2] == 4:
            # Shape (N, 4, 4) -> already correct
            pass
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")
    elif poses.dtype == object:
        # Handle object array of matrices
        poses_list = []
        valid_frame_ids = []
        for i, pose in enumerate(poses):
            if (
                pose is not None
                and not np.any(np.isnan(pose))
                and not np.any(np.isinf(pose))
            ):
                poses_list.append(pose)
                valid_frame_ids.append(frame_ids[i])
        poses = np.array(poses_list)
        frame_ids = np.array(valid_frame_ids)
    else:
        raise ValueError(
            f"Unexpected pose format: shape={poses.shape}, dtype={poses.dtype}"
        )

    return poses, frame_ids


def load_poses_from_text(pose_file: str):
    """
    Load poses from TUM format text file.

    Args:
        pose_file: Path to pose text file (obj_0_pose.txt)

    Returns:
        poses: (N, 4, 4) array of pose matrices
        frame_ids: (N,) array of frame IDs (inferred from line numbers)
    """
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    poses = []

    with open(pose_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # TUM format: timestamp tx ty tz qx qy qz qw
            # timestamp is not used but kept for format compatibility
            _ = float(parts[0])  # timestamp
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )

            # Convert quaternion to rotation matrix
            # Normalize quaternion
            q_norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            if q_norm > 1e-6:
                qx, qy, qz, qw = qx / q_norm, qy / q_norm, qz / q_norm, qw / q_norm

            # Quaternion to rotation matrix
            R = np.array(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qw * qz),
                        2 * (qx * qz + qw * qy),
                    ],
                    [
                        2 * (qx * qy + qw * qz),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qw * qx),
                    ],
                    [
                        2 * (qx * qz - qw * qy),
                        2 * (qy * qz + qw * qx),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ]
            )

            # Create pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = [tx, ty, tz]

            poses.append(pose)

    if len(poses) == 0:
        raise ValueError(f"No valid poses found in file: {pose_file}")

    # For text files, assume sequential frame IDs starting from 0
    # Note: This assumes poses are saved in the same order as frames are processed
    frame_ids = np.arange(len(poses))

    return np.array(poses), frame_ids


def evaluate_ho3d_single(
    data_path: str,
    video_name: str,
    results_dir: str,
    metadata_path: str = None,
    pose_file: str = None,
):
    """
    Evaluate HO3D single object tracking results.

    Args:
        data_path: Path to HO3D dataset root
        video_name: Name of the video sequence (e.g., "MPM10", "AP10")
        results_dir: Directory containing results (where plots will be saved)
        metadata_path: Path to metadata NPZ file (optional, will be inferred if not provided)
        pose_file: Path to pose text file (optional, alternative to metadata)
    """
    video_path = os.path.join(data_path, os.path.join("evaluation/", video_name))
    reader = Ho3dReader(video_path, data_path)
    video_name = reader.get_video_name()

    out_folder = os.path.join(results_dir, video_name, "")
    os.makedirs(out_folder, exist_ok=True)

    # Load predicted poses
    if pose_file is not None:
        print(f"Loading poses from text file: {pose_file}")
        pred_poses, pred_frame_ids = load_poses_from_text(pose_file)
    elif metadata_path is not None:
        print(f"Loading poses from metadata: {metadata_path}")
        pred_poses, pred_frame_ids = load_poses_from_metadata(metadata_path)
    else:
        # Try to infer metadata path from results directory
        possible_metadata_paths = [
            os.path.join(out_folder, "meta_data", "meata_data.npz"),
            os.path.join(results_dir, video_name, "meta_data", "meata_data.npz"),
            os.path.join(results_dir, "meta_data", "meata_data.npz"),
            os.path.join(out_folder, "meata_data.npz"),
        ]

        # Also try pose file
        possible_pose_files = [
            os.path.join(out_folder, "poses", "obj_0_pose.txt"),
            os.path.join(results_dir, video_name, "poses", "obj_0_pose.txt"),
            os.path.join(out_folder, "obj_0_pose.txt"),
        ]

        metadata_path = None
        for path in possible_metadata_paths:
            if os.path.exists(path):
                metadata_path = path
                break

        pose_file = None
        if metadata_path is None:
            for path in possible_pose_files:
                if os.path.exists(path):
                    pose_file = path
                    break

        if metadata_path is not None:
            print(f"Found metadata file: {metadata_path}")
            pred_poses, pred_frame_ids = load_poses_from_metadata(metadata_path)
        elif pose_file is not None:
            print(f"Found pose file: {pose_file}")
            pred_poses, pred_frame_ids = load_poses_from_text(pose_file)
        else:
            raise FileNotFoundError(
                f"Could not find metadata or pose file. Tried:\n"
                f"  Metadata: {possible_metadata_paths}\n"
                f"  Pose files: {possible_pose_files}\n"
                f"Please provide --metadata_path or --pose_file"
            )

    print(f"Loaded {len(pred_poses)} predicted poses")

    # Load ground truth poses
    gt_poses = []
    gt_ids = []
    num_frames = len(reader.color_files)

    for i in range(num_frames):
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is not None:
            gt_poses.append(gt_pose)
            gt_ids.append(i)

    gt_poses = np.array(gt_poses)
    gt_ids = np.array(gt_ids)

    print(f"Loaded {len(gt_poses)} ground truth poses")

    # Match predicted poses with ground truth poses by frame ID
    # Find common frame IDs
    if len(pred_frame_ids) != len(pred_poses):
        # If frame_ids don't match, assume sequential
        pred_frame_ids = np.arange(len(pred_poses))

    # Match by frame ID
    common_ids = np.intersect1d(pred_frame_ids, gt_ids)
    if len(common_ids) == 0:
        raise ValueError(
            f"No common frame IDs found between predicted ({pred_frame_ids[:5]}...) "
            f"and GT ({gt_ids[:5]}...)"
        )

    print(f"Found {len(common_ids)} common frame IDs")

    # Get poses for common frame IDs
    pred_poses_matched = []
    gt_poses_matched = []
    matched_ids = []

    for frame_id in common_ids:
        pred_idx = np.where(pred_frame_ids == frame_id)[0]
        gt_idx = np.where(gt_ids == frame_id)[0]

        if len(pred_idx) > 0 and len(gt_idx) > 0:
            pred_poses_matched.append(pred_poses[pred_idx[0]])
            gt_poses_matched.append(gt_poses[gt_idx[0]])
            matched_ids.append(frame_id)

    pred_poses_matched = np.array(pred_poses_matched)
    gt_poses_matched = np.array(gt_poses_matched)

    print(f"Matched {len(pred_poses_matched)} poses")

    # Align first frame
    pred_poses_aligned = (
        pred_poses_matched @ inverse_SE3(pred_poses_matched[0]) @ gt_poses_matched[0]
    )

    # Compute errors
    adi_errs = []
    add_errs = []
    mesh = reader.get_gt_mesh()

    for i in range(len(pred_poses_aligned)):
        adi = adi_err(pred_poses_aligned[i], gt_poses_matched[i], mesh.vertices.copy())
        add = add_err(pred_poses_aligned[i], gt_poses_matched[i], mesh.vertices.copy())
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
        pred_poses=pred_poses_aligned,
        gt_poses=gt_poses_matched,
        video_name=video_name,
        output_dir=out_folder,
        save_plots=True,
        show_plots=False,
    )

    plot_pose_error_comparison(
        pred_poses=pred_poses_aligned,
        gt_poses=gt_poses_matched,
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

    print(f"Evaluation plots saved to: {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/justin/data/HO3D_V3/",
        help="Path to HO3D dataset root",
    )
    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        default="MPM10",
        help="Name of the video sequence (e.g., MPM10, AP10)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ho3d_single",
        help="Directory containing results (where plots will be saved)",
    )
    parser.add_argument(
        "--metadata_path",
        "-m",
        default="/home/justin/code/point-to-pose/debug/pipeline/meta_data/meta_data.npz",
        type=str,
        help="Path to metadata NPZ file (optional, will be inferred if not provided)",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default=None,
        help="Path to pose text file (optional, alternative to metadata)",
    )

    args = parser.parse_args()

    evaluate_ho3d_single(
        data_path=args.data_path,
        video_name=args.video_name,
        results_dir=args.results_dir,
        metadata_path=args.metadata_path,
        pose_file=args.pose_file,
    )
