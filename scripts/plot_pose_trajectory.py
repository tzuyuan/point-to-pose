#!/usr/bin/env python3
"""
Script to load estimated poses from meta_data and GT poses from dataset,
then plot xyz position and roll/pitch/yaw angles over time.
"""
import sys
from pathlib import Path

# Add project root to Python path
script_file = Path(__file__).resolve()
project_root = script_file.parents[1]  # scripts -> point-to-pose
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Import Ho3dReader and transform utilities after path is set up
try:
    from point2pose.io.sources.dataset.datareader import Ho3dReader
    from point2pose.utils.transform import inverse_SE3
except ImportError as e:
    print(f"Warning: Could not import point2pose modules: {e}")
    print(
        "The script can still work if you provide pose files instead of using GT poses."
    )
    Ho3dReader = None
    inverse_SE3 = None


def rotation_matrix_to_euler(rot_matrix):
    """
    Convert rotation matrix to roll, pitch, yaw (Euler angles in radians).
    Uses ZYX convention (yaw-pitch-roll).

    Args:
        rot_matrix: (3, 3) rotation matrix

    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    r = R.from_matrix(rot_matrix)
    # Use 'xyz' intrinsic rotations (equivalent to 'zyx' extrinsic)
    # This gives roll (x), pitch (y), yaw (z)
    euler = r.as_euler("xyz", degrees=False)
    return euler[0], euler[1], euler[2]  # roll, pitch, yaw


def load_poses_from_meta_data(meta_data_path):
    """
    Load estimated poses from meta_data.npz file.

    Args:
        meta_data_path: Path to meta_data.npz file

    Returns:
        poses: List of (4, 4) pose matrices, or None if not found
        frame_ids: List of frame IDs
    """
    if not os.path.exists(meta_data_path):
        print(f"Warning: meta_data file not found at {meta_data_path}")
        return None, None

    try:
        data = np.load(meta_data_path, allow_pickle=True)

        # Try to find pose data in meta_data
        # Common field names: 'obj_pose', 'pose', 'poses', etc.
        poses = None
        frame_ids = None

        # Check for pose in fixed fields
        if "obj_pose" in data:
            poses = data["obj_pose"]
        elif "pose" in data:
            poses = data["pose"]
        elif "poses" in data:
            poses = data["poses"]

        # Handle object array case (if poses are stored as object array)
        if poses is not None and poses.dtype == object:
            # Convert object array to list of matrices
            poses_list = []
            for i in range(len(poses)):
                if poses[i] is not None:
                    poses_list.append(np.array(poses[i]))
                else:
                    poses_list.append(None)
            poses = poses_list

        # Get frame IDs
        if "frame_id" in data:
            frame_ids = data["frame_id"]
        elif "frame_ids" in data:
            frame_ids = data["frame_ids"]

        # If poses are stored as a list/array of 4x4 matrices
        if poses is not None:
            # If it's already a list (from object array handling), keep it
            if isinstance(poses, list):
                # Convert list to numpy array if all elements are arrays
                try:
                    poses_array = np.array([p for p in poses if p is not None])
                    if poses_array.ndim == 3 and poses_array.shape[1:] == (4, 4):
                        # Reconstruct list with None values preserved
                        poses_list = []
                        valid_idx = 0
                        for p in poses:
                            if p is None:
                                poses_list.append(None)
                            else:
                                poses_list.append(poses_array[valid_idx])
                                valid_idx += 1
                        poses = poses_list
                except (ValueError, TypeError):
                    pass  # Keep as list
            else:
                poses = np.array(poses)
                # Ensure it's (N, 4, 4) shape
                if poses.ndim == 2 and poses.shape == (4, 4):
                    poses = poses[np.newaxis, ...]
                elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
                    pass  # Already correct shape
                else:
                    print(f"Warning: Unexpected pose shape: {poses.shape}")
                    return None, None

        return poses, frame_ids

    except (IOError, ValueError, KeyError) as e:
        print(f"Error loading meta_data: {e}")
        return None, None


def load_poses_from_pose_file(pose_file_path):
    """
    Load poses from TUM format pose file.

    Args:
        pose_file_path: Path to obj_i_pose.txt file

    Returns:
        poses: List of (4, 4) pose matrices
        timestamps: List of timestamps
    """
    if not os.path.exists(pose_file_path):
        return None, None

    poses = []
    timestamps = []

    try:
        with open(pose_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )

                # Convert quaternion to rotation matrix
                r = R.from_quat([qx, qy, qz, qw])
                rot_matrix = r.as_matrix()

                # Build 4x4 pose matrix
                pose = np.eye(4)
                pose[:3, 3] = [tx, ty, tz]
                pose[:3, :3] = rot_matrix

                poses.append(pose)
                timestamps.append(timestamp)

        return np.array(poses), np.array(timestamps)

    except (IOError, ValueError) as e:
        print(f"Error loading pose file: {e}")
        return None, None


def load_gt_poses(reader, frame_indices=None):
    """
    Load ground truth poses from dataset reader.

    Args:
        reader: Ho3dReader instance
        frame_indices: Optional list of frame indices to load

    Returns:
        poses: List of (4, 4) pose matrices
        valid_indices: List of frame indices where GT pose is available
    """
    poses = []
    valid_indices = []

    if frame_indices is None:
        frame_indices = range(len(reader))

    for i in frame_indices:
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is not None:
            poses.append(gt_pose)
            valid_indices.append(i)
        else:
            # Pad with None to maintain alignment
            poses.append(None)
            valid_indices.append(i)

    return poses, valid_indices


def extract_xyz_rpy(poses):
    """
    Extract xyz translation and roll/pitch/yaw from pose matrices.

    Args:
        poses: List or array of (4, 4) pose matrices

    Returns:
        xyz: (N, 3) array of x, y, z positions
        rpy: (N, 3) array of roll, pitch, yaw angles (in degrees)
    """
    xyz = []
    rpy = []

    for pose in poses:
        if pose is None:
            xyz.append([np.nan, np.nan, np.nan])
            rpy.append([np.nan, np.nan, np.nan])
        else:
            # Extract translation
            tx, ty, tz = pose[:3, 3]
            xyz.append([tx, ty, tz])

            # Extract rotation and convert to Euler angles
            rot_matrix = pose[:3, :3]
            roll, pitch, yaw = rotation_matrix_to_euler(rot_matrix)
            rpy.append([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])

    return np.array(xyz), np.array(rpy)


def plot_pose_trajectory(
    pred_poses,
    gt_poses,
    frame_ids=None,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(16, 12),
):
    """
    Plot xyz position and roll/pitch/yaw angles over time for both estimated and GT poses.

    Args:
        pred_poses: List/array of (4, 4) predicted pose matrices
        gt_poses: List/array of (4, 4) GT pose matrices (can contain None)
        frame_ids: Optional list of frame IDs
        video_name: Name of the video/sequence
        output_dir: Directory to save plots
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        figsize: Figure size
    """
    # Extract xyz and rpy
    pred_xyz, pred_rpy = extract_xyz_rpy(pred_poses)
    gt_xyz, gt_rpy = extract_xyz_rpy(gt_poses)

    # Check if GT poses are available
    has_gt = gt_poses is not None and any(p is not None for p in gt_poses)

    # Create frame indices
    if frame_ids is None:
        frame_ids = np.arange(len(pred_poses))
    else:
        frame_ids = np.array(frame_ids)

    # Create figure with subplots
    # Layout: Left column (xyz positions), Right column (roll/pitch/yaw angles)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Left column: Position plots (X, Y, Z)
    # Plot 1: X position (left, top)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        frame_ids,
        pred_xyz[:, 0],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax1.plot(
            frame_ids,
            gt_xyz[:, 0],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("X Position (m)", fontsize=12)
    ax1.set_title("X Position Over Time", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Y position (left, middle)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        frame_ids,
        pred_xyz[:, 1],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax2.plot(
            frame_ids,
            gt_xyz[:, 1],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Y Position (m)", fontsize=12)
    ax2.set_title("Y Position Over Time", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Z position (left, bottom)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(
        frame_ids,
        pred_xyz[:, 2],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax3.plot(
            frame_ids,
            gt_xyz[:, 2],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax3.set_xlabel("Frame", fontsize=12)
    ax3.set_ylabel("Z Position (m)", fontsize=12)
    ax3.set_title("Z Position Over Time", fontsize=14, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Right column: Rotation plots (Roll, Pitch, Yaw)
    # Plot 4: Roll (right, top)
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.plot(
        frame_ids,
        pred_rpy[:, 0],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax4.plot(
            frame_ids,
            gt_rpy[:, 0],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax4.set_xlabel("Frame", fontsize=12)
    ax4.set_ylabel("Roll (degrees)", fontsize=12)
    ax4.set_title("Roll Angle Over Time", fontsize=14, fontweight="bold")
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Pitch (right, middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        frame_ids,
        pred_rpy[:, 1],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax5.plot(
            frame_ids,
            gt_rpy[:, 1],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax5.set_xlabel("Frame", fontsize=12)
    ax5.set_ylabel("Pitch (degrees)", fontsize=12)
    ax5.set_title("Pitch Angle Over Time", fontsize=14, fontweight="bold")
    ax5.legend(loc="best", fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Yaw (right, bottom)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(
        frame_ids,
        pred_rpy[:, 2],
        label="Estimated",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    if has_gt:
        ax6.plot(
            frame_ids,
            gt_rpy[:, 2],
            label="GT",
            color="coral",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
    ax6.set_xlabel("Frame", fontsize=12)
    ax6.set_ylabel("Yaw (degrees)", fontsize=12)
    ax6.set_title("Yaw Angle Over Time", fontsize=14, fontweight="bold")
    ax6.legend(loc="best", fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Overall title
    title_str = "Pose Trajectory Comparison"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "pose_trajectory.png"
        if video_name:
            plot_filename = f"pose_trajectory_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved pose trajectory plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot pose trajectories from meta_data and GT"
    )
    parser.add_argument(
        "--meta_data_path",
        default="/home/justin/code/point-to-pose/debug/pipeline/meta_data/meata_data.npz",
        type=str,
        help="Path to meta_data.npz file or pose file (obj_i_pose.txt)",
    )
    parser.add_argument(
        "--data_path",
        default="/home/justin/data/HO3D_V3/",
        type=str,
        help="Path to HO3D dataset root directory",
    )
    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        default=None,
        help="Video/sequence name (e.g., MPM10). If not provided, will try to infer from meta_data or skip GT loading.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/justin/code/point-to-pose/debug/pose_trajectory/",
        help="Directory to save plots (default: same as meta_data directory)",
    )
    parser.add_argument(
        "--pose_file",
        action="store_true",
        help="If set, treat meta_data_path as a pose file (obj_i_pose.txt) instead of meta_data.npz",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.meta_data_path)

    # Load estimated poses
    if args.pose_file:
        print(f"Loading poses from pose file: {args.meta_data_path}")
        pred_poses, _ = load_poses_from_pose_file(args.meta_data_path)
        frame_ids = None
    else:
        print(f"Loading poses from meta_data: {args.meta_data_path}")
        pred_poses, frame_ids = load_poses_from_meta_data(args.meta_data_path)

    if pred_poses is None:
        print("Error: Could not load estimated poses. Exiting.")
        return

    print(f"Loaded {len(pred_poses)} estimated poses")

    # Load GT poses if video_name and data_path are provided
    gt_poses = None
    if args.video_name is not None and args.data_path is not None:
        if Ho3dReader is None:
            print("Warning: Ho3dReader is not available. Skipping GT pose loading.")
            print("Plotting only estimated poses...")
        else:
            try:
                video_path = os.path.join(args.data_path, "evaluation", args.video_name)
                reader = Ho3dReader(video_path, args.data_path)

                print(f"Loading GT poses for {len(reader)} frames...")
                gt_poses, _ = load_gt_poses(reader)
                print(f"Loaded {len(gt_poses)} GT poses")
            except (
                FileNotFoundError,
                IOError,
                KeyError,
                ValueError,
                AttributeError,
            ) as e:
                print(f"Warning: Could not load GT poses: {e}")
                print("Plotting only estimated poses...")
                gt_poses = None
    else:
        if args.video_name is None:
            print("Warning: --video_name not provided. Skipping GT pose loading.")
        if args.data_path is None:
            print("Warning: --data_path not provided. Skipping GT pose loading.")
        print("Plotting only estimated poses...")

    # Handle GT poses
    if gt_poses is None:
        # Create dummy GT poses (all None) for plotting
        gt_poses = [None] * len(pred_poses)
        video_name_for_plot = args.video_name or "estimated_only"
    else:
        video_name_for_plot = args.video_name

    # Align lengths if GT poses are available
    has_valid_gt = gt_poses is not None and any(p is not None for p in gt_poses)

    if has_valid_gt:
        min_len = min(len(pred_poses), len(gt_poses))
        pred_poses = pred_poses[:min_len]
        gt_poses = gt_poses[:min_len]

        # Convert to numpy arrays - filter out None values and create proper arrays
        # First, find valid indices (where both poses are not None)
        if isinstance(pred_poses, list):
            pred_poses_list = pred_poses
        else:
            pred_poses_list = list(pred_poses)

        if isinstance(gt_poses, list):
            gt_poses_list = gt_poses
        else:
            gt_poses_list = list(gt_poses)

        # Convert to numpy arrays, handling None values
        pred_poses_array = []
        gt_poses_array = []
        for i in range(min_len):
            if pred_poses_list[i] is not None:
                pred_poses_array.append(np.array(pred_poses_list[i]))
            else:
                pred_poses_array.append(None)
            if gt_poses_list[i] is not None:
                gt_poses_array.append(np.array(gt_poses_list[i]))
            else:
                gt_poses_array.append(None)

        # Find first valid frame for alignment
        first_valid_idx = None
        for i in range(min_len):
            if (
                pred_poses_array[i] is not None
                and gt_poses_array[i] is not None
                and isinstance(pred_poses_array[i], np.ndarray)
                and isinstance(gt_poses_array[i], np.ndarray)
                and pred_poses_array[i].shape == (4, 4)
                and gt_poses_array[i].shape == (4, 4)
                and not np.any(np.isnan(pred_poses_array[i]))
                and not np.any(np.isnan(gt_poses_array[i]))
            ):
                first_valid_idx = i
                break

        # Align first frame: align predicted poses to GT by setting first predicted pose = first GT pose
        # This ensures both trajectories start at the same pose for better comparison
        if first_valid_idx is not None and inverse_SE3 is not None:
            print(
                f"Aligning first frame (frame {first_valid_idx}): setting first predicted pose = first GT pose..."
            )
            # Get first valid poses
            pred_first = pred_poses_array[first_valid_idx]
            gt_first = gt_poses_array[first_valid_idx]

            # Alignment: pred_poses = pred_poses @ inv(pred_poses[0]) @ gt_poses[0]
            # This transforms all predicted poses so that the first one matches the first GT
            alignment_transform = inverse_SE3(pred_first) @ gt_first

            # Apply alignment to all valid predicted poses
            for i in range(min_len):
                if pred_poses_array[i] is not None and isinstance(
                    pred_poses_array[i], np.ndarray
                ):
                    if pred_poses_array[i].shape == (4, 4):
                        pred_poses_array[i] = pred_poses_array[i] @ alignment_transform

            pred_poses = pred_poses_array
            gt_poses = gt_poses_array
            print("First frame alignment completed.")
        else:
            if first_valid_idx is None:
                print("Warning: Cannot align first frame (no valid first poses found).")
            if inverse_SE3 is None:
                print(
                    "Warning: inverse_SE3 not available. Skipping first frame alignment."
                )
            # Still convert to arrays even if we can't align
            pred_poses = pred_poses_array
            gt_poses = gt_poses_array
    else:
        min_len = len(pred_poses)
        # Convert to numpy array if it's a list
        if isinstance(pred_poses, list):
            # Handle list of poses (may contain None)
            pred_poses_array = []
            for p in pred_poses:
                if p is not None:
                    pred_poses_array.append(np.array(p))
                else:
                    pred_poses_array.append(None)
            pred_poses = pred_poses_array

    if frame_ids is None:
        frame_ids = np.arange(min_len)
    else:
        frame_ids = frame_ids[:min_len]

    print(f"Plotting {min_len} frames...")

    # Plot trajectories
    plot_pose_trajectory(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        frame_ids=frame_ids,
        video_name=video_name_for_plot,
        output_dir=args.output_dir,
        save_plots=True,
        show_plots=args.show_plots,
    )

    print("Done!")


if __name__ == "__main__":
    main()
