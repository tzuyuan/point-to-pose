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
        return None, None, None, None

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

        # Helper function to reconstruct ragged fields
        def reconstruct_ragged_field(data, field_name):
            """Reconstruct a ragged field from _data, _offsets, _lengths."""
            data_key = f"{field_name}_data"
            offsets_key = f"{field_name}_offsets"
            lengths_key = f"{field_name}_lengths"

            if data_key not in data:
                return None

            data_flat = data[data_key]
            offsets = data[offsets_key]
            lengths = data[lengths_key]

            # Reconstruct ragged array
            result = []
            for i in range(len(lengths)):
                start = offsets[i]
                end = start + lengths[i]
                if lengths[i] > 0:
                    result.append(data_flat[start:end])
                else:
                    result.append(np.array([]))
            return result

        # Get key point frames (frame IDs when key points were added)
        key_point_frames = None
        # Try direct field first
        if "obj_key_point_frames" in data:
            key_point_frames_raw = data["obj_key_point_frames"]
            if (
                isinstance(key_point_frames_raw, np.ndarray)
                and len(key_point_frames_raw) > 0
            ):
                # If it's a regular array, use it directly
                if key_point_frames_raw.ndim == 1:
                    key_point_frames = np.unique(key_point_frames_raw).tolist()
                elif key_point_frames_raw.ndim == 0:
                    key_point_frames = [int(key_point_frames_raw)]
                else:
                    # Flatten and get unique
                    key_point_frames = np.unique(
                        key_point_frames_raw.flatten()
                    ).tolist()
        # Try ragged field
        elif "obj_key_point_frames_data" in data:
            key_point_frames_ragged = reconstruct_ragged_field(
                data, "obj_key_point_frames"
            )
            if key_point_frames_ragged is not None:
                # Collect all unique frame IDs from all frames
                all_frames = []
                for kp_frames in key_point_frames_ragged:
                    if len(kp_frames) > 0:
                        all_frames.extend(kp_frames.flatten().tolist())
                if len(all_frames) > 0:
                    key_point_frames = sorted(list(set(all_frames)))

        # Filter out invalid frame IDs
        if key_point_frames is not None:
            key_point_frames = [f for f in key_point_frames if f >= 0]
            if len(key_point_frames) == 0:
                key_point_frames = None

        # Get key frames (frames where new points were sampled)
        key_frames = None
        # Try direct field first
        if "is_key_frame" in data:
            is_key_frame = data["is_key_frame"]
            if isinstance(is_key_frame, np.ndarray):
                if frame_ids is not None:
                    min_len = min(len(frame_ids), len(is_key_frame))
                    frame_ids_subset = frame_ids[:min_len]
                    is_key_frame_subset = is_key_frame[:min_len]
                    # Handle boolean or integer array
                    if is_key_frame_subset.dtype == bool:
                        key_frames = frame_ids_subset[is_key_frame_subset].tolist()
                    else:
                        # Treat non-zero as True
                        key_frames = frame_ids_subset[is_key_frame_subset != 0].tolist()
                    if len(key_frames) == 0:
                        key_frames = None
            elif isinstance(is_key_frame, (list, tuple)):
                if frame_ids is not None:
                    min_len = min(len(frame_ids), len(is_key_frame))
                    key_frames = [
                        frame_ids[i]
                        for i, kf in enumerate(is_key_frame[:min_len])
                        if kf
                    ]
                    if len(key_frames) == 0:
                        key_frames = None
        # Try ragged field
        elif "is_key_frame_data" in data:
            is_key_frame_ragged = reconstruct_ragged_field(data, "is_key_frame")
            if is_key_frame_ragged is not None and frame_ids is not None:
                key_frames = []
                for i, kf_data in enumerate(is_key_frame_ragged):
                    if i >= len(frame_ids):
                        break
                    # Check if this frame is a key frame
                    # For scalar values stored in ragged array
                    if kf_data.size > 0:
                        kf_value = kf_data[0]
                        # Check if value is True/non-zero
                        if isinstance(kf_value, (bool, np.bool_)):
                            if kf_value:
                                key_frames.append(frame_ids[i])
                        elif isinstance(
                            kf_value, (int, float, np.integer, np.floating)
                        ):
                            if kf_value != 0:
                                key_frames.append(frame_ids[i])
                        elif isinstance(kf_data, np.ndarray) and kf_data.dtype == bool:
                            if np.any(kf_data):
                                key_frames.append(frame_ids[i])
                if len(key_frames) == 0:
                    key_frames = None

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
                    return None, None, None, None

        return poses, frame_ids, key_point_frames, key_frames

    except (IOError, ValueError, KeyError) as e:
        print(f"Error loading meta_data: {e}")
        return None, None, None, None


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


def compute_pose_errors(pred_poses, gt_poses):
    """
    Compute individual position and rotation errors between predicted and GT poses.

    Args:
        pred_poses: List/array of (4, 4) predicted pose matrices
        gt_poses: List/array of (4, 4) GT pose matrices (can contain None)

    Returns:
        x_errors: (N,) array of X position errors (meters)
        y_errors: (N,) array of Y position errors (meters)
        z_errors: (N,) array of Z position errors (meters)
        roll_errors: (N,) array of roll angle errors (degrees)
        pitch_errors: (N,) array of pitch angle errors (degrees)
        yaw_errors: (N,) array of yaw angle errors (degrees)
    """
    x_errors = []
    y_errors = []
    z_errors = []
    roll_errors = []
    pitch_errors = []
    yaw_errors = []

    for pred_pose, gt_pose in zip(pred_poses, gt_poses):
        if pred_pose is None or gt_pose is None:
            x_errors.append(np.nan)
            y_errors.append(np.nan)
            z_errors.append(np.nan)
            roll_errors.append(np.nan)
            pitch_errors.append(np.nan)
            yaw_errors.append(np.nan)
            continue

        # Position errors: individual component differences
        pred_trans = pred_pose[:3, 3]
        gt_trans = gt_pose[:3, 3]
        x_errors.append(abs(pred_trans[0] - gt_trans[0]))
        y_errors.append(abs(pred_trans[1] - gt_trans[1]))
        z_errors.append(abs(pred_trans[2] - gt_trans[2]))

        # Rotation errors: individual Euler angle differences
        pred_rot = R.from_matrix(pred_pose[:3, :3])
        gt_rot = R.from_matrix(gt_pose[:3, :3])

        # Get Euler angles for both
        pred_euler = pred_rot.as_euler("xyz", degrees=False)
        gt_euler = gt_rot.as_euler("xyz", degrees=False)

        # Compute angle differences (handle wrap-around)
        roll_diff = pred_euler[0] - gt_euler[0]
        pitch_diff = pred_euler[1] - gt_euler[1]
        yaw_diff = pred_euler[2] - gt_euler[2]

        # Wrap to [-pi, pi] range
        roll_diff = np.arctan2(np.sin(roll_diff), np.cos(roll_diff))
        pitch_diff = np.arctan2(np.sin(pitch_diff), np.cos(pitch_diff))
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

        roll_errors.append(np.abs(np.rad2deg(roll_diff)))
        pitch_errors.append(np.abs(np.rad2deg(pitch_diff)))
        yaw_errors.append(np.abs(np.rad2deg(yaw_diff)))

    return (
        np.array(x_errors),
        np.array(y_errors),
        np.array(z_errors),
        np.array(roll_errors),
        np.array(pitch_errors),
        np.array(yaw_errors),
    )


def plot_pose_trajectory(
    pred_poses,
    gt_poses,
    frame_ids=None,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(16, 12),
    key_point_frames=None,
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
        key_point_frames: Optional list of frame IDs where key points were added
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

    # Note: key_point_frames is kept as a parameter for future enhancements
    # but is not currently used in the main trajectory plot
    _ = key_point_frames

    # Create figure with subplots
    # Layout: 3 rows x 2 columns (only position and rotation plots, no errors)
    # Rows: X, Y, Z positions, Roll, Pitch, Yaw rotations
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


def plot_error_trajectory(
    pred_poses,
    gt_poses,
    frame_ids=None,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(16, 12),
    key_point_frames=None,
    key_frames=None,
):
    """
    Plot error over time for x, y, z position and roll, pitch, yaw rotation.

    Args:
        pred_poses: List/array of (4, 4) predicted pose matrices
        gt_poses: List/array of (4, 4) GT pose matrices (can contain None)
        frame_ids: Optional list of frame IDs
        video_name: Name of the video/sequence
        output_dir: Directory to save plots
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        figsize: Figure size
        key_point_frames: Optional list of frame IDs where key points were added
        key_frames: Optional list of frame IDs that are key frames
    """
    # Check if GT poses are available
    has_gt = gt_poses is not None and any(p is not None for p in gt_poses)

    if not has_gt:
        print("Warning: No GT poses available. Cannot plot errors.")
        return None

    # Create frame indices
    if frame_ids is None:
        frame_ids = np.arange(len(pred_poses))
    else:
        frame_ids = np.array(frame_ids)

    # Compute individual errors
    x_errors, y_errors, z_errors, roll_errors, pitch_errors, yaw_errors = (
        compute_pose_errors(pred_poses, gt_poses)
    )

    # Create figure with subplots
    # Layout: 3 rows x 2 columns
    # Left column: X, Y, Z position errors
    # Right column: Roll, Pitch, Yaw rotation errors
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Helper function to add vertical lines for key points and key frames
    def add_key_lines(ax, frame_ids, key_point_frames, key_frames):
        """Add vertical lines for key point additions and key frames."""
        has_lines = False
        if (
            key_point_frames is not None
            and len(key_point_frames) > 0
            and len(frame_ids) > 0
        ):
            for idx, kp_frame in enumerate(key_point_frames):
                if kp_frame >= frame_ids[0] and kp_frame <= frame_ids[-1]:
                    ax.axvline(
                        x=kp_frame,
                        color="green",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Key Point Added" if idx == 0 and not has_lines else "",
                    )
                    has_lines = True

        if key_frames is not None and len(key_frames) > 0 and len(frame_ids) > 0:
            for idx, kf_frame in enumerate(key_frames):
                if kf_frame >= frame_ids[0] and kf_frame <= frame_ids[-1]:
                    ax.axvline(
                        x=kf_frame,
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Key Frame" if idx == 0 and not has_lines else "",
                    )
                    has_lines = True

        return has_lines

    # Plot 1: X position error (left, top)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        frame_ids, x_errors, label="X Error", color="darkred", linewidth=2, alpha=0.8
    )
    has_lines = add_key_lines(ax1, frame_ids, key_point_frames, key_frames)
    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("X Position Error (m)", fontsize=12)
    ax1.set_title("X Position Error Over Time", fontsize=14, fontweight="bold")
    if has_lines:
        ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Y position error (left, middle)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        frame_ids, y_errors, label="Y Error", color="darkred", linewidth=2, alpha=0.8
    )
    add_key_lines(ax2, frame_ids, key_point_frames, key_frames)
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Y Position Error (m)", fontsize=12)
    ax2.set_title("Y Position Error Over Time", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Plot 3: Z position error (left, bottom)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(
        frame_ids, z_errors, label="Z Error", color="darkred", linewidth=2, alpha=0.8
    )
    add_key_lines(ax3, frame_ids, key_point_frames, key_frames)
    ax3.set_xlabel("Frame", fontsize=12)
    ax3.set_ylabel("Z Position Error (m)", fontsize=12)
    ax3.set_title("Z Position Error Over Time", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # Plot 4: Roll error (right, top)
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.plot(
        frame_ids,
        roll_errors,
        label="Roll Error",
        color="darkblue",
        linewidth=2,
        alpha=0.8,
    )
    add_key_lines(ax4, frame_ids, key_point_frames, key_frames)
    ax4.set_xlabel("Frame", fontsize=12)
    ax4.set_ylabel("Roll Error (degrees)", fontsize=12)
    ax4.set_title("Roll Error Over Time", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # Plot 5: Pitch error (right, middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        frame_ids,
        pitch_errors,
        label="Pitch Error",
        color="darkblue",
        linewidth=2,
        alpha=0.8,
    )
    add_key_lines(ax5, frame_ids, key_point_frames, key_frames)
    ax5.set_xlabel("Frame", fontsize=12)
    ax5.set_ylabel("Pitch Error (degrees)", fontsize=12)
    ax5.set_title("Pitch Error Over Time", fontsize=14, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale("log")

    # Plot 6: Yaw error (right, bottom)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(
        frame_ids,
        yaw_errors,
        label="Yaw Error",
        color="darkblue",
        linewidth=2,
        alpha=0.8,
    )
    add_key_lines(ax6, frame_ids, key_point_frames, key_frames)
    ax6.set_xlabel("Frame", fontsize=12)
    ax6.set_ylabel("Yaw Error (degrees)", fontsize=12)
    ax6.set_title("Yaw Error Over Time", fontsize=14, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale("log")

    # Overall title
    title_str = "Pose Error Trajectory"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "pose_error_trajectory.png"
        if video_name:
            plot_filename = f"pose_error_trajectory_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved pose error trajectory plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def plot_keypoint_error_analysis(
    pred_poses,
    gt_poses,
    frame_ids=None,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(18, 10),
    key_point_frames=None,
    key_frames=None,
    window_size=5,
):
    """
    Analyze error changes around key point additions to see if they correlate with error increases.

    Args:
        pred_poses: List/array of (4, 4) predicted pose matrices
        gt_poses: List/array of (4, 4) GT pose matrices (can contain None)
        frame_ids: Optional list of frame IDs
        video_name: Name of the video/sequence
        output_dir: Directory to save plots
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        figsize: Figure size
        key_point_frames: Optional list of frame IDs where key points were added
        key_frames: Optional list of frame IDs that are key frames (unused, kept for API consistency)
        window_size: Number of frames before/after key point addition to analyze
    """
    # Note: key_frames is kept for API consistency but not used in this analysis
    _ = key_frames
    # Check if GT poses are available
    has_gt = gt_poses is not None and any(p is not None for p in gt_poses)

    if not has_gt:
        print(
            "Warning: No GT poses available. Cannot analyze key point error correlation."
        )
        return None

    if key_point_frames is None or len(key_point_frames) == 0:
        print(
            "Warning: No key point frames available. Cannot analyze key point error correlation."
        )
        return None

    # Create frame indices
    if frame_ids is None:
        frame_ids = np.arange(len(pred_poses))
    else:
        frame_ids = np.array(frame_ids)

    # Compute individual errors
    x_errors, y_errors, z_errors, roll_errors, pitch_errors, yaw_errors = (
        compute_pose_errors(pred_poses, gt_poses)
    )

    # Compute total position and rotation errors for overall analysis
    pos_errors = np.sqrt(x_errors**2 + y_errors**2 + z_errors**2)
    rot_errors = np.sqrt(roll_errors**2 + pitch_errors**2 + yaw_errors**2)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # Helper function to compute error changes around key point additions
    def compute_error_changes(key_point_frames, errors, frame_ids, window_size):
        """Compute error changes before/after key point additions."""
        error_changes = []
        error_before = []
        error_after = []
        valid_frames = []

        for kp_frame in key_point_frames:
            if kp_frame < window_size or kp_frame >= len(frame_ids) - window_size:
                continue  # Skip if not enough frames before/after

            # Find frame index
            frame_idx = np.where(frame_ids == kp_frame)[0]
            if len(frame_idx) == 0:
                continue
            frame_idx = frame_idx[0]

            # Get errors in window before and after
            before_start = max(0, frame_idx - window_size)
            after_end = min(len(errors), frame_idx + window_size + 1)

            # Average error before (excluding the key point frame itself)
            before_errors = errors[before_start:frame_idx]
            after_errors = errors[frame_idx + 1 : after_end]

            if len(before_errors) > 0 and len(after_errors) > 0:
                before_mean = np.nanmean(before_errors)
                after_mean = np.nanmean(after_errors)
                error_change = after_mean - before_mean
                error_changes.append(error_change)
                error_before.append(before_mean)
                error_after.append(after_mean)
                valid_frames.append(kp_frame)

        return (
            np.array(error_changes),
            np.array(error_before),
            np.array(error_after),
            valid_frames,
        )

    # Compute error changes for position and rotation
    pos_changes, pos_before, pos_after, valid_pos_frames = compute_error_changes(
        key_point_frames, pos_errors, frame_ids, window_size
    )
    rot_changes, rot_before, rot_after, _ = compute_error_changes(
        key_point_frames, rot_errors, frame_ids, window_size
    )

    # Plot 1: Position error over time with key point additions highlighted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        frame_ids,
        pos_errors,
        label="Position Error",
        color="darkred",
        linewidth=1.5,
        alpha=0.7,
    )
    # Highlight key point addition frames
    for kp_frame in key_point_frames:
        if kp_frame >= frame_ids[0] and kp_frame <= frame_ids[-1]:
            ax1.axvline(
                x=kp_frame, color="green", linestyle=":", linewidth=1.5, alpha=0.6
            )
    ax1.set_xlabel("Frame", fontsize=11)
    ax1.set_ylabel("Position Error (m)", fontsize=11)
    ax1.set_title(
        "Position Error with Key Point Additions", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Rotation error over time with key point additions highlighted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        frame_ids,
        rot_errors,
        label="Rotation Error",
        color="darkblue",
        linewidth=1.5,
        alpha=0.7,
    )
    # Highlight key point addition frames
    for kp_frame in key_point_frames:
        if kp_frame >= frame_ids[0] and kp_frame <= frame_ids[-1]:
            ax2.axvline(
                x=kp_frame, color="green", linestyle=":", linewidth=1.5, alpha=0.6
            )
    ax2.set_xlabel("Frame", fontsize=11)
    ax2.set_ylabel("Rotation Error (degrees)", fontsize=11)
    ax2.set_title(
        "Rotation Error with Key Point Additions", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Plot 3: Error change distribution (position)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(pos_changes) > 0:
        colors = ["red" if x > 0 else "green" for x in pos_changes]
        ax3.bar(range(len(pos_changes)), pos_changes, color=colors, alpha=0.6)
        ax3.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax3.set_xlabel("Key Point Addition Event", fontsize=11)
        ax3.set_ylabel("Error Change (m)", fontsize=11)
        ax3.set_title(
            "Position Error Change After Key Point Addition",
            fontsize=12,
            fontweight="bold",
        )
        ax3.grid(True, alpha=0.3, axis="y")
        # Add statistics text
        increase_count = np.sum(pos_changes > 0)
        decrease_count = np.sum(pos_changes < 0)
        mean_change = np.mean(pos_changes)
        ax3.text(
            0.05,
            0.95,
            f"Increase: {increase_count}\nDecrease: {decrease_count}\nMean: {mean_change:.4f}m",
            transform=ax3.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

    # Plot 4: Error change distribution (rotation)
    ax4 = fig.add_subplot(gs[1, 2])
    if len(rot_changes) > 0:
        colors = ["red" if x > 0 else "green" for x in rot_changes]
        ax4.bar(range(len(rot_changes)), rot_changes, color=colors, alpha=0.6)
        ax4.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax4.set_xlabel("Key Point Addition Event", fontsize=11)
        ax4.set_ylabel("Error Change (degrees)", fontsize=11)
        ax4.set_title(
            "Rotation Error Change After Key Point Addition",
            fontsize=12,
            fontweight="bold",
        )
        ax4.grid(True, alpha=0.3, axis="y")
        # Add statistics text
        increase_count = np.sum(rot_changes > 0)
        decrease_count = np.sum(rot_changes < 0)
        mean_change = np.mean(rot_changes)
        ax4.text(
            0.05,
            0.95,
            f"Increase: {increase_count}\nDecrease: {decrease_count}\nMean: {mean_change:.4f}°",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)

    # Plot 5: Before vs After scatter (position)
    ax5 = fig.add_subplot(gs[1, 0])
    if len(pos_before) > 0 and len(pos_after) > 0:
        ax5.scatter(pos_before, pos_after, alpha=0.6, s=50)
        # Add diagonal line (y=x)
        max_val = max(np.nanmax(pos_before), np.nanmax(pos_after))
        min_val = min(np.nanmin(pos_before), np.nanmin(pos_after))
        ax5.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.5,
            label="No Change",
        )
        ax5.set_xlabel("Error Before (m)", fontsize=11)
        ax5.set_ylabel("Error After (m)", fontsize=11)
        ax5.set_title("Position Error: Before vs After", fontsize=12, fontweight="bold")
        ax5.legend(loc="best", fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale("log")
        ax5.set_yscale("log")
    else:
        ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)

    # Plot 6: Before vs After scatter (rotation)
    ax6 = fig.add_subplot(gs[1, 1])
    if len(rot_before) > 0 and len(rot_after) > 0:
        ax6.scatter(rot_before, rot_after, alpha=0.6, s=50)
        # Add diagonal line (y=x)
        max_val = max(np.nanmax(rot_before), np.nanmax(rot_after))
        min_val = min(np.nanmin(rot_before), np.nanmin(rot_after))
        ax6.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.5,
            label="No Change",
        )
        ax6.set_xlabel("Error Before (degrees)", fontsize=11)
        ax6.set_ylabel("Error After (degrees)", fontsize=11)
        ax6.set_title("Rotation Error: Before vs After", fontsize=12, fontweight="bold")
        ax6.legend(loc="best", fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale("log")
        ax6.set_yscale("log")
    else:
        ax6.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax6.transAxes)

    # Plot 7: Error trajectory around key point additions (position) - first few examples
    ax7 = fig.add_subplot(gs[2, 0])
    num_examples = min(5, len(key_point_frames))
    for i, kp_frame in enumerate(key_point_frames[:num_examples]):
        frame_idx = np.where(frame_ids == kp_frame)[0]
        if len(frame_idx) == 0:
            continue
        frame_idx = frame_idx[0]

        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(frame_ids), frame_idx + window_size + 1)
        window_frames = frame_ids[start_idx:end_idx]
        window_errors = pos_errors[start_idx:end_idx]

        # Normalize frame to be relative to key point addition (0)
        relative_frames = window_frames - kp_frame
        ax7.plot(
            relative_frames,
            window_errors,
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.6,
            label=f"Event {i+1}" if i < 3 else "",
        )
        ax7.axvline(x=0, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax7.set_xlabel("Frames Relative to Key Point Addition", fontsize=11)
    ax7.set_ylabel("Position Error (m)", fontsize=11)
    ax7.set_title(
        f"Position Error Around Key Point Additions (first {num_examples})",
        fontsize=12,
        fontweight="bold",
    )
    ax7.legend(loc="best", fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale("log")

    # Plot 8: Error trajectory around key point additions (rotation) - first few examples
    ax8 = fig.add_subplot(gs[2, 1])
    for i, kp_frame in enumerate(key_point_frames[:num_examples]):
        frame_idx = np.where(frame_ids == kp_frame)[0]
        if len(frame_idx) == 0:
            continue
        frame_idx = frame_idx[0]

        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(frame_ids), frame_idx + window_size + 1)
        window_frames = frame_ids[start_idx:end_idx]
        window_errors = rot_errors[start_idx:end_idx]

        # Normalize frame to be relative to key point addition (0)
        relative_frames = window_frames - kp_frame
        ax8.plot(
            relative_frames,
            window_errors,
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.6,
            label=f"Event {i+1}" if i < 3 else "",
        )
        ax8.axvline(x=0, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax8.set_xlabel("Frames Relative to Key Point Addition", fontsize=11)
    ax8.set_ylabel("Rotation Error (degrees)", fontsize=11)
    ax8.set_title(
        f"Rotation Error Around Key Point Additions (first {num_examples})",
        fontsize=12,
        fontweight="bold",
    )
    ax8.legend(loc="best", fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale("log")

    # Plot 9: Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    stats_text = "Key Point Addition Analysis\n"
    stats_text += "=" * 40 + "\n\n"
    stats_text += f"Total Key Point Events: {len(key_point_frames)}\n"
    stats_text += f"Analyzed Events: {len(valid_pos_frames)}\n"
    stats_text += f"Window Size: ±{window_size} frames\n\n"

    if len(pos_changes) > 0:
        stats_text += "Position Error:\n"
        stats_text += f"  Mean Change: {np.mean(pos_changes):.6f} m\n"
        stats_text += f"  Median Change: {np.median(pos_changes):.6f} m\n"
        stats_text += f"  Std Dev: {np.std(pos_changes):.6f} m\n"
        stats_text += f"  Increases: {np.sum(pos_changes > 0)} ({100*np.sum(pos_changes > 0)/len(pos_changes):.1f}%)\n"
        stats_text += f"  Decreases: {np.sum(pos_changes < 0)} ({100*np.sum(pos_changes < 0)/len(pos_changes):.1f}%)\n\n"

    if len(rot_changes) > 0:
        stats_text += "Rotation Error:\n"
        stats_text += f"  Mean Change: {np.mean(rot_changes):.6f}°\n"
        stats_text += f"  Median Change: {np.median(rot_changes):.6f}°\n"
        stats_text += f"  Std Dev: {np.std(rot_changes):.6f}°\n"
        stats_text += f"  Increases: {np.sum(rot_changes > 0)} ({100*np.sum(rot_changes > 0)/len(rot_changes):.1f}%)\n"
        stats_text += f"  Decreases: {np.sum(rot_changes < 0)} ({100*np.sum(rot_changes < 0)/len(rot_changes):.1f}%)\n"

    ax9.text(
        0.1,
        0.95,
        stats_text,
        transform=ax9.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )

    # Overall title
    title_str = "Key Point Addition vs Error Analysis"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "keypoint_error_analysis.png"
        if video_name:
            plot_filename = f"keypoint_error_analysis_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved key point error analysis plot to: {plot_path}")

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
    key_point_frames = None
    key_frames = None
    if args.pose_file:
        print(f"Loading poses from pose file: {args.meta_data_path}")
        pred_poses, _ = load_poses_from_pose_file(args.meta_data_path)
        frame_ids = None
    else:
        print(f"Loading poses from meta_data: {args.meta_data_path}")
        pred_poses, frame_ids, key_point_frames, key_frames = load_poses_from_meta_data(
            args.meta_data_path
        )

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

    # Print key point frame info
    if key_point_frames is not None and len(key_point_frames) > 0:
        print(
            f"Found {len(key_point_frames)} frames where key points were added: {key_point_frames}"
        )
    else:
        print("No key point frame information found in meta_data")

    # Print key frame info
    if key_frames is not None and len(key_frames) > 0:
        print(f"Found {len(key_frames)} key frames: {key_frames}")
    else:
        print("No key frame information found in meta_data")

    # Plot trajectories
    plot_pose_trajectory(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        frame_ids=frame_ids,
        video_name=video_name_for_plot,
        output_dir=args.output_dir,
        save_plots=True,
        show_plots=args.show_plots,
        key_point_frames=key_point_frames,
    )

    # Plot error trajectories (separate figure)
    plot_error_trajectory(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        frame_ids=frame_ids,
        video_name=video_name_for_plot,
        output_dir=args.output_dir,
        save_plots=True,
        show_plots=args.show_plots,
        key_point_frames=key_point_frames,
        key_frames=key_frames,
    )

    # Plot key point error correlation analysis (separate figure)
    if key_point_frames is not None and len(key_point_frames) > 0:
        plot_keypoint_error_analysis(
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            frame_ids=frame_ids,
            video_name=video_name_for_plot,
            output_dir=args.output_dir,
            save_plots=True,
            show_plots=args.show_plots,
            key_point_frames=key_point_frames,
            key_frames=key_frames,
        )
    else:
        print("Skipping key point error analysis (no key point frames available)")

    print("Done!")


if __name__ == "__main__":
    main()
