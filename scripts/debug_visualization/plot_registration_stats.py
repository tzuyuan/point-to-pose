#!/usr/bin/env python3
"""
Plot registration statistics (inlier, uncertainty, residual) for each point in registration at a given frame.
Similar to visualize_register_pcd.py in terms of data access.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from point2pose.io.sources.dataset.datareader import Ho3dReader, YcbineoatReader
    from point2pose.utils.transform import inverse_SE3
except ImportError:
    Ho3dReader = None
    YcbineoatReader = None
    inverse_SE3 = None


def find_meata_data_path(
    register_folder: str,
    meta_data_path_override: Optional[str] = None,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """Find meta_data.npz file in expected locations (also checks meata_data.npz for backward compatibility).

    Args:
        register_folder: Path to register folder
        meta_data_path_override: Direct path to meta_data.npz (takes highest priority)
        results_dir: Results directory (e.g., /path/to/results/ho3d_single)
        video_name: Video sequence name (e.g., MPM10)
    """
    # If override path is provided, use it directly
    if meta_data_path_override is not None:
        if os.path.exists(meta_data_path_override):
            return meta_data_path_override
        else:
            print(
                f"Warning: Specified meta_data_path does not exist: {meta_data_path_override}"
            )
            # Continue to search in other locations

    # Get project root by finding the scripts directory's parent
    # __file__ is scripts/debug_visualization/plot_registration_stats.py
    # script_dir is scripts/debug_visualization
    # os.path.dirname(script_dir) is scripts
    # os.path.dirname(os.path.dirname(script_dir)) is point-to-pose (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # This gives point-to-pose

    # Try meta_data.npz first (correct spelling)
    meta_data_paths = []

    # NEW: Check new results folder structure first (highest priority after override)
    if results_dir and video_name:
        new_structure_path = os.path.join(
            results_dir, video_name, "meta_data", "meta_data.npz"
        )
        meta_data_paths.append(new_structure_path)
        # Also try with default results directory if not provided
        default_results_dir = os.path.join(project_root, "results", "ho3d_single")
        if results_dir != default_results_dir:
            meta_data_paths.append(
                os.path.join(
                    default_results_dir, video_name, "meta_data", "meta_data.npz"
                )
            )
    elif results_dir:
        # If only results_dir provided, check all video subdirectories
        if os.path.exists(results_dir):
            for item in os.listdir(results_dir):
                video_path = os.path.join(results_dir, item)
                if os.path.isdir(video_path):
                    meta_data_path = os.path.join(
                        video_path, "meta_data", "meta_data.npz"
                    )
                    meta_data_paths.append(meta_data_path)

    # Existing paths for backward compatibility
    meta_data_paths.extend(
        [
            # Relative to register folder
            os.path.join(register_folder, "meta_data.npz"),
            os.path.join(os.path.dirname(register_folder), "meta_data.npz"),
            # Relative to project root (common output location)
            os.path.join(project_root, "meta_data", "meta_data.npz"),
            os.path.join(project_root, "debug", "meta_data", "meta_data.npz"),
            os.path.join(
                project_root, "debug", "pipeline", "meta_data", "meta_data.npz"
            ),  # The actual location
            # Relative paths from register folder
            os.path.join(
                os.path.dirname(os.path.dirname(register_folder)),
                "meta_data",
                "meta_data.npz",
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(register_folder)),
                "debug",
                "meta_data",
                "meta_data.npz",
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(register_folder)),
                "debug",
                "pipeline",
                "meta_data",
                "meta_data.npz",
            ),
            # Current working directory (where pipeline might have been run)
            os.path.join(os.getcwd(), "meta_data", "meta_data.npz"),
            os.path.join(
                os.getcwd(), "debug", "pipeline", "meta_data", "meta_data.npz"
            ),
        ]
    )

    # Also check old typo for backward compatibility
    meata_data_paths = [
        os.path.join(register_folder, "meata_data.npz"),
        os.path.join(os.path.dirname(register_folder), "meata_data.npz"),
        os.path.join(project_root, "meta_data", "meata_data.npz"),
        os.path.join(project_root, "debug", "meta_data", "meata_data.npz"),
        os.path.join(project_root, "debug", "pipeline", "meta_data", "meata_data.npz"),
        os.path.join(
            os.path.dirname(os.path.dirname(register_folder)),
            "debug",
            "pipeline",
            "meta_data",
            "meata_data.npz",
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(register_folder)),
            "meta_data",
            "meata_data.npz",
        ),
    ]

    all_paths = meta_data_paths + meata_data_paths
    for path in all_paths:
        if os.path.exists(path):
            print(f"Found meta_data.npz at: {path}")
            return path

    # Print debug info if not found
    print(
        f"Debug: Searched {len(all_paths)} locations, register_folder={register_folder}"
    )
    print(f"  Project root: {project_root}")
    print(f"  Current working directory: {os.getcwd()}")
    return None


def unpack_ragged(name: str, store: dict, dim: int = -1) -> list:
    """Unpack ragged array data from NPZ storage format.

    Args:
        name: Base name of the ragged field (e.g., "reg_key_points")
        store: Dictionary loaded from NPZ file
        dim: Dimension to reshape to (-1 for 1D, 2 for 2D points, 3 for 3D points)

    Returns:
        List of arrays, one per frame
    """
    data_key = f"{name}_data"
    offsets_key = f"{name}_offsets"
    lengths_key = f"{name}_lengths"

    if data_key not in store or offsets_key not in store or lengths_key not in store:
        return []

    data = store[data_key]
    offsets = store[offsets_key]
    lengths = store[lengths_key]

    out = []
    for off, L in zip(offsets, lengths):
        flat_data = data[off : off + L]

        # Handle empty or invalid data
        if L == 0 or len(flat_data) == 0:
            # Return empty array with appropriate shape
            if dim == 3:
                out.append(np.array([]).reshape(0, 3))
            elif dim == 2:
                out.append(np.array([]).reshape(0, 2))
            else:
                out.append(np.array([]))
            continue

        # Check if data is valid (not just placeholder values like -1)
        # If it's a single element that can't be reshaped, return empty array
        if dim == 3:
            if len(flat_data) < 3 or len(flat_data) % 3 != 0:
                # Can't reshape to 3D, return empty array
                out.append(np.array([]).reshape(0, 3))
                continue
            reshaped_data = flat_data.reshape(-1, 3)
        elif dim == 2:
            if len(flat_data) < 2 or len(flat_data) % 2 != 0:
                # Can't reshape to 2D, return empty array
                out.append(np.array([]).reshape(0, 2))
                continue
            reshaped_data = flat_data.reshape(-1, 2)
        else:
            reshaped_data = flat_data  # Keep as 1D
        out.append(reshaped_data)
    return out


def load_registration_stats(
    register_folder: str,
    object_number: int,
    frame_number: int,
    mode: str = "auto",
    meta_data_path_override: Optional[str] = None,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        str,
        Optional[np.ndarray],
    ]
]:
    """Load registration statistics for a given frame.

    Args:
        register_folder: Path to register folder
        object_number: Object number (currently only supports object 0, kept for compatibility)
        frame_number: Frame number to extract stats for
        mode: Registration mode ("f2f", "f2m", or "auto" to auto-detect)
        meta_data_path_override: Direct path to meta_data.npz (optional)
        results_dir: Results directory for new structure (optional)
        video_name: Video sequence name for new structure (optional)

    Returns:
        Tuple of (residuals, inliers, uncertainties, key_points/prev3d, curr3d, keyframe_ids, reg_key_points_idx, mode, key_points_first_frame) if data is found, None otherwise.
        All arrays have the same length (number of registration points).
        mode is either "f2f" or "f2m".
    """
    _ = object_number  # Currently unused, kept for compatibility
    meta_data_path = find_meata_data_path(
        register_folder,
        meta_data_path_override=meta_data_path_override,
        results_dir=results_dir,
        video_name=video_name,
    )
    if meta_data_path is None:
        print("No meta_data.npz found in expected locations")
        print(f"  Searched relative to register_folder: {register_folder}")
        print(f"  Current working directory: {os.getcwd()}")
        print(
            "  Please ensure meta_data.npz exists, or specify --register_folder with the correct path"
        )
        return None

    try:
        data = np.load(meta_data_path, allow_pickle=True)
        print(f"Loaded meta_data.npz from: {meta_data_path}")

        # Find frame index
        if "frame_id" not in data:
            print("No frame_id field found in meta_data.npz")
            return None

        frame_ids = data["frame_id"]
        frame_idx = None

        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"Frame {frame_number} not found in meta_data.npz")
            print(f"Available frames: {frame_ids[:10]}...")  # Show first 10
            return None

        print(f"Found frame {frame_number} at index {frame_idx}")

        # Detect mode: check which fields are present
        reg_prev3d_list = unpack_ragged("reg_prev3d", data, dim=3)
        reg_key_points_list = unpack_ragged("reg_key_points", data, dim=3)

        if mode == "auto":
            # Auto-detect mode based on which field has data at this frame
            has_prev3d = (
                frame_idx < len(reg_prev3d_list)
                and reg_prev3d_list[frame_idx] is not None
                and len(reg_prev3d_list[frame_idx]) > 0
                and reg_prev3d_list[frame_idx].size > 0
                and not (
                    reg_prev3d_list[frame_idx].ndim > 0
                    and np.all(reg_prev3d_list[frame_idx].flatten() == -1)
                )
            )
            has_key_points = (
                frame_idx < len(reg_key_points_list)
                and reg_key_points_list[frame_idx] is not None
                and len(reg_key_points_list[frame_idx]) > 0
                and reg_key_points_list[frame_idx].size > 0
                and not (
                    reg_key_points_list[frame_idx].ndim > 0
                    and np.all(reg_key_points_list[frame_idx].flatten() == -1)
                )
            )

            if has_prev3d and not has_key_points:
                detected_mode = "f2f"
            elif has_key_points and not has_prev3d:
                detected_mode = "f2m"
            elif has_prev3d and has_key_points:
                # If both exist, prefer the one with valid data (not all -1)
                detected_mode = "f2f"  # Default to f2f if both present
            else:
                detected_mode = "f2m"  # Default to f2m
        else:
            detected_mode = mode

        print(f"Using registration mode: {detected_mode}")

        # Use unpack_ragged to extract data (following notebook pattern)
        reg_residuals_list = unpack_ragged("reg_residuals", data, dim=-1)
        reg_inliers_list = unpack_ragged("reg_inliers", data, dim=-1)
        reg_curr3d_list = unpack_ragged("reg_curr3d", data, dim=3)
        reg_key_points_idx_list = unpack_ragged("reg_key_points_idx", data, dim=-1)
        uncertainties_list = unpack_ragged("uncertainties", data, dim=-1)
        obj_key_point_frames_list = unpack_ragged("obj_key_point_frames", data, dim=-1)

        # Extract data for this frame
        if frame_idx >= len(reg_residuals_list):
            print(f"Frame index {frame_idx} out of range for reg_residuals")
            return None

        residuals = reg_residuals_list[frame_idx].astype(float)
        inliers = (
            reg_inliers_list[frame_idx].astype(bool)
            if frame_idx < len(reg_inliers_list)
            else np.ones(len(residuals), dtype=bool)
        )

        # Load source points based on mode
        key_points_first_frame = None  # keep untransformed copy for GT alignment
        if detected_mode == "f2f":
            key_points = (
                reg_prev3d_list[frame_idx] if frame_idx < len(reg_prev3d_list) else None
            )
            print("Loaded f2f mode: using reg_prev3d as source points")
        else:  # f2m mode
            if frame_idx < len(reg_key_points_list):
                key_points_first_frame = reg_key_points_list[frame_idx]
                key_points = key_points_first_frame
            else:
                key_points = None
            print("Loaded f2m mode: using reg_key_points as source points")

        curr3d = (
            reg_curr3d_list[frame_idx] if frame_idx < len(reg_curr3d_list) else None
        )
        reg_key_points_idx = (
            reg_key_points_idx_list[frame_idx].astype(int)
            if frame_idx < len(reg_key_points_idx_list)
            else None
        )

        if residuals is None or len(residuals) == 0:
            print("Warning: Could not extract reg_residuals (empty)")
            if residuals is None:
                residuals = np.array([])
            # return None  # Don't return None, continue with empty residuals

        if inliers is None or len(inliers) == 0:
            # Determine expected length
            expected_len = len(residuals)
            if expected_len == 0:
                if key_points is not None:
                    expected_len = len(key_points)
                elif curr3d is not None:
                    expected_len = len(curr3d)

            print(
                f"Warning: Could not extract reg_inliers, assuming all {expected_len} points are inliers"
            )
            inliers = np.ones(expected_len, dtype=bool)

        # Extract uncertainties using reg_key_points_idx (following notebook pattern)
        uncertainties = None
        if reg_key_points_idx is not None and frame_idx < len(uncertainties_list):
            frame_uncertainties = uncertainties_list[frame_idx]
            if len(reg_key_points_idx) == len(residuals):
                uncertainties = frame_uncertainties[reg_key_points_idx].astype(float)
            else:
                print(
                    f"Warning: reg_key_points_idx length ({len(reg_key_points_idx)}) != residuals length ({len(residuals)})"
                )

        if uncertainties is None:
            print(
                "Warning: Could not extract uncertainties, will not plot uncertainty stats"
            )

        # Extract frame IDs for ALL key points using unpack_ragged (following notebook pattern)
        obj_key_points_list = unpack_ragged("obj_key_points", data, dim=3)
        keyframe_ids = None

        if frame_idx < len(obj_key_point_frames_list) and frame_idx < len(
            obj_key_points_list
        ):
            # Get all key points and their frame IDs for this frame
            all_key_points_frame = obj_key_points_list[frame_idx]
            all_key_frame_ids = obj_key_point_frames_list[frame_idx].astype(int)

            # Filter NaN points (same as load_all_key_points_with_frame_ids)
            # if np.any(np.isnan(all_key_points_frame)):
            #     valid_mask = ~np.isnan(all_key_points_frame).any(axis=1)
            #     all_key_points_frame = all_key_points_frame[valid_mask]
            #     keyframe_ids = all_key_frame_ids[valid_mask]
            # else:
            keyframe_ids = all_key_frame_ids

            # Apply transformation if obj_pose exists (same as load_all_key_points_with_frame_ids)
            if "obj_pose" in data:
                obj_pose = data["obj_pose"][frame_idx]
                if not (np.any(np.isnan(obj_pose)) or np.any(np.isinf(obj_pose))):
                    pts_h = np.hstack(
                        [all_key_points_frame, np.ones((len(all_key_points_frame), 1))]
                    )
                    pts_transformed_homo = (obj_pose @ pts_h.T).T
                    all_key_points_frame = pts_transformed_homo[:, :3]

                    # Filter NaN after transformation
                    # if np.any(np.isnan(all_key_points_frame)):
                    #     valid_mask = ~np.isnan(all_key_points_frame).any(axis=1)
                    # keyframe_ids = keyframe_ids[valid_mask]

            keyframe_ids = np.asarray(keyframe_ids).astype(int, copy=False)
            print(f"Loaded {len(keyframe_ids)} keyframe IDs for all key points")

        if keyframe_ids is None:
            print("Warning: Could not extract keyframe_ids")

        # Transform key_points based on mode
        # Note: curr3d should NOT be transformed - it's already in the current frame coordinate system
        obj_pose = None
        if "obj_pose" in data:
            obj_pose = data["obj_pose"][frame_idx]

        # For f2m mode: transform key_points with obj_pose (first->current)
        # For f2f mode: prev3d is already in current frame coordinates, no transformation needed
        if detected_mode == "f2m" and key_points is not None and obj_pose is not None:
            if not (np.any(np.isnan(obj_pose)) or np.any(np.isinf(obj_pose))):
                kp_h = np.hstack([key_points, np.ones((len(key_points), 1))])
                key_points = (obj_pose @ kp_h.T).T[:, :3]
                print("Transformed key_points (f2m mode) using obj_pose")
        elif detected_mode == "f2f":
            print("Using prev3d directly (f2f mode, no transformation needed)")

        # Note: curr3d should NOT be transformed - it's already in the current frame coordinate system

        # Ensure all arrays have the same length
        if len(residuals) != len(inliers):
            min_len = min(len(residuals), len(inliers))
            residuals = residuals[:min_len]
            inliers = inliers[:min_len]
            print(f"Warning: Truncated to min length: {min_len}")

        if uncertainties is not None and len(uncertainties) != len(residuals):
            min_len = min(len(residuals), len(uncertainties))
            residuals = residuals[:min_len]
            inliers = inliers[:min_len]
            uncertainties = uncertainties[:min_len]
            print(f"Warning: Truncated uncertainties to match length: {min_len}")

        # if keyframe_ids is not None and len(keyframe_ids) != len(residuals):
        #     min_len = min(len(residuals), len(keyframe_ids))
        #     residuals = residuals[:min_len]
        #     inliers = inliers[:min_len]
        #     if uncertainties is not None:
        #         uncertainties = uncertainties[:min_len]
        #     keyframe_ids = keyframe_ids[:min_len]
        #     print(f"Warning: Truncated keyframe_ids to match length: {min_len}")

        if keyframe_ids is not None:
            unique_frames = np.unique(keyframe_ids)
            print(
                f"  - Keyframe IDs: {unique_frames[:10]}..."
                if len(unique_frames) > 10
                else f"  - Keyframe IDs: {unique_frames}"
            )
            print(f"  - Number of unique keyframes: {len(unique_frames)}")

        print("Loaded registration stats:")
        print(f"  - Number of points: {len(residuals)}")
        if len(inliers) > 0:
            print(
                f"  - Inliers: {np.sum(inliers)} ({100*np.sum(inliers)/len(inliers):.1f}%)"
            )
            print(
                f"  - Outliers: {np.sum(~inliers)} ({100*np.sum(~inliers)/len(inliers):.1f}%)"
            )
        else:
            print("  - Inliers: 0")
        if uncertainties is not None:
            print(
                f"  - Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]"
            )
        if len(residuals) > 0:
            print(f"  - Residual range: [{residuals.min():.4f}, {residuals.max():.4f}]")
            if np.sum(inliers) > 0:
                print(f"  - Mean residual (inliers): {np.mean(residuals[inliers]):.4f}")
            if np.sum(~inliers) > 0:
                print(
                    f"  - Mean residual (outliers): {np.mean(residuals[~inliers]):.4f}"
                )
        else:
            print("  - Residual range: N/A (empty)")

        # Note: curr3d are already in the current frame coordinate system and should NOT be transformed.
        # For f2m: The registration transformation (obj_pose) transforms key_points from first frame to current frame.
        # For f2f: prev3d is already in the current frame coordinate system.
        # curr3d are the target points in the current frame and remain as-is for visualization.

        return (
            residuals,
            inliers,
            uncertainties,
            key_points,
            curr3d,
            keyframe_ids,
            reg_key_points_idx,
            detected_mode,
            key_points_first_frame,
        )

    except (IOError, ValueError, KeyError) as e:
        print(f"Error loading registration stats: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_registration_stats(
    residuals: np.ndarray,
    inliers: np.ndarray,
    uncertainties: Optional[np.ndarray],
    keyframe_ids: Optional[np.ndarray],
    frame_number: int,
    reg_key_points_idx: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
    key_points: Optional[np.ndarray] = None,
    curr3d: Optional[np.ndarray] = None,
    register_folder: Optional[str] = None,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    ho3d_root: Optional[str] = None,
):
    """Create plots showing registration statistics.

    Args:
        residuals: Array of residuals for each registration point
        inliers: Boolean array indicating inliers
        uncertainties: Optional array of uncertainties for each registration point
        keyframe_ids: Optional array of keyframe IDs for each registration point
        frame_number: Frame number for title
        reg_key_points_idx: Optional array of indices of registration key points
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m") for title
        key_points: Optional source points (N, 3) for error vs uncertainty plot
        curr3d: Optional target points (N, 3) for error vs uncertainty plot
        register_folder: Optional register folder path for GT pose loading
        results_dir: Optional results directory for GT pose loading
        video_name: Optional video name for GT pose loading
        ho3d_root: Optional HO3D root directory for GT pose loading
    """
    num_points = len(residuals)
    num_inliers = np.sum(inliers)
    num_outliers = num_points - num_inliers

    # Determine layout based on available data
    has_unc = uncertainties is not None
    has_kfids = keyframe_ids is not None

    reg_keyframe_ids = None
    if keyframe_ids is not None and reg_key_points_idx is not None:
        reg_idx_array = np.asarray(reg_key_points_idx, dtype=int)
        # Ensure indices are within bounds
        if np.all(reg_idx_array >= 0) and np.all(reg_idx_array < len(keyframe_ids)):
            reg_keyframe_ids = np.asarray(keyframe_ids)[reg_idx_array]
        else:
            print(
                f"Warning: reg_key_points_idx out of bounds. keyframe_ids length: {len(keyframe_ids)}, "
                f"reg_idx min: {reg_idx_array.min()}, max: {reg_idx_array.max()}"
            )
            reg_keyframe_ids = None

    # Check if we can add error vs uncertainty plot (needs key_points, curr3d, uncertainties, and GT pose)
    has_error_vs_unc = (
        key_points is not None
        and curr3d is not None
        and uncertainties is not None
        and register_folder is not None
    )

    # Calculate number of plots needed
    num_plots = 4  # base plots: inlier/outlier, residuals hist (all), residuals hist (split), residuals sorted
    if has_unc:
        num_plots += 2  # uncertainty hist, residual vs uncertainty (inlier/outlier)
    if has_unc and has_kfids:
        num_plots += 1  # residual vs uncertainty (colored by keyframe)
    if reg_keyframe_ids is not None:
        num_plots += 1  # residuals vs frame ID
    if has_error_vs_unc:
        num_plots += 1  # error vs uncertainty (GT-aligned)

    # Create figure with appropriate layout
    if num_plots <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    elif num_plots <= 9:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
    else:
        # Fallback: use a flexible grid
        n_cols = 3
        n_rows = (num_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

    fig.suptitle(
        f"Registration Statistics ({mode.upper()}) - Frame {frame_number} (N={num_points})",
        fontsize=14,
    )

    # 1. Inlier/Outlier distribution
    ax = axes[0]
    ax.bar(["Inliers", "Outliers"], [num_inliers, num_outliers], color=["green", "red"])
    ax.set_ylabel("Count")
    ax.set_title("Inlier/Outlier Distribution")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate([num_inliers, num_outliers]):
        ax.text(i, v, str(v), ha="center", va="bottom")

    # 2. Residuals histogram (all points)
    ax = axes[1]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Distribution (All Points)")
    ax.grid(True, alpha=0.3)
    ax.axvline(
        np.mean(residuals),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(residuals):.4f}",
    )
    ax.legend()

    # 3. Residuals histogram (inliers vs outliers)
    ax = axes[2]
    if num_inliers > 0:
        ax.hist(
            residuals[inliers],
            bins=30,
            alpha=0.7,
            label="Inliers",
            color="green",
            edgecolor="black",
        )
    if num_outliers > 0:
        ax.hist(
            residuals[~inliers],
            bins=30,
            alpha=0.7,
            label="Outliers",
            color="red",
            edgecolor="black",
        )
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Distribution (Inliers vs Outliers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Residuals sorted by value
    ax = axes[3]
    sorted_indices = np.argsort(residuals)
    sorted_residuals = residuals[sorted_indices]
    sorted_inliers = inliers[sorted_indices]
    colors = ["green" if i else "red" for i in sorted_inliers]
    ax.scatter(range(num_points), sorted_residuals, c=colors, alpha=0.6, s=10)
    ax.set_xlabel("Point Index (Sorted by Residual)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals Sorted by Value")
    ax.grid(True, alpha=0.3)
    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Inliers"),
        Patch(facecolor="red", label="Outliers"),
    ]
    ax.legend(handles=legend_elements)

    # 5. Uncertainty histogram (if available)
    if uncertainties is not None:
        ax = axes[4]
        ax.hist(uncertainties, bins=50, alpha=0.7, edgecolor="black", color="blue")
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Count")
        ax.set_title("Uncertainty Distribution")
        ax.grid(True, alpha=0.3)
        ax.axvline(
            np.mean(uncertainties),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(uncertainties):.4f}",
        )
        ax.legend()

        # 6. Residuals vs Uncertainty scatter plot (colored by inlier/outlier)
        ax = axes[5]
        colors_scatter = ["green" if i else "red" for i in inliers]
        ax.scatter(
            uncertainties,
            residuals,
            c=colors_scatter,
            alpha=0.6,
            s=20,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Uncertainty (Inlier/Outlier)")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements)

        # 7. Residuals vs Uncertainty scatter plot (colored by keyframe_id)
        if reg_keyframe_ids is not None:
            ax = axes[6]

            # Get unique keyframe IDs and create color map
            # Normalize reg_keyframe_ids to 1-D int array (avoid ndarray elements)
            def _to_int_scalar(x):
                try:
                    # ndarray/list -> take first element; else cast directly
                    if isinstance(x, (np.ndarray, list, tuple)):
                        return int(np.asarray(x).reshape(-1)[0])
                    return int(x)
                except Exception:
                    return int(np.asarray(x).reshape(-1)[0])

            reg_kf_int = np.asarray(
                [_to_int_scalar(x) for x in reg_keyframe_ids], dtype=int
            )
            unique_frames = np.unique(reg_kf_int)

            # Use matplotlib colormap similar to visualize_register_pcd.py
            try:
                import matplotlib

                cmap_get = getattr(
                    getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap"
                )
                tab10_colors = cmap_get("tab10")

                # Map frame IDs to color indices (avoiding red which is index 3)
                frame_to_color = {}
                for i, fid in enumerate(sorted(unique_frames.tolist())):
                    color_idx = i if i < 3 else i + 1  # Skip red (index 3)
                    frame_to_color[int(fid)] = tab10_colors(color_idx % 10)[:3]
            except (AttributeError, ImportError, KeyError):
                # Fallback palette excluding red
                palette = [
                    (0.2, 0.2, 1.0),  # blue
                    (0.0, 0.7, 0.3),  # green
                    (1.0, 0.6, 0.0),  # orange
                    (0.6, 0.0, 0.8),  # purple
                    (0.0, 0.7, 0.7),  # cyan
                    (0.6, 0.6, 0.0),  # yellow
                ]
                frame_to_color = {
                    int(fid): palette[i % len(palette)]
                    for i, fid in enumerate(sorted(unique_frames.tolist()))
                }

            # Create color array for scatter plot
            point_colors = np.array(
                [frame_to_color.get(int(fid), (0.5, 0.5, 0.5)) for fid in reg_kf_int]
            )

            ax.scatter(
                uncertainties,
                residuals,
                c=point_colors,
                alpha=0.6,
                s=20,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.set_xlabel("Uncertainty")
            ax.set_ylabel("Residual")
            ax.set_title("Residuals vs Uncertainty (Colored by Keyframe)")
            ax.grid(True, alpha=0.3)

            # Add legend for keyframes (limit to first 10 for readability)
            legend_frames = sorted(unique_frames.tolist())[:10]
            legend_elements_kf = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=frame_to_color[int(fid)],
                    markersize=8,
                    label=f"Frame {fid}",
                )
                for fid in legend_frames
            ]
            if len(unique_frames) > 10:
                legend_elements_kf.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="gray",
                        markersize=8,
                        label=f"... ({len(unique_frames) - 10} more)",
                    )
                )
            ax.legend(handles=legend_elements_kf, loc="best", fontsize=8, ncol=2)

    # 8. Residuals vs Frame ID scatter plot
    # Calculate plot index: 4 base plots + (2 if has_unc) + (1 if has_unc and has_kfids)
    plot_idx = 4
    if has_unc:
        plot_idx += 2
    if has_unc and has_kfids:
        plot_idx += 1

    if reg_keyframe_ids is not None:
        ax = axes[plot_idx]

        # Normalize reg_keyframe_ids to 1-D int array
        def _to_int_scalar(x):
            try:
                if isinstance(x, (np.ndarray, list, tuple)):
                    return int(np.asarray(x).reshape(-1)[0])
                return int(x)
            except Exception:
                return int(np.asarray(x).reshape(-1)[0])

        reg_kf_int = np.asarray(
            [_to_int_scalar(x) for x in reg_keyframe_ids], dtype=int
        )

        # Ensure same length
        min_len = min(len(residuals), len(reg_kf_int))
        residuals_for_plot = residuals[:min_len]
        reg_kf_int_for_plot = reg_kf_int[:min_len]
        inliers_for_plot = inliers[:min_len] if inliers is not None else None

        # Color by inlier/outlier
        if inliers_for_plot is not None:
            colors_scatter = ["green" if i else "red" for i in inliers_for_plot]
            ax.scatter(
                reg_kf_int_for_plot,
                residuals_for_plot,
                c=colors_scatter,
                alpha=0.6,
                s=20,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.legend(handles=legend_elements)
        else:
            ax.scatter(
                reg_kf_int_for_plot,
                residuals_for_plot,
                alpha=0.6,
                s=20,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Frame ID")
        ax.grid(True, alpha=0.3)

    # 9. Error vs Uncertainty plot (GT-aligned)
    if has_error_vs_unc:
        # Calculate plot index
        plot_idx = 4
        if has_unc:
            plot_idx += 2
        if has_unc and has_kfids:
            plot_idx += 1
        if reg_keyframe_ids is not None:
            plot_idx += 1

        ax = axes[plot_idx]

        # Try to extract video_name and results_dir from register_folder if not provided
        actual_results_dir = results_dir
        actual_video_name = video_name
        if (
            actual_results_dir is None or actual_video_name is None
        ) and register_folder:
            detected_results_dir, detected_video_name = extract_video_info_from_path(
                register_folder
            )
            if detected_results_dir and actual_results_dir is None:
                actual_results_dir = detected_results_dir
            if detected_video_name and actual_video_name is None:
                actual_video_name = detected_video_name

        # Get GT pose for current frame
        gt_pose = get_gt_pose_for_frame(
            register_folder,
            frame_number,
            results_dir=actual_results_dir,
            video_name=actual_video_name,
            ho3d_root=ho3d_root,
        )

        if gt_pose is not None:
            # Ensure arrays have the same length
            min_len = min(len(key_points), len(curr3d), len(uncertainties))
            key_points_aligned = key_points[:min_len]
            curr3d_aligned = curr3d[:min_len]
            uncertainties_aligned = uncertainties[:min_len]

            # Get inliers if available (need to check if inliers parameter exists)
            inliers_aligned = None
            if "inliers" in locals() or "inliers" in globals():
                # Try to get inliers from the function's scope
                try:
                    # inliers is passed to plot_registration_stats, check if we can access it
                    pass  # Will handle below
                except:
                    pass

            # Transform source points using GT pose
            # Note: gt_pose is already transformed relative to first frame (first frame = identity)
            if mode == "f2m":
                # For f2m mode: key_points are in first frame coordinates
                # Since first frame is now identity, transform is just gt_pose
                transform = gt_pose
            else:
                # f2f mode: key_points are from previous frame
                prev_frame_gt_pose = get_gt_pose_for_frame(
                    register_folder,
                    frame_number - 1 if frame_number > 0 else 0,
                    results_dir=actual_results_dir,
                    video_name=actual_video_name,
                    ho3d_root=ho3d_root,
                )
                if prev_frame_gt_pose is not None and frame_number > 0:
                    # Both poses are already transformed relative to first frame
                    # Transform from prev frame to current: gt_pose @ inverse_SE3(prev_frame_gt_pose)
                    if inverse_SE3 is not None:
                        transform = gt_pose @ inverse_SE3(prev_frame_gt_pose)
                    else:
                        prev_frame_gt_pose_inv = np.linalg.inv(prev_frame_gt_pose)
                        transform = gt_pose @ prev_frame_gt_pose_inv
                else:
                    transform = gt_pose

            # Transform source points to current frame using GT transform
            # For f2m: key_points are in first frame coordinates, transform to current frame
            key_points_h = np.hstack(
                [key_points_aligned, np.ones((len(key_points_aligned), 1))]
            )
            key_points_transformed = (transform @ key_points_h.T).T[:, :3]

            # curr3d is already in current frame coordinates, no transformation needed
            # Compute error as Euclidean distance between transformed source and target
            errors = np.linalg.norm(key_points_transformed - curr3d_aligned, axis=1)

            # Scatter plot: error vs uncertainty
            ax.scatter(
                uncertainties_aligned,
                errors,
                alpha=0.6,
                s=20,
                edgecolors="black",
                linewidths=0.5,
            )

            ax.set_xlabel("Uncertainty", fontsize=10)
            ax.set_ylabel("Error (m)", fontsize=10)
            ax.set_title("Error vs Uncertainty (GT-aligned)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_error = np.mean(errors)
            mean_uncertainty = np.mean(uncertainties_aligned)
            ax.axhline(
                mean_error,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=1,
                label=f"Mean Error: {mean_error:.4f}m",
            )
            ax.axvline(
                mean_uncertainty,
                color="blue",
                linestyle="--",
                alpha=0.7,
                linewidth=1,
                label=f"Mean Unc: {mean_uncertainty:.4f}",
            )
            ax.legend(fontsize=8)
        else:
            # If GT pose not available, show a message
            ax.text(
                0.5,
                0.5,
                "GT pose not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_title("Error vs Uncertainty (GT-aligned)", fontsize=10)
            ax.set_xlabel("Uncertainty", fontsize=10)
            ax.set_ylabel("Error (m)", fontsize=10)
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        plt.close()  # Close figure to allow next plot to be created
    else:
        # Don't show yet - will show at the end with all figures
        pass


def load_point_cloud(
    register_folder: str, object_number: int, frame_number: int
) -> Optional[np.ndarray]:
    """Load a point cloud from the register folder.

    Returns:
        numpy array of points (N, 3) or None if not found
    """
    _ = object_number  # Used in filename format string
    filename = f"obj_{object_number}_frame_{frame_number}.ply"
    filepath = os.path.join(register_folder, filename)

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            print(f"Warning: Point cloud {filename} is empty")
            return None
        return np.asarray(pcd.points)
    except ImportError:
        print("Warning: open3d not available, cannot load point cloud")
        return None
    except (IOError, ValueError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def load_all_key_point_frame_ids_raw(
    register_folder: str,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load ALL key point frame IDs (unfiltered, length = obj_key_points_lengths/3).

    Returns a 1-D int array aligned to original key point indexing so reg_key_points_idx
    can be used directly without bounds issues from filtering.
    Uses unpack_ragged following notebook pattern.

    Args:
        register_folder: Path to register folder
        frame_number: Frame number
        results_dir: Results directory for new structure (optional)
        video_name: Video sequence name for new structure (optional)
    """
    meta_data_path = find_meata_data_path(
        register_folder,
        meta_data_path_override=None,
        results_dir=results_dir,
        video_name=video_name,
    )
    if meta_data_path is None:
        print("No meta_data.npz found for loading raw key point frame ids")
        return None

    try:
        data = np.load(meta_data_path, allow_pickle=True)
        if "frame_id" not in data:
            return None

        frame_ids = data["frame_id"]
        frame_idx = None
        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break
        if frame_idx is None:
            return None

        # Use unpack_ragged to get frame IDs (following notebook pattern)
        obj_key_point_frames_list = unpack_ragged("obj_key_point_frames", data, dim=-1)

        if frame_idx < len(obj_key_point_frames_list):
            return obj_key_point_frames_list[frame_idx].astype(int)
        return None
    except Exception as e:
        print(f"Error loading raw key point frame ids: {e}")
        return None


def load_all_key_points_with_frame_ids(
    register_folder: str,
    object_number: int,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load all key points and their frame IDs from meta_data.npz.

    Uses unpack_ragged following notebook pattern.

    Args:
        register_folder: Path to register folder
        object_number: Object number (currently unused, kept for compatibility)
        frame_number: Frame number
        results_dir: Results directory for new structure (optional)
        video_name: Video sequence name for new structure (optional)

    Returns:
        Tuple of (key_points, frame_ids) if found, None otherwise.
        key_points: (N, 3) array of 3D points
        frame_ids: (N,) array of frame IDs for each point
    """
    _ = object_number  # Currently unused, kept for compatibility
    meta_data_path = find_meata_data_path(
        register_folder,
        meta_data_path_override=None,
        results_dir=results_dir,
        video_name=video_name,
    )
    if meta_data_path is None:
        print("No meta_data.npz found for loading all key points")
        return None

    try:
        data = np.load(meta_data_path, allow_pickle=True)

        if "frame_id" not in data:
            print("Error: No frame_id in meta_data.npz")
            return None

        frame_ids = data["frame_id"]
        frame_idx = None
        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"Error: Frame {frame_number} not found in meta_data.npz")
            return None

        # Use unpack_ragged to extract data (following notebook pattern)
        obj_key_points_list = unpack_ragged("obj_key_points", data, dim=3)
        obj_key_point_frames_list = unpack_ragged("obj_key_point_frames", data, dim=-1)

        if frame_idx >= len(obj_key_points_list) or frame_idx >= len(
            obj_key_point_frames_list
        ):
            print(f"Error: frame_idx ({frame_idx}) out of range")
            return None

        pts = obj_key_points_list[frame_idx]
        key_point_frame_ids = obj_key_point_frames_list[frame_idx].astype(int)

        # Filter NaN points
        if np.any(np.isnan(pts)):
            valid_mask = ~np.isnan(pts).any(axis=1)
            pts = pts[valid_mask]
            key_point_frame_ids = key_point_frame_ids[valid_mask]

        # Transform to current frame using obj_pose
        if "obj_pose" in data:
            obj_pose = data["obj_pose"][frame_idx]
            if not (np.any(np.isnan(obj_pose)) or np.any(np.isinf(obj_pose))):
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                pts_transformed_homo = (obj_pose @ pts_h.T).T
                pts = pts_transformed_homo[:, :3]

                # Filter NaN after transformation
                if np.any(np.isnan(pts)):
                    valid_mask = ~np.isnan(pts).any(axis=1)
                    pts = pts[valid_mask]
                    key_point_frame_ids = key_point_frame_ids[valid_mask]

        # Ensure same length
        min_len = min(len(pts), len(key_point_frame_ids))
        pts = pts[:min_len]
        key_point_frame_ids = key_point_frame_ids[:min_len]

        return pts, key_point_frame_ids

    except (IOError, ValueError, KeyError) as e:
        print(f"Error loading all key points: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_gt_key_points(
    register_folder: str,
    object_number: int,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    ho3d_root: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load ground truth key points by transforming key point map using GT pose.

    Args:
        register_folder: Path to register folder
        object_number: Object number (currently unused, kept for compatibility)
        frame_number: Frame number to load GT key points for
        results_dir: Results directory for new structure (optional)
        video_name: Video sequence name for new structure (optional)
        ho3d_root: HO3D dataset root directory (optional, will try to infer)

    Returns:
        GT key points in current frame coordinate system (N, 3) or None if not found
    """
    _ = object_number  # Currently unused, kept for compatibility

    # Load key point map from meta_data.npz (stored in first frame coordinate system)
    meta_data_path = find_meata_data_path(
        register_folder,
        meta_data_path_override=None,
        results_dir=results_dir,
        video_name=video_name,
    )
    if meta_data_path is None:
        print("No meta_data.npz found for loading GT key points")
        return None

    try:
        data = np.load(meta_data_path, allow_pickle=True)

        # Get key point map (stored in first frame coordinate system)
        obj_key_points_list = unpack_ragged("obj_key_points", data, dim=3)

        # Find frame index for the CURRENT frame to get the full accumulated map
        if "frame_id" not in data:
            print("No frame_id field found in meta_data.npz")
            return None

        frame_ids = data["frame_id"]
        current_frame_idx = None

        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                current_frame_idx = i
                break

        if current_frame_idx is None:
            print(
                f"Frame {frame_number} not found in meta_data.npz, falling back to last frame"
            )
            current_frame_idx = len(frame_ids) - 1

        if current_frame_idx >= len(obj_key_points_list):
            print(f"Frame index {current_frame_idx} out of range for obj_key_points")
            return None

        key_point_map = obj_key_points_list[
            current_frame_idx
        ]  # Key points in first frame coords, but accumulated up to current frame

        if len(key_point_map) == 0:
            print("Key point map is empty")
            return None

        # Load GT pose for current frame using datareader
        # Try to infer video directory from ho3d_root first (HO3D dataset location)
        # Then fall back to results_dir if needed
        video_dir = None

        # Priority 1: ho3d_root/video_name (actual HO3D dataset location)
        if ho3d_root and video_name:
            potential_video_dir = os.path.join(ho3d_root, video_name)
            # Verify it's a valid HO3D video directory (should have rgb/ subdirectory)
            if os.path.exists(potential_video_dir) and os.path.exists(
                os.path.join(potential_video_dir, "rgb")
            ):
                video_dir = potential_video_dir

        # Priority 2: results_dir/video_name (fallback, but usually doesn't have rgb/)
        if video_dir is None and results_dir and video_name:
            potential_video_dir = os.path.join(results_dir, video_name)
            # Only use if it has rgb/ subdirectory (unlikely but check anyway)
            if os.path.exists(potential_video_dir) and os.path.exists(
                os.path.join(potential_video_dir, "rgb")
            ):
                video_dir = potential_video_dir

        # If still not found and video_name is provided, try common locations
        if video_dir is None and video_name:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            potential_paths = [
                os.path.join(project_root, "data", "ho3d", video_name),
                os.path.join(project_root, "data", video_name),
                os.path.join(
                    "/mnt",
                    "9a72c439-d0a7-45e8-8d20-d7a235d02763",
                    "DATASET",
                    "HO3D",
                    video_name,
                ),
            ]
            # Also try with ho3d_root if provided
            if ho3d_root:
                potential_paths.insert(0, os.path.join(ho3d_root, video_name))

            for path in potential_paths:
                # Verify it's a valid HO3D video directory (should have rgb/ subdirectory)
                if os.path.exists(path) and os.path.exists(os.path.join(path, "rgb")):
                    video_dir = path
                    break

        if video_dir is None or not os.path.exists(video_dir):
            print(
                f"Warning: Could not find video directory for {video_name}, cannot load GT pose"
            )
            if ho3d_root:
                searched_path = os.path.join(ho3d_root, video_name)
                exists = os.path.exists(searched_path)
                has_rgb = (
                    os.path.exists(os.path.join(searched_path, "rgb"))
                    if exists
                    else False
                )
                print(
                    f"  Tried: {searched_path} (exists: {exists}, has rgb/: {has_rgb})"
                )
            if results_dir:
                searched_path = os.path.join(results_dir, video_name)
                exists = os.path.exists(searched_path)
                has_rgb = (
                    os.path.exists(os.path.join(searched_path, "rgb"))
                    if exists
                    else False
                )
                print(
                    f"  Tried: {searched_path} (exists: {exists}, has rgb/: {has_rgb})"
                )
            return None

        # Verify video_dir has rgb/ subdirectory before proceeding
        rgb_dir = os.path.join(video_dir, "rgb")
        if not os.path.exists(rgb_dir):
            print(
                f"Warning: Video directory {video_dir} does not contain rgb/ subdirectory, cannot load GT pose"
            )
            return None

        # Try to infer ho3d_root if not provided or if provided ho3d_root doesn't have models
        # The ho3d_root should contain a "models" directory
        actual_ho3d_root = ho3d_root
        if ho3d_root is None or not os.path.exists(os.path.join(ho3d_root, "models")):
            # Common HO3D root locations
            potential_roots = []
            if ho3d_root:
                # If ho3d_root was provided but doesn't have models, try parent directory
                potential_roots.append(os.path.dirname(ho3d_root))
            if video_dir:
                potential_roots.extend(
                    [
                        os.path.dirname(
                            video_dir
                        ),  # Video dir might be directly under ho3d_root
                        os.path.join(os.path.dirname(video_dir), ".."),
                    ]
                )
            potential_roots.extend(
                [
                    os.path.join(
                        "/mnt",
                        "9a72c439-d0a7-45e8-8d20-d7a235d02763",
                        "DATASET",
                        "HO3D",
                    ),
                ]
            )
            for root in potential_roots:
                if (
                    root
                    and os.path.exists(root)
                    and os.path.exists(os.path.join(root, "models"))
                ):
                    actual_ho3d_root = root
                    break

        # If we still don't have a valid ho3d_root, use the provided one anyway
        if actual_ho3d_root is None:
            actual_ho3d_root = ho3d_root

        # Create datareader
        reader = None
        if actual_ho3d_root and os.path.exists(actual_ho3d_root) and video_dir:
            try:
                print(
                    f"Creating Ho3dReader with video_dir={video_dir}, ho3d_root={actual_ho3d_root}"
                )
                reader = Ho3dReader(video_dir, actual_ho3d_root)
            except Exception as e:
                print(f"Warning: Could not create Ho3dReader: {e}")
                import traceback

                traceback.print_exc()

        if reader is None:
            print("Warning: Could not create datareader, cannot load GT pose")
            return None

        # Get GT pose for current frame
        if frame_number >= len(reader):
            print(
                f"Frame {frame_number} out of range for datareader (length: {len(reader)})"
            )
            return None

        gt_pose_current_raw = reader.get_gt_pose(frame_number)
        if gt_pose_current_raw is None:
            print(f"GT pose not available for frame {frame_number}")
            return None

        # Get GT pose for first frame (for reference)
        # Note: We need the transform from first frame to current frame
        # because the key_point_map is stored in the first frame's coordinate system.
        # This is true even for accumulated maps - new points are transformed back to frame 0 before adding.

        # Find index of frame 0 (or whatever the reference frame is)
        # Assuming the map reference frame is the one at index 0 of the sequence
        first_frame_idx = 0
        gt_pose_first = reader.get_gt_pose(first_frame_idx)

        if gt_pose_first is None:
            # If first frame doesn't have GT (e.g. tracking started mid-sequence), try to find first valid GT
            for i in range(len(reader)):
                pose = reader.get_gt_pose(i)
                if pose is not None:
                    gt_pose_first = pose
                    first_frame_idx = i
                    print(
                        f"Using frame {i} as reference for GT pose (first frame was None)"
                    )
                    break

            if gt_pose_first is None:
                print(
                    "GT pose not available for any frame, using current frame pose directly"
                )
                # Transform key points directly with current GT pose
                # Assuming key points are in object frame, transform to camera frame
                key_points_h = np.hstack(
                    [key_point_map, np.ones((len(key_point_map), 1))]
                )
                gt_key_points = (gt_pose_current_raw @ key_points_h.T).T[:, :3]
                return gt_key_points

        # Transform GT poses by inverse of first frame's pose (assuming first frame is identity)
        if inverse_SE3 is not None:
            # gt_pose_current = inverse_SE3(gt_pose_first) @ gt_pose_current_raw
            gt_pose_current = gt_pose_current_raw @ inverse_SE3(gt_pose_first)
            gt_pose_first_transformed = (
                inverse_SE3(gt_pose_first) @ gt_pose_first
            )  # This should be identity
        else:
            # Fallback to numpy inverse if inverse_SE3 not available
            gt_pose_first_inv = np.linalg.inv(gt_pose_first)
            gt_pose_current = gt_pose_first_inv @ gt_pose_current_raw
            gt_pose_first_transformed = (
                gt_pose_first_inv @ gt_pose_first
            )  # This should be identity

        # Transform key points from first frame to current frame using transformed GT poses
        # Key points are in first frame coordinate system (reference frame)
        # Since first frame is now identity, transform is just gt_pose_current
        # Transform: gt_pose_current @ key_points (where gt_pose_current is already relative to first frame)
        transform = gt_pose_current

        key_points_h = np.hstack([key_point_map, np.ones((len(key_point_map), 1))])
        gt_key_points = (transform @ key_points_h.T).T[:, :3]

        # Filter NaN points
        if np.any(np.isnan(gt_key_points)):
            valid_mask = ~np.isnan(gt_key_points).any(axis=1)
            gt_key_points = gt_key_points[valid_mask]

        print(f"Loaded {len(gt_key_points)} GT key points for frame {frame_number}")
        return gt_key_points

    except Exception as e:
        print(f"Error loading GT key points: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_gt_keypoint_map_for_frame(
    register_folder: str,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    ho3d_root: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Get GT keypoint map in the CURRENT frame coordinate system for a given frame.

    This mirrors the notebook's use of a GT-corrected keypoint map: we start from the
    stored keypoint map in the first-frame coordinates and transform it into the
    current frame using the GT pose.
    """
    meta_data_path = find_meata_data_path(
        register_folder,
        meta_data_path_override=None,
        results_dir=results_dir,
        video_name=video_name,
    )
    if meta_data_path is None:
        print("No meta_data.npz found for GT keypoint map")
        return None

    try:
        data = np.load(meta_data_path, allow_pickle=True)

        if "frame_id" not in data:
            print("No frame_id field found in meta_data.npz")
            return None

        frame_ids = data["frame_id"]
        frame_idx = None
        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"Frame {frame_number} not found in meta_data.npz")
            return None

        # Keypoint map in first-frame coordinates (no filtering to preserve indices)
        obj_key_points_list = unpack_ragged("obj_key_points", data, dim=3)
        if frame_idx >= len(obj_key_points_list):
            print(f"Frame index {frame_idx} out of range for obj_key_points")
            return None

        key_point_map_first = obj_key_points_list[frame_idx]
        if key_point_map_first is None or len(key_point_map_first) == 0:
            print("Key point map is empty")
            return None

        # GT pose from first frame to current frame
        gt_pose_current = get_gt_pose_for_frame(
            register_folder,
            frame_number,
            results_dir=results_dir,
            video_name=video_name,
            ho3d_root=ho3d_root,
        )
        if gt_pose_current is None:
            print("Warning: Could not load GT pose for keypoint map")
            return None

        # Transform map from first frame to current frame
        kp_h = np.hstack([key_point_map_first, np.ones((len(key_point_map_first), 1))])
        gt_key_points = (gt_pose_current @ kp_h.T).T[:, :3]
        return gt_key_points
    except Exception as e:
        print(f"Error loading GT keypoint map: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_3d_correspondences(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    curr3d: np.ndarray,
    all_key_points: np.ndarray,
    all_key_point_frame_ids: np.ndarray,
    pair_keyframe_ids: Optional[np.ndarray],
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
):
    """Create a 3D plot showing registered points, all key points (colored by frame ID), and correspondences.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        curr3d: Current 3D points (N, 3)
        all_key_points: All key points (M, 3)
        all_key_point_frame_ids: Frame IDs for all key points (M,)
        pair_keyframe_ids: Optional keyframe IDs for registration pairs (N,)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.3,
            s=1,
            label="Registered Point Cloud",
        )

    # Plot all key points colored by frame ID
    unique_frames = np.unique(all_key_point_frame_ids)

    # Create color map for keyframe IDs
    try:
        import matplotlib

        cmap_get = getattr(getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap")
        tab10_colors = cmap_get("tab10")

        frame_to_color = {}
        for i, fid in enumerate(sorted(unique_frames)):
            color_idx: int = i if i < 3 else i + 1  # Skip red (index 3)
            frame_to_color[fid] = tab10_colors(color_idx % 10)[:3]
    except (AttributeError, ImportError, KeyError):
        # Fallback palette
        palette = [
            (0.2, 0.2, 1.0),  # blue
            (0.0, 0.7, 0.3),  # green
            (1.0, 0.6, 0.0),  # orange
            (0.6, 0.0, 0.8),  # purple
            (0.0, 0.7, 0.7),  # cyan
            (0.6, 0.6, 0.0),  # yellow
        ]
        frame_to_color = {
            fid: palette[i % len(palette)]
            for i, fid in enumerate(sorted(unique_frames))
        }

    # Plot key points grouped by frame ID
    for fid in sorted(unique_frames):
        mask = all_key_point_frame_ids == fid
        if np.any(mask):
            ax.scatter(
                all_key_points[mask, 0],
                all_key_points[mask, 1],
                all_key_points[mask, 2],
                c=[frame_to_color[fid]],
                alpha=0.7,
                s=30,
                label=f"Keyframe {fid}",
                marker="o",
                edgecolors="black",
                linewidths=0.5,
            )

    # Draw correspondence lines
    # Note: key_points are already transformed in load_registration_stats,
    # and curr3d are in the current frame coordinate system (not transformed)
    if len(key_points) == len(curr3d) and len(key_points) > 0:
        if inliers is not None and len(inliers) == len(key_points):
            # Draw lines for inliers (green) and outliers (red) separately
            for i in range(len(key_points)):
                if inliers[i]:
                    # Green line for inliers
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "g-",
                        alpha=0.5,
                        linewidth=2.0,
                    )
                else:
                    # Red line for outliers
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "r-",
                        alpha=0.5,
                        linewidth=2.0,
                    )
        else:
            # If no inliers info, draw all lines in green
            for i in range(len(key_points)):
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "g-",
                    alpha=0.5,
                    linewidth=2.0,
                )

    # Also draw the pair points colored by their keyframe IDs (matches -c)
    if pair_keyframe_ids is not None and len(pair_keyframe_ids) == len(key_points):
        try:
            import matplotlib

            cmap_get = getattr(
                getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap"
            )
            tab10_colors = cmap_get("tab10")
            frame_to_pair_color = {}
            for i, fid in enumerate(sorted(np.unique(pair_keyframe_ids))):
                color_idx = i if i < 3 else i + 1  # avoid red slot
                frame_to_pair_color[fid] = tab10_colors(color_idx % 10)[:3]
        except Exception:
            palette = [
                (0.2, 0.2, 1.0),
                (0.0, 0.7, 0.3),
                (1.0, 0.6, 0.0),
                (0.6, 0.0, 0.8),
                (0.0, 0.7, 0.7),
                (0.6, 0.6, 0.0),
            ]
            frame_to_pair_color = {
                fid: palette[i % len(palette)]
                for i, fid in enumerate(sorted(np.unique(pair_keyframe_ids)))
            }

        pair_colors = np.array([frame_to_pair_color[f] for f in pair_keyframe_ids])

        # Plot inliers and outliers separately with different markers
        if inliers is not None:
            inlier_mask = inliers
            outlier_mask = ~inliers

            # Plot inliers for key_points (source)
            if np.any(inlier_mask):
                ax.scatter(
                    key_points[inlier_mask, 0],
                    key_points[inlier_mask, 1],
                    key_points[inlier_mask, 2],
                    c=pair_colors[inlier_mask],
                    s=60,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label="Registered source (inliers)",
                )

            # Plot outliers for key_points (source) with different marker
            if np.any(outlier_mask):
                ax.scatter(
                    key_points[outlier_mask, 0],
                    key_points[outlier_mask, 1],
                    key_points[outlier_mask, 2],
                    c=pair_colors[outlier_mask],
                    s=70,
                    marker="X",
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.9,
                    label="Registered source (outliers)",
                )

            # Plot inliers for curr3d (target)
            if np.any(inlier_mask):
                ax.scatter(
                    curr3d[inlier_mask, 0],
                    curr3d[inlier_mask, 1],
                    curr3d[inlier_mask, 2],
                    c=pair_colors[inlier_mask],
                    s=60,
                    marker="^",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label="Registered target (inliers)",
                )

            # Plot outliers for curr3d (target) with different marker
            if np.any(outlier_mask):
                ax.scatter(
                    curr3d[outlier_mask, 0],
                    curr3d[outlier_mask, 1],
                    curr3d[outlier_mask, 2],
                    c=pair_colors[outlier_mask],
                    s=70,
                    marker="s",
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.9,
                    label="Registered target (outliers)",
                )
        else:
            # If no inliers info, plot all points with default markers
            ax.scatter(
                key_points[:, 0],
                key_points[:, 1],
                key_points[:, 2],
                c=pair_colors,
                s=60,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                label="Registered source (colored by keyframe)",
            )
            ax.scatter(
                curr3d[:, 0],
                curr3d[:, 1],
                curr3d[:, 2],
                c=pair_colors,
                s=60,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                label="Registered target",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    source_label = "prev3d" if mode == "f2f" else "key_points"
    ax.set_title(
        f"3D Registration Visualization ({mode.upper()}) - Frame {frame_number}\n"
        f"Source Points ({source_label}) Colored by Keyframe ID, Lines Show Correspondences"
    )

    # Limit legend to first 10 frames for readability
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:  # 10 keyframes + point cloud + potential other items
        ax.legend(handles[:12], labels[:12], loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)

    # Set equal aspect ratio
    all_points = np.vstack([all_key_points, curr3d])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])
    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions - manually set aspect
            # The equal limits above should help, but we can't guarantee perfect aspect
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot to: {output_path}")
        plt.close()  # Close figure to allow next plot to be created
    else:
        # Don't show yet - will show at the end with all figures
        pass


def plot_3d_correspondences_uncertainty(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    curr3d: np.ndarray,
    all_key_points: np.ndarray,
    all_key_point_frame_ids: np.ndarray,
    uncertainties: Optional[np.ndarray],
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
    reg_key_points_idx: Optional[np.ndarray] = None,
):
    """Create a 3D plot showing registered points colored by uncertainty, similar to plot_3d_correspondences.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        curr3d: Current 3D points (N, 3)
        all_key_points: All key points (M, 3)
        all_key_point_frame_ids: Frame IDs for all key points (M,)
        uncertainties: Array of uncertainties for each registration point (N,)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.3,
            s=1,
            label="Registered Point Cloud",
        )

    # Determine which key points are used in registration
    reg_indices_set = set()
    if reg_key_points_idx is not None:
        reg_indices_set = set(reg_key_points_idx.flatten())

    # Plot all key points - color by frame ID if used in registration, grey otherwise
    unique_frames = np.unique(all_key_point_frame_ids)

    # Create color map for keyframe IDs
    try:
        import matplotlib

        cmap_get = getattr(getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap")
        tab10_colors = cmap_get("tab10")

        frame_to_color = {}
        for i, fid in enumerate(sorted(unique_frames)):
            color_idx: int = i if i < 3 else i + 1  # Skip red (index 3)
            frame_to_color[fid] = tab10_colors(color_idx % 10)[:3]
    except (AttributeError, ImportError, KeyError):
        # Fallback palette
        palette = [
            (0.2, 0.2, 1.0),  # blue
            (0.0, 0.7, 0.3),  # green
            (1.0, 0.6, 0.0),  # orange
            (0.6, 0.0, 0.8),  # purple
            (0.0, 0.7, 0.7),  # cyan
            (0.6, 0.6, 0.0),  # yellow
        ]
        frame_to_color = {
            fid: palette[i % len(palette)]
            for i, fid in enumerate(sorted(unique_frames))
        }

    # Plot key points grouped by frame ID, but color non-registration points grey
    grey_color = (0.5, 0.5, 0.5)  # Grey color
    grey_labeled = False  # Track if we've added grey to legend

    for fid in sorted(unique_frames):
        mask = all_key_point_frame_ids == fid
        if np.any(mask):
            # Separate points used in registration from those not used
            frame_indices = np.where(mask)[0]
            reg_mask = np.array([idx in reg_indices_set for idx in frame_indices])
            non_reg_mask = ~reg_mask

            # Plot registration points colored by frame ID
            if np.any(reg_mask):
                reg_frame_indices = frame_indices[reg_mask]
                ax.scatter(
                    all_key_points[reg_frame_indices, 0],
                    all_key_points[reg_frame_indices, 1],
                    all_key_points[reg_frame_indices, 2],
                    c=[frame_to_color[fid]],
                    alpha=0.7,
                    s=30,
                    label=f"Keyframe {fid}",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                )

            # Plot non-registration points in grey
            if np.any(non_reg_mask):
                non_reg_frame_indices = frame_indices[non_reg_mask]
                label = "Other key points" if not grey_labeled else ""
                ax.scatter(
                    all_key_points[non_reg_frame_indices, 0],
                    all_key_points[non_reg_frame_indices, 1],
                    all_key_points[non_reg_frame_indices, 2],
                    c=[grey_color],
                    alpha=0.5,
                    s=20,
                    label=label,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.3,
                )
                grey_labeled = True

    # Draw correspondence lines (same as original)
    if len(key_points) > 0 and len(key_points) == len(curr3d):
        if inliers is not None and len(inliers) == len(key_points):
            for i in range(len(key_points)):
                if inliers[i]:
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "g-",
                        alpha=0.5,
                        linewidth=2.0,
                    )
                else:
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "r-",
                        alpha=0.5,
                        linewidth=2.0,
                    )
        else:
            for i in range(len(key_points)):
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "g-",
                    alpha=0.5,
                    linewidth=2.0,
                )

    # Color points by uncertainty using a colormap
    if uncertainties is not None and len(uncertainties) == len(key_points):
        try:
            import matplotlib

            cmap_get = getattr(
                getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap"
            )
            # Use a colormap that shows uncertainty well (viridis or coolwarm)
            uncertainty_cmap = cmap_get("viridis")

            # Normalize uncertainties for colormap
            uncertainty_normalized = (uncertainties - uncertainties.min()) / (
                uncertainties.max() - uncertainties.min() + 1e-10
            )
            uncertainty_colors = uncertainty_cmap(uncertainty_normalized)

            # Plot inliers and outliers separately with different markers
            if inliers is not None:
                inlier_mask = inliers
                outlier_mask = ~inliers

                # Plot inliers for key_points (source)
                if np.any(inlier_mask):
                    ax.scatter(
                        key_points[inlier_mask, 0],
                        key_points[inlier_mask, 1],
                        key_points[inlier_mask, 2],
                        c=uncertainty_colors[inlier_mask],
                        s=60,
                        marker="o",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.9,
                        label="Registered source (inliers)",
                    )

                # Plot outliers for key_points (source)
                if np.any(outlier_mask):
                    ax.scatter(
                        key_points[outlier_mask, 0],
                        key_points[outlier_mask, 1],
                        key_points[outlier_mask, 2],
                        c=uncertainty_colors[outlier_mask],
                        s=70,
                        marker="X",
                        edgecolors="black",
                        linewidths=0.7,
                        alpha=0.9,
                        label="Registered source (outliers)",
                    )

                # Plot inliers for curr3d (target)
                if np.any(inlier_mask):
                    ax.scatter(
                        curr3d[inlier_mask, 0],
                        curr3d[inlier_mask, 1],
                        curr3d[inlier_mask, 2],
                        c=uncertainty_colors[inlier_mask],
                        s=60,
                        marker="^",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.9,
                        label="Registered target (inliers)",
                    )

                # Plot outliers for curr3d (target)
                if np.any(outlier_mask):
                    ax.scatter(
                        curr3d[outlier_mask, 0],
                        curr3d[outlier_mask, 1],
                        curr3d[outlier_mask, 2],
                        c=uncertainty_colors[outlier_mask],
                        s=70,
                        marker="s",
                        edgecolors="black",
                        linewidths=0.7,
                        alpha=0.9,
                        label="Registered target (outliers)",
                    )
            else:
                # If no inliers info, plot all points
                ax.scatter(
                    key_points[:, 0],
                    key_points[:, 1],
                    key_points[:, 2],
                    c=uncertainty_colors,
                    s=60,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label="Registered source (colored by uncertainty)",
                )
                ax.scatter(
                    curr3d[:, 0],
                    curr3d[:, 1],
                    curr3d[:, 2],
                    c=uncertainty_colors,
                    s=60,
                    marker="^",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label="Registered target",
                )

            # Add colorbar for uncertainty
            sm = plt.cm.ScalarMappable(
                cmap=uncertainty_cmap,
                norm=plt.Normalize(vmin=uncertainties.min(), vmax=uncertainties.max()),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label("Uncertainty", rotation=270, labelpad=15)
        except Exception as e:
            print(f"Warning: Could not create uncertainty colormap: {e}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D Registration Visualization ({mode.upper()}) - Frame {frame_number}\n"
        f"Points Colored by Uncertainty, Lines Show Correspondences"
    )

    # Limit legend to first 10 frames for readability
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        ax.legend(handles[:12], labels[:12], loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)

    # Set equal aspect ratio
    all_points = np.vstack([all_key_points, curr3d])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])
    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions - manually set aspect
            # The equal limits above should help, but we can't guarantee perfect aspect
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D uncertainty plot to: {output_path}")
        plt.close()
    else:
        pass


def plot_3d_correspondences_residuals(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    curr3d: np.ndarray,
    all_key_points: np.ndarray,
    all_key_point_frame_ids: np.ndarray,
    residuals: Optional[np.ndarray],
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
    reg_key_points_idx: Optional[np.ndarray] = None,
):
    """Create a 3D plot showing registered points colored by residuals, similar to plot_3d_correspondences.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        curr3d: Current 3D points (N, 3)
        all_key_points: All key points (M, 3)
        all_key_point_frame_ids: Frame IDs for all key points (M,)
        residuals: Array of residuals for each registration point (N,)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.3,
            s=1,
            label="Registered Point Cloud",
        )

    # Determine which key points are used in registration
    reg_indices_set = set()
    if reg_key_points_idx is not None:
        reg_indices_set = set(reg_key_points_idx.flatten())

    # Plot all key points - color by frame ID if used in registration, grey otherwise
    unique_frames = np.unique(all_key_point_frame_ids)

    # Create color map for keyframe IDs
    try:
        import matplotlib

        cmap_get = getattr(getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap")
        tab10_colors = cmap_get("tab10")

        frame_to_color = {}
        for i, fid in enumerate(sorted(unique_frames)):
            color_idx: int = i if i < 3 else i + 1  # Skip red (index 3)
            frame_to_color[fid] = tab10_colors(color_idx % 10)[:3]
    except (AttributeError, ImportError, KeyError):
        # Fallback palette
        palette = [
            (0.2, 0.2, 1.0),  # blue
            (0.0, 0.7, 0.3),  # green
            (1.0, 0.6, 0.0),  # orange
            (0.6, 0.0, 0.8),  # purple
            (0.0, 0.7, 0.7),  # cyan
            (0.6, 0.6, 0.0),  # yellow
        ]
        frame_to_color = {
            fid: palette[i % len(palette)]
            for i, fid in enumerate(sorted(unique_frames))
        }

    # Plot key points grouped by frame ID, but color non-registration points grey
    grey_color = (0.5, 0.5, 0.5)  # Grey color
    grey_labeled = False  # Track if we've added grey to legend

    for fid in sorted(unique_frames):
        mask = all_key_point_frame_ids == fid
        if np.any(mask):
            # Separate points used in registration from those not used
            frame_indices = np.where(mask)[0]
            reg_mask = np.array([idx in reg_indices_set for idx in frame_indices])
            non_reg_mask = ~reg_mask

            # Plot registration points colored by frame ID
            if np.any(reg_mask):
                reg_frame_indices = frame_indices[reg_mask]
                ax.scatter(
                    all_key_points[reg_frame_indices, 0],
                    all_key_points[reg_frame_indices, 1],
                    all_key_points[reg_frame_indices, 2],
                    c=[frame_to_color[fid]],
                    alpha=0.7,
                    s=30,
                    label=f"Keyframe {fid}",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                )

            # Plot non-registration points in grey
            if np.any(non_reg_mask):
                non_reg_frame_indices = frame_indices[non_reg_mask]
                label = "Other key points" if not grey_labeled else ""
                ax.scatter(
                    all_key_points[non_reg_frame_indices, 0],
                    all_key_points[non_reg_frame_indices, 1],
                    all_key_points[non_reg_frame_indices, 2],
                    c=[grey_color],
                    alpha=0.5,
                    s=20,
                    label=label,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.3,
                )
                grey_labeled = True

    # Filter to inliers only if available
    if inliers is not None:
        inlier_mask = inliers
        key_points_filtered = key_points[inlier_mask]
        curr3d_filtered = curr3d[inlier_mask]
        residuals_filtered = residuals[inlier_mask] if residuals is not None else None
    else:
        # If no inliers info, use all points
        inlier_mask = None
        key_points_filtered = key_points
        curr3d_filtered = curr3d
        residuals_filtered = residuals

    # Draw correspondence lines (only for inliers)
    if len(key_points_filtered) == len(curr3d_filtered):
        for i in range(len(key_points_filtered)):
            ax.plot(
                [key_points_filtered[i, 0], curr3d_filtered[i, 0]],
                [key_points_filtered[i, 1], curr3d_filtered[i, 1]],
                [key_points_filtered[i, 2], curr3d_filtered[i, 2]],
                "g-",
                alpha=0.5,
                linewidth=2.0,
            )

    # Color points by residuals using a colormap (inliers only)
    if residuals_filtered is not None and len(residuals_filtered) == len(
        key_points_filtered
    ):
        try:
            import matplotlib

            cmap_get = getattr(
                getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap"
            )
            # Use a colormap that shows residuals well (coolwarm or RdYlGn)
            residual_cmap = cmap_get("coolwarm")

            # Normalize residuals for colormap (using only inlier residuals)
            residual_normalized = (residuals_filtered - residuals_filtered.min()) / (
                residuals_filtered.max() - residuals_filtered.min() + 1e-10
            )
            residual_colors = residual_cmap(residual_normalized)

            # Plot inliers only
            # Plot key_points (source)
            ax.scatter(
                key_points_filtered[:, 0],
                key_points_filtered[:, 1],
                key_points_filtered[:, 2],
                c=residual_colors,
                s=60,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                label="Registered source (inliers only)",
            )

            # Plot curr3d (target)
            ax.scatter(
                curr3d_filtered[:, 0],
                curr3d_filtered[:, 1],
                curr3d_filtered[:, 2],
                c=residual_colors,
                s=60,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                label="Registered target (inliers only)",
            )

            # Add colorbar for residuals (using inlier residual range)
            sm = plt.cm.ScalarMappable(
                cmap=residual_cmap,
                norm=plt.Normalize(
                    vmin=residuals_filtered.min(), vmax=residuals_filtered.max()
                ),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label("Residual (inliers only)", rotation=270, labelpad=15)
        except Exception as e:
            print(f"Warning: Could not create residual colormap: {e}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if inliers is not None:
        ax.set_title(
            f"3D Registration Visualization ({mode.upper()}) - Frame {frame_number}\n"
            f"Points Colored by Residual (Inliers Only), Lines Show Correspondences"
        )
    else:
        ax.set_title(
            f"3D Registration Visualization ({mode.upper()}) - Frame {frame_number}\n"
            f"Points Colored by Residual, Lines Show Correspondences"
        )

    # Limit legend to first 10 frames for readability
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        ax.legend(handles[:12], labels[:12], loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)

    # Set equal aspect ratio
    all_points = np.vstack([all_key_points, curr3d])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])
    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions - manually set aspect
            # The equal limits above should help, but we can't guarantee perfect aspect
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D residual plot to: {output_path}")
        plt.close()
    else:
        pass


def plot_3d_registration_points_only(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    curr3d: np.ndarray,
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
):
    """Create a 3D plot showing only the registration points (source and target) with correspondences.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        curr3d: Current 3D points (N, 3)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available (for context)
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.2,
            s=1,
            label="Point Cloud (context)",
        )

    # Ensure arrays have the same length
    min_len = min(len(key_points), len(curr3d))
    key_points = key_points[:min_len]
    curr3d = curr3d[:min_len]
    if inliers is not None:
        inliers = inliers[:min_len]

    # Draw correspondence lines
    if len(key_points) > 0 and len(key_points) == len(curr3d):
        if inliers is not None and len(inliers) == len(key_points):
            inlier_mask = inliers
            outlier_mask = ~inliers

            # Draw lines for inliers (green) and outliers (red) separately
            for i in range(len(key_points)):
                if inlier_mask[i]:
                    # Green line for inliers
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "g-",
                        alpha=0.6,
                        linewidth=1.5,
                    )
                else:
                    # Red line for outliers
                    ax.plot(
                        [key_points[i, 0], curr3d[i, 0]],
                        [key_points[i, 1], curr3d[i, 1]],
                        [key_points[i, 2], curr3d[i, 2]],
                        "r-",
                        alpha=0.6,
                        linewidth=1.5,
                    )
        else:
            # If no inliers info, draw all lines in green
            for i in range(len(key_points)):
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "g-",
                    alpha=0.6,
                    linewidth=1.5,
                )

    # Plot source points (key_points or prev3d)
    if inliers is not None:
        inlier_mask = inliers
        outlier_mask = ~inliers

        # Plot inliers for source points
        if np.any(inlier_mask):
            ax.scatter(
                key_points[inlier_mask, 0],
                key_points[inlier_mask, 1],
                key_points[inlier_mask, 2],
                c="green",
                s=80,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
                label="Source (inliers)",
            )

        # Plot outliers for source points with different marker
        if np.any(outlier_mask):
            ax.scatter(
                key_points[outlier_mask, 0],
                key_points[outlier_mask, 1],
                key_points[outlier_mask, 2],
                c="red",
                s=100,
                marker="X",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.9,
                label="Source (outliers)",
            )
    else:
        # If no inliers info, plot all source points
        ax.scatter(
            key_points[:, 0],
            key_points[:, 1],
            key_points[:, 2],
            c="blue",
            s=80,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            label="Source",
        )

    # Plot target points (curr3d)
    if inliers is not None:
        inlier_mask = inliers
        outlier_mask = ~inliers

        # Plot inliers for target points
        if np.any(inlier_mask):
            ax.scatter(
                curr3d[inlier_mask, 0],
                curr3d[inlier_mask, 1],
                curr3d[inlier_mask, 2],
                c="green",
                s=80,
                marker="^",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
                label="Target (inliers)",
            )

        # Plot outliers for target points with different marker
        if np.any(outlier_mask):
            ax.scatter(
                curr3d[outlier_mask, 0],
                curr3d[outlier_mask, 1],
                curr3d[outlier_mask, 2],
                c="red",
                s=100,
                marker="s",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.9,
                label="Target (outliers)",
            )
    else:
        # If no inliers info, plot all target points
        ax.scatter(
            curr3d[:, 0],
            curr3d[:, 1],
            curr3d[:, 2],
            c="orange",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            label="Target",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    source_label = "prev3d" if mode == "f2f" else "key_points"
    num_inliers = np.sum(inliers) if inliers is not None else len(key_points)
    num_outliers = np.sum(~inliers) if inliers is not None else 0
    ax.set_title(
        f"Registration Points Only ({mode.upper()}) - Frame {frame_number}\n"
        f"Source ({source_label}) ↔ Target (curr3d) | "
        f"Inliers: {num_inliers}, Outliers: {num_outliers}"
    )

    ax.legend(loc="upper left", fontsize=10)

    # Set equal aspect ratio
    all_points = np.vstack([key_points, curr3d])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])
    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions - manually set aspect
            # The equal limits above should help, but we can't guarantee perfect aspect
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D registration points only plot to: {output_path}")
        plt.close()
    else:
        pass


def plot_3d_with_gt_key_points(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    curr3d: np.ndarray,
    gt_key_points: np.ndarray,
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
    mode: str = "f2m",
):
    """Create a 3D plot showing source points, target points, and GT key points.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        curr3d: Current 3D points (N, 3)
        gt_key_points: Ground truth key points (M, 3)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.2,
            s=1,
            label="Point Cloud (context)",
        )

    # Ensure arrays have the same length
    min_len = min(len(key_points), len(curr3d))
    key_points = key_points[:min_len]
    curr3d = curr3d[:min_len]
    if inliers is not None:
        inliers = inliers[:min_len]

    # Draw correspondence lines
    if inliers is not None:
        inlier_mask = inliers
        outlier_mask = ~inliers

        # Draw lines for inliers (green) and outliers (red) separately
        for i in range(len(key_points)):
            if inlier_mask[i]:
                # Green line for inliers
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "g-",
                    alpha=0.4,
                    linewidth=1.5,
                )
            else:
                # Red line for outliers
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "r-",
                    alpha=0.4,
                    linewidth=1.5,
                )
    else:
        # If no inliers info, draw all lines in green
        for i in range(len(key_points)):
            ax.plot(
                [key_points[i, 0], curr3d[i, 0]],
                [key_points[i, 1], curr3d[i, 1]],
                [key_points[i, 2], curr3d[i, 2]],
                "g-",
                alpha=0.4,
                linewidth=1.5,
            )

    # Plot source points (key_points or prev3d)
    if inliers is not None:
        inlier_mask = inliers
        outlier_mask = ~inliers

        # Plot inliers for source points
        if np.any(inlier_mask):
            ax.scatter(
                key_points[inlier_mask, 0],
                key_points[inlier_mask, 1],
                key_points[inlier_mask, 2],
                c="green",
                s=80,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
                label="Source (inliers)",
            )

        # Plot outliers for source points with different marker
        if np.any(outlier_mask):
            ax.scatter(
                key_points[outlier_mask, 0],
                key_points[outlier_mask, 1],
                key_points[outlier_mask, 2],
                c="red",
                s=100,
                marker="X",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.9,
                label="Source (outliers)",
            )
    else:
        # If no inliers info, plot all source points
        ax.scatter(
            key_points[:, 0],
            key_points[:, 1],
            key_points[:, 2],
            c="blue",
            s=80,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            label="Source",
        )

    # Plot target points (curr3d)
    if inliers is not None:
        inlier_mask = inliers
        outlier_mask = ~inliers

        # Plot inliers for target points
        if np.any(inlier_mask):
            ax.scatter(
                curr3d[inlier_mask, 0],
                curr3d[inlier_mask, 1],
                curr3d[inlier_mask, 2],
                c="green",
                s=80,
                marker="^",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
                label="Target (inliers)",
            )

        # Plot outliers for target points with different marker
        if np.any(outlier_mask):
            ax.scatter(
                curr3d[outlier_mask, 0],
                curr3d[outlier_mask, 1],
                curr3d[outlier_mask, 2],
                c="red",
                s=100,
                marker="s",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.9,
                label="Target (outliers)",
            )
    else:
        # If no inliers info, plot all target points
        ax.scatter(
            curr3d[:, 0],
            curr3d[:, 1],
            curr3d[:, 2],
            c="orange",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            label="Target",
        )

    # Plot GT key points
    if gt_key_points is not None and len(gt_key_points) > 0:
        ax.scatter(
            gt_key_points[:, 0],
            gt_key_points[:, 1],
            gt_key_points[:, 2],
            c="purple",
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            label="GT Key Points",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    source_label = "prev3d" if mode == "f2f" else "key_points"
    num_inliers = np.sum(inliers) if inliers is not None else len(key_points)
    num_outliers = np.sum(~inliers) if inliers is not None else 0
    ax.set_title(
        f"Registration with GT Key Points ({mode.upper()}) - Frame {frame_number}\n"
        f"Source ({source_label}) ↔ Target (curr3d) | "
        f"Inliers: {num_inliers}, Outliers: {num_outliers} | "
        f"GT Key Points: {len(gt_key_points) if gt_key_points is not None else 0}"
    )

    ax.legend(loc="upper left", fontsize=10)

    # Set equal aspect ratio
    all_points = np.vstack([key_points, curr3d])
    if gt_key_points is not None and len(gt_key_points) > 0:
        all_points = np.vstack([all_points, gt_key_points])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])
    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions - manually set aspect
            # The equal limits above should help, but we can't guarantee perfect aspect
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot with GT key points to: {output_path}")
        plt.close()
    else:
        pass


def plot_3d_key_points_vs_gt(
    register_folder: str,
    object_number: int,
    frame_number: int,
    key_points: np.ndarray,
    gt_key_points: np.ndarray,
    output_path: Optional[str] = None,
    mode: str = "f2m",
):
    """Create a 3D plot showing only current key points vs GT key points.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3) - for f2m mode, or prev3d (N, 3) for f2f mode
        gt_key_points: Ground truth key points (M, 3)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m")
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load registered point cloud if available (for context)
    pcd_points = load_point_cloud(register_folder, object_number, frame_number)
    if pcd_points is not None:
        ax.scatter(
            pcd_points[:, 0],
            pcd_points[:, 1],
            pcd_points[:, 2],
            c="lightgray",
            alpha=0.2,
            s=1,
            label="Point Cloud (context)",
        )

    # Plot current key points (source)
    ax.scatter(
        key_points[:, 0],
        key_points[:, 1],
        key_points[:, 2],
        c="blue",
        s=60,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        label=f"Current Key Points ({'prev3d' if mode == 'f2f' else 'key_points'})",
    )

    # Plot GT key points
    if gt_key_points is not None and len(gt_key_points) > 0:
        ax.scatter(
            gt_key_points[:, 0],
            gt_key_points[:, 1],
            gt_key_points[:, 2],
            c="red",
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            label="GT Key Points",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"Key Points vs GT ({mode.upper()}) - Frame {frame_number}\n"
        f"Current: {len(key_points)} | GT: {len(gt_key_points) if gt_key_points is not None else 0}"
    )

    ax.legend(loc="upper left", fontsize=10)

    # Set equal aspect ratio
    all_points = key_points.copy()
    if gt_key_points is not None and len(gt_key_points) > 0:
        all_points = np.vstack([all_points, gt_key_points])
    if pcd_points is not None:
        all_points = np.vstack([all_points, pcd_points])

    if len(all_points) > 0:
        # Calculate ranges for each axis
        x_range = all_points[:, 0].max() - all_points[:, 0].min()
        y_range = all_points[:, 1].max() - all_points[:, 1].min()
        z_range = all_points[:, 2].max() - all_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0

        # Calculate midpoints
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        # Set equal limits for all axes
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set equal aspect ratio using set_box_aspect (if available, matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Key Points vs GT plot to: {output_path}")
        plt.close()
    else:
        pass


def get_gt_pose_for_frame(
    register_folder: str,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    ho3d_root: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Get ground truth pose for a given frame.

    Args:
        register_folder: Path to register folder
        frame_number: Frame number to get GT pose for
        results_dir: Results directory for new structure (optional)
        video_name: Video sequence name for new structure (optional)
        ho3d_root: HO3D dataset root directory (optional, will try to infer)

    Returns:
        GT pose matrix (4, 4) or None if not found
    """
    # Try to infer video directory from ho3d_root first (HO3D dataset location)
    # Then fall back to results_dir if needed
    video_dir = None

    # Priority 1: ho3d_root/video_name (actual HO3D dataset location)
    if ho3d_root and video_name:
        potential_video_dir = os.path.join(ho3d_root, video_name)
        # Verify it's a valid HO3D video directory (should have rgb/ subdirectory)
        if os.path.exists(potential_video_dir) and os.path.exists(
            os.path.join(potential_video_dir, "rgb")
        ):
            video_dir = potential_video_dir

    # Priority 2: results_dir/video_name (fallback, but usually doesn't have rgb/)
    if video_dir is None and results_dir and video_name:
        potential_video_dir = os.path.join(results_dir, video_name)
        # Only use if it has rgb/ subdirectory (unlikely but check anyway)
        if os.path.exists(potential_video_dir) and os.path.exists(
            os.path.join(potential_video_dir, "rgb")
        ):
            video_dir = potential_video_dir

    # If still not found and video_name is provided, try common locations
    if video_dir is None and video_name:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        potential_paths = [
            os.path.join(project_root, "data", "ho3d", video_name),
            os.path.join(project_root, "data", video_name),
            os.path.join(
                "/mnt",
                "9a72c439-d0a7-45e8-8d20-d7a235d02763",
                "DATASET",
                "HO3D",
                video_name,
            ),
        ]
        # Also try with ho3d_root if provided
        if ho3d_root:
            potential_paths.insert(0, os.path.join(ho3d_root, video_name))

        for path in potential_paths:
            # Verify it's a valid HO3D video directory (should have rgb/ subdirectory)
            if os.path.exists(path) and os.path.exists(os.path.join(path, "rgb")):
                video_dir = path
                break

    if video_dir is None or not os.path.exists(video_dir):
        print(
            f"Warning: Could not find video directory for {video_name}, cannot load GT pose"
        )
        if video_name:
            print(f"  Searched paths (must contain rgb/ subdirectory):")
            if ho3d_root:
                searched_path = os.path.join(ho3d_root, video_name)
                exists = os.path.exists(searched_path)
                has_rgb = (
                    os.path.exists(os.path.join(searched_path, "rgb"))
                    if exists
                    else False
                )
                print(f"    - {searched_path} (exists: {exists}, has rgb/: {has_rgb})")
            if results_dir:
                searched_path = os.path.join(results_dir, video_name)
                exists = os.path.exists(searched_path)
                has_rgb = (
                    os.path.exists(os.path.join(searched_path, "rgb"))
                    if exists
                    else False
                )
                print(f"    - {searched_path} (exists: {exists}, has rgb/: {has_rgb})")
        return None

    # Verify video_dir has rgb/ subdirectory before proceeding
    rgb_dir = os.path.join(video_dir, "rgb")
    if not os.path.exists(rgb_dir):
        print(
            f"Warning: Video directory {video_dir} does not contain rgb/ subdirectory, cannot load GT pose"
        )
        return None

    # Try to infer ho3d_root if not provided or if provided ho3d_root doesn't have models
    actual_ho3d_root = ho3d_root
    if ho3d_root is None or not os.path.exists(os.path.join(ho3d_root, "models")):
        potential_roots = []
        if ho3d_root:
            potential_roots.append(os.path.dirname(ho3d_root))
        if video_dir:
            potential_roots.extend(
                [
                    os.path.dirname(video_dir),
                    os.path.join(os.path.dirname(video_dir), ".."),
                ]
            )
        potential_roots.extend(
            [
                os.path.join(
                    "/mnt",
                    "9a72c439-d0a7-45e8-8d20-d7a235d02763",
                    "DATASET",
                    "HO3D",
                ),
            ]
        )
        for root in potential_roots:
            if (
                root
                and os.path.exists(root)
                and os.path.exists(os.path.join(root, "models"))
            ):
                actual_ho3d_root = root
                break

    # If we still don't have a valid ho3d_root, use the provided one anyway
    if actual_ho3d_root is None:
        actual_ho3d_root = ho3d_root

    # Create datareader
    reader = None
    if actual_ho3d_root and os.path.exists(actual_ho3d_root) and video_dir:
        try:
            print(
                f"Creating Ho3dReader with video_dir={video_dir}, ho3d_root={actual_ho3d_root}"
            )
            reader = Ho3dReader(video_dir, actual_ho3d_root)
        except Exception as e:
            print(f"Warning: Could not create Ho3dReader: {e}")
            import traceback

            traceback.print_exc()
            return None

    if reader is None:
        print("Warning: Could not create datareader, cannot load GT pose")
        print(f"  video_dir: {video_dir}")
        print(f"  actual_ho3d_root: {actual_ho3d_root}")
        print(
            f"  ho3d_root exists: {os.path.exists(actual_ho3d_root) if actual_ho3d_root else False}"
        )
        print(
            f"  video_dir exists: {os.path.exists(video_dir) if video_dir else False}"
        )
        return None

    # Get GT pose for current frame
    if frame_number >= len(reader):
        print(
            f"Frame {frame_number} out of range for datareader (length: {len(reader)})"
        )
        return None

    gt_pose = reader.get_gt_pose(frame_number)
    if gt_pose is None:
        print(f"Warning: GT pose not available for frame {frame_number}")
        return None

    # Transform GT pose by inverse of first frame's pose (assuming first frame is identity)
    # Get first frame GT pose
    first_frame_idx = 0
    gt_pose_first = reader.get_gt_pose(first_frame_idx)

    if gt_pose_first is None:
        # If first frame doesn't have GT, try to find first valid GT pose
        for i in range(len(reader)):
            pose = reader.get_gt_pose(i)
            if pose is not None:
                gt_pose_first = pose
                first_frame_idx = i
                break

    if gt_pose_first is not None:
        # Transform: gt_pose @ inverse_SE3(gt_pose_first) to make first frame identity
        # This gives transform from first frame to current frame
        if inverse_SE3 is not None:
            gt_pose_transformed = gt_pose @ inverse_SE3(gt_pose_first)
        else:
            # Fallback to numpy inverse if inverse_SE3 not available
            gt_pose_first_inv = np.linalg.inv(gt_pose_first)
            gt_pose_transformed = gt_pose @ gt_pose_first_inv
        return gt_pose_transformed
    else:
        # If no first frame pose available, return original (shouldn't happen in practice)
        print(
            f"Warning: Could not find first frame GT pose, returning untransformed pose"
        )
        return gt_pose


def plot_error_vs_uncertainty(
    key_points: np.ndarray,
    key_points_first_frame: Optional[np.ndarray],
    curr3d: np.ndarray,
    uncertainties: np.ndarray,
    reg_key_points_idx: Optional[np.ndarray],
    register_folder: str,
    frame_number: int,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    ho3d_root: Optional[str] = None,
    output_path: Optional[str] = None,
    mode: str = "f2m",
):
    """Plot error vs uncertainty after aligning source and target points using GT pose.

    Args:
        key_points: Source points (N, 3) - registration key points (already in current frame for plotting)
        key_points_first_frame: Untransformed source points in first-frame coordinates (used for GT alignment in f2m)
        curr3d: Target points (N, 3) - current frame 3D points
        uncertainties: Uncertainty values for each point (N,)
        reg_key_points_idx: Indices into the global keypoint map for each registration pair
        register_folder: Path to register folder
        frame_number: Frame number
        results_dir: Results directory (optional)
        video_name: Video sequence name (optional)
        ho3d_root: HO3D dataset root directory (optional)
        output_path: Optional path to save the figure
        mode: Registration mode ("f2f" or "f2m") for title
    """
    # Ensure arrays have the same length
    min_len = min(len(curr3d), len(uncertainties))
    curr3d = curr3d[:min_len]
    uncertainties = uncertainties[:min_len]
    if key_points is not None and len(key_points) >= min_len:
        key_points = key_points[:min_len]

    if min_len == 0:
        print("Warning: No points available for error vs uncertainty plot")
        return

    # If we don't have keypoint indices, fall back to simple source/target distance
    if reg_key_points_idx is None:
        print(
            "Warning: reg_key_points_idx is None, falling back to key_points vs curr3d error"
        )
        if key_points is None or len(key_points) < min_len:
            print("  key_points not available, skipping error vs uncertainty plot")
            return
        errors = np.linalg.norm(key_points - curr3d, axis=1)
    else:
        # Use GT keypoint map in the CURRENT frame and compare against registered targets.
        # This mirrors the notebook logic: measure how far each registered point is from
        # the GT keypoint it should correspond to.
        gt_map = get_gt_keypoint_map_for_frame(
            register_folder,
            frame_number,
            results_dir=results_dir,
            video_name=video_name,
            ho3d_root=ho3d_root,
        )
        if gt_map is None:
            print(
                "Warning: Could not load GT keypoint map, falling back to key_points vs curr3d error"
            )
            if key_points is None or len(key_points) < min_len:
                print("  key_points not available, skipping error vs uncertainty plot")
                return
            errors = np.linalg.norm(key_points - curr3d, axis=1)
        else:
            reg_idx_array = np.asarray(reg_key_points_idx, dtype=int)
            reg_idx_array = reg_idx_array[:min_len]
            if (
                np.any(reg_idx_array < 0)
                or np.any(reg_idx_array >= len(gt_map))
                or len(reg_idx_array) != min_len
            ):
                print(
                    "Warning: reg_key_points_idx out of bounds for GT map, "
                    "falling back to key_points vs curr3d error"
                )
                if key_points is None or len(key_points) < min_len:
                    print(
                        "  key_points not available, skipping error vs uncertainty plot"
                    )
                    return
                errors = np.linalg.norm(key_points - curr3d, axis=1)
            else:
                corresponding_gt = gt_map[reg_idx_array]
                errors = np.linalg.norm(corresponding_gt - curr3d, axis=1)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot: error vs uncertainty
    ax.scatter(
        uncertainties,
        errors,
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Error (m)", fontsize=12)
    ax.set_title(
        f"Point Error vs Uncertainty ({mode.upper()}) - Frame {frame_number} (N={min_len})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(errors)
    mean_uncertainty = np.mean(uncertainties)
    ax.axhline(
        mean_error,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Error: {mean_error:.4f}m",
    )
    ax.axvline(
        mean_uncertainty,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Uncertainty: {mean_uncertainty:.4f}",
    )
    ax.legend()

    plt.tight_layout()

    if isinstance(output_path, str) and len(output_path) > 0:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved error vs uncertainty plot to: {output_path}")
        plt.close()
    else:
        pass


def extract_video_info_from_path(
    register_folder: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract results_dir and video_name from register_folder path.

    Assumes structure: results_dir/video_name/register or results_dir/video_name/...

    Args:
        register_folder: Path to register folder

    Returns:
        Tuple of (results_dir, video_name) if detected, (None, None) otherwise
    """
    register_folder = os.path.abspath(register_folder)
    path_parts = register_folder.split(os.sep)

    # Look for common patterns:
    # 1. .../results/ho3d_single/MPM10/register -> results_dir=.../results/ho3d_single, video_name=MPM10
    # 2. .../results/ho3d_single/MPM10/.../register -> same
    # 3. .../results/.../video_name/register -> extract video_name and results_dir

    # Try to find "results" in the path
    results_idx = None
    for i, part in enumerate(path_parts):
        if part == "results":
            results_idx = i
            break

    if results_idx is not None and results_idx + 2 < len(path_parts):
        # Found results directory, check if next two levels exist
        # results_dir should be up to results_idx+1, video_name at results_idx+2
        potential_results_dir = os.sep.join(path_parts[: results_idx + 2])
        potential_video_name = path_parts[results_idx + 2]

        # Verify this structure makes sense (video_name directory exists)
        potential_video_path = os.path.join(potential_results_dir, potential_video_name)
        if os.path.exists(potential_video_path) and os.path.isdir(potential_video_path):
            return potential_results_dir, potential_video_name

    # Alternative: if register_folder is directly under a video_name directory
    # e.g., /path/to/video_name/register
    parent_dir = os.path.dirname(register_folder)
    if parent_dir and parent_dir != register_folder:
        parent_name = os.path.basename(parent_dir)
        grandparent_dir = os.path.dirname(parent_dir)

        # Check if parent looks like a video name (not a generic name like "register" or "debug")
        generic_names = {"register", "debug", "results", "meta_data", "output"}
        if parent_name not in generic_names and grandparent_dir:
            # Check if grandparent contains "results"
            if "results" in grandparent_dir:
                # Try to find results directory
                grandparent_parts = grandparent_dir.split(os.sep)
                for i, part in enumerate(grandparent_parts):
                    if part == "results" and i + 1 < len(grandparent_parts):
                        results_dir = os.sep.join(grandparent_parts[: i + 2])
                        if os.path.exists(results_dir):
                            return results_dir, parent_name

    return None, None


def main(args):
    """Main function."""
    # Construct register folder path
    if args.register_folder:
        register_folder = args.register_folder
    else:
        # Use default debug/register folder
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        register_folder = os.path.join(project_root, "debug", "register")

    if not os.path.exists(register_folder):
        print(f"Error: Register folder {register_folder} does not exist")
        return

    # Auto-detect video sequence info from register_folder path if not provided
    mode = getattr(args, "mode", "auto")
    meta_data_path = getattr(args, "meta_data_path", None)
    results_dir = getattr(args, "results_dir", None)
    video_name = getattr(args, "video_name", None)
    ho3d_root = getattr(args, "ho3d_root", None)

    # If results_dir or video_name not provided, try to extract from register_folder path
    if (results_dir is None or video_name is None) and register_folder:
        detected_results_dir, detected_video_name = extract_video_info_from_path(
            register_folder
        )
        if detected_results_dir and results_dir is None:
            results_dir = detected_results_dir
            print(f"Auto-detected results_dir from path: {results_dir}")
        if detected_video_name and video_name is None:
            video_name = detected_video_name
            print(f"Auto-detected video_name from path: {video_name}")
    result = load_registration_stats(
        register_folder,
        args.object_number,
        args.frame_number,
        mode=mode,
        meta_data_path_override=meta_data_path,
        results_dir=results_dir,
        video_name=video_name,
    )

    if result is None:
        print("Failed to load registration statistics")
        return

    (
        residuals,
        inliers,
        uncertainties,
        key_points,
        curr3d,
        keyframe_ids,
        reg_key_points_idx,
        detected_mode,
        key_points_first_frame,
    ) = result
    # key_points and curr3d are currently unused but kept for future use
    # _ = (
    #     key_points,
    #     curr3d,
    #     keyframe_ids,
    #     reg_key_points_idx,
    # )  # keyframe_ids used in plot function

    # Determine output path
    output_path = None
    if args.output:
        output_path = args.output
    elif args.save:
        output_dir = (
            os.path.dirname(register_folder)
            if args.register_folder
            else os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "debug",
            )
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"reg_stats_obj_{args.object_number}_frame_{args.frame_number}.png",
        )

    # Plot statistics
    stats_output_path = output_path
    if output_path:
        # If output path is specified, create separate paths for stats and 3D plots
        base_path = os.path.splitext(output_path)[0]
        stats_output_path = f"{base_path}_stats.png"
        plot_3d_output_path = f"{base_path}_3d.png"
        plot_3d_uncertainty_output_path = f"{base_path}_3d_uncertainty.png"
        plot_3d_residuals_output_path = f"{base_path}_3d_residuals.png"
        plot_3d_points_only_output_path = f"{base_path}_3d_points_only.png"
        plot_3d_gt_output_path = f"{base_path}_3d_gt.png"
        plot_3d_kp_vs_gt_output_path = f"{base_path}_3d_kp_vs_gt.png"
        plot_error_vs_uncertainty_output_path = f"{base_path}_error_vs_uncertainty.png"
    else:
        plot_3d_output_path = None
        plot_3d_uncertainty_output_path = None
        plot_3d_residuals_output_path = None
        plot_3d_points_only_output_path = None
        plot_3d_gt_output_path = None
        plot_3d_kp_vs_gt_output_path = None
        plot_error_vs_uncertainty_output_path = None

    # Create statistics plot (don't show yet)
    if len(residuals) > 0:
        plot_registration_stats(
            residuals,
            inliers,
            uncertainties,
            keyframe_ids,
            args.frame_number,
            reg_key_points_idx=reg_key_points_idx,
            output_path=stats_output_path,
            mode=detected_mode,
            key_points=key_points,
            curr3d=curr3d,
            register_folder=register_folder,
            results_dir=results_dir,
            video_name=video_name,
            ho3d_root=ho3d_root,
        )
    else:
        print("Warning: Skipping registration stats plot due to empty residuals")

    # Load data for 3D visualization
    if key_points is not None and curr3d is not None:
        all_key_data = load_all_key_points_with_frame_ids(
            register_folder,
            args.object_number,
            args.frame_number,
            results_dir=results_dir,
            video_name=video_name,
        )
        if all_key_data is not None:
            all_key_points, all_key_point_frame_ids = all_key_data

            # Compute registration pair keyframe IDs from RAW (unfiltered) frame id list
            pair_keyframe_ids = None
            if reg_key_points_idx is not None:
                raw_ids = load_all_key_point_frame_ids_raw(
                    register_folder,
                    args.frame_number,
                    results_dir=results_dir,
                    video_name=video_name,
                )
                if raw_ids is not None:
                    reg_idx_array = np.asarray(reg_key_points_idx, dtype=int)
                    if np.all(reg_idx_array >= 0) and np.all(
                        reg_idx_array < len(raw_ids)
                    ):
                        pair_keyframe_ids = raw_ids[reg_idx_array]
                    else:
                        print(
                            f"Warning: reg_key_points_idx out of bounds for 3D coloring. raw_ids length: {len(raw_ids)}, "
                            f"reg_idx min: {reg_idx_array.min()}, max: {reg_idx_array.max()}"
                        )
                else:
                    print(
                        "Warning: Could not load raw key point frame ids for 3D coloring"
                    )

            plot_3d_correspondences(
                register_folder,
                args.object_number,
                args.frame_number,
                key_points,
                curr3d,
                all_key_points,
                all_key_point_frame_ids,
                pair_keyframe_ids,
                inliers,
                plot_3d_output_path,
                mode=detected_mode,
            )

            # Create uncertainty-colored 3D visualization
            plot_3d_correspondences_uncertainty(
                register_folder,
                args.object_number,
                args.frame_number,
                key_points,
                curr3d,
                all_key_points,
                all_key_point_frame_ids,
                uncertainties,
                inliers,
                plot_3d_uncertainty_output_path,
                mode=detected_mode,
                reg_key_points_idx=reg_key_points_idx,
            )

            # Create residual-colored 3D visualization
            if len(residuals) > 0:
                plot_3d_correspondences_residuals(
                    register_folder,
                    args.object_number,
                    args.frame_number,
                    key_points,
                    curr3d,
                    all_key_points,
                    all_key_point_frame_ids,
                    residuals,
                    inliers,
                    plot_3d_residuals_output_path,
                    mode=detected_mode,
                    reg_key_points_idx=reg_key_points_idx,
                )

            # Create registration points only visualization
            plot_3d_registration_points_only(
                register_folder,
                args.object_number,
                args.frame_number,
                key_points,
                curr3d,
                inliers,
                plot_3d_points_only_output_path,
                mode=detected_mode,
            )

            # Load GT key points and create visualization
            gt_key_points = load_gt_key_points(
                register_folder,
                args.object_number,
                args.frame_number,
                results_dir=results_dir,
                video_name=video_name,
                ho3d_root=ho3d_root,
            )
            if gt_key_points is not None and len(gt_key_points) > 0:
                plot_3d_with_gt_key_points(
                    register_folder,
                    args.object_number,
                    args.frame_number,
                    key_points,
                    curr3d,
                    gt_key_points,
                    inliers,
                    plot_3d_gt_output_path,
                    mode=detected_mode,
                )

                # Plot key points vs GT key points (new function)
                plot_3d_key_points_vs_gt(
                    register_folder,
                    args.object_number,
                    args.frame_number,
                    key_points,
                    gt_key_points,
                    plot_3d_kp_vs_gt_output_path,
                    mode=detected_mode,
                )
            else:
                print("Warning: Could not load GT key points for visualization")
        else:
            print("Warning: Could not load all key points for 3D visualization")
    else:
        print("Warning: key_points or curr3d not available for 3D visualization")
        # Still try to create registration points only plot if we have the data
        if key_points is not None and curr3d is not None:
            plot_3d_registration_points_only(
                register_folder,
                args.object_number,
                args.frame_number,
                key_points,
                curr3d,
                inliers,
                plot_3d_points_only_output_path if output_path else None,
                mode=detected_mode,
            )

            # Try to load GT key points even if all_key_points failed
            gt_key_points = load_gt_key_points(
                register_folder,
                args.object_number,
                args.frame_number,
                results_dir=results_dir,
                video_name=video_name,
                ho3d_root=ho3d_root,
            )
            if gt_key_points is not None and len(gt_key_points) > 0:
                plot_3d_with_gt_key_points(
                    register_folder,
                    args.object_number,
                    args.frame_number,
                    key_points,
                    curr3d,
                    gt_key_points,
                    inliers,
                    plot_3d_gt_output_path if output_path else None,
                    mode=detected_mode,
                )

                # Plot key points vs GT key points (new function)
                plot_3d_key_points_vs_gt(
                    register_folder,
                    args.object_number,
                    args.frame_number,
                    key_points,
                    gt_key_points,
                    plot_3d_kp_vs_gt_output_path if output_path else None,
                    mode=detected_mode,
                )

    # Plot error vs uncertainty (using GT pose to align points)
    # This can be done regardless of whether we have all_key_points
    if (
        key_points is not None
        and curr3d is not None
        and uncertainties is not None
        and len(uncertainties) > 0
    ):
        plot_error_vs_uncertainty(
            key_points,
            key_points_first_frame,
            curr3d,
            uncertainties,
            reg_key_points_idx,
            register_folder,
            args.frame_number,
            results_dir=results_dir,
            video_name=video_name,
            ho3d_root=ho3d_root,
            output_path=plot_error_vs_uncertainty_output_path if output_path else None,
            mode=detected_mode,
        )
    elif uncertainties is None or len(uncertainties) == 0:
        print("Warning: Uncertainties not available for error vs uncertainty plot")
    elif key_points is None or curr3d is None:
        print(
            "Warning: key_points or curr3d not available for error vs uncertainty plot"
        )

    # Show all figures at once if not saving
    if not output_path:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot registration statistics (inlier, uncertainty, residual) for each point in registration"
    )

    parser.add_argument(
        "--object_number",
        "-o",
        type=int,
        required=True,
        help="Object number to visualize (e.g., 0, 1, 2...)",
    )

    parser.add_argument(
        "--frame_number",
        "-f",
        type=int,
        required=True,
        help="Frame number to plot statistics for",
    )

    parser.add_argument(
        "--register_folder",
        "-i",
        type=str,
        default=None,
        help="Path to register folder (optional, defaults to project_root/debug/register)",
    )

    parser.add_argument(
        "--meta_data_path",
        "-d",
        type=str,
        default=None,
        help="Direct path to meta_data.npz file (optional, will auto-search if not provided)",
    )

    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default=None,
        help="Results directory (e.g., /path/to/results/ho3d_single). Used with video_name to find meta_data in new structure.",
    )

    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        default=None,
        help="Video sequence name (e.g., MPM10). Used with results_dir to find meta_data in new structure.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (optional, if not specified, will show interactively)",
    )

    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save plot to default location in debug folder",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="auto",
        choices=["auto", "f2f", "f2m"],
        help="Registration mode: 'auto' to auto-detect, 'f2f' for frame-to-frame, 'f2m' for frame-to-map (default: auto)",
    )

    parser.add_argument(
        "--ho3d_root",
        type=str,
        default="/home/justin/data/HO3D_V3/evaluation",
        help="HO3D dataset root directory (optional, will try to infer from video path)",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
