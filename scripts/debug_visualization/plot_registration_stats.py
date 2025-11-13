#!/usr/bin/env python3
"""
Plot registration statistics (inlier, uncertainty, residual) for each point in registration at a given frame.
Similar to visualize_register_pcd.py in terms of data access.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def find_meata_data_path(register_folder: str) -> Optional[str]:
    """Find meata_data.npz file in expected locations."""
    meata_data_paths = [
        os.path.join(register_folder, "meata_data.npz"),
        os.path.join(os.path.dirname(register_folder), "meata_data.npz"),
        os.path.join(
            os.path.dirname(os.path.dirname(register_folder)),
            "debug",
            "pipeline",
            "meta_data",
            "meata_data.npz",
        ),
    ]

    for path in meata_data_paths:
        if os.path.exists(path):
            return path

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
    register_folder: str, object_number: int, frame_number: int, mode: str = "auto"
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
    ]
]:
    """Load registration statistics for a given frame.

    Args:
        register_folder: Path to register folder
        object_number: Object number (currently only supports object 0, kept for compatibility)
        frame_number: Frame number to extract stats for
        mode: Registration mode ("f2f", "f2m", or "auto" to auto-detect)

    Returns:
        Tuple of (residuals, inliers, uncertainties, key_points/prev3d, curr3d, keyframe_ids, reg_key_points_idx, mode) if data is found, None otherwise.
        All arrays have the same length (number of registration points).
        mode is either "f2f" or "f2m".
    """
    _ = object_number  # Currently unused, kept for compatibility
    meata_data_path = find_meata_data_path(register_folder)
    if meata_data_path is None:
        print("No meata_data.npz found in expected locations")
        return None

    try:
        data = np.load(meata_data_path, allow_pickle=True)
        print(f"Loaded meata_data.npz from: {meata_data_path}")

        # Find frame index
        if "frame_id" not in data:
            print("No frame_id field found in meata_data.npz")
            return None

        frame_ids = data["frame_id"]
        frame_idx = None

        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"Frame {frame_number} not found in meata_data.npz")
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
        if detected_mode == "f2f":
            key_points = (
                reg_prev3d_list[frame_idx] if frame_idx < len(reg_prev3d_list) else None
            )
            print("Loaded f2f mode: using reg_prev3d as source points")
        else:  # f2m mode
            key_points = (
                reg_key_points_list[frame_idx]
                if frame_idx < len(reg_key_points_list)
                else None
            )
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
            print("Could not extract reg_residuals")
            return None

        if inliers is None or len(inliers) == 0:
            print("Could not extract reg_inliers, assuming all are inliers")
            inliers = np.ones(len(residuals), dtype=bool)

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
        print(
            f"  - Inliers: {np.sum(inliers)} ({100*np.sum(inliers)/len(inliers):.1f}%)"
        )
        print(
            f"  - Outliers: {np.sum(~inliers)} ({100*np.sum(~inliers)/len(inliers):.1f}%)"
        )
        if uncertainties is not None:
            print(
                f"  - Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]"
            )
        print(f"  - Residual range: [{residuals.min():.4f}, {residuals.max():.4f}]")
        print(f"  - Mean residual (inliers): {np.mean(residuals[inliers]):.4f}")
        if np.sum(~inliers) > 0:
            print(f"  - Mean residual (outliers): {np.mean(residuals[~inliers]):.4f}")

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

    # Calculate number of plots needed
    num_plots = 4  # base plots: inlier/outlier, residuals hist (all), residuals hist (split), residuals sorted
    if has_unc:
        num_plots += 2  # uncertainty hist, residual vs uncertainty (inlier/outlier)
    if has_unc and has_kfids:
        num_plots += 1  # residual vs uncertainty (colored by keyframe)
    if reg_keyframe_ids is not None:
        num_plots += 1  # residuals vs frame ID

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
    register_folder: str, frame_number: int
) -> Optional[np.ndarray]:
    """Load ALL key point frame IDs (unfiltered, length = obj_key_points_lengths/3).

    Returns a 1-D int array aligned to original key point indexing so reg_key_points_idx
    can be used directly without bounds issues from filtering.
    Uses unpack_ragged following notebook pattern.
    """
    meata_data_path = find_meata_data_path(register_folder)
    if meata_data_path is None:
        print("No meata_data.npz found for loading raw key point frame ids")
        return None

    try:
        data = np.load(meata_data_path, allow_pickle=True)
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
    register_folder: str, object_number: int, frame_number: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load all key points and their frame IDs from meata_data.npz.

    Uses unpack_ragged following notebook pattern.

    Returns:
        Tuple of (key_points, frame_ids) if found, None otherwise.
        key_points: (N, 3) array of 3D points
        frame_ids: (N,) array of frame IDs for each point
    """
    _ = object_number  # Currently unused, kept for compatibility
    meata_data_path = find_meata_data_path(register_folder)
    if meata_data_path is None:
        print("No meata_data.npz found for loading all key points")
        return None

    try:
        data = np.load(meata_data_path, allow_pickle=True)

        if "frame_id" not in data:
            print("Error: No frame_id in meata_data.npz")
            return None

        frame_ids = data["frame_id"]
        frame_idx = None
        for i, fid in enumerate(frame_ids):
            if fid == frame_number:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"Error: Frame {frame_number} not found in meata_data.npz")
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
    if len(key_points) == len(curr3d):
        if inliers is not None:
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
    if len(key_points) == len(curr3d):
        if inliers is not None:
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

    # Load registration statistics
    mode = getattr(args, "mode", "auto")
    result = load_registration_stats(
        register_folder, args.object_number, args.frame_number, mode=mode
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
    else:
        plot_3d_output_path = None
        plot_3d_uncertainty_output_path = None
        plot_3d_residuals_output_path = None
        plot_3d_points_only_output_path = None

    # Create statistics plot (don't show yet)
    plot_registration_stats(
        residuals,
        inliers,
        uncertainties,
        keyframe_ids,
        args.frame_number,
        reg_key_points_idx=reg_key_points_idx,
        output_path=stats_output_path,
        mode=detected_mode,
    )

    # Load data for 3D visualization
    if key_points is not None and curr3d is not None:
        all_key_data = load_all_key_points_with_frame_ids(
            register_folder, args.object_number, args.frame_number
        )
        if all_key_data is not None:
            all_key_points, all_key_point_frame_ids = all_key_data

            # Compute registration pair keyframe IDs from RAW (unfiltered) frame id list
            pair_keyframe_ids = None
            if reg_key_points_idx is not None:
                raw_ids = load_all_key_point_frame_ids_raw(
                    register_folder, args.frame_number
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

    parsed_args = parser.parse_args()

    main(parsed_args)
