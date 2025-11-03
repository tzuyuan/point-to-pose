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


def load_registration_stats(
    register_folder: str, object_number: int, frame_number: int
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]
]:
    """Load registration statistics for a given frame.

    Args:
        register_folder: Path to register folder
        object_number: Object number (currently only supports object 0, kept for compatibility)
        frame_number: Frame number to extract stats for

    Returns:
        Tuple of (residuals, inliers, uncertainties, key_points, curr3d, keyframe_ids) if data is found, None otherwise.
        All arrays have the same length (number of registration points).
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

        # Extract residuals
        residuals = None
        if (
            "reg_residuals_data" in data
            and "reg_residuals_offsets" in data
            and "reg_residuals_lengths" in data
        ):
            offsets = data["reg_residuals_offsets"]
            lengths = data["reg_residuals_lengths"]
            data_array = data["reg_residuals_data"]

            if frame_idx < len(offsets):
                start_idx = offsets[frame_idx]
                end_idx = start_idx + lengths[frame_idx]
                residuals = data_array[start_idx:end_idx].astype(float)

        if residuals is None:
            print("Could not extract reg_residuals")
            return None

        # Extract inliers
        inliers = None
        if (
            "reg_inliers_data" in data
            and "reg_inliers_offsets" in data
            and "reg_inliers_lengths" in data
        ):
            offsets = data["reg_inliers_offsets"]
            lengths = data["reg_inliers_lengths"]
            data_array = data["reg_inliers_data"]

            if frame_idx < len(offsets):
                start_idx = offsets[frame_idx]
                end_idx = start_idx + lengths[frame_idx]
                inliers = data_array[start_idx:end_idx].astype(bool)

        if inliers is None:
            print("Could not extract reg_inliers, assuming all are inliers")
            inliers = np.ones(len(residuals), dtype=bool)

        # Extract uncertainties using reg_key_points_idx
        uncertainties = None
        if (
            "reg_key_points_idx_data" in data
            and "reg_key_points_idx_offsets" in data
            and "reg_key_points_idx_lengths" in data
            and "uncertainties_data" in data
            and "uncertainties_offsets" in data
            and "uncertainties_lengths" in data
        ):
            # Get indices into uncertainty array
            idx_offsets = data["reg_key_points_idx_offsets"]
            idx_lengths = data["reg_key_points_idx_lengths"]
            idx_data_array = data["reg_key_points_idx_data"]

            # Get uncertainty array for this frame
            unc_offsets = data["uncertainties_offsets"]
            unc_lengths = data["uncertainties_lengths"]
            unc_data_array = data["uncertainties_data"]

            if frame_idx < len(idx_offsets) and frame_idx < len(unc_offsets):
                # Get registration point indices
                idx_start = idx_offsets[frame_idx]
                idx_end = idx_start + idx_lengths[frame_idx]
                reg_idx = idx_data_array[idx_start:idx_end].astype(int)

                # Get uncertainty array for this frame
                unc_start = unc_offsets[frame_idx]
                unc_end = unc_start + unc_lengths[frame_idx]
                frame_uncertainties = unc_data_array[unc_start:unc_end]

                # Index into frame uncertainties
                if len(reg_idx) == len(residuals):
                    uncertainties = frame_uncertainties[reg_idx].astype(float)
                else:
                    print(
                        f"Warning: reg_key_points_idx length ({len(reg_idx)}) != residuals length ({len(residuals)})"
                    )

        if uncertainties is None:
            print(
                "Warning: Could not extract uncertainties, will not plot uncertainty stats"
            )

        # Extract keyframe IDs for registration points
        keyframe_ids = None
        if (
            "reg_key_points_lengths" in data
            and "obj_key_point_frames_offsets" in data
            and "obj_key_point_frames_data" in data
        ):
            if frame_idx < len(data["reg_key_points_lengths"]):
                key_points_len = data["reg_key_points_lengths"][frame_idx]
                n_kps = key_points_len // 3
                if frame_idx < len(
                    data["obj_key_point_frames_offsets"]
                ) and n_kps == len(residuals):
                    frame_ids_offset = data["obj_key_point_frames_offsets"][frame_idx]
                    keyframe_ids = data["obj_key_point_frames_data"][
                        frame_ids_offset : frame_ids_offset + n_kps
                    ].astype(int)
                    if len(keyframe_ids) != len(residuals):
                        print(
                            f"Warning: keyframe_ids length ({len(keyframe_ids)}) != residuals length ({len(residuals)})"
                        )
                        keyframe_ids = None

        if keyframe_ids is None:
            print("Warning: Could not extract keyframe_ids")

        # Extract key_points and curr3d for reference (optional)
        key_points = None
        if (
            "reg_key_points_data" in data
            and "reg_key_points_offsets" in data
            and "reg_key_points_lengths" in data
        ):
            offsets = data["reg_key_points_offsets"]
            lengths = data["reg_key_points_lengths"]
            data_array = data["reg_key_points_data"]

            if frame_idx < len(offsets):
                start_idx = offsets[frame_idx]
                end_idx = start_idx + lengths[frame_idx]
                key_points_flat = data_array[start_idx:end_idx]
                if len(key_points_flat) % 3 == 0:
                    key_points = key_points_flat.reshape(-1, 3)

        curr3d = None
        if (
            "reg_curr3d_data" in data
            and "reg_curr3d_offsets" in data
            and "reg_curr3d_lengths" in data
        ):
            offsets = data["reg_curr3d_offsets"]
            lengths = data["reg_curr3d_lengths"]
            data_array = data["reg_curr3d_data"]

            if frame_idx < len(offsets):
                start_idx = offsets[frame_idx]
                end_idx = start_idx + lengths[frame_idx]
                curr3d_flat = data_array[start_idx:end_idx]
                if len(curr3d_flat) % 3 == 0:
                    curr3d = curr3d_flat.reshape(-1, 3)

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

        if keyframe_ids is not None and len(keyframe_ids) != len(residuals):
            min_len = min(len(residuals), len(keyframe_ids))
            residuals = residuals[:min_len]
            inliers = inliers[:min_len]
            if uncertainties is not None:
                uncertainties = uncertainties[:min_len]
            keyframe_ids = keyframe_ids[:min_len]
            print(f"Warning: Truncated keyframe_ids to match length: {min_len}")

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
        # The registration transformation (obj_pose) transforms key_points from first frame to current frame.
        # curr3d are the target points in the current frame and remain as-is for visualization.

        return residuals, inliers, uncertainties, key_points, curr3d, keyframe_ids

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
    output_path: Optional[str] = None,
):
    """Create plots showing registration statistics.

    Args:
        residuals: Array of residuals for each registration point
        inliers: Boolean array indicating inliers
        uncertainties: Optional array of uncertainties for each registration point
        keyframe_ids: Optional array of keyframe IDs for each registration point
        frame_number: Frame number for title
        output_path: Optional path to save the figure
    """
    num_points = len(residuals)
    num_inliers = np.sum(inliers)
    num_outliers = num_points - num_inliers

    # Determine layout based on available data
    has_unc = uncertainties is not None
    has_kfids = keyframe_ids is not None

    # Calculate number of plots needed
    num_plots = 4  # base plots: inlier/outlier, residuals hist (all), residuals hist (split), residuals sorted
    if has_unc:
        num_plots += 2  # uncertainty hist, residual vs uncertainty (inlier/outlier)
    if has_unc and has_kfids:
        num_plots += 1  # residual vs uncertainty (colored by keyframe)

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
        f"Registration Statistics - Frame {frame_number} (N={num_points})", fontsize=14
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
        if keyframe_ids is not None:
            ax = axes[6]

            # Get unique keyframe IDs and create color map
            unique_frames = np.unique(keyframe_ids)

            # Use matplotlib colormap similar to visualize_register_pcd.py
            try:
                import matplotlib

                cmap_get = getattr(
                    getattr(matplotlib, "colormaps", matplotlib.cm), "get_cmap"
                )
                tab10_colors = cmap_get("tab10")

                # Map frame IDs to color indices (avoiding red which is index 3)
                frame_to_color = {}
                for i, fid in enumerate(sorted(unique_frames)):
                    color_idx = i if i < 3 else i + 1  # Skip red (index 3)
                    frame_to_color[fid] = tab10_colors(color_idx % 10)[:3]
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
                    fid: palette[i % len(palette)]
                    for i, fid in enumerate(sorted(unique_frames))
                }

            # Create color array for scatter plot
            point_colors = np.array([frame_to_color[fid] for fid in keyframe_ids])

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
            legend_frames = sorted(unique_frames)[:10]
            legend_elements_kf = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=frame_to_color[fid],
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

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if output_path:
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


def load_all_key_points_with_frame_ids(
    register_folder: str, object_number: int, frame_number: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load all key points and their frame IDs from meata_data.npz.

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

        # Extract key points
        required_keys = [
            "obj_key_points_data",
            "obj_key_points_offsets",
            "obj_key_points_lengths",
        ]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            print(f"Error: Missing keys in meata_data.npz: {missing_keys}")
            return None

        offsets = data["obj_key_points_offsets"]
        lengths = data["obj_key_points_lengths"]
        flat = data["obj_key_points_data"]

        if frame_idx >= len(offsets):
            print(f"Error: frame_idx ({frame_idx}) >= len(offsets) ({len(offsets)})")
            return None

        start_idx = int(offsets[frame_idx])
        end_idx = start_idx + int(lengths[frame_idx])
        pts_flat = flat[start_idx:end_idx]

        if len(pts_flat) % 3 != 0:
            print(f"Error: obj_key_points length {len(pts_flat)} not divisible by 3")
            return None

        pts = pts_flat.reshape(-1, 3)

        # Filter NaN points
        if np.any(np.isnan(pts)):
            valid_mask = ~np.isnan(pts).any(axis=1)
            pts = pts[valid_mask]

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

        # Extract frame IDs for key points
        if (
            "obj_key_point_frames_offsets" in data
            and "obj_key_point_frames_data" in data
        ):
            frame_ids_offset = data["obj_key_point_frames_offsets"][frame_idx]
            n_kps = len(pts)
            key_point_frame_ids = data["obj_key_point_frames_data"][
                frame_ids_offset : frame_ids_offset + n_kps
            ].astype(int)

            # Ensure same length
            min_len = min(len(pts), len(key_point_frame_ids))
            pts = pts[:min_len]
            key_point_frame_ids = key_point_frame_ids[:min_len]
        else:
            print("Warning: obj_key_point_frames not found, cannot color by frame ID")
            return None

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
    inliers: Optional[np.ndarray],
    output_path: Optional[str] = None,
):
    """Create a 3D plot showing registered points, all key points (colored by frame ID), and correspondences.

    Args:
        register_folder: Path to register folder
        object_number: Object number
        frame_number: Frame number
        key_points: Registration key points (N, 3)
        curr3d: Current 3D points (N, 3)
        all_key_points: All key points (M, 3)
        all_key_point_frame_ids: Frame IDs for all key points (M,)
        inliers: Optional inlier mask (N,)
        output_path: Optional path to save the figure
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
            color_idx = i if i < 3 else i + 1  # Skip red (index 3)
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
    if len(key_points) == len(curr3d):
        # Transform key_points if needed (they should already be transformed in load_registration_stats)
        # But let's use them as-is since they're already transformed
        for i in range(len(key_points)):
            # Only draw lines for inliers if inliers is provided
            if inliers is None or inliers[i]:
                ax.plot(
                    [key_points[i, 0], curr3d[i, 0]],
                    [key_points[i, 1], curr3d[i, 1]],
                    [key_points[i, 2], curr3d[i, 2]],
                    "k-",
                    alpha=0.3,
                    linewidth=0.5,
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D Registration Visualization - Frame {frame_number}\n"
        f"Key Points Colored by Keyframe ID, Lines Show Correspondences"
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
        max_range = (
            np.array(
                [
                    all_points[:, 0].max() - all_points[:, 0].min(),
                    all_points[:, 1].max() - all_points[:, 1].min(),
                    all_points[:, 2].max() - all_points[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot to: {output_path}")
        plt.close()  # Close figure to allow next plot to be created
    else:
        # Don't show yet - will show at the end with all figures
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
    result = load_registration_stats(
        register_folder, args.object_number, args.frame_number
    )

    if result is None:
        print("Failed to load registration statistics")
        return

    residuals, inliers, uncertainties, key_points, curr3d, keyframe_ids = result
    # key_points and curr3d are currently unused but kept for future use
    _ = (key_points, curr3d, keyframe_ids)  # keyframe_ids used in plot function

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
        # If output path is specified, create separate paths for stats and 3D plot
        base_path = os.path.splitext(output_path)[0]
        stats_output_path = f"{base_path}_stats.png"
        plot_3d_output_path = f"{base_path}_3d.png"
    else:
        plot_3d_output_path = None

    # Create statistics plot (don't show yet)
    plot_registration_stats(
        residuals,
        inliers,
        uncertainties,
        keyframe_ids,
        args.frame_number,
        stats_output_path,
    )

    # Load data for 3D visualization
    if key_points is not None and curr3d is not None:
        all_key_data = load_all_key_points_with_frame_ids(
            register_folder, args.object_number, args.frame_number
        )
        if all_key_data is not None:
            all_key_points, all_key_point_frame_ids = all_key_data
            plot_3d_correspondences(
                register_folder,
                args.object_number,
                args.frame_number,
                key_points,
                curr3d,
                all_key_points,
                all_key_point_frame_ids,
                inliers,
                plot_3d_output_path,
            )
        else:
            print("Warning: Could not load all key points for 3D visualization")
    else:
        print("Warning: key_points or curr3d not available for 3D visualization")

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

    parsed_args = parser.parse_args()

    main(parsed_args)
