import numpy as np
from scipy.spatial import cKDTree
import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from point2pose.utils.transform import to_homo

# Set non-interactive backend for headless environments
# This allows plots to be saved without a display
# Users can override by setting MPLBACKEND environment variable before import
try:
    if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
except (RuntimeError, ImportError):
    pass  # If backend is already set, continue


def add_err(pred, gt, model_pts):
    ### Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
    #
    # NVIDIA CORPORATION and its licensors retain all intellectual property
    # and proprietary rights in and to this software, related documentation
    # and any modifications thereto.  Any use, reproduction, disclosure or
    # distribution of this software and related documentation without an express
    # license agreement from NVIDIA CORPORATION is strictly prohibited.
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e


def adi_err(pred, gt, model_pts):
    # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
    #
    # NVIDIA CORPORATION and its licensors retain all intellectual property
    # and proprietary rights in and to this software, related documentation
    # and any modifications thereto.  Any use, reproduction, disclosure or
    # distribution of this software and related documentation without an express
    # license agreement from NVIDIA CORPORATION is strictly prohibited.
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e


def compute_auc(rec, max_val=0.1):
    """https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/2df96b720e8e499b9f0d5fcebfbae2bcfa51ab19/eval_ycb.py#L45"""
    if len(rec) == 0:
        return 0
    rec = np.sort(np.array(rec))
    n = len(rec)
    prec = np.arange(1, n + 1) / float(n)
    rec = rec.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(rec < max_val)[0]
    rec = rec[index]
    prec = prec[index]

    mrec = [0, *list(rec), max_val]
    mpre = [0, *list(prec), prec[-1]]

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    i = np.where(mrec[1:] != mrec[0 : len(mrec) - 1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val
    return ap


def plot_evaluation_results(
    add_s_errs,
    add_errs,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(15, 10),
):
    """
    Plot comprehensive evaluation results for ADD-S and ADD errors.

    Args:
        add_s_errs: Array of ADD-S errors (in meters)
        add_errs: Array of ADD errors (in meters)
        video_name: Name of the video/sequence (optional)
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
        figsize: Figure size tuple
    """
    add_s_errs = np.array(add_s_errs)
    add_errs = np.array(add_errs)

    # Convert to centimeters for display
    add_s_errs_cm = add_s_errs * 100
    add_errs_cm = add_errs * 100

    # Compute statistics
    add_s_mean = add_s_errs_cm.mean()
    add_s_std = add_s_errs_cm.std()
    add_s_median = np.median(add_s_errs_cm)
    add_s_min = add_s_errs_cm.min()
    add_s_max = add_s_errs_cm.max()

    add_mean = add_errs_cm.mean()
    add_std = add_errs_cm.std()
    add_median = np.median(add_errs_cm)
    add_min = add_errs_cm.min()
    add_max = add_errs_cm.max()

    # Compute AUC
    add_s_auc = compute_auc(add_s_errs) * 100
    add_auc = compute_auc(add_errs) * 100

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Error distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        add_s_errs_cm,
        bins=30,
        alpha=0.7,
        label="ADD-S",
        color="steelblue",
        edgecolor="black",
    )
    ax1.hist(
        add_errs_cm, bins=30, alpha=0.7, label="ADD", color="coral", edgecolor="black"
    )
    ax1.axvline(
        add_s_mean,
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=f"ADD-S mean: {add_s_mean:.2f} cm",
    )
    ax1.axvline(
        add_mean,
        color="coral",
        linestyle="--",
        linewidth=2,
        label=f"ADD mean: {add_mean:.2f} cm",
    )
    ax1.set_xlabel("Error (cm)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Error Distribution", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error over frames (time series)
    ax2 = fig.add_subplot(gs[0, 1])
    frames = np.arange(len(add_s_errs_cm))
    ax2.plot(
        frames,
        add_s_errs_cm,
        label="ADD-S",
        color="steelblue",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.plot(frames, add_errs_cm, label="ADD", color="coral", linewidth=1.5, alpha=0.8)
    ax2.axhline(
        add_s_mean,
        color="steelblue",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="ADD-S mean",
    )
    ax2.axhline(
        add_mean,
        color="coral",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="ADD mean",
    )
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Error (cm)", fontsize=12)
    ax2.set_title("Error Over Frames", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot comparison
    ax3 = fig.add_subplot(gs[1, 0])
    box_data = [add_s_errs_cm, add_errs_cm]
    bp = ax3.boxplot(
        box_data,
        labels=["ADD-S", "ADD"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    bp["boxes"][1].set_facecolor("lightcoral")
    ax3.set_ylabel("Error (cm)", fontsize=12)
    ax3.set_title("Error Distribution Comparison", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: AUC curve
    ax4 = fig.add_subplot(gs[1, 1])
    max_val = 0.1  # 10cm threshold
    thresholds = np.linspace(0, max_val * 100, 100)  # in cm

    # Compute recall for ADD-S (percentage of frames with error <= threshold)
    add_s_sorted = np.sort(
        add_s_errs_cm / 100
    )  # convert back to meters for AUC computation
    add_s_recall = []
    for thresh in thresholds / 100:  # convert to meters
        recall = np.sum(add_s_sorted <= thresh) / len(add_s_sorted)
        add_s_recall.append(recall)

    # Compute recall for ADD (percentage of frames with error <= threshold)
    add_sorted = np.sort(add_errs_cm / 100)  # convert back to meters
    add_recall = []
    for thresh in thresholds / 100:  # convert to meters
        recall = np.sum(add_sorted <= thresh) / len(add_sorted)
        add_recall.append(recall)

    ax4.plot(
        thresholds,
        add_s_recall,
        label=f"ADD-S (AUC: {add_s_auc:.2f}%)",
        color="steelblue",
        linewidth=2,
    )
    ax4.plot(
        thresholds,
        add_recall,
        label=f"ADD (AUC: {add_auc:.2f}%)",
        color="coral",
        linewidth=2,
    )
    ax4.axvline(
        5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="5cm threshold"
    )
    ax4.axvline(
        10, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="10cm threshold"
    )
    ax4.set_xlabel("Threshold (cm)", fontsize=12)
    ax4.set_ylabel("Recall", fontsize=12)
    ax4.set_title("AUC Curve (Recall vs Threshold)", fontsize=14, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_val * 100)
    ax4.set_ylim(0, 1.05)

    # Plot 5: Cumulative distribution
    ax5 = fig.add_subplot(gs[2, 0])
    add_s_sorted_cm = np.sort(add_s_errs_cm)
    add_sorted_cm = np.sort(add_errs_cm)
    y_add_s = np.arange(1, len(add_s_sorted_cm) + 1) / len(add_s_sorted_cm)
    y_add = np.arange(1, len(add_sorted_cm) + 1) / len(add_sorted_cm)
    ax5.plot(
        add_s_sorted_cm, y_add_s * 100, label="ADD-S", color="steelblue", linewidth=2
    )
    ax5.plot(add_sorted_cm, y_add * 100, label="ADD", color="coral", linewidth=2)
    ax5.axvline(
        5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="5cm threshold"
    )
    ax5.axvline(
        10, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="10cm threshold"
    )
    ax5.set_xlabel("Error (cm)", fontsize=12)
    ax5.set_ylabel("Cumulative Percentage (%)", fontsize=12)
    ax5.set_title("Cumulative Distribution Function", fontsize=14, fontweight="bold")
    ax5.legend(loc="lower right", fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)

    # Plot 6: Statistics table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Create statistics table
    stats_data = [
        ["Metric", "ADD-S", "ADD"],
        ["Mean (cm)", f"{add_s_mean:.2f}", f"{add_mean:.2f}"],
        ["Std (cm)", f"{add_s_std:.2f}", f"{add_std:.2f}"],
        ["Median (cm)", f"{add_s_median:.2f}", f"{add_median:.2f}"],
        ["Min (cm)", f"{add_s_min:.2f}", f"{add_min:.2f}"],
        ["Max (cm)", f"{add_s_max:.2f}", f"{add_max:.2f}"],
        ["AUC (%)", f"{add_s_auc:.2f}", f"{add_auc:.2f}"],
        ["Num Frames", f"{len(add_s_errs_cm)}", f"{len(add_errs_cm)}"],
    ]

    table = ax6.table(
        cellText=stats_data[1:],
        colLabels=stats_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style the header
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style alternating rows
    for i in range(1, len(stats_data)):
        for j in range(len(stats_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax6.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)

    # Overall title
    title_str = "Evaluation Results"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "evaluation_results.png"
        if video_name:
            plot_filename = f"evaluation_results_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved evaluation plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def plot_error_over_time(
    add_s_errs,
    add_errs,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(12, 6),
):
    """
    Plot error over time (frames) with additional analysis.

    Args:
        add_s_errs: Array of ADD-S errors (in meters)
        add_errs: Array of ADD errors (in meters)
        video_name: Name of the video/sequence (optional)
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
        figsize: Figure size tuple
    """
    add_s_errs = np.array(add_s_errs) * 100  # Convert to cm
    add_errs = np.array(add_errs) * 100

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    frames = np.arange(len(add_s_errs))

    # Plot 1: Error over time with moving average
    ax1 = axes[0]
    window_size = max(10, len(add_s_errs) // 20)  # Adaptive window size

    if len(add_s_errs) > window_size:
        # Moving average
        add_s_ma = np.convolve(
            add_s_errs, np.ones(window_size) / window_size, mode="same"
        )
        add_ma = np.convolve(add_errs, np.ones(window_size) / window_size, mode="same")

        ax1.plot(
            frames,
            add_s_errs,
            label="ADD-S",
            color="steelblue",
            alpha=0.3,
            linewidth=0.5,
        )
        ax1.plot(frames, add_errs, label="ADD", color="coral", alpha=0.3, linewidth=0.5)
        ax1.plot(
            frames,
            add_s_ma,
            label=f"ADD-S (MA {window_size})",
            color="steelblue",
            linewidth=2,
        )
        ax1.plot(
            frames, add_ma, label=f"ADD (MA {window_size})", color="coral", linewidth=2
        )
    else:
        ax1.plot(frames, add_s_errs, label="ADD-S", color="steelblue", linewidth=1.5)
        ax1.plot(frames, add_errs, label="ADD", color="coral", linewidth=1.5)

    ax1.axhline(
        add_s_errs.mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"ADD-S mean: {add_s_errs.mean():.2f} cm",
    )
    ax1.axhline(
        add_errs.mean(),
        color="coral",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"ADD mean: {add_errs.mean():.2f} cm",
    )
    ax1.set_ylabel("Error (cm)", fontsize=12)
    ax1.set_title(
        "Error Over Frames (with Moving Average)", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error difference
    ax2 = axes[1]
    error_diff = add_errs - add_s_errs
    ax2.plot(frames, error_diff, color="purple", linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax2.axhline(
        error_diff.mean(),
        color="purple",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Mean diff: {error_diff.mean():.2f} cm",
    )
    ax2.fill_between(
        frames,
        0,
        error_diff,
        where=(error_diff > 0),
        alpha=0.3,
        color="red",
        label="ADD > ADD-S",
    )
    ax2.fill_between(
        frames,
        0,
        error_diff,
        where=(error_diff < 0),
        alpha=0.3,
        color="green",
        label="ADD < ADD-S",
    )
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Error Difference (cm)", fontsize=12)
    ax2.set_title("ADD - ADD-S Error Difference", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    title_str = "Error Analysis Over Time"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "error_over_time.png"
        if video_name:
            plot_filename = f"error_over_time_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved error over time plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def compute_translation_error(pred_pose, gt_pose):
    """
    Compute translation error between predicted and ground truth poses.

    Args:
        pred_pose: (4, 4) predicted pose matrix
        gt_pose: (4, 4) ground truth pose matrix

    Returns:
        translation_error: Euclidean distance in meters
        translation_error_components: (3,) array of x, y, z errors in meters
    """
    pred_trans = pred_pose[:3, 3]
    gt_trans = gt_pose[:3, 3]

    translation_error = np.linalg.norm(pred_trans - gt_trans)
    translation_error_components = pred_trans - gt_trans

    return translation_error, translation_error_components


def compute_rotation_error(pred_pose, gt_pose):
    """
    Compute rotation error between predicted and ground truth poses.

    Args:
        pred_pose: (4, 4) predicted pose matrix
        gt_pose: (4, 4) ground truth pose matrix

    Returns:
        rotation_error_deg: Rotation error in degrees (geodesic distance on SO(3))
        rotation_error_rad: Rotation error in radians
        axis_angle_error: (3,) axis-angle representation of rotation error
    """
    pred_rot = pred_pose[:3, :3]
    gt_rot = gt_pose[:3, :3]

    # Relative rotation: R_error = R_pred^T @ R_gt
    # This gives the rotation needed to go from pred to gt
    rel_rot = pred_rot.T @ gt_rot

    # Convert rotation matrix to axis-angle representation
    # Trace of rotation matrix: tr(R) = 1 + 2*cos(θ)
    trace = np.trace(rel_rot)
    # Clamp to valid range [-1, 3] for numerical stability
    trace = np.clip(trace, -1.0, 3.0)

    # Angle from trace: cos(θ) = (tr(R) - 1) / 2
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    # Handle the case when angle is very small (near identity)
    if angle_rad < 1e-6:
        axis_angle = np.zeros(3)
        rotation_error_rad = 0.0
    else:
        # Axis from skew-symmetric part: (R - R^T) / (2*sin(θ))
        skew_sym = (rel_rot - rel_rot.T) / 2.0
        sin_angle = np.sin(angle_rad)
        if sin_angle > 1e-6:
            axis = (
                np.array([skew_sym[2, 1], skew_sym[0, 2], skew_sym[1, 0]]) / sin_angle
            )
            axis_angle = axis * angle_rad
        else:
            # Fallback for very small angles
            axis_angle = np.zeros(3)

        rotation_error_rad = angle_rad

    rotation_error_deg = np.rad2deg(rotation_error_rad)

    return rotation_error_deg, rotation_error_rad, axis_angle


def plot_pose_errors(
    pred_poses,
    gt_poses,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(16, 12),
):
    """
    Plot comprehensive pose error analysis including translation and rotation errors.

    Args:
        pred_poses: (N, 4, 4) array of predicted pose matrices
        gt_poses: (N, 4, 4) array of ground truth pose matrices
        video_name: Name of the video/sequence (optional)
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
        figsize: Figure size tuple
    """
    pred_poses = np.array(pred_poses)
    gt_poses = np.array(gt_poses)

    n_frames = len(pred_poses)
    frames = np.arange(n_frames)

    # Compute errors for all frames
    trans_errors = []
    trans_errors_x = []
    trans_errors_y = []
    trans_errors_z = []
    rot_errors_deg = []
    rot_errors_rad = []
    axis_angle_errors_x = []
    axis_angle_errors_y = []
    axis_angle_errors_z = []

    for i in range(n_frames):
        trans_err, trans_err_components = compute_translation_error(
            pred_poses[i], gt_poses[i]
        )
        rot_err_deg, rot_err_rad, axis_angle = compute_rotation_error(
            pred_poses[i], gt_poses[i]
        )

        trans_errors.append(trans_err)
        trans_errors_x.append(trans_err_components[0])
        trans_errors_y.append(trans_err_components[1])
        trans_errors_z.append(trans_err_components[2])
        rot_errors_deg.append(rot_err_deg)
        rot_errors_rad.append(rot_err_rad)
        axis_angle_errors_x.append(axis_angle[0])
        axis_angle_errors_y.append(axis_angle[1])
        axis_angle_errors_z.append(axis_angle[2])

    # Convert to numpy arrays and to centimeters for translation
    trans_errors = np.array(trans_errors) * 100  # cm
    trans_errors_x = np.array(trans_errors_x) * 100  # cm
    trans_errors_y = np.array(trans_errors_y) * 100  # cm
    trans_errors_z = np.array(trans_errors_z) * 100  # cm
    rot_errors_deg = np.array(rot_errors_deg)
    rot_errors_rad = np.array(rot_errors_rad)
    axis_angle_errors_x = np.array(axis_angle_errors_x)
    axis_angle_errors_y = np.array(axis_angle_errors_y)
    axis_angle_errors_z = np.array(axis_angle_errors_z)

    # Compute statistics
    trans_mean = trans_errors.mean()
    trans_std = trans_errors.std()
    trans_median = np.median(trans_errors)

    rot_mean = rot_errors_deg.mean()
    rot_std = rot_errors_deg.std()
    rot_median = np.median(rot_errors_deg)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Plot 1: Translation error magnitude over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        frames,
        trans_errors,
        color="steelblue",
        linewidth=1.5,
        alpha=0.8,
        label="Translation Error",
    )
    ax1.axhline(
        trans_mean,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {trans_mean:.2f} cm",
    )
    ax1.fill_between(
        frames,
        trans_mean - trans_std,
        trans_mean + trans_std,
        alpha=0.2,
        color="steelblue",
        label=f"±1 Std: {trans_std:.2f} cm",
    )
    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("Translation Error (cm)", fontsize=12)
    ax1.set_title("Translation Error Magnitude", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Translation error components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(frames, trans_errors_x, label="X", color="red", linewidth=1.5, alpha=0.8)
    ax2.plot(frames, trans_errors_y, label="Y", color="green", linewidth=1.5, alpha=0.8)
    ax2.plot(frames, trans_errors_z, label="Z", color="blue", linewidth=1.5, alpha=0.8)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Translation Error (cm)", fontsize=12)
    ax2.set_title("Translation Error Components", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rotation error over time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        frames,
        rot_errors_deg,
        color="coral",
        linewidth=1.5,
        alpha=0.8,
        label="Rotation Error",
    )
    ax3.axhline(
        rot_mean,
        color="coral",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {rot_mean:.2f}°",
    )
    ax3.fill_between(
        frames,
        rot_mean - rot_std,
        rot_mean + rot_std,
        alpha=0.2,
        color="coral",
        label=f"±1 Std: {rot_std:.2f}°",
    )
    ax3.set_xlabel("Frame", fontsize=12)
    ax3.set_ylabel("Rotation Error (degrees)", fontsize=12)
    ax3.set_title("Rotation Error (Geodesic Distance)", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Rotation error axis-angle components
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        frames,
        axis_angle_errors_x,
        label="X-axis",
        color="red",
        linewidth=1.5,
        alpha=0.8,
    )
    ax4.plot(
        frames,
        axis_angle_errors_y,
        label="Y-axis",
        color="green",
        linewidth=1.5,
        alpha=0.8,
    )
    ax4.plot(
        frames,
        axis_angle_errors_z,
        label="Z-axis",
        color="blue",
        linewidth=1.5,
        alpha=0.8,
    )
    ax4.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax4.set_xlabel("Frame", fontsize=12)
    ax4.set_ylabel("Axis-Angle Error (rad)", fontsize=12)
    ax4.set_title(
        "Rotation Error Axis-Angle Components", fontsize=14, fontweight="bold"
    )
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Error distribution comparison
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(
        trans_errors,
        bins=30,
        alpha=0.7,
        label="Translation",
        color="steelblue",
        edgecolor="black",
    )
    ax5_twin = ax5.twinx()
    ax5_twin.hist(
        rot_errors_deg,
        bins=30,
        alpha=0.7,
        label="Rotation",
        color="coral",
        edgecolor="black",
    )
    ax5.axvline(
        trans_mean,
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=f"Trans mean: {trans_mean:.2f} cm",
    )
    ax5_twin.axvline(
        rot_mean,
        color="coral",
        linestyle="--",
        linewidth=2,
        label=f"Rot mean: {rot_mean:.2f}°",
    )
    ax5.set_xlabel("Translation Error (cm)", fontsize=12, color="steelblue")
    ax5.set_ylabel("Frequency (Translation)", fontsize=12, color="steelblue")
    ax5_twin.set_ylabel("Frequency (Rotation)", fontsize=12, color="coral")
    ax5.set_title("Error Distribution", fontsize=14, fontweight="bold")
    ax5.tick_params(axis="x", labelcolor="steelblue")
    ax5.tick_params(axis="y", labelcolor="steelblue")
    ax5_twin.tick_params(axis="y", labelcolor="coral")
    ax5.grid(True, alpha=0.3)
    # Combine legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # Plot 6: Statistics table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Compute component statistics
    trans_x_mean = np.abs(trans_errors_x).mean()
    trans_y_mean = np.abs(trans_errors_y).mean()
    trans_z_mean = np.abs(trans_errors_z).mean()

    axis_x_mean = np.abs(axis_angle_errors_x).mean()
    axis_y_mean = np.abs(axis_angle_errors_y).mean()
    axis_z_mean = np.abs(axis_angle_errors_z).mean()

    stats_data = [
        ["Metric", "Translation", "Rotation"],
        ["Mean", f"{trans_mean:.2f} cm", f"{rot_mean:.2f}°"],
        ["Std", f"{trans_std:.2f} cm", f"{rot_std:.2f}°"],
        ["Median", f"{trans_median:.2f} cm", f"{rot_median:.2f}°"],
        ["Max", f"{trans_errors.max():.2f} cm", f"{rot_errors_deg.max():.2f}°"],
        ["Min", f"{trans_errors.min():.2f} cm", f"{rot_errors_deg.min():.2f}°"],
        ["", "", ""],
        ["Component Means", "", ""],
        ["|X|", f"{trans_x_mean:.2f} cm", f"{np.rad2deg(axis_x_mean):.2f}°"],
        ["|Y|", f"{trans_y_mean:.2f} cm", f"{np.rad2deg(axis_y_mean):.2f}°"],
        ["|Z|", f"{trans_z_mean:.2f} cm", f"{np.rad2deg(axis_z_mean):.2f}°"],
        ["", "", ""],
        ["Num Frames", f"{n_frames}", f"{n_frames}"],
    ]

    table = ax6.table(
        cellText=stats_data[1:],
        colLabels=stats_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style the header
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style section headers
    for i in range(1, len(stats_data)):
        if stats_data[i][0] in ["Component Means"]:
            for j in range(len(stats_data[0])):
                table[(i - 1, j)].set_facecolor("#e0e0e0")
                table[(i - 1, j)].set_text_props(weight="bold")
        elif stats_data[i][0] == "":
            for j in range(len(stats_data[0])):
                table[(i - 1, j)].set_facecolor("#ffffff")

    # Style alternating rows
    row_idx = 0
    for i in range(1, len(stats_data)):
        if stats_data[i][0] != "":
            if row_idx % 2 == 1:
                for j in range(len(stats_data[0])):
                    table[(i - 1, j)].set_facecolor("#f0f0f0")
            row_idx += 1

    ax6.set_title("Pose Error Statistics", fontsize=14, fontweight="bold", pad=20)

    # Overall title
    title_str = "Pose Error Analysis"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "pose_errors.png"
        if video_name:
            plot_filename = f"pose_errors_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved pose error plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def plot_pose_error_comparison(
    pred_poses,
    gt_poses,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    figsize=(14, 8),
):
    """
    Plot side-by-side comparison of translation and rotation errors with moving averages.

    Args:
        pred_poses: (N, 4, 4) array of predicted pose matrices
        gt_poses: (N, 4, 4) array of ground truth pose matrices
        video_name: Name of the video/sequence (optional)
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
        figsize: Figure size tuple
    """
    pred_poses = np.array(pred_poses)
    gt_poses = np.array(gt_poses)

    n_frames = len(pred_poses)
    frames = np.arange(n_frames)

    # Compute errors
    trans_errors = []
    rot_errors_deg = []

    for i in range(n_frames):
        trans_err, _ = compute_translation_error(pred_poses[i], gt_poses[i])
        rot_err_deg, _, _ = compute_rotation_error(pred_poses[i], gt_poses[i])
        trans_errors.append(trans_err)
        rot_errors_deg.append(rot_err_deg)

    trans_errors = np.array(trans_errors) * 100  # cm
    rot_errors_deg = np.array(rot_errors_deg)

    # Compute moving averages
    window_size = max(10, n_frames // 20)
    if n_frames > window_size:
        trans_ma = np.convolve(
            trans_errors, np.ones(window_size) / window_size, mode="same"
        )
        rot_ma = np.convolve(
            rot_errors_deg, np.ones(window_size) / window_size, mode="same"
        )
    else:
        trans_ma = trans_errors
        rot_ma = rot_errors_deg

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Translation error with moving average
    ax1 = axes[0]
    ax1.plot(
        frames,
        trans_errors,
        color="steelblue",
        alpha=0.3,
        linewidth=0.5,
        label="Translation Error",
    )
    ax1.plot(
        frames,
        trans_ma,
        color="steelblue",
        linewidth=2,
        label=f"Moving Average (window={window_size})",
    )
    ax1.axhline(
        trans_errors.mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {trans_errors.mean():.2f} cm",
    )
    ax1.fill_between(
        frames,
        trans_errors.mean() - trans_errors.std(),
        trans_errors.mean() + trans_errors.std(),
        alpha=0.2,
        color="steelblue",
        label=f"±1 Std: {trans_errors.std():.2f} cm",
    )
    ax1.set_ylabel("Translation Error (cm)", fontsize=12)
    ax1.set_title("Translation Error Over Frames", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rotation error with moving average
    ax2 = axes[1]
    ax2.plot(
        frames,
        rot_errors_deg,
        color="coral",
        alpha=0.3,
        linewidth=0.5,
        label="Rotation Error",
    )
    ax2.plot(
        frames,
        rot_ma,
        color="coral",
        linewidth=2,
        label=f"Moving Average (window={window_size})",
    )
    ax2.axhline(
        rot_errors_deg.mean(),
        color="coral",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {rot_errors_deg.mean():.2f}°",
    )
    ax2.fill_between(
        frames,
        rot_errors_deg.mean() - rot_errors_deg.std(),
        rot_errors_deg.mean() + rot_errors_deg.std(),
        alpha=0.2,
        color="coral",
        label=f"±1 Std: {rot_errors_deg.std():.2f}°",
    )
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Rotation Error (degrees)", fontsize=12)
    ax2.set_title("Rotation Error Over Frames", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    title_str = "Pose Error Comparison"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "pose_error_comparison.png"
        if video_name:
            plot_filename = f"pose_error_comparison_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved pose error comparison plot to: {plot_path}")

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def plot_recall_vs_threshold(
    add_s_errs,
    add_errs,
    video_name=None,
    output_dir=None,
    save_plots=True,
    show_plots=False,
    max_threshold=10.0,
    figsize=(12, 8),
):
    """
    Plot recall vs threshold curves for ADD-S and ADD errors.

    Recall is defined as the percentage of frames with error <= threshold.

    Args:
        add_s_errs: Array of ADD-S errors (in meters)
        add_errs: Array of ADD errors (in meters)
        video_name: Name of the video/sequence (optional)
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
        max_threshold: Maximum threshold in centimeters (default: 10.0 cm)
        figsize: Figure size tuple
    """
    add_s_errs = np.array(add_s_errs)
    add_errs = np.array(add_errs)

    # Convert to centimeters
    add_s_errs_cm = add_s_errs * 100
    add_errs_cm = add_errs * 100

    # Generate thresholds in cm
    thresholds = np.linspace(0, max_threshold, 200)

    # Compute recall for ADD-S (percentage of frames with error <= threshold)
    add_s_sorted = np.sort(add_s_errs_cm)
    add_s_recall = []
    for thresh in thresholds:
        recall = np.sum(add_s_sorted <= thresh) / len(add_s_sorted)
        add_s_recall.append(recall * 100)  # Convert to percentage

    # Compute recall for ADD (percentage of frames with error <= threshold)
    add_sorted = np.sort(add_errs_cm)
    add_recall = []
    for thresh in thresholds:
        recall = np.sum(add_sorted <= thresh) / len(add_sorted)
        add_recall.append(recall * 100)  # Convert to percentage

    # Compute recall at common thresholds
    common_thresholds = [1.0, 2.0, 5.0, 10.0]  # in cm
    add_s_recall_at_thresh = {}
    add_recall_at_thresh = {}

    for thresh in common_thresholds:
        if thresh <= max_threshold:
            add_s_recall_at_thresh[thresh] = (
                np.sum(add_s_sorted <= thresh) / len(add_s_sorted) * 100
            )
            add_recall_at_thresh[thresh] = (
                np.sum(add_sorted <= thresh) / len(add_sorted) * 100
            )

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Recall vs threshold curves
    ax1 = axes[0]
    ax1.plot(
        thresholds,
        add_s_recall,
        label="ADD-S",
        color="steelblue",
        linewidth=2.5,
        alpha=0.9,
    )
    ax1.plot(
        thresholds,
        add_recall,
        label="ADD",
        color="coral",
        linewidth=2.5,
        alpha=0.9,
    )

    # Add markers at common thresholds
    for thresh in common_thresholds:
        if thresh <= max_threshold:
            # Find the index in thresholds array
            idx = np.argmin(np.abs(thresholds - thresh))
            if thresh == common_thresholds[0]:
                ax1.plot(
                    thresh,
                    add_s_recall[idx],
                    marker="o",
                    markersize=8,
                    color="steelblue",
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    label=f"ADD-S @ {thresh}cm",
                )
                ax1.plot(
                    thresh,
                    add_recall[idx],
                    marker="s",
                    markersize=8,
                    color="coral",
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    label=f"ADD @ {thresh}cm",
                )
            else:
                ax1.plot(
                    thresh,
                    add_s_recall[idx],
                    marker="o",
                    markersize=8,
                    color="steelblue",
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )
                ax1.plot(
                    thresh,
                    add_recall[idx],
                    marker="s",
                    markersize=8,
                    color="coral",
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )
            # Add vertical line
            ax1.axvline(
                thresh,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.4,
            )
            # Add text annotation
            ax1.text(
                thresh,
                max(add_s_recall[idx], add_recall[idx]) + 2,
                f"{thresh}cm",
                ha="center",
                fontsize=9,
                alpha=0.7,
            )

    ax1.set_xlabel("Threshold (cm)", fontsize=12)
    ax1.set_ylabel("Recall (%)", fontsize=12)
    ax1.set_title("Recall vs Threshold", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_threshold)
    ax1.set_ylim(0, 105)
    ax1.set_yticks(np.arange(0, 101, 10))

    # Plot 2: Recall difference (ADD - ADD-S)
    ax2 = axes[1]
    recall_diff = np.array(add_recall) - np.array(add_s_recall)
    ax2.plot(
        thresholds,
        recall_diff,
        color="purple",
        linewidth=2,
        alpha=0.8,
        label="ADD - ADD-S",
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.fill_between(
        thresholds,
        0,
        recall_diff,
        where=(recall_diff > 0),
        alpha=0.3,
        color="green",
        label="ADD > ADD-S",
    )
    ax2.fill_between(
        thresholds,
        0,
        recall_diff,
        where=(recall_diff < 0),
        alpha=0.3,
        color="red",
        label="ADD < ADD-S",
    )

    # Add markers at common thresholds
    for thresh in common_thresholds:
        if thresh <= max_threshold:
            idx = np.argmin(np.abs(thresholds - thresh))
            ax2.plot(
                thresh,
                recall_diff[idx],
                marker="o",
                markersize=6,
                color="purple",
                markeredgecolor="white",
                markeredgewidth=1,
            )
            ax2.axvline(
                thresh,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.4,
            )

    ax2.set_xlabel("Threshold (cm)", fontsize=12)
    ax2.set_ylabel("Recall Difference (%)", fontsize=12)
    ax2.set_title("Recall Difference (ADD - ADD-S)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_threshold)

    title_str = "Recall vs Threshold Analysis"
    if video_name:
        title_str += f" - {video_name}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save plot
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = "recall_vs_threshold.png"
        if video_name:
            plot_filename = f"recall_vs_threshold_{video_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved recall vs threshold plot to: {plot_path}")

    # Print recall statistics at common thresholds
    print("\nRecall Statistics:")
    print(f"{'Threshold (cm)':<15} {'ADD-S Recall (%)':<20} {'ADD Recall (%)':<20}")
    print("-" * 55)
    for thresh in common_thresholds:
        if thresh <= max_threshold:
            print(
                f"{thresh:<15.1f} {add_s_recall_at_thresh[thresh]:<20.2f} {add_recall_at_thresh[thresh]:<20.2f}"
            )

    # Show plot
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig
