"""AP-series only: effect of texture on ADD / ADD-S AUC.

The AP* sequences in HO3D all use the same object (019_pitcher_base), so they
form a controlled comparison: same shape, same mesh, just different viewpoints
with different amounts of visible discriminative texture. Within this family:
AP12 is the textureless extreme (mostly the smooth white pitcher base in view);
AP14/AP13 see the textured label. This script renders AP-only figures.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir",
                        default="/home/justin/results/eccv_point2pose/texture_dependency_analysis")
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to <analysis-dir>/ap_only/")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "ap_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(analysis_dir / "summary.csv")
    summary = summary[summary["video"].str.startswith("AP")].copy()
    summary = summary.sort_values("tx_grad_mean").reset_index(drop=True)

    df_frame = pd.read_csv(analysis_dir / "per_frame_all.csv")
    df_frame = df_frame[df_frame["video"].str.startswith("AP")].copy()

    print("AP-series summary (sorted by tx_grad ascending = most→least textureless):")
    cols = ["video", "tx_grad_mean", "mean_n_pts_unc020_and_grad>50",
            "mean_e2d_px", "mean_add_cm", "mean_adi_cm", "add_auc", "adi_auc"]
    print(summary[cols].to_string(index=False))

    # FIG 1 — headline: bar chart of ADD AUC and ADD-S AUC, sorted by tx_grad.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(summary))
    w = 0.38
    bars1 = ax.bar(x - w/2, summary["add_auc"], width=w, color="#4C72B0", label="ADD AUC")
    bars2 = ax.bar(x + w/2, summary["adi_auc"], width=w, color="#DD8452", label="ADD-S AUC")
    for b, v in zip(bars1, summary["add_auc"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=8)
    for b, v in zip(bars2, summary["adi_auc"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    labels = [f"{r['video']}\ntx={r['tx_grad_mean']:.0f}" for _, r in summary.iterrows()]
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 105)
    ax.set_title(
        "AP-series (same object: 019_pitcher_base) — pose AUC vs texturelessness\n"
        "left → right: textureless → textured"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "AP_auc_bar.png", dpi=150)
    plt.close(fig)

    # FIG 2 — same idea but showing ADD-cm and ADD-S-cm.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars1 = ax.bar(x - w/2, summary["mean_add_cm"], width=w, color="#4C72B0", label="mean ADD")
    bars2 = ax.bar(x + w/2, summary["mean_adi_cm"], width=w, color="#DD8452", label="mean ADD-S")
    for b, v in zip(bars1, summary["mean_add_cm"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8)
    for b, v in zip(bars2, summary["mean_adi_cm"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean error (cm)")
    ax.set_title(
        "AP-series — pose error (cm) vs texturelessness (lower is better)"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "AP_err_bar.png", dpi=150)
    plt.close(fig)

    # FIG 3 — scatter: tx_grad vs ADD AUC and tx_grad vs mean e2d.
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    pt_color = "#4C72B0"

    # Per-video, per-panel label offsets to avoid overlap on tight clusters.
    label_offsets = {
        "add_auc": {
            "AP12": (8, 6),
            "AP10": (8, 6),
            "AP11": (-22, -16),
            "AP14": (8, 6),
            "AP13": (8, -14),
        },
        "mean_e2d_px": {
            "AP12": (8, 6),
            "AP10": (8, 6),
            "AP11": (8, 6),
            "AP14": (8, -12),
            "AP13": (8, 6),
        },
    }
    for ax, ycol, ylabel in [
        (axes[0], "mean_e2d_px", "Mean 2D tracking error (px)"),
        (axes[1], "add_auc", "ADD AUC"),
    ]:
        x_arr = summary["tx_grad_mean"].values
        y_arr = summary[ycol].values
        ax.scatter(x_arr, y_arr, s=70, color=pt_color, zorder=3)
        for _, r in summary.iterrows():
            xo, yo = label_offsets[ycol].get(r["video"], (8, 6))
            ax.annotate(
                r["video"],
                (r["tx_grad_mean"], r[ycol]),
                fontsize=11,
                fontweight="bold",
                color="black",
                xytext=(xo, yo), textcoords="offset points",
            )
        ax.set_xlabel("Mean Sobel magnitude (high means more texture)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[1].set_ylim(40, 100)
    e_min = summary["mean_e2d_px"].min(); e_max = summary["mean_e2d_px"].max()
    pad = (e_max - e_min) * 0.15
    axes[0].set_ylim(max(0, e_min - pad), e_max + pad)
    axes[0].set_title("2D point tracker error  ↓")
    axes[1].set_title("Pose tracking AUC  ↑")
    fig.tight_layout()
    fig.savefig(out_dir / "AP_scatter_tx_grad_vs_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # FIG 4 — count-based metric vs ADD AUC and ADD-S AUC.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ycol, ylabel in [
        (axes[0], "add_auc", "ADD AUC"),
        (axes[1], "adi_auc", "ADD-S AUC"),
    ]:
        sx = summary["mean_n_pts_unc020_and_grad>50"].values
        ax.scatter(sx, summary[ycol], s=60, color="#4C72B0", zorder=3)
        for _, r in summary.iterrows():
            color = "red" if r["video"] == "AP12" else "black"
            ax.annotate(r["video"], (r["mean_n_pts_unc020_and_grad>50"], r[ycol]),
                        fontsize=10, color=color, xytext=(5, 5),
                        textcoords="offset points")
        ax.set_xlabel("Mean #pts (unc<0.2 AND grad>50) per frame")
        ax.set_ylabel(ylabel)
        ax.set_ylim(40, 100)
        ax.grid(alpha=0.3)
        r, _ = pearsonr(sx, summary[ycol]); rho, _ = spearmanr(sx, summary[ycol])
        ax.text(0.05, 0.95, f"N={len(summary)}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.suptitle("AP-series: confident-textured points (count-based metric) vs pose AUC",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "AP_scatter_count_vs_auc.png", dpi=150)
    plt.close(fig)

    # FIG 5 — combined panel: x = tx_grad (sorted), bars for each AUC; line for e2d.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.bar(x - w/2, summary["add_auc"], width=w, color="#4C72B0", label="ADD AUC")
    bars2 = ax1.bar(x + w/2, summary["adi_auc"], width=w, color="#DD8452", label="ADD-S AUC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("AUC", color="black")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["mean_e2d_px"], color="#C44E52", marker="o",
             linewidth=2, label="mean e2d (px)")
    ax2.set_ylabel("mean e2d (px)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.legend(loc="upper right")
    ax1.set_title(
        "AP-series: texturelessness (left→right: textureless→textured)\n"
        "Bars: ADD/ADD-S AUC. Red line: 2D point-tracker error e2d."
    )
    fig.tight_layout()
    fig.savefig(out_dir / "AP_combined_auc_e2d.png", dpi=150)
    plt.close(fig)

    # FIG 6 — within-AP per-frame scatter e2d vs ADD, colored by sequence.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ycol, ylabel in [
        (axes[0], "add_cm", "ADD (cm)"),
        (axes[1], "adi_cm", "ADD-S (cm)"),
    ]:
        for v, sub in df_frame.groupby("video"):
            color = "red" if v == "AP12" else None
            ax.scatter(sub["mean_e2d_px"], sub[ycol], s=8, alpha=0.5, label=v, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("Mean per-frame 2D tracking error e2d (px)")
        ax.set_ylabel(ylabel)
        m = np.isfinite(df_frame["mean_e2d_px"]) & np.isfinite(df_frame[ycol])
        r, _ = pearsonr(df_frame.loc[m, "mean_e2d_px"], df_frame.loc[m, ycol])
        rho, _ = spearmanr(df_frame.loc[m, "mean_e2d_px"], df_frame.loc[m, ycol])
        ax.text(0.05, 0.95, f"pooled N={m.sum()}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("AP-series only: per-frame 2D tracking error vs pose error", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "AP_per_frame_e2d_vs_pose.png", dpi=150)
    plt.close(fig)

    # Save AP-only summary CSV.
    summary.to_csv(out_dir / "AP_summary.csv", index=False)

    # Print headline correlations.
    print(f"\nAP-only correlations (N={len(summary)}):")
    for xcol in ["tx_grad_mean", "mean_n_pts_unc020_and_grad>50", "mean_n_pts_grad>60"]:
        for ycol in ["add_auc", "adi_auc", "mean_e2d_px", "mean_add_cm"]:
            r, _ = pearsonr(summary[xcol], summary[ycol])
            rho, _ = spearmanr(summary[xcol], summary[ycol])
            print(f"  {xcol} vs {ycol}: Pearson={r:.3f}, Spearman={rho:.3f}")

    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
