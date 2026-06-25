"""Zoomed MPM13 timeseries around failure regions.

Reuses MPM13_per_frame_e2d_counts.csv (e2d threshold counts) and
per_frame_MPM13.csv (tx_grad, ADD), and renders zoomed views.
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


def render_zoom(df_main: pd.DataFrame, df_counts: pd.DataFrame,
                fmin: int, fmax: int, label: str, out_path: Path,
                show_title: bool = True):
    main = df_main[(df_main["frame"] >= fmin) & (df_main["frame"] <= fmax)].copy()
    counts = df_counts[(df_counts["frame"] >= fmin) & (df_counts["frame"] <= fmax)].copy()

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    color_e2d = "#C44E52"      # red
    color_npts = "#1F77B4"     # blue
    color_sobel = "#5A8A4C"    # green
    color_add = "#8B3A3A"      # darker red

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.2, w_pad=0.08)

    # Panel 1: mean e2d (left, red) + #well-tracked points (right, blue).
    lw = 1.6
    ax1.plot(counts["frame"], counts["mean_e2d_px"],
             color=color_e2d, linewidth=lw)
    ax1.set_ylabel("2D tracking error (px)", color=color_e2d)
    ax1.tick_params(axis="y", which="both", labelcolor=color_e2d, color=color_e2d)
    ax1.grid(alpha=0.3, axis="x")
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color(color_e2d)

    ax1b = ax1.twinx()
    if "n_e2d_lt_10" in counts.columns:
        ax1b.plot(counts["frame"], counts["n_e2d_lt_10"],
                  color=color_npts, linewidth=lw)
    ax1b.set_ylabel("# points with e < 10 px", color=color_npts)
    ax1b.tick_params(axis="y", labelcolor=color_npts)
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_color(color_npts)

    # Panel 2: ADD (left, red) + mean Sobel (right, green).
    ax2.plot(main["frame"], main["add_cm"],
             color=color_add, linewidth=lw)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                alpha=0.4, zorder=1)
    ax2.set_ylabel("ADD error (cm)", color=color_add)
    ax2.tick_params(axis="y", which="both", labelcolor=color_add, color=color_add)
    ax2.grid(alpha=0.3, axis="x")
    for spine in ("top",):
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color(color_add)
    # "frame" placed inline at the left of the tick row instead of centered.
    ax2.set_xlabel("")
    ax2.annotate("frame", xy=(-0.005, -0.01), xycoords="axes fraction",
                 ha="right", va="top", fontsize=18)

    ax2b = ax2.twinx()
    ax2b.plot(main["frame"], main["tx_grad"],
              color=color_sobel, linewidth=lw)
    ax2b.set_ylabel("Sobel magnitude", color=color_sobel)
    ax2b.tick_params(axis="y", which="both", labelcolor=color_sobel, color=color_sobel)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_color(color_sobel)

    if show_title:
        fig.suptitle("Texture Analysis on HO3D MPM13", fontsize=22)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir",
                        default="/home/justin/results/eccv_point2pose/texture_dependency_analysis")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "ap_only"

    df_main = pd.read_csv(analysis_dir / "per_frame_MPM13.csv")
    df_counts = pd.read_csv(out_dir / "MPM13_per_frame_e2d_counts.csv")

    # Zoom 1: main failure window (frames ~300-800).
    render_zoom(df_main, df_counts, fmin=300, fmax=800, label="main",
                out_path=out_dir / "MPM13_zoom_main_failure.png")

    # Zoom 2: late failure window (frames ~1380-1500).
    render_zoom(df_main, df_counts, fmin=1380, fmax=1530, label="late",
                out_path=out_dir / "MPM13_zoom_late_failure.png")

    # Zoom 3: tight focus on the onset of the main failure (~frames 380-450).
    render_zoom(df_main, df_counts, fmin=370, fmax=470, label="onset",
                out_path=out_dir / "MPM13_zoom_main_onset.png",
                show_title=False)


if __name__ == "__main__":
    main()
