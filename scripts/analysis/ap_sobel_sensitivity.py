"""AP-series Sobel kernel-size sensitivity + count-based scatter.

Re-runs the texturelessness metric on AP10-14 with Sobel ksize in {3, 5, 7}
and renders one AP-scatter figure per kernel size (file name encodes ksize).
Also renders the count-based scatter (mean #pts unc<0.2 AND grad>50) styled
to match the polished AP_scatter_tx_grad_vs_auc layout.

Plot style follows the locked-in convention:
  - left panel: 2D point tracker error (↓), right panel: Pose tracking AUC (↑)
  - uniform marker color, no connecting lines, no stats annotations
  - kernel size never written on the figure; only in the file name
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture_dependency import (  # noqa: E402
    DEFAULT_OUT_DIR, DEFAULT_DATA_ROOT,
    find_ho3d_paths, read_rgb, read_mask,
)

AP_VIDEOS = ["AP10", "AP11", "AP12", "AP13", "AP14"]
KSIZES = [3, 5, 7]
PT_COLOR = "#4C72B0"

LABEL_OFFSETS = {
    "left": {  # e2d
        "AP12": (8, 6), "AP10": (8, 6), "AP11": (8, 6),
        "AP14": (8, -12), "AP13": (8, 6),
    },
    "right": {  # AUC
        "AP12": (8, 6), "AP10": (8, 6), "AP11": (-22, -16),
        "AP14": (8, 6), "AP13": (8, -14),
    },
}


def compute_tx_grad_per_frame(video: str, data_root: Path, ksize: int) -> pd.Series:
    paths = find_ho3d_paths(data_root, video)
    rgb_files = sorted(paths.rgb_dir.glob("*.jpg"))
    H, W = read_rgb(rgb_files[0]).shape[:2]
    out = []
    for f in rgb_files:
        rgb = read_rgb(f)
        mask_idx = int(f.stem)
        mask = read_mask(paths.mask_dir / f"{mask_idx:05d}.png", (H, W))
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=ksize)
        grad = np.abs(sx) + np.abs(sy)
        m = mask > 0
        out.append(float(grad[m].mean()) if m.sum() >= 16 else float("nan"))
    return pd.Series(out)


def render_scatter(summary: pd.DataFrame, x_col: str, x_label: str, out_path: Path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for side, ax, ycol, ylabel, title in [
        ("left",  axes[0], "mean_e2d_px", "Mean 2D tracking error (px)",
         "2D point tracker error  ↓"),
        ("right", axes[1], "add_auc",     "ADD AUC",
         "Pose tracking AUC  ↑"),
    ]:
        x = summary[x_col].values
        y = summary[ycol].values
        ax.scatter(x, y, s=70, color=PT_COLOR, zorder=3)
        for _, r in summary.iterrows():
            xo, yo = LABEL_OFFSETS[side].get(r["video"], (8, 6))
            ax.annotate(r["video"], (r[x_col], r[ycol]),
                        fontsize=11, fontweight="bold",
                        xytext=(xo, yo), textcoords="offset points")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].set_ylim(40, 100)
    e_min, e_max = summary["mean_e2d_px"].min(), summary["mean_e2d_px"].max()
    pad = (e_max - e_min) * 0.15
    axes[0].set_ylim(max(0, e_min - pad), e_max + pad)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "ap_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    # AP-only summary from the existing table.
    summary_full = pd.read_csv(analysis_dir / "summary.csv")
    summary = summary_full[summary_full["video"].isin(AP_VIDEOS)].copy()

    # Compute tx_grad at ksize 3 / 5 / 7 (per-frame mean over object mask).
    for k in KSIZES:
        col = f"tx_grad_k{k}"
        means = []
        for v in summary["video"]:
            print(f"[ksize={k}] {v}")
            s = compute_tx_grad_per_frame(v, Path(args.data_root), ksize=k)
            means.append(s.mean())
        summary[col] = means

    summary = summary.sort_values("tx_grad_k3").reset_index(drop=True)
    print("\nAP-series tx_grad at multiple kernel sizes:")
    cols = ["video", "tx_grad_k3", "tx_grad_k5", "tx_grad_k7",
            "mean_e2d_px", "mean_add_cm", "add_auc"]
    print(summary[cols].to_string(index=False))

    # Save augmented summary for reference.
    summary.to_csv(out_dir / "AP_summary_with_ksizes.csv", index=False)

    # Render one scatter per kernel size.
    for k in KSIZES:
        render_scatter(
            summary,
            x_col=f"tx_grad_k{k}",
            x_label="Mean Sobel magnitude (high means more texture)",
            out_path=out_dir / f"AP_scatter_tx_grad_vs_auc_k{k}.png",
        )

    # Count-based scatter: mean #pts (unc<0.2 AND grad>50) vs metrics.
    # The count column is already in summary_full from add_texture_counts.py.
    count_col = "mean_n_pts_unc020_and_grad>50"
    if count_col not in summary.columns:
        # join from full summary
        summary = summary.merge(summary_full[["video", count_col]], on="video", how="left")
    render_scatter(
        summary.sort_values(count_col).reset_index(drop=True),
        x_col=count_col,
        x_label="Mean # points (confident AND in textured pixels)",
        out_path=out_dir / "AP_scatter_count_vs_auc.png",
    )


if __name__ == "__main__":
    main()
