"""Re-render plots and band tables from existing CSVs without reprocessing.

Reads `summary.csv`, `per_frame_all.csv`, `per_point_all.csv` from the analysis
directory and re-runs `render_plots` with chosen thresholds. Use this after the
calibration histogram has been inspected to pick `--tx-low` and `--tx-high`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture_dependency import render_plots, DEFAULT_OUT_DIR, bootstrap_ci  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--unc-thres", type=float, default=0.2)
    parser.add_argument("--ptx-thresholds", type=float, nargs="*",
                        default=[10, 25, 50])
    parser.add_argument("--tx-low", type=float, required=True)
    parser.add_argument("--tx-high", type=float, required=True)
    parser.add_argument("--cited-videos", nargs="*", default=["AP12"])
    parser.add_argument("--filter-e2d-min", type=float, default=0.5,
                        help="Drop frames with mean_e2d_px below this (alignment artifacts)")
    args = parser.parse_args()

    out_dir = Path(args.analysis_dir)
    summary_df = pd.read_csv(out_dir / "summary.csv")
    df_frame = pd.read_csv(out_dir / "per_frame_all.csv")
    df_point = pd.read_csv(out_dir / "per_point_all.csv")

    df_frame = df_frame[
        df_frame["mean_e2d_px"].isna()
        | (df_frame["mean_e2d_px"] >= args.filter_e2d_min)
    ].copy()

    render_plots(
        df_frame_all=df_frame,
        df_point_all=df_point,
        summary_df=summary_df,
        out_dir=out_dir,
        unc_thres=args.unc_thres,
        ptx_thresholds=args.ptx_thresholds,
        tx_low=args.tx_low,
        tx_high=args.tx_high,
        cited_videos=args.cited_videos,
    )

    print("\n=== Headline correlations (after artifact filter) ===")
    sx = summary_df["tx_grad_mean"].values
    for col, label in [
        ("mean_e2d_px", "tx_grad vs mean e2d (sequence-level)"),
        ("add_auc", "tx_grad vs ADD AUC (sequence-level)"),
        ("mean_add_cm", "tx_grad vs mean ADD (sequence-level)"),
    ]:
        sy = summary_df[col].values.astype(float)
        m = np.isfinite(sx) & np.isfinite(sy)
        r, _ = pearsonr(sx[m], sy[m])
        rho, _ = spearmanr(sx[m], sy[m])
        rho_b, lo, hi = bootstrap_ci(sx[m], sy[m], spearmanr)
        print(f"  {label}: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f} [{lo:.2f},{hi:.2f}]")

    fx = df_frame["mean_e2d_px"].values
    fy = df_frame["add_cm"].values
    m = np.isfinite(fx) & np.isfinite(fy)
    r, _ = pearsonr(fx[m], fy[m])
    rho, _ = spearmanr(fx[m], fy[m])
    print(f"  frame-level e2d vs ADD: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f}")
    fx = df_frame["n_unc_020"].values.astype(float)
    fy = df_frame["add_cm"].values
    m = np.isfinite(fx) & np.isfinite(fy)
    r, _ = pearsonr(fx[m], fy[m])
    rho, _ = spearmanr(fx[m], fy[m])
    print(f"  frame-level n_unc<0.2 vs ADD: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f}")


if __name__ == "__main__":
    main()
