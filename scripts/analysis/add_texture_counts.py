"""Augment per-frame CSVs with count-based texture metrics.

For each sequence and each frame t, compute (without re-running pose):
  n_textured_mask_px_T  := #pixels in object mask with |grad| > T
  frac_mask_textured_T  := n_textured_mask_px_T / mask_area_px
  n_pts_in_textured_T   := #visible tracked points whose pixel grad > T
  n_unc_textured_020    := #visible tracked points with both unc<0.2 AND grad>50

Then re-render scatter plots: count-based texture availability vs e2d / ADD.

Reads existing per_frame_<video>.csv files and appends columns; rewrites the
combined per_frame_all.csv.
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
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture_dependency import (  # noqa: E402
    DEFAULT_OUT_DIR, DEFAULT_RESULTS_ROOT, DEFAULT_DATA_ROOT,
    find_ho3d_paths, load_meta, ragged, read_rgb, read_mask, bootstrap_ci,
)

GRAD_THRESHOLDS = [30, 60, 100]


def compute_grad_image(rgb: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(sx) + np.abs(sy)


def process_video(
    video_name: str, results_root: Path, data_root: Path, out_dir: Path,
):
    csv_path = out_dir / f"per_frame_{video_name}.csv"
    npz_path = results_root / video_name / "meta_data" / "meta_data.npz"
    if not csv_path.exists() or not npz_path.exists():
        print(f"[skip] {video_name}: missing csv or npz")
        return None

    paths = find_ho3d_paths(data_root, video_name)
    if paths is None:
        print(f"[skip] {video_name}: no dataset folder")
        return None

    df = pd.read_csv(csv_path)
    npz = load_meta(npz_path)
    rgb_files = sorted(paths.rgb_dir.glob("*.jpg"))
    n_frames = min(len(df), len(rgb_files))
    df = df.iloc[:n_frames].copy()

    H, W = read_rgb(rgb_files[0]).shape[:2]
    new_cols = {f"n_mask_grad>{T}": np.zeros(n_frames, dtype=np.int64) for T in GRAD_THRESHOLDS}
    new_cols.update({f"frac_mask_grad>{T}": np.zeros(n_frames, dtype=np.float32) for T in GRAD_THRESHOLDS})
    new_cols.update({f"n_pts_grad>{T}": np.zeros(n_frames, dtype=np.int64) for T in GRAD_THRESHOLDS})
    new_cols["n_pts_unc020_and_grad>50"] = np.zeros(n_frames, dtype=np.int64)

    print(f"[run] {video_name} ({n_frames} frames)")
    for t in range(n_frames):
        rgb_id = str(df.iloc[t]["frame_id"]).zfill(4)
        mask_idx = int(rgb_id)
        rgb = read_rgb(rgb_files[t])
        mask = read_mask(paths.mask_dir / f"{mask_idx:05d}.png", (H, W))
        grad = compute_grad_image(rgb)

        m = mask > 0
        for T in GRAD_THRESHOLDS:
            in_mask_textured = m & (grad > T)
            n = int(in_mask_textured.sum())
            new_cols[f"n_mask_grad>{T}"][t] = n
            mask_area = int(m.sum())
            new_cols[f"frac_mask_grad>{T}"][t] = n / max(mask_area, 1)

        # Per-point texture at current frame: sample grad at predicted 2D for visible pts.
        track2d_t = ragged(npz, "track2d", t).reshape(-1, 2)
        vis_t = ragged(npz, "visibles", t).astype(bool)
        if track2d_t.shape[0] > 0:
            xy = track2d_t
            xs = np.clip(xy[:, 0].round().astype(np.int64), 0, W - 1)
            ys = np.clip(xy[:, 1].round().astype(np.int64), 0, H - 1)
            grads_at_pts = grad[ys, xs]
            in_img = (
                (xy[:, 0] >= 0) & (xy[:, 0] < W)
                & (xy[:, 1] >= 0) & (xy[:, 1] < H)
                & np.isfinite(xy).all(axis=1)
            )
            vis_and_in = vis_t & in_img
            for T in GRAD_THRESHOLDS:
                new_cols[f"n_pts_grad>{T}"][t] = int(((grads_at_pts > T) & vis_and_in).sum())
            unc_t = ragged(npz, "uncertainties", t)
            new_cols["n_pts_unc020_and_grad>50"][t] = int(
                ((unc_t < 0.2) & (grads_at_pts > 50) & vis_and_in).sum()
            )

    for k, v in new_cols.items():
        df[k] = v
    df.to_csv(csv_path, index=False)
    return df


def render_count_plots(out_dir: Path, cited_videos: list[str]):
    df = pd.read_csv(out_dir / "per_frame_all.csv")

    # Sequence-level summary aggregated from per-frame, including new count cols.
    summary_df = pd.read_csv(out_dir / "summary.csv").set_index("video")
    extra = []
    for v, sub in df.groupby("video"):
        row = {"video": v}
        for T in GRAD_THRESHOLDS:
            row[f"mean_n_mask_grad>{T}"] = sub[f"n_mask_grad>{T}"].mean()
            row[f"mean_frac_mask_grad>{T}"] = sub[f"frac_mask_grad>{T}"].mean()
            row[f"mean_n_pts_grad>{T}"] = sub[f"n_pts_grad>{T}"].mean()
        row["mean_n_pts_unc020_and_grad>50"] = sub["n_pts_unc020_and_grad>50"].mean()
        extra.append(row)
    extra_df = pd.DataFrame(extra).set_index("video")
    # Drop overlapping columns from prior runs before joining.
    overlap = [c for c in extra_df.columns if c in summary_df.columns]
    if overlap:
        summary_df = summary_df.drop(columns=overlap)
    summary_df = summary_df.join(extra_df, how="left").reset_index()
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    cited = set(cited_videos)

    # P8 - per-frame: n_pts_grad>60 vs ADD, faceted by sequence.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, ycol, ylabel in [
        (axes[0, 0], "add_cm", "ADD (cm)"),
        (axes[0, 1], "mean_e2d_px", "mean e2d (px)"),
        (axes[1, 0], "add_cm", "ADD (cm)"),
        (axes[1, 1], "mean_e2d_px", "mean e2d (px)"),
    ]:
        pass

    # P8a/b: n_mask_grad>60 (image-level texture availability)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for v, sub in df.groupby("video"):
        c = "red" if v in cited else None
        for ax, ycol in [(axes[0], "mean_e2d_px"), (axes[1], "add_cm")]:
            ax.scatter(sub["n_mask_grad>60"], sub[ycol], s=6, alpha=0.4, label=v, color=c)
    for ax, ylabel, ycol in [
        (axes[0], "mean e2d (px)", "mean_e2d_px"),
        (axes[1], "ADD (cm)", "add_cm"),
    ]:
        ax.set_xlabel("# textured mask pixels (grad>50)")
        ax.set_ylabel(ylabel)
        x = df["n_mask_grad>60"].values.astype(float)
        y = df[ycol].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 4:
            r, _ = pearsonr(x[m], y[m]); rho, _ = spearmanr(x[m], y[m])
            ax.text(0.05, 0.95,
                    f"pooled N={m.sum()}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("P8a: textured-mask-pixels vs e2d")
    axes[1].set_title("P8b: textured-mask-pixels vs ADD")
    fig.tight_layout()
    fig.savefig(out_dir / "P8_n_mask_textured_vs_metrics.png", dpi=150)
    plt.close(fig)

    # P9: per-frame n_pts_grad>60 vs ADD/e2d.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for v, sub in df.groupby("video"):
        c = "red" if v in cited else None
        for ax, ycol in [(axes[0], "mean_e2d_px"), (axes[1], "add_cm")]:
            ax.scatter(sub["n_pts_grad>60"], sub[ycol], s=6, alpha=0.4, label=v, color=c)
    for ax, ylabel, ycol in [
        (axes[0], "mean e2d (px)", "mean_e2d_px"),
        (axes[1], "ADD (cm)", "add_cm"),
    ]:
        ax.set_xlabel("# tracked points in textured pixels (grad>50)")
        ax.set_ylabel(ylabel)
        x = df["n_pts_grad>60"].values.astype(float)
        y = df[ycol].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 4:
            r, _ = pearsonr(x[m], y[m]); rho, _ = spearmanr(x[m], y[m])
            ax.text(0.05, 0.95,
                    f"pooled N={m.sum()}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("P9a: tracked-points-in-textured vs e2d")
    axes[1].set_title("P9b: tracked-points-in-textured vs ADD")
    fig.tight_layout()
    fig.savefig(out_dir / "P9_n_pts_textured_vs_metrics.png", dpi=150)
    plt.close(fig)

    # P10: combined-criterion (unc<0.2 AND grad>50) vs ADD.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for v, sub in df.groupby("video"):
        c = "red" if v in cited else None
        for ax, ycol in [(axes[0], "mean_e2d_px"), (axes[1], "add_cm")]:
            ax.scatter(sub["n_pts_unc020_and_grad>50"], sub[ycol], s=6, alpha=0.4, label=v, color=c)
    for ax, ylabel, ycol in [
        (axes[0], "mean e2d (px)", "mean_e2d_px"),
        (axes[1], "ADD (cm)", "add_cm"),
    ]:
        ax.set_xlabel("# pts (unc<0.2 AND grad>50)")
        ax.set_ylabel(ylabel)
        x = df["n_pts_unc020_and_grad>50"].values.astype(float)
        y = df[ycol].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 4:
            r, _ = pearsonr(x[m], y[m]); rho, _ = spearmanr(x[m], y[m])
            ax.text(0.05, 0.95,
                    f"pooled N={m.sum()}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("P10a: confident-AND-textured pts vs e2d")
    axes[1].set_title("P10b: confident-AND-textured pts vs ADD")
    fig.tight_layout()
    fig.savefig(out_dir / "P10_n_pts_unc_and_grad_vs_metrics.png", dpi=150)
    plt.close(fig)

    # P11: sequence-level scatter, n_textured_pts vs e2d/AUC.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sx = summary_df["mean_n_pts_unc020_and_grad>50"].values.astype(float)
    sy1 = summary_df["mean_e2d_px"].values.astype(float)
    sy2 = summary_df["add_auc"].values.astype(float)
    for ax, y, ylabel in [
        (axes[0], sy1, "Mean e2d (px)"),
        (axes[1], sy2, "ADD AUC"),
    ]:
        ax.scatter(sx, y, s=40)
        for i, v in enumerate(summary_df["video"]):
            color = "red" if v in cited else "black"
            ax.annotate(v, (sx[i], y[i]), fontsize=8, color=color)
        ax.set_xlabel("Mean #pts (unc<0.2 AND grad>50)")
        ax.set_ylabel(ylabel)
        m = np.isfinite(sx) & np.isfinite(y)
        if m.sum() > 4:
            r, _ = pearsonr(sx[m], y[m]); rho, _ = spearmanr(sx[m], y[m])
            rho_b, lo, hi = bootstrap_ci(sx[m], y[m], spearmanr)
            ax.text(0.05, 0.95,
                    f"N={m.sum()}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f} [{lo:.2f},{hi:.2f}]",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[0].set_title("P11a: confident-textured pts vs e2d (sequence)")
    axes[1].set_title("P11b: confident-textured pts vs ADD AUC (sequence)")
    fig.tight_layout()
    fig.savefig(out_dir / "P11_seq_n_pts_unc_grad_vs_metrics.png", dpi=150)
    plt.close(fig)

    # Failure-mode timeseries: extend P4-style plots to MPM10/MPM13/SM1.
    failure_videos = ["MPM10", "MPM13", "SM1", "AP12"]
    for v in failure_videos:
        sub = df[df["video"] == v].sort_values("frame")
        if sub.empty:
            continue
        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(sub["frame"], sub["mean_e2d_px"], color="C0")
        axes[0].set_ylabel("mean e2d (px)")
        axes[0].set_yscale("log")
        axes[1].plot(sub["frame"], sub["n_unc_020"], color="C1", label="unc<0.2")
        axes[1].plot(sub["frame"], sub["n_pts_unc020_and_grad>50"], color="C5", label="unc<0.2 AND grad>50")
        axes[1].plot(sub["frame"], sub["n_pts_grad>60"], color="C6", linestyle="--", label="any grad>50")
        axes[1].set_ylabel("#tracked points")
        axes[1].legend(fontsize=7, loc="upper right")
        axes[2].plot(sub["frame"], sub["n_mask_grad>60"], color="C7")
        axes[2].set_ylabel("#textured mask px (grad>50)")
        axes[3].plot(sub["frame"], sub["mask_area_px"], color="gray", label="mask area")
        axes[3].set_ylabel("mask area px")
        axes[4].plot(sub["frame"], sub["add_cm"], color="C2")
        axes[4].set_ylabel("ADD (cm)")
        axes[4].set_xlabel("frame")
        axes[0].set_title(f"Failure analysis: {v}")
        fig.tight_layout()
        fig.savefig(out_dir / f"Pfail_{v}.png", dpi=150)
        plt.close(fig)

    # Headline correlations on count metrics.
    print("\n=== Count-based metric correlations (frame-level pooled) ===")
    for xcol in [
        "n_mask_grad>60",
        "n_pts_grad>60",
        "n_pts_unc020_and_grad>50",
        "n_unc_020",
    ]:
        x = df[xcol].values.astype(float)
        for ycol, ylabel in [("add_cm", "ADD"), ("mean_e2d_px", "e2d")]:
            y = df[ycol].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            r, _ = pearsonr(x[m], y[m]); rho, _ = spearmanr(x[m], y[m])
            print(f"  {xcol} vs {ylabel}: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f}")

    print("\n=== Count-based metric correlations (sequence-level) ===")
    for xcol in [
        "mean_n_mask_grad>60",
        "mean_n_pts_grad>60",
        "mean_n_pts_unc020_and_grad>50",
        "mean_n_unc_020",
    ]:
        x = summary_df[xcol].values.astype(float)
        for ycol, ylabel in [
            ("mean_add_cm", "mean ADD"),
            ("add_auc", "ADD AUC"),
            ("mean_e2d_px", "mean e2d"),
        ]:
            y = summary_df[ycol].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 4:
                continue
            r, _ = pearsonr(x[m], y[m]); rho, _ = spearmanr(x[m], y[m])
            rho_b, lo, hi = bootstrap_ci(x[m], y[m], spearmanr)
            print(f"  {xcol} vs {ylabel}: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f} [{lo:.2f},{hi:.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--cited-videos", nargs="*", default=["AP12"])
    parser.add_argument("--skip-recompute", action="store_true",
                        help="Only re-render plots (per-frame CSVs must already have count cols).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)

    if args.videos:
        videos = args.videos
    else:
        videos = sorted(p.stem.replace("per_frame_", "")
                        for p in out_dir.glob("per_frame_*.csv")
                        if "all" not in p.stem)

    if not args.skip_recompute:
        for v in videos:
            process_video(v, results_root, data_root, out_dir)
        # Rebuild per_frame_all from ALL per_frame_<v>.csv files (augmented or not).
        all_paths = sorted(p for p in out_dir.glob("per_frame_*.csv")
                           if p.stem != "per_frame_all")
        all_dfs = [pd.read_csv(p) for p in all_paths]
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(out_dir / "per_frame_all.csv", index=False)
            print(f"\nWrote per_frame_all.csv with {len(combined)} rows")

    # Only render if every per_frame CSV has the count columns.
    sample = pd.read_csv(out_dir / "per_frame_all.csv", nrows=1)
    if "n_mask_grad>60" in sample.columns:
        render_count_plots(out_dir, args.cited_videos)
    else:
        print("[skip] render_count_plots: not all videos augmented yet "
              "(re-run without --videos to process all).")


if __name__ == "__main__":
    main()
