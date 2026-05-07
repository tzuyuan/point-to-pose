"""MPM13 per-frame timeseries: e2d statistics, tx_grad, ADD error.

Computes per-frame counts of points whose 2D-tracking error is below a
threshold (only computable post-hoc since per-point e2d isn't stored), and
overlays them with mean e2d, tx_grad, and ADD on a shared time axis.
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
    DEFAULT_OUT_DIR, DEFAULT_RESULTS_ROOT, DEFAULT_DATA_ROOT,
    find_ho3d_paths, load_meta, ragged, project_points, get_gt_pose,
    load_intrinsics_and_mesh, recover_sample_frames, compute_canonical_anchors,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="MPM13")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=str(Path(DEFAULT_OUT_DIR) / "ap_only"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    video = args.video
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_ho3d_paths(Path(args.data_root), video)
    if paths is None:
        raise SystemExit(f"No HO3D folder for {video}")

    npz_path = Path(args.results_root) / video / "meta_data" / "meta_data.npz"
    npz = load_meta(npz_path)
    K, _, rgb_files = load_intrinsics_and_mesh(paths)

    n_frames = min(len(npz["frame_id"]), len(rgb_files))
    rgb_id_strs = [Path(p).stem for p in rgb_files[:n_frames]]

    gt_poses = [get_gt_pose(paths.meta_dir, s) for s in rgb_id_strs]
    sample_frame = recover_sample_frames(npz, n_frames)
    X_can, valid_anchor = compute_canonical_anchors(npz, sample_frame, gt_poses)

    H, W = cv2.imread(str(rgb_files[0])).shape[:2]

    # Pull per-frame add_cm and tx_grad from existing CSV.
    csv = pd.read_csv(Path(args.out_dir).parent / f"per_frame_{video}.csv")

    # Compute per-frame counts of points with e2d < threshold.
    counts = {f"n_e2d_lt_{int(t)}": np.zeros(n_frames, dtype=np.int64)
              for t in args.thresholds}
    mean_e2d_recompute = np.full(n_frames, np.nan, dtype=np.float64)
    n_eval = np.zeros(n_frames, dtype=np.int64)

    for t in range(n_frames):
        T_gt_t = gt_poses[t]
        if T_gt_t is None:
            continue
        track2d_t = ragged(npz, "track2d", t).reshape(-1, 2)
        vis_t = ragged(npz, "visibles", t).astype(bool)
        valid_t = ragged(npz, "valid", t).astype(bool)
        n_t = track2d_t.shape[0]
        keep = valid_anchor[:n_t]
        if not keep.any():
            continue
        gs = np.where(keep)[0]
        gt2d = project_points(X_can[gs], K, T_gt_t)
        pred2d = track2d_t[gs]
        in_img = (
            (gt2d[:, 0] >= 0) & (gt2d[:, 0] < W)
            & (gt2d[:, 1] >= 0) & (gt2d[:, 1] < H)
            & np.isfinite(gt2d).all(axis=1)
            & np.isfinite(pred2d).all(axis=1)
        )
        m = vis_t[gs] & valid_t[gs] & in_img
        if not m.any():
            continue
        e2d = np.linalg.norm(pred2d[m] - gt2d[m], axis=1)
        mean_e2d_recompute[t] = float(e2d.mean())
        n_eval[t] = int(m.sum())
        for thr in args.thresholds:
            counts[f"n_e2d_lt_{int(thr)}"][t] = int((e2d < thr).sum())

    frames = np.arange(n_frames)

    # 4-row figure: mean e2d, #pts e2d<thr, tx_grad, ADD.
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(frames, mean_e2d_recompute, color="C0", linewidth=1.0)
    axes[0].set_ylabel("Mean e2d (px)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for i, thr in enumerate(args.thresholds):
        col = cmap(0.2 + 0.6 * i / max(len(args.thresholds) - 1, 1))
        axes[1].plot(frames, counts[f"n_e2d_lt_{int(thr)}"],
                     color=col, linewidth=1.0,
                     label=f"e2d < {int(thr)} px")
    axes[1].set_ylabel("# points with e2d < threshold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(csv["frame"], csv["tx_grad"], color="C7", linewidth=1.0)
    axes[2].set_ylabel("tx_grad")
    axes[2].grid(alpha=0.3)

    axes[3].plot(csv["frame"], csv["add_cm"], color="C2", linewidth=1.0)
    axes[3].set_ylabel("ADD (cm)")
    axes[3].set_xlabel("frame")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{video}_e2d_threshold_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Also save the per-frame counts as CSV.
    df_out = pd.DataFrame({
        "frame": frames,
        "frame_id": rgb_id_strs[:n_frames],
        "mean_e2d_px": mean_e2d_recompute,
        "n_eval_points": n_eval,
        **{k: v for k, v in counts.items()},
    })
    df_out.to_csv(out_dir / f"{video}_per_frame_e2d_counts.csv", index=False)


if __name__ == "__main__":
    main()
