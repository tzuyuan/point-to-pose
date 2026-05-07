"""Runtime analysis for the modular pipeline on HO3D + YCBMultiTrack.

Reads per-frame timings from canonical pipeline runs and produces:

  1. Per-module mean/median/p95 table at the default config.
  2. Stacked bar plot of mean per-module time (per video + aggregated).
  3. Box plot of per-frame distribution per module.
  4. Scatter (with binned mean) of 2D-tracker time vs the *actual* number of
     tracked points each frame — within-run scaling.
  5. Per-frame timelines: total_ms and per-module ms vs frame_id, per video.
  6. Cotracker3 vs TAPIR comparison: tracker time vs num_active_tracks, plus
     ADD/ADD-S AUC table from each run's quality.json.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_COLUMNS = [
    ("SAM2 (segmenter)", "segmenter_ms"),
    ("2D tracker (TAPIR)", "tracker_ms"),
    ("Register", "register_ms"),
    ("Keypoint addition (keyframe)", "keyframe_ms"),
    ("Local graph opt", "local_opt_ms"),
    ("Global graph opt - TSDF", "graph_opt_only_ms"),
    ("TSDF integration", "tsdf_ms"),
]

STACK_MODULES = [
    ("SAM2", "segmenter_ms"),
    ("2D tracker", "tracker_ms"),
    ("Register", "register_ms"),
    ("Keypoint addition", "keyframe_ms"),
    ("Local opt", "local_opt_ms"),
    ("Global graph opt - TSDF", "graph_opt_only_ms"),
    ("TSDF", "tsdf_ms"),
    ("Other", "other_ms"),
]


def discover_default_runs(root: str, default_n: int) -> pd.DataFrame:
    """Find timings.csv under root/n{default_n:03d}/<video>/timings.csv."""
    pattern = os.path.join(root, f"n{default_n:03d}", "*", "timings.csv")
    rows = []
    for csv_path in glob.glob(pattern):
        m = re.search(r"/n(\d+)/([^/]+)/timings\.csv$", csv_path)
        if not m:
            continue
        rows.append(
            {
                "num_points": int(m.group(1)),
                "video": m.group(2),
                "csv_path": csv_path,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["video", "csv_path"])
    return pd.DataFrame(rows).sort_values("video").reset_index(drop=True)


def load_runs(runs: pd.DataFrame, drop_warmup: int = 1) -> pd.DataFrame:
    frames = []
    for _, r in runs.iterrows():
        df = pd.read_csv(r["csv_path"])
        if drop_warmup > 0 and len(df) > drop_warmup:
            df = df.iloc[drop_warmup:].copy()
        df["video"] = r["video"]
        df["num_points"] = r["num_points"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Derived columns.
    out["graph_opt_only_ms"] = (out["global_opt_ms"] - out["tsdf_ms"]).clip(lower=0.0)
    accounted = (
        out["frontend_total_ms"]
        + out["track_table_ms"]
        + out["track_compact_ms"]
        + out["recovery_ms"]
        + out["local_opt_ms"]
        + out["keyframe_ms"]
        + out["global_opt_ms"]
        + out["logging_ms"]
    )
    out["other_ms"] = (out["total_ms"] - accounted).clip(lower=0.0)
    return out


def _summary_stats(frame: pd.DataFrame, cols) -> pd.DataFrame:
    rows = {
        "mean": frame[cols].mean(),
        "median": frame[cols].median(),
        "std": frame[cols].std(),
        "p95": frame[cols].quantile(0.95),
    }
    return pd.DataFrame(rows).T  # rows are stats, cols are modules


def write_summary_table(df: pd.DataFrame, out_path: str):
    cols = [c for _, c in MODULE_COLUMNS] + ["total_ms"]
    pieces = []
    for video, sub in df.groupby("video"):
        agg = _summary_stats(sub, cols)
        agg.insert(0, "video", video)
        pieces.append(agg.reset_index().rename(columns={"index": "stat"}))
    agg_all = _summary_stats(df, cols)
    agg_all.insert(0, "video", "ALL")
    pieces.append(agg_all.reset_index().rename(columns={"index": "stat"}))
    out = pd.concat(pieces, ignore_index=True)
    out.to_csv(out_path, index=False)

    md_path = out_path.replace(".csv", ".md")
    rename = {col: name for name, col in MODULE_COLUMNS}
    rename["total_ms"] = "Total"
    md_df = out.rename(columns=rename).round(2)
    Path(md_path).write_text(
        "# Per-module runtime (ms) at default config (sampler.num_points=30)\n\n"
        + md_df.to_markdown(index=False)
        + "\n"
    )
    print(f"Wrote {out_path} and {md_path}")


def plot_module_breakdown(df: pd.DataFrame, out_path: str):
    videos = sorted(df["video"].unique())
    labels = list(videos) + ["aggregated"]
    means = []
    for v in videos:
        means.append(df[df["video"] == v][[c for _, c in STACK_MODULES]].mean())
    means.append(df[[c for _, c in STACK_MODULES]].mean())
    means_df = pd.DataFrame(means, index=labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(labels))
    cmap = plt.get_cmap("tab10")
    for i, (name, col) in enumerate(STACK_MODULES):
        vals = means_df[col].values
        ax.bar(labels, vals, bottom=bottom, label=name, color=cmap(i % 10))
        bottom = bottom + vals
    ax.set_ylabel("Mean per-frame time (ms)")
    ax.set_title("Per-module runtime breakdown — default config (num_points=30)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_module_boxplot(df: pd.DataFrame, out_path: str):
    cols = [c for _, c in MODULE_COLUMNS]
    names = [n for n, _ in MODULE_COLUMNS]
    data = [df[c].values for c in cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=names, showfliers=False)
    ax.set_ylabel("Per-frame time (ms)")
    ax.set_title("Per-module per-frame distribution — default config (num_points=30)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_tracker_vs_num_points(df: pd.DataFrame, out_path: str):
    """Scatter of tracker_ms vs num_active_tracks per frame, with binned mean
    overlay and a linear fit."""
    sub = df[(df["num_active_tracks"] > 0) & (df["tracker_ms"] > 0)].copy()
    if sub.empty:
        print("[skip] no rows with positive num_active_tracks / tracker_ms")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    videos = sorted(sub["video"].unique())
    for i, v in enumerate(videos):
        s = sub[sub["video"] == v]
        ax.scatter(
            s["num_active_tracks"], s["tracker_ms"],
            s=8, alpha=0.25, color=cmap(i), label=f"{v} (per-frame)",
        )

    # Binned mean across all videos
    bins = np.unique(sub["num_active_tracks"].values)
    if len(bins) > 30:
        bins = np.linspace(sub["num_active_tracks"].min(),
                           sub["num_active_tracks"].max(), 30)
    sub["bin"] = pd.cut(sub["num_active_tracks"], bins=bins, include_lowest=True)
    binned = sub.groupby("bin", observed=True).agg(
        n=("num_active_tracks", "mean"),
        t_mean=("tracker_ms", "mean"),
        t_std=("tracker_ms", "std"),
    ).dropna()
    ax.errorbar(binned["n"], binned["t_mean"], yerr=binned["t_std"],
                marker="o", color="black", capsize=3, lw=1.5,
                label="binned mean ± 1σ")

    # Linear fit
    x = sub["num_active_tracks"].values.astype(float)
    y = sub["tracker_ms"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="red", linestyle="--",
            label=f"linear fit: {slope*1000:.3f} μs/pt + {intercept:.1f} ms")

    ax.set_xlabel("Number of active tracked points (per frame)")
    ax.set_ylabel("2D tracker time per frame (ms)")
    ax.set_title("TAPIR tracker runtime vs number of tracked points")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_tracker_vs_num_points_loglog(df: pd.DataFrame, out_path: str):
    """Same data on log-log axes — exposes the asymptotic exponent."""
    sub = df[(df["num_active_tracks"] > 0) & (df["tracker_ms"] > 0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, v in enumerate(sorted(sub["video"].unique())):
        s = sub[sub["video"] == v]
        ax.scatter(s["num_active_tracks"], s["tracker_ms"],
                   s=8, alpha=0.25, color=cmap(i), label=v)

    x = np.log10(sub["num_active_tracks"].values.astype(float))
    y = np.log10(sub["tracker_ms"].values.astype(float))
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(sub["num_active_tracks"].min(), sub["num_active_tracks"].max(), 100)
    ax.plot(xs, (10 ** b) * xs ** a, color="red", linestyle="--",
            label=f"power-law fit: t ∝ N^{a:.2f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of active tracked points (per frame)")
    ax.set_ylabel("2D tracker time per frame (ms)")
    ax.set_title("TAPIR tracker runtime vs num tracks (log-log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_frame_timeline(df: pd.DataFrame, out_path: str, label: str = "TAPIR"):
    """Per-frame timeline of total_ms and a few key modules."""
    videos = sorted(df["video"].unique())
    fig, axes = plt.subplots(len(videos), 1, figsize=(11, 3.2 * len(videos)),
                             sharex=False, squeeze=False)
    cmap = plt.get_cmap("tab10")
    series = [
        ("Total", "total_ms", "black"),
        ("2D tracker", "tracker_ms", cmap(1)),
        ("SAM2", "segmenter_ms", cmap(0)),
        ("Register", "register_ms", cmap(2)),
        ("Keyframe addition", "keyframe_ms", cmap(3)),
        ("TSDF", "tsdf_ms", cmap(5)),
    ]
    for ax, video in zip(axes[:, 0], videos):
        sub = df[df["video"] == video].sort_values("frame_id")
        for name, col, color in series:
            ax.plot(sub["frame_id"], sub[col], color=color, lw=0.8,
                    alpha=0.9 if name == "Total" else 0.6, label=name)
        ax.set_title(f"{label} per-frame timing — {video}")
        ax.set_ylabel("ms")
        ax.set_xlabel("Frame index")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def discover_alt_run(root: str, subdir: str) -> pd.DataFrame:
    """Find timings.csv directly under root/<subdir>/<video>/timings.csv (no n*).

    Used for cotracker3 ('cotracker3/...') and YCB ('ycb_tapir/...')."""
    pattern = os.path.join(root, subdir, "*", "timings.csv")
    rows = []
    for csv_path in glob.glob(pattern):
        m = re.search(r"/([^/]+)/([^/]+)/timings\.csv$", csv_path)
        if not m:
            continue
        rows.append({"video": m.group(2), "csv_path": csv_path})
    if not rows:
        return pd.DataFrame(columns=["video", "csv_path"])
    return pd.DataFrame(rows).sort_values("video").reset_index(drop=True)


def load_alt_runs(runs: pd.DataFrame, drop_warmup: int = 1) -> pd.DataFrame:
    frames = []
    for _, r in runs.iterrows():
        df = pd.read_csv(r["csv_path"])
        if drop_warmup > 0 and len(df) > drop_warmup:
            df = df.iloc[drop_warmup:].copy()
        df["video"] = r["video"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["graph_opt_only_ms"] = (out["global_opt_ms"] - out["tsdf_ms"]).clip(lower=0.0)
    accounted = (
        out["frontend_total_ms"] + out["track_table_ms"] + out["track_compact_ms"]
        + out["recovery_ms"] + out["local_opt_ms"] + out["keyframe_ms"]
        + out["global_opt_ms"] + out["logging_ms"]
    )
    out["other_ms"] = (out["total_ms"] - accounted).clip(lower=0.0)
    return out


def plot_tracker_compare(tapir_df: pd.DataFrame, ct3_df: pd.DataFrame, out_path: str):
    """Overlay TAPIR vs cotracker3 on tracker_ms vs num_active_tracks."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    def _scatter_and_fit(df, color, label):
        sub = df[(df["num_active_tracks"] > 0) & (df["tracker_ms"] > 0)]
        if sub.empty:
            return None
        ax.scatter(sub["num_active_tracks"], sub["tracker_ms"],
                   s=6, alpha=0.18, color=color, label=label)
        x = sub["num_active_tracks"].values.astype(float)
        y = sub["tracker_ms"].values.astype(float)
        slope, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + b, color=color, lw=2)
        return (slope, b, len(sub))

    tapir_fit = _scatter_and_fit(tapir_df, "tab:blue", "TAPIR")
    ct3_fit = _scatter_and_fit(ct3_df, "tab:orange", "cotracker3")

    ax.set_xlabel("Number of active tracked points (per frame)")
    ax.set_ylabel("2D tracker time per frame (ms)")
    ax.set_title("Tracker runtime v.s. Number of Tracks")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return tapir_fit, ct3_fit


def plot_tracker_compare_resolutions(root: str, out_path: str, drop_warmup: int = 1,
                                     videos=None):
    """Same style as plot_tracker_compare, but the series is one line per resolution.

    Reads resolution_sweep/tapir/<HxW>/<video>/timings.csv. By default combines all
    videos found at each resolution; pass `videos` to restrict.
    """
    runs = discover_resolution_sweep(root)
    if runs.empty:
        return
    if videos is not None:
        runs = runs[runs["video"].isin(videos)]
        if runs.empty:
            return

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    legend_handles = []

    def _scatter_and_fit(sub, color, label):
        if sub.empty:
            return
        ax.scatter(sub["num_active_tracks"], sub["tracker_ms"],
                   s=6, alpha=0.18, color=color)
        x = sub["num_active_tracks"].values.astype(float)
        y = sub["tracker_ms"].values.astype(float)
        slope, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + b, color=color, lw=2)
        # opaque marker for the legend so the dot color matches the line color
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="", color=color,
                   markersize=6, label=label)
        )

    by_res = sorted(runs["height"].unique())
    for i, h in enumerate(by_res):
        frames = []
        for _, r in runs[runs["height"] == h].iterrows():
            df = pd.read_csv(r["csv_path"])
            if drop_warmup > 0 and len(df) > drop_warmup:
                df = df.iloc[drop_warmup:]
            df["video"] = r["video"]
            frames.append(df)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        sub = df[(df["num_active_tracks"] > 0) & (df["tracker_ms"] > 0)]
        _scatter_and_fit(sub, cmap(i), f"TAPIR ({h}x{h})")

    # cotracker3 (HO3D, 480x640 per its native config)
    ct3_runs = discover_alt_run(root, "cotracker3")
    if not ct3_runs.empty:
        ct3_df = load_alt_runs(ct3_runs, drop_warmup=drop_warmup)
        sub = ct3_df[(ct3_df["num_active_tracks"] > 0) & (ct3_df["tracker_ms"] > 0)]
        _scatter_and_fit(sub, cmap(len(by_res)), "cotracker3 (480x640)")

    ax.set_xlabel("Number of active tracked points (per frame)")
    ax.set_ylabel("2D tracker time per frame (ms)")
    ax.set_title("Tracker runtime v.s. Number of Tracks")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def collect_quality(root: str, sub_pattern: str, label: str) -> list:
    """Read quality.json from each run under sub_pattern (e.g. 'n030/*' or 'cotracker3/*')."""
    import json as _json
    rows = []
    for q_path in sorted(glob.glob(os.path.join(root, sub_pattern, "quality.json"))):
        try:
            with open(q_path) as f:
                data = _json.load(f)
        except Exception:
            continue
        video = os.path.basename(os.path.dirname(q_path))
        if isinstance(data, dict) and "add_s_auc" in data:
            data = {video: data}
        for obj_name, q in data.items():
            row = {"tracker": label, "video": video, "object": obj_name}
            row.update({k: q.get(k, None) for k in
                         ["add_s_err_mean_cm", "add_err_mean_cm",
                          "add_s_auc", "add_auc"]})
            rows.append(row)
    return rows


def write_quality_table(root: str, out_path: str):
    rows = []
    rows += collect_quality(root, "n030/*", "TAPIR")
    rows += collect_quality(root, "cotracker3/*", "cotracker3_online")
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    md = out_path.replace(".csv", ".md")
    Path(md).write_text(
        "# Quality (ADD / ADD-S) — TAPIR vs cotracker3_online on HO3D\n\n"
        + df.round(2).to_markdown(index=False) + "\n"
    )
    print(f"Wrote {out_path} and {md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/justin/code/point-to-pose/results/runtime_analysis_20260505",
    )
    parser.add_argument("--default_num_points", type=int, default=30)
    parser.add_argument("--drop_warmup", type=int, default=1)
    args = parser.parse_args()

    runs = discover_default_runs(args.root, args.default_num_points)
    if runs.empty:
        print(f"No timings.csv at {args.root}/n{args.default_num_points:03d}/*/timings.csv",
              file=sys.stderr)
        sys.exit(1)
    print("Discovered canonical runs:")
    print(runs.to_string(index=False))

    df = load_runs(runs, drop_warmup=args.drop_warmup)
    print(f"Total rows after warmup drop: {len(df)}")

    fig_dir = os.path.join(args.root, "figures")
    tab_dir = os.path.join(args.root, "tables")
    agg_dir = os.path.join(args.root, "aggregated")
    for d in (fig_dir, tab_dir, agg_dir):
        os.makedirs(d, exist_ok=True)

    df.to_csv(os.path.join(agg_dir, "timings_default.csv"), index=False)
    print(f"Wrote {agg_dir}/timings_default.csv")

    write_summary_table(df, os.path.join(tab_dir, "per_module_summary.csv"))
    plot_module_breakdown(df, os.path.join(fig_dir, "module_breakdown.png"))
    plot_module_boxplot(df, os.path.join(fig_dir, "module_boxplot.png"))
    plot_tracker_vs_num_points(df, os.path.join(fig_dir, "tracker_vs_num_points.png"))
    plot_tracker_vs_num_points_loglog(
        df, os.path.join(fig_dir, "tracker_vs_num_points_loglog.png")
    )
    plot_per_frame_timeline(
        df, os.path.join(fig_dir, "per_frame_timeline_ho3d_tapir.png"),
        label="TAPIR (HO3D)",
    )

    # Cotracker3 (HO3D) — optional
    ct3_runs = discover_alt_run(args.root, "cotracker3")
    if not ct3_runs.empty:
        ct3_df = load_alt_runs(ct3_runs, drop_warmup=args.drop_warmup)
        ct3_df.to_csv(os.path.join(agg_dir, "timings_cotracker3.csv"), index=False)
        plot_per_frame_timeline(
            ct3_df, os.path.join(fig_dir, "per_frame_timeline_ho3d_cotracker3.png"),
            label="cotracker3_online (HO3D)",
        )
        plot_tracker_compare(
            df, ct3_df, os.path.join(fig_dir, "tracker_compare_tapir_vs_cotracker3.png")
        )

    # YCB single-object — optional
    ycb_runs = discover_alt_run(args.root, "ycb_tapir")
    if not ycb_runs.empty:
        ycb_df = load_alt_runs(ycb_runs, drop_warmup=args.drop_warmup)
        ycb_df.to_csv(os.path.join(agg_dir, "timings_ycb_tapir.csv"), index=False)
        plot_per_frame_timeline(
            ycb_df, os.path.join(fig_dir, "per_frame_timeline_ycb_tapir.png"),
            label="TAPIR (YCBMultiTrack)",
        )

    # Quality table comparing TAPIR vs cotracker3 (uses quality.json files)
    write_quality_table(args.root, os.path.join(tab_dir, "quality_tapir_vs_cotracker3.csv"))

    # Resolution sweep — optional
    write_resolution_sweep_outputs(args.root, fig_dir, tab_dir, args.drop_warmup)
    plot_tracker_compare_resolutions(
        args.root,
        os.path.join(fig_dir, "tracker_compare_resolutions.png"),
        drop_warmup=args.drop_warmup,
    )


# -----------------------------------------------------------------------------
# Resolution sweep
# -----------------------------------------------------------------------------

def discover_resolution_sweep(root: str) -> pd.DataFrame:
    """Find timings.csv under root/resolution_sweep/tapir/<HxW>/<video>/timings.csv."""
    pattern = os.path.join(root, "resolution_sweep", "tapir", "*", "*", "timings.csv")
    rows = []
    for csv_path in glob.glob(pattern):
        m = re.search(r"/(\d+)x(\d+)/([^/]+)/timings\.csv$", csv_path)
        if not m:
            continue
        rows.append({
            "height": int(m.group(1)),
            "width": int(m.group(2)),
            "video": m.group(3),
            "csv_path": csv_path,
        })
    if not rows:
        return pd.DataFrame(columns=["height", "width", "video", "csv_path"])
    return pd.DataFrame(rows).sort_values(["height", "video"]).reset_index(drop=True)


def _load_quality(csv_path: str) -> dict:
    """Read sibling quality.json and return a flat dict (handles both single and multi-obj formats)."""
    import json as _json
    q_path = os.path.join(os.path.dirname(csv_path), "quality.json")
    if not os.path.exists(q_path):
        return {}
    try:
        with open(q_path) as f:
            data = _json.load(f)
    except Exception:
        return {}
    if isinstance(data, dict) and "add_s_auc" in data:
        return data
    if isinstance(data, dict) and len(data) > 0:
        # multi-object — return the first (we use single-obj sequences)
        first = next(iter(data.values()))
        if isinstance(first, dict):
            return first
    return {}


def write_resolution_sweep_outputs(root: str, fig_dir: str, tab_dir: str, drop_warmup: int):
    runs = discover_resolution_sweep(root)
    if runs.empty:
        return
    print("Discovered resolution-sweep runs:")
    print(runs.to_string(index=False))

    rows = []
    for _, r in runs.iterrows():
        df = pd.read_csv(r["csv_path"])
        if drop_warmup > 0 and len(df) > drop_warmup:
            df = df.iloc[drop_warmup:]
        q = _load_quality(r["csv_path"])
        rows.append({
            "video": r["video"],
            "resolution": f"{r['height']}x{r['width']}",
            "h": int(r["height"]),
            "w": int(r["width"]),
            "tracker_ms_mean": float(df["tracker_ms"].mean()),
            "total_ms_mean": float(df["total_ms"].mean()),
            "add_s_err_cm": q.get("add_s_err_mean_cm"),
            "add_err_cm": q.get("add_err_mean_cm"),
            "add_s_auc": q.get("add_s_auc"),
            "add_auc": q.get("add_auc"),
            "n_frames": int(len(df)),
        })
    table = pd.DataFrame(rows).sort_values(["video", "h"]).reset_index(drop=True)
    csv_path = os.path.join(tab_dir, "resolution_sweep.csv")
    md_path = os.path.join(tab_dir, "resolution_sweep.md")
    table.to_csv(csv_path, index=False)
    md_df = table.drop(columns=["h", "w"]).round(2)
    Path(md_path).write_text(
        "# TAPIR resolution sweep — runtime + quality\n\n"
        "Mean per-frame tracker_ms and pipeline total_ms (frame 0 dropped as warm-up). "
        "ADD/ADD-S in cm; AUC in percent.\n\n"
        + md_df.to_markdown(index=False) + "\n"
    )
    print(f"Wrote {csv_path} and {md_path}")

    # Plots
    cmap = plt.get_cmap("tab10")
    videos = sorted(table["video"].unique())

    # 1) Runtime vs resolution
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, v in enumerate(videos):
        sub = table[table["video"] == v].sort_values("h")
        ax.plot(sub["h"], sub["tracker_ms_mean"], marker="o", color=cmap(i), label=v)
    ax.set_xlabel("Tracker input resolution (square, pixels per side)")
    ax.set_ylabel("Mean 2D tracker time per frame (ms)")
    ax.set_title("TAPIR runtime vs input resolution")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "resolution_runtime.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {os.path.join(fig_dir, 'resolution_runtime.png')}")

    # 2) Quality vs resolution
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, v in enumerate(videos):
        sub = table[table["video"] == v].sort_values("h").dropna(subset=["add_s_auc"])
        if sub.empty:
            continue
        ax.plot(sub["h"], sub["add_s_auc"], marker="o", color=cmap(i), label=f"{v} ADD-S AUC")
        if sub["add_auc"].notna().any():
            ax.plot(sub["h"], sub["add_auc"], marker="s", color=cmap(i), linestyle="--",
                    alpha=0.7, label=f"{v} ADD AUC")
    ax.set_xlabel("Tracker input resolution (square, pixels per side)")
    ax.set_ylabel("AUC (%)")
    ax.set_title("TAPIR quality vs input resolution")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "resolution_quality.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {os.path.join(fig_dir, 'resolution_quality.png')}")

    # 3) Pareto: runtime vs quality
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, v in enumerate(videos):
        sub = table[table["video"] == v].sort_values("h").dropna(subset=["add_s_auc"])
        if sub.empty:
            continue
        ax.plot(sub["tracker_ms_mean"], sub["add_s_auc"],
                marker="o", color=cmap(i), label=v)
        for _, row in sub.iterrows():
            ax.annotate(
                row["resolution"],
                (row["tracker_ms_mean"], row["add_s_auc"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
                color=cmap(i),
            )
    ax.set_xlabel("Mean 2D tracker time per frame (ms)")
    ax.set_ylabel("ADD-S AUC (%)")
    ax.set_title("TAPIR resolution Pareto: runtime vs quality")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "resolution_pareto.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {os.path.join(fig_dir, 'resolution_pareto.png')}")


if __name__ == "__main__":
    main()
