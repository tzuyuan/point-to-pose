"""Aggregate tracker-sweep pose results into comparison tables.

Only frame-count-verified (complete) sequence outputs are included. Produces:
  - YCB: per-object ADD/ADD-S AUC on the common sequence set (sequences every
    tracker completed) + per-tracker full-coverage averages.
  - HO3D: per-sequence table, full-set mean, and the ECCV 8-video subset mean
    (AP14, MPM10-14, SB11, SB13) comparable to the published run.

Usage:
    python experiments/tracker_sweep/aggregate_results.py [--out report.md]
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

ROOT = "/home/justin/results/tracker_pose_benchmark"
TRACKERS = ["tapir", "cotracker3_online", "tapnext", "litetracker", "trackon"]
ECCV8 = {"AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13"}

YCB_DATA = "/home/justin/data/YCBMultiTrack_new_eccv_submission"
HO3D_DATA = "/home/justin/data/HO3D_V3/evaluation"


def complete_sequences(dataset_dir, data_root, img_sub="jpg"):
    """tracker -> set of sequences with full-length metadata."""
    out = {}
    for trk in TRACKERS:
        seqs = set()
        for meta in glob.glob(f"{dataset_dir}/{trk}/*/meta_data/meta_data.npz"):
            seq = meta.split("/")[-3]
            n_exp = len(glob.glob(f"{data_root}/{seq}/{img_sub}/*"))
            try:
                n = len(
                    np.atleast_1d(
                        np.load(meta, allow_pickle=True)["frame_id"]
                    )
                )
            except Exception:
                n = -1
            if n_exp > 0 and n >= n_exp - 5:
                seqs.add(seq)
        out[trk] = seqs
    return out


def ycb_tables(lines):
    comp = complete_sequences(f"{ROOT}/ycbmultitrack_real", YCB_DATA, "jpg")
    all_seqs = sorted(set().union(*comp.values()))
    common = sorted(set.intersection(*[comp[t] for t in TRACKERS]))

    lines.append("## YCBMultiTrack (ECCV-submission GT)\n")
    lines.append(f"Sequences completed per tracker: "
                 + ", ".join(f"{t}={len(comp[t])}/{len(all_seqs)}" for t in TRACKERS))
    excl = [s for s in all_seqs if s not in common]
    lines.append(f"\nCommon set ({len(common)} seqs); excluded from common table: {excl}\n")

    # load per-object metrics
    per_obj = {t: {} for t in TRACKERS}
    for t in TRACKERS:
        path = f"{ROOT}/ycbmultitrack_real/{t}/summary/per_object_metrics.csv"
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r["video_name"] in comp[t]:
                per_obj[t][(r["video_name"], r["object_name"])] = (
                    float(r["add_s_auc"]),
                    float(r["add_auc"]),
                    float(r["add_s_err_cm"]),
                    float(r["add_err_cm"]),
                )

    def table(seq_filter, title):
        lines.append(f"### {title}\n")
        lines.append("| tracker | ADD-S AUC | ADD AUC | ADD-S err (cm) | ADD err (cm) | objs |")
        lines.append("|---|---|---|---|---|---|")
        for t in TRACKERS:
            vals = [v for (seq, _), v in per_obj[t].items() if seq in seq_filter]
            if not vals:
                lines.append(f"| {t} | - | - | - | - | 0 |")
                continue
            arr = np.array(vals)
            lines.append(
                f"| {t} | {arr[:,0].mean():.2f} | {arr[:,1].mean():.2f} | "
                f"{arr[:,2].mean():.2f} | {arr[:,3].mean():.2f} | {len(vals)} |"
            )
        lines.append("")

    table(set(common), f"Common sequence set ({len(common)} seqs, per-object mean)")
    table(set(all_seqs), "Full own-coverage (not directly comparable across trackers)")


def ho3d_tables(lines):
    comp = complete_sequences(f"{ROOT}/ho3d", HO3D_DATA, "rgb")
    lines.append("## HO3D (ECCV-final config: cluster-RANSAC mad, tapir@320)\n")
    lines.append("Sequences completed per tracker: "
                 + ", ".join(f"{t}={len(comp[t])}" for t in TRACKERS) + "\n")

    # metrics recomputed from verified metadata (summary_results.txt only
    # covers each run's newly-processed sequences, so it has holes)
    per_seq = {t: {} for t in TRACKERS}
    recomputed = f"{ROOT}/ho3d_recomputed_metrics.csv"
    if os.path.exists(recomputed):
        for r in csv.DictReader(open(recomputed)):
            t, name = r["tracker"], r["seq"]
            if t in per_seq and name in comp.get(t, set()):
                per_seq[t][name] = (
                    float(r["add_s_auc"]),
                    float(r["add_auc"]),
                    np.nan,  # mesh CD not recomputed
                )

    all_seqs = sorted(set().union(*[set(per_seq[t]) for t in TRACKERS]))
    lines.append("Per-sequence ADD AUC (* = ECCV 8-video subset):\n")
    header = "| seq | " + " | ".join(TRACKERS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRACKERS) + 1))
    for s in all_seqs:
        mark = "*" if s in ECCV8 else ""
        row = [f"{per_seq[t][s][1]:.1f}" if s in per_seq[t] else "-" for t in TRACKERS]
        lines.append(f"| {s}{mark} | " + " | ".join(row) + " |")
    lines.append("")

    for subset, title in [
        (ECCV8, "ECCV 8-video subset mean (comparable to published 94.85 / 78.92)"),
        (set(all_seqs), "Full-set mean (own coverage)"),
    ]:
        lines.append(f"### {title}\n")
        lines.append("| tracker | ADD-S AUC | ADD AUC | mesh CD (cm) | seqs |")
        lines.append("|---|---|---|---|---|")
        for t in TRACKERS:
            vals = [v for s, v in per_seq[t].items() if s in subset]
            if not vals:
                lines.append(f"| {t} | - | - | - | 0 |")
                continue
            arr = np.array(vals)
            cd = np.nanmean(arr[:, 2])
            lines.append(
                f"| {t} | {arr[:,0].mean():.2f} | {arr[:,1].mean():.2f} | "
                f"{cd:.2f} | {len(vals)} |"
            )
        lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/final_report.md")
    args = ap.parse_args()

    lines = ["# Tracker pose-benchmark results (ECCV configs, verified-complete runs only)\n"]
    ycb_tables(lines)
    ho3d_tables(lines)
    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
