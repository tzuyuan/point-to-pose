"""Benchmark point-tracking ACCURACY against GT object poses (e2d protocol).

Same protocol as the ECCV texture-dependency analysis: query points sampled on
the object mask in the first frame are lifted to the OBJECT frame using the
depth image and the GT pose (T_cam_obj), then reprojected into every frame with
that frame's GT pose. The reprojection is the ground-truth track; tracker
output is compared against it.

GT visibility of a point at frame t: projected inside the image, positive
depth, and consistent with the measured depth map (within --occl-tol meters;
falls back to the dilated GT mask where depth is missing). Frames are scored
only where the object is in the image (is_obj_in_image.npy) and the GT pose is
finite.

Metrics (TAP-Vid conventions, over GT-visible points):
  - e2d mean / median pixel error
  - delta_avg: fraction of points within {1,2,4,8,16} px, averaged
  - OA: predicted-visibility accuracy vs GT visibility
  - latency (mean / p95 ms)

Outputs per sequence: results.json, summary.md, per_frame_<tracker>.csv,
e2d_timeseries.png, and an overlay video per tracker (GT = small cross,
prediction = dot; green if within 8 px, red otherwise, gray if predicted
occluded).

Example:
    python scripts/benchmark_trackers_gt.py \
        --data /home/justin/data/YCBMultiTrack_recalib/006_mustard_bottle \
        --obj 006_mustard_bottle \
        --trackers tapir,tapnext,trackon,litetracker,cotracker3_online \
        --n-frames 800 --n-points 24
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from benchmark_trackers import (  # noqa: E402
    DEFAULT_CONFIGS,
    make_frame,
    sample_mask_points,
)

DELTA_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)

# dataviz default categorical palette (light mode, fixed slot order)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948"]


def load_sequence(data_dir, obj, n_frames, stride):
    rgb = sorted(glob.glob(os.path.join(data_dir, "jpg", "*")))
    mask = sorted(glob.glob(os.path.join(data_dir, "masks", obj, "*")))
    depth = sorted(glob.glob(os.path.join(data_dir, "depth", "*")))
    pose = sorted(glob.glob(os.path.join(data_dir, "annotated_poses", obj, "*.txt")))
    assert len(rgb) == len(mask) == len(depth) == len(pose), (
        f"count mismatch rgb={len(rgb)} mask={len(mask)} "
        f"depth={len(depth)} pose={len(pose)}"
    )
    in_image_path = os.path.join(
        data_dir, "is_obj_in_image_labels", obj, "is_obj_in_image.npy"
    )
    in_image = (
        np.load(in_image_path).astype(bool)
        if os.path.exists(in_image_path)
        else np.ones(len(rgb), dtype=bool)
    )
    sl = slice(None, None, stride)
    rgb, mask, depth, pose = rgb[sl], mask[sl], depth[sl], pose[sl]
    in_image = in_image[sl]
    K = np.loadtxt(os.path.join(data_dir, "cam_K.txt"))
    return (
        rgb[:n_frames],
        mask[:n_frames],
        depth[:n_frames],
        pose[:n_frames],
        in_image[:n_frames],
        K,
    )


def lift_points_to_object(pts_xy, depth0, T0, K, depth_factor=1000.0):
    """[N,2] pixel points + depth + GT pose (T_cam_obj) -> [N,3] object-frame
    points and a validity mask (depth present at the query pixel)."""
    u = pts_xy[:, 0].round().astype(int)
    v = pts_xy[:, 1].round().astype(int)
    z = depth0[v, u].astype(np.float64) / depth_factor
    valid = z > 0
    ones = np.ones_like(z)
    rays = np.linalg.inv(K) @ np.stack([pts_xy[:, 0], pts_xy[:, 1], ones])
    x_cam = rays * z  # [3, N]
    R, t = T0[:3, :3], T0[:3, 3:4]
    x_obj = R.T @ (x_cam - t)
    return x_obj.T, valid


def project_points(x_obj, T, K):
    """[N,3] object points + T_cam_obj -> [N,2] pixels and [N] camera z."""
    x_cam = (T[:3, :3] @ x_obj.T + T[:3, 3:4]).T
    z = x_cam[:, 2]
    uv = (K @ x_cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
    return uv, z


def gt_visibility(uv, z, depth_t, mask_t, occl_tol, depth_factor=1000.0):
    """GT-visible: in image, in front of camera, depth-consistent (or inside
    the dilated mask where depth is missing)."""
    h, w = depth_t.shape
    u = uv[:, 0].round().astype(int)
    v = uv[:, 1].round().astype(int)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    vis = np.zeros(len(uv), dtype=bool)
    ui, vi = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    d_meas = depth_t[vi, ui].astype(np.float64) / depth_factor
    mask_dil = cv2.dilate((mask_t > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    depth_ok = (d_meas > 0) & (np.abs(d_meas - z) < occl_tol)
    mask_ok = (d_meas == 0) & (mask_dil[vi, ui] > 0)
    vis[in_img] = (depth_ok | mask_ok)[in_img]
    return vis


def run_tracker_gt(name, config, seq, x_obj, pts0, out_dir, save_video=True):
    from point2pose.core.module_registry import TRACKER

    rgb_files, mask_files, depth_files, pose_files, in_image, K = seq
    rgb0 = cv2.cvtColor(cv2.imread(rgb_files[0]), cv2.COLOR_BGR2RGB)
    h, w = rgb0.shape[:2]

    tracker = TRACKER.get(name)(dict(config))
    tracker.initialize(make_frame(rgb0, 0))
    tracker.add_query_points(make_frame(rgb0, 0), pts0)

    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            os.path.join(out_dir, f"{name}_gt.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h),
        )

    per_frame = []
    all_err, all_within = [], {thr: [] for thr in DELTA_THRESHOLDS}
    oa_hits, oa_total = 0, 0
    latencies = []

    for t, rf in enumerate(rgb_files):
        rgb = cv2.cvtColor(cv2.imread(rf), cv2.COLOR_BGR2RGB)

        start = time.time()
        tracks, unc, vis = tracker.track_once(make_frame(rgb, t))
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

        T = np.loadtxt(pose_files[t])
        has_gt = bool(in_image[t]) and np.all(np.isfinite(T))
        if not has_gt:
            per_frame.append((t, 0, np.nan, np.nan))
            continue

        depth_t = cv2.imread(depth_files[t], cv2.IMREAD_UNCHANGED)
        mask_t = cv2.imread(mask_files[t], cv2.IMREAD_GRAYSCALE)
        uv_gt, z_gt = project_points(x_obj, T, K)
        gt_vis = gt_visibility(uv_gt, z_gt, depth_t, mask_t, occl_tol=0.03)

        pred_vis = vis.astype(bool).reshape(-1)
        err = np.linalg.norm(tracks - uv_gt, axis=1)

        oa_hits += int((pred_vis == gt_vis).sum())
        oa_total += len(gt_vis)

        if gt_vis.sum() > 0:
            e = err[gt_vis]
            all_err.append(e)
            for thr in DELTA_THRESHOLDS:
                all_within[thr].append((e < thr).astype(np.float64))
            per_frame.append(
                (t, int(gt_vis.sum()), float(e.mean()), float(np.median(e)))
            )
        else:
            per_frame.append((t, 0, np.nan, np.nan))

        if writer is not None:
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for i in range(len(tracks)):
                gx, gy = uv_gt[i].round().astype(int)
                px, py = tracks[i].round().astype(int)
                if gt_vis[i] and 0 <= gx < w and 0 <= gy < h:
                    cv2.drawMarker(frame_bgr, (gx, gy), (255, 255, 0),
                                   cv2.MARKER_CROSS, 8, 1)
                if 0 <= px < w and 0 <= py < h:
                    color = (
                        (128, 128, 128) if not pred_vis[i]
                        else ((0, 200, 0) if err[i] < 8 else (0, 0, 255))
                    )
                    cv2.circle(frame_bgr, (px, py), 4, color,
                               -1 if pred_vis[i] else 1)
            cv2.putText(frame_bgr, f"{name} t={t}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    with open(os.path.join(out_dir, f"per_frame_{name}.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "n_gt_visible", "e2d_mean_px", "e2d_median_px"])
        wcsv.writerows(per_frame)

    err_cat = np.concatenate(all_err) if all_err else np.array([np.nan])
    lat = np.array(latencies[5:]) * 1000
    deltas = {
        f"delta_{int(thr)}px": float(np.concatenate(v).mean())
        for thr, v in all_within.items()
    }
    result = {
        "tracker": name,
        "e2d_mean_px": float(err_cat.mean()),
        "e2d_median_px": float(np.median(err_cat)),
        **deltas,
        "delta_avg": float(np.mean(list(deltas.values()))),
        "occlusion_accuracy": float(oa_hits / max(oa_total, 1)),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "n_eval_point_frames": int(sum(len(e) for e in all_err)),
    }
    del tracker
    torch.cuda.empty_cache()
    return result, per_frame


def plot_timeseries(per_frame_by_tracker, out_path, smooth=15):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    for i, (name, rows) in enumerate(per_frame_by_tracker.items()):
        frames = np.array([r[0] for r in rows])
        med = np.array([r[3] for r in rows], dtype=np.float64)
        ok = np.isfinite(med)
        med_s = np.convolve(
            np.where(ok, med, np.nan), np.ones(smooth) / smooth, mode="same"
        )
        ax.plot(frames, med_s, lw=2, label=name,
                color=SERIES_COLORS[i % len(SERIES_COLORS)])
    ax.set_xlabel("frame")
    ax.set_ylabel("median e2d error (px)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Tracking error vs GT reprojection (rolling median)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--trackers", default="tapir,tapnext,trackon,litetracker")
    ap.add_argument("--n-frames", type=int, default=800)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=24)
    ap.add_argument("--occl-tol", type=float, default=0.03)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="debug/tracker_benchmark_gt")
    args = ap.parse_args()

    seq = load_sequence(args.data, args.obj, args.n_frames, args.frame_stride)
    rgb_files, mask_files, depth_files, pose_files, in_image, K = seq
    print(f"GT benchmark on {len(rgb_files)} frames from {args.data}")

    # query points: sampled on the frame-0 mask, keep only those with depth
    rgb0_mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    depth0 = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    T0 = np.loadtxt(pose_files[0])
    pts = sample_mask_points(rgb0_mask, args.n_points * 2)
    x_obj, valid = lift_points_to_object(pts, depth0, T0, K)
    pts0, x_obj = pts[valid][: args.n_points], x_obj[valid][: args.n_points]
    print(f"{len(pts0)} query points with valid depth")

    # sanity: reprojection at the query frame must be ~0
    uv0, _ = project_points(x_obj, T0, K)
    print(f"query-frame reprojection error: {np.abs(uv0 - pts0).max():.2e} px")

    os.makedirs(args.out, exist_ok=True)

    import point2pose.modules.tracker  # noqa: F401

    results, per_frame_by_tracker = [], {}
    for name in args.trackers.split(","):
        name = name.strip()
        print(f"=== {name} ===")
        res, per_frame = run_tracker_gt(
            name, DEFAULT_CONFIGS[name], seq, x_obj, pts0, args.out,
            save_video=not args.no_video,
        )
        print(json.dumps(res, indent=2))
        results.append(res)
        per_frame_by_tracker[name] = per_frame

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_timeseries(
        per_frame_by_tracker, os.path.join(args.out, "e2d_timeseries.png")
    )

    lines = [
        "| tracker | e2d mean | e2d med | d1 | d2 | d4 | d8 | d16 | d_avg | OA | ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['tracker']} | {r['e2d_mean_px']:.2f} | "
            f"{r['e2d_median_px']:.2f} | {r['delta_1px']:.3f} | "
            f"{r['delta_2px']:.3f} | {r['delta_4px']:.3f} | "
            f"{r['delta_8px']:.3f} | {r['delta_16px']:.3f} | "
            f"{r['delta_avg']:.3f} | {r['occlusion_accuracy']:.3f} | "
            f"{r['latency_ms_mean']:.1f} |"
        )
    summary = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
