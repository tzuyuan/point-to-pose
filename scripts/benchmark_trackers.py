"""Benchmark registered point trackers on a recorded RGB sequence with masks.

Samples query points inside the object mask of the first frame, streams the
sequence through each tracker via the common Tracker interface, and reports:
  - per-frame latency (mean / p95, excluding warmup)
  - mask-inlier rate: fraction of predicted-visible points that land inside
    the (dilated) GT object mask, averaged over frames
  - survival rate: fraction of points inside the mask in the last 50 frames
  - visibility rate

Also writes an overlay video per tracker (green = visible & in mask,
red = visible & outside mask, gray = predicted occluded) and a summary
markdown/json file.

Example:
    python scripts/benchmark_trackers.py \
        --data /home/justin/data/YCBMultiTrack_recalib/006_mustard_bottle \
        --obj 006_mustard_bottle \
        --trackers tapir,tapnext,trackon,litetracker,cotracker3_online \
        --n-frames 800 --n-points 24
"""

import argparse
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


DEFAULT_CONFIGS = {
    "tapir": dict(
        device="cuda",
        resize_height=256,
        resize_width=256,
        checkpoint_path="checkpoints/tapir/causal_bootstapir_checkpoint.pt",
    ),
    "tapnext": dict(
        device="cuda",
        checkpoint_path="checkpoints/tapnext/tapnextpp_ckpt.pt",
    ),
    "trackon": dict(
        device="cuda",
        checkpoint_path="checkpoints/trackon/trackon2_dinov2_checkpoint.pt",
    ),
    "litetracker": dict(
        device="cuda",
        checkpoint_path="/home/justin/code/co-tracker-realtime/checkpoints/scaled_online.pth",
    ),
    "cotracker3_online": dict(
        device="cuda",
        resize_height=480,
        resize_width=640,
        window_len=16,
        checkpoint_path="/home/justin/code/co-tracker-realtime/checkpoints/scaled_online.pth",
    ),
}


def make_frame(rgb, fid):
    f = types.SimpleNamespace()
    f.rgb = rgb
    f.id = fid
    return f


def sample_mask_points(mask, n_points, margin_px=6, seed=0):
    """Uniformly sample n_points inside the eroded mask, [x, y]."""
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel)
    ys, xs = np.nonzero(eroded)
    if len(xs) < n_points:
        ys, xs = np.nonzero(mask > 0)
    rng = np.random.default_rng(seed)
    # farthest-point-style selection for spatial spread
    idx = [rng.integers(len(xs))]
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    d = np.full(len(pts), np.inf)
    for _ in range(n_points - 1):
        d = np.minimum(d, np.linalg.norm(pts - pts[idx[-1]], axis=1))
        idx.append(int(np.argmax(d)))
    return pts[idx]


def run_tracker(name, config, rgb_files, mask_files, n_points, out_dir):
    from point2pose.core.module_registry import TRACKER

    rgb0 = cv2.cvtColor(cv2.imread(rgb_files[0]), cv2.COLOR_BGR2RGB)
    mask0 = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    pts0 = sample_mask_points(mask0, n_points)

    tracker = TRACKER.get(name)(dict(config))
    tracker.initialize(make_frame(rgb0, 0))
    tracker.add_query_points(make_frame(rgb0, 0), pts0)

    h, w = rgb0.shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(out_dir, f"{name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h),
    )
    dilate_kernel = np.ones((11, 11), np.uint8)

    latencies, inlier_rates, vis_rates = [], [], []
    survival_window = []
    for t, (rf, mf) in enumerate(zip(rgb_files, mask_files)):
        rgb = cv2.cvtColor(cv2.imread(rf), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)

        start = time.time()
        tracks, unc, vis = tracker.track_once(make_frame(rgb, t))
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

        mask_dil = cv2.dilate((mask > 0).astype(np.uint8), dilate_kernel)
        xi = np.clip(tracks[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(tracks[:, 1].round().astype(int), 0, h - 1)
        in_mask = mask_dil[yi, xi] > 0
        vis = vis.astype(bool).reshape(-1)

        if mask_dil.sum() > 0:  # only score frames where the object is present
            if vis.sum() > 0:
                inlier_rates.append(in_mask[vis].mean())
            vis_rates.append(vis.mean())
            if t >= len(rgb_files) - 50:
                survival_window.append(in_mask.mean())

        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for i in range(len(tracks)):
            color = (
                (128, 128, 128)
                if not vis[i]
                else ((0, 200, 0) if in_mask[i] else (0, 0, 255))
            )
            cv2.circle(frame_bgr, (xi[i], yi[i]), 4, color, -1 if vis[i] else 1)
        cv2.putText(
            frame_bgr, f"{name} t={t}", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame_bgr)

    writer.release()
    lat = np.array(latencies[5:]) * 1000
    result = {
        "tracker": name,
        "frames": len(rgb_files),
        "points": int(n_points),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / lat.mean()),
        "mask_inlier_rate": float(np.mean(inlier_rates)),
        "survival_rate_last50": float(np.mean(survival_window)),
        "visibility_rate": float(np.mean(vis_rates)),
    }
    del tracker
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sequence dir with jpg/ and masks/")
    ap.add_argument("--obj", required=True, help="mask subdirectory name")
    ap.add_argument("--trackers", default="tapir,tapnext,trackon,litetracker")
    ap.add_argument("--n-frames", type=int, default=800)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=24)
    ap.add_argument("--out", default="debug/tracker_benchmark")
    args = ap.parse_args()

    rgb_files = sorted(glob.glob(os.path.join(args.data, "jpg", "*")))
    mask_files = sorted(
        glob.glob(os.path.join(args.data, "masks", args.obj, "*"))
    )
    assert len(rgb_files) == len(mask_files), "rgb/mask count mismatch"
    rgb_files = rgb_files[:: args.frame_stride][: args.n_frames]
    mask_files = mask_files[:: args.frame_stride][: args.n_frames]
    print(f"Benchmarking on {len(rgb_files)} frames from {args.data}")

    os.makedirs(args.out, exist_ok=True)

    import point2pose.modules.tracker  # noqa: F401  (registers trackers)

    results = []
    for name in args.trackers.split(","):
        name = name.strip()
        print(f"=== {name} ===")
        res = run_tracker(
            name,
            DEFAULT_CONFIGS[name],
            rgb_files,
            mask_files,
            args.n_points,
            args.out,
        )
        print(json.dumps(res, indent=2))
        results.append(res)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # markdown summary
    lines = [
        "| tracker | latency (ms) | p95 | FPS | mask inlier | survival(last50) | visible |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['tracker']} | {r['latency_ms_mean']:.1f} | "
            f"{r['latency_ms_p95']:.1f} | {r['fps']:.0f} | "
            f"{r['mask_inlier_rate']:.3f} | {r['survival_rate_last50']:.3f} | "
            f"{r['visibility_rate']:.3f} |"
        )
    summary = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
