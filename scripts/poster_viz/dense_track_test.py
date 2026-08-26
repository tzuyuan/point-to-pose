"""Dense-point tracking on a recorded RGB sequence (CoTracker-demo style).

Seeds a dense grid of query points on the object at frame 0 (isolated via
depth), streams the sequence through a registered tracker, saves tracks to
npz, and writes a rainbow overlay video.

Usage: dense_track_test.py [tracker] [grid_step_px]
"""
import os
import sys
import time
import types

import cv2
import numpy as np

sys.path.insert(0, "/home/justin/code/point-to-pose")

DATA = "/home/justin/data/test"
OUT = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
TRACKER_NAME = sys.argv[1] if len(sys.argv) > 1 else "tapnext"
GRID = int(sys.argv[2]) if len(sys.argv) > 2 else 9

CONFIGS = {
    "tapnext": dict(device="cuda",
                    checkpoint_path="checkpoints/tapnext/tapnextpp_ckpt.pt"),
    "tapir": dict(device="cuda", resize_height=256, resize_width=256,
                  checkpoint_path="checkpoints/tapir/causal_bootstapir_checkpoint.pt"),
    "litetracker": dict(device="cuda",
                        checkpoint_path="/home/justin/code/co-tracker-realtime/checkpoints/scaled_online.pth"),
}

n_frames = len([f for f in os.listdir(f"{DATA}/rgb") if f.endswith(".png")])
print(f"{n_frames} frames, tracker={TRACKER_NAME}, grid={GRID}px")


def frame_ns(rgb, fid):
    f = types.SimpleNamespace()
    f.rgb = rgb
    f.id = fid
    return f


def rgb_at(t):
    return cv2.cvtColor(cv2.imread(f"{DATA}/rgb/{t:06d}.png"), cv2.COLOR_BGR2RGB)


# ---- dense queries on the object at t=0 (GrabCut mask, verified visually) ----
mask = np.load(f"{SCRATCH}/../toy_mask0.npy")
mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
ys, xs = np.nonzero(mask)
print(f"object mask: {mask.sum()} px, bbox x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}")
cv2.imwrite(f"{SCRATCH}/test_mask0.png", mask * 255)

gy, gx = np.mgrid[0:480:GRID, 0:640:GRID]
keep = mask[gy.ravel(), gx.ravel()] > 0
pts0 = np.stack([gx.ravel()[keep], gy.ravel()[keep]], 1).astype(np.float32)
print(f"{len(pts0)} dense query points")

# ---- track ----
from point2pose.core.module_registry import TRACKER  # noqa: E402
import point2pose.modules.tracker  # noqa: E402,F401  (registers trackers)

os.chdir("/home/justin/code/point-to-pose")
tracker = TRACKER.get(TRACKER_NAME)(dict(CONFIGS[TRACKER_NAME]))
rgb0 = rgb_at(0)
tracker.initialize(frame_ns(rgb0, 0))
tracker.add_query_points(frame_ns(rgb0, 0), pts0)

all_tracks = np.zeros((n_frames, len(pts0), 2), np.float32)
all_vis = np.zeros((n_frames, len(pts0)), bool)
t0 = time.time()
for t in range(n_frames):
    tracks, unc, vis = tracker.track_once(frame_ns(rgb_at(t), t))
    all_tracks[t] = tracks[: len(pts0)]
    all_vis[t] = vis.astype(bool).reshape(-1)[: len(pts0)]
    if t % 100 == 0:
        print(f"t={t} visible={all_vis[t].mean():.2f} ({time.time()-t0:.0f}s)")

np.savez_compressed(f"{SCRATCH}/dense_tracks_{TRACKER_NAME}.npz",
                    tracks=all_tracks, visibles=all_vis, queries=pts0)
print("tracking done", f"{time.time()-t0:.0f}s")

# ---- rainbow colors by query position ----
g = pts0[:, 0] / 640 * 0.5 + pts0[:, 1] / 480 * 0.5
g = (g - g.min()) / max(np.ptp(g), 1e-6)
from matplotlib import cm  # noqa: E402
COLS = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        for c in cm.gist_rainbow(g)[:, :3]]

# ---- overlay video ----
S = 1
vw = cv2.VideoWriter(f"{OUT}/dense_track_{TRACKER_NAME}.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
for t in range(n_frames):
    im = cv2.imread(f"{DATA}/rgb/{t:06d}.png")
    for j in range(len(pts0)):
        if not all_vis[t, j] or not np.isfinite(all_tracks[t, j]).all():
            continue
        x, y = all_tracks[t, j].astype(int)
        if 0 <= x < 640 and 0 <= y < 480:
            cv2.circle(im, (x, y), 4, COLS[j], -1, cv2.LINE_AA)
            cv2.circle(im, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
    vw.write(im)
vw.release()
print("saved", f"{OUT}/dense_track_{TRACKER_NAME}.mp4")

# ---- report visibility over time (find the occlusion window) ----
vf = all_vis.mean(1)
occ = vf < 0.08
runs = []
t = 0
while t < n_frames:
    if occ[t]:
        s = t
        while t < n_frames and occ[t]:
            t += 1
        if t - s >= 15:
            runs.append((s, t))
    else:
        t += 1
print("visibility mean:", np.round(vf.mean(), 2),
      "| occlusion windows:", runs)
