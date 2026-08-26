"""Relocalization strip from a dense_track_test npz.

Usage: reentry_from_npz.py <tracker> <t_before> <t_occ> <t_snap> <t_after>
"""
import os
import sys

import cv2
import numpy as np
from matplotlib import cm

DATA = "/home/justin/data/test"
OUT = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
NAME = sys.argv[1] if len(sys.argv) > 1 else "tapnext"
TS = [int(a) for a in sys.argv[2:6]] if len(sys.argv) > 5 else [300, 350, 395, 440]
T_BEFORE, T_OCC, T_SNAP, T_AFTER = TS
S = 2
DOT_R = 7

d = np.load(f"{SCRATCH}/dense_tracks_{NAME}.npz")
tr, vis, q = d["tracks"], d["visibles"], d["queries"]
n = len(q)

g = q[:, 0] / 640 * 0.5 + q[:, 1] / 480 * 0.5
g = (g - g.min()) / max(np.ptp(g), 1e-6)
COLS = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        for c in cm.gist_rainbow(g)[:, :3]]

# shared crop over the object's positions at the shown times
pp = np.vstack([tr[t][vis[t]] for t in (T_BEFORE, T_SNAP, T_AFTER)])
cx0, cy0 = np.maximum(pp.min(0) - 60, 0).astype(int)
cx1, cy1 = pp.max(0) + 60
w, h = cx1 - cx0, cy1 - cy0
if w / h > 4 / 3:
    cy0, cy1 = cy0 - (w * 3 / 4 - h) / 2, cy1 + (w * 3 / 4 - h) / 2
else:
    cx0, cx1 = cx0 - (h * 4 / 3 - w) / 2, cx1 + (h * 4 / 3 - w) / 2
cx0, cy0 = max(0, int(cx0)), max(0, int(cy0))
cx1, cy1 = min(640, int(cx1)), min(480, int(cy1))
print("crop", (cx0, cy0), (cx1, cy1))


def render(T, mode, out_name):
    im = cv2.imread(f"{DATA}/rgb/{T:06d}.png")
    canvas = cv2.resize(im, None, fx=S, fy=S, interpolation=cv2.INTER_LANCZOS4)
    if mode == "ghost":
        # freeze all ghosts at the object's last fully-visible configuration
        vf = vis.mean(1)
        t_ref = max(t for t in range(T + 1) if vf[t] >= 0.6)
        ov = canvas.copy()
        for j in range(n):
            if not vis[t_ref, j] or not np.isfinite(tr[t_ref, j]).all():
                continue
            x, y = (tr[t_ref, j] * S).astype(int)
            c = tuple(int(0.6 * v + 0.4 * 150) for v in COLS[j])
            cv2.circle(ov, (x, y), DOT_R, c, 2, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.30, ov, 0.70, 0)
    else:
        for j in range(n):
            if not vis[T, j] or not np.isfinite(tr[T, j]).all():
                continue
            x, y = (tr[T, j] * S).astype(int)
            cv2.circle(canvas, (x, y), DOT_R, COLS[j], -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), DOT_R, (255, 255, 255), 1, cv2.LINE_AA)
    crop = canvas[cy0 * S:cy1 * S, cx0 * S:cx1 * S]
    cv2.imwrite(f"{OUT}/{out_name}", crop)
    print("saved", out_name, f"(visible {vis[T].mean():.2f})")
    return crop


tag = f"test_{NAME}_{T_BEFORE}"
tiles = [
    render(T_BEFORE, "dots", f"reloc_{tag}_1_tracked.png"),
    render(T_OCC, "ghost", f"reloc_{tag}_2_occluded.png"),
    render(T_SNAP, "dots", f"reloc_{tag}_3_snap.png"),
    render(T_AFTER, "dots", f"reloc_{tag}_4_resumed.png"),
]
hgt = min(t.shape[0] for t in tiles)
gap = np.full((hgt, 10, 3), 255, np.uint8)
strip = np.concatenate(sum(([t[:hgt], gap] for t in tiles), [])[:-1], axis=1)
bar = np.full((26, strip.shape[1], 3), 245, np.uint8)
span = T_AFTER - T_BEFORE
occ = vis.mean(1) < 0.08
t = T_BEFORE
while t <= T_AFTER:
    if occ[t]:
        s0 = t
        while t <= T_AFTER and occ[t]:
            t += 1
        x0 = int((s0 - T_BEFORE) / span * strip.shape[1])
        x1 = int((t - T_BEFORE) / span * strip.shape[1])
        bar[:, x0:x1] = (170, 170, 170)
    else:
        t += 1
for T in TS:
    x = int((T - T_BEFORE) / span * (strip.shape[1] - 4))
    bar[:, x:x + 4] = (60, 120, 230)
strip = np.concatenate([strip, bar], axis=0)
cv2.imwrite(f"{OUT}/reloc_{tag}_strip.png", strip)
print("saved", f"{OUT}/reloc_{tag}_strip.png")
