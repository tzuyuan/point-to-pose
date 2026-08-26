"""Point-sampling visualization: at a sampling event, newly sampled query
points (amber rings + cross) vs already-tracked points (small dots), with the
object mask contour.

Usage: panel_sampling.py <seq> <t_event> [<t_event> ...]
       panel_sampling.py <seq> events   # list sampling-event frames
"""
import sys

import os

import cv2
import numpy as np
from matplotlib import cm

from common import Run, OUT_DIR, upscale

CONTOUR = os.environ.get("CONTOUR", "1") != "0"

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP14"
S = 2
NEW_C = (0, 176, 255)      # amber (BGR) for freshly sampled points
OLD_C = (98, 211, 60)      # green for existing tracked points

r = Run(f"ho3d_{SEQ}" if not SEQ.startswith("ycb") else SEQ)
counts = (r.d["track2d_lengths"] // 2).astype(int)
growth = np.diff(counts, prepend=counts[0])
events = np.where(growth > 0)[0]

if len(sys.argv) > 2 and sys.argv[2] == "events":
    print("sampling events (t, n_new):",
          [(int(t), int(growth[t])) for t in events])
    sys.exit(0)

for t in [int(a) for a in sys.argv[2:]] or [int(events[len(events) // 2])]:
    if growth[t] <= 0:
        near = events[np.argmin(np.abs(events - t))]
        print(f"t={t} has no sampling; using nearest event t={near}")
        t = int(near)
    p = r.track2d(t)
    vis = r.visibles(t)
    n_new = int(growth[t])
    new_idx = np.arange(len(p) - n_new, len(p))
    old_idx = np.where(vis[: len(p) - n_new])[0]

    canvas = upscale(r.rgb(t), S)

    # light mask contour ("sampling inside the mask")
    m = r.mask(t, 0)
    if CONTOUR and m is not None:
        mu = cv2.resize(m.astype(np.uint8), None, fx=S, fy=S,
                        interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)

    # only the freshly sampled points, rainbow-colored like the trail figure
    q = p[new_idx]
    ok = np.isfinite(q).all(1)
    g = q[:, 0] / r.W * 0.75 + q[:, 1] / r.H * 0.25
    lo, hi = np.nanpercentile(g[ok], 3), np.nanpercentile(g[ok], 97)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    cols = [(int(c[2]*255), int(c[1]*255), int(c[0]*255))
            for c in cm.gist_rainbow(g)[:, :3]]
    for k, j in enumerate(new_idx):
        if not ok[k]:
            continue
        x, y = (p[j] * S).astype(int)
        cv2.circle(canvas, (x, y), 8, cols[k], -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)

    out = f"{OUT_DIR}/sampling_{SEQ}_t{t}.png"
    cv2.imwrite(out, canvas)
    print(f"saved {out} ({n_new} new samples, {len(old_idx)} existing visible)")
