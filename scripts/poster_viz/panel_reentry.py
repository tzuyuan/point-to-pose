"""Occlusion -> re-entry strip for the 2D point tracker (<=4 frames).

Same tracks keep the same rainbow color across all panels (color = identity).
  panel 1  tracked      filled dots + short trails
  panel 2  occluded     hollow ghost rings at the tracker's predictions
  panel 3  snap-back    same colors land back on the same object parts
  panel 4  resumed      trails flowing again

Usage: panel_reentry.py <dataset> <obj> <t_before> <t_occ> <t_snap> <t_after>
"""
import sys

import cv2
import numpy as np
from matplotlib import cm

from common import Run, OUT_DIR, upscale

DS = sys.argv[1] if len(sys.argv) > 1 else "ycb3"
OBJ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TS = [int(a) for a in sys.argv[3:7]] if len(sys.argv) > 6 else [1090, 1155, 1205, 1245]
T_BEFORE, T_OCC, T_SNAP, T_AFTER = TS
TRAIL = 28
STEP = 2
MAX_PTS = 40
S = 2
DOT_R = 9
LINE_W = 5

r = Run(DS)
ids = r.track_obj_ids_voted()


def positions(t):
    p = r.track2d(t)
    return p, r.visibles(t), r.valid(t)


# ---- pick the proof set: tracks of OBJ visible at t_before AND t_snap ----
pb, vb, valb = positions(T_BEFORE)
ps, vs, vals = positions(T_SNAP)
k = min(len(pb), len(ps))
sel = np.where((ids[:k] == OBJ) & vb[:k] & vs[:k] & valb[:k] & vals[:k]
               & np.isfinite(pb[:k]).all(1) & np.isfinite(ps[:k]).all(1))[0]
# spatial-consistency at both ends
for pts_t in (pb, ps):
    med = np.median(pts_t[sel], axis=0)
    d = np.linalg.norm(pts_t[sel] - med, axis=1)
    sel = sel[d < max(4 * np.median(d), 80.0)]
if len(sel) > MAX_PTS:
    sel = np.random.default_rng(1).choice(sel, MAX_PTS, replace=False)
sel = np.sort(sel)
print(f"proof set: {len(sel)} tracks survive {T_BEFORE} -> {T_SNAP}")

# ---- one color per track, anchored at t_before position ----
g = pb[sel, 0] / r.W * 0.75 + pb[sel, 1] / r.H * 0.25
lo, hi = np.percentile(g, 3), np.percentile(g, 97)
g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
COLS = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        for c in cm.gist_rainbow(g)[:, :3]]

# ---- shared crop covering the object at all four times ----
allp = np.vstack([pb[sel], ps[sel], positions(T_AFTER)[0][sel]])
cx0, cy0 = allp.min(0) - 60
cx1, cy1 = allp.max(0) + 60
# expand to 4:3
w, h = cx1 - cx0, cy1 - cy0
if w / h > 4 / 3:
    pad = w * 3 / 4 - h
    cy0 -= pad / 2
    cy1 += pad / 2
else:
    pad = h * 4 / 3 - w
    cx0 -= pad / 2
    cx1 += pad / 2
cx0, cy0 = max(0, int(cx0)), max(0, int(cy0))
cx1, cy1 = min(r.W, int(cx1)), min(r.H, int(cy1))
print(f"crop: ({cx0},{cy0})-({cx1},{cy1})")


def trails_hist(T):
    frames = list(range(max(0, T - TRAIL), T + 1, STEP))
    if frames[-1] != T:
        frames.append(T)
    hist = np.full((len(frames), len(sel), 2), np.nan, np.float32)
    hvis = np.zeros((len(frames), len(sel)), bool)
    for i, t in enumerate(frames):
        p, v, _ = positions(t)
        m = min(len(p), max(sel) + 1)
        ok = sel[sel < m]
        idx = np.searchsorted(sel, ok)
        hist[i, idx] = p[ok]
        hvis[i, idx] = v[ok]
    return hist, hvis


def render(T, mode, out_name):
    canvas = upscale(r.rgb(T), S)
    p, v, _ = positions(T)

    if mode in ("trail", "after"):
        hist, hvis = trails_hist(T)
        steps = np.linalg.norm(np.diff(hist, axis=0), axis=2)
        nseg = hist.shape[0] - 1
        for i in range(nseg):
            a = 0.35 + 0.65 * (i + 1) / nseg
            seg = canvas.copy()
            drew = False
            for j in range(len(sel)):
                if steps[i, j] > 24 or not (hvis[i, j] and hvis[i + 1, j]):
                    continue
                q0, q1 = hist[i, j] * S, hist[i + 1, j] * S
                if not np.isfinite([q0, q1]).all():
                    continue
                cv2.line(seg, tuple(q0.astype(int)), tuple(q1.astype(int)),
                         COLS[j], LINE_W, cv2.LINE_AA)
                drew = True
            if drew:
                canvas = cv2.addWeighted(canvas, 1 - a, seg, a, 0)

    if mode == "ghost":
        # freeze each ghost at its last-visible position (identities held),
        # desaturated ring so it doesn't read as a live measurement
        ov = canvas.copy()
        for j, tj in enumerate(sel):
            q = None
            for tb in range(T, T_BEFORE - 1, -1):
                pv, vv, _ = positions(tb)
                if tj < len(pv) and vv[tj] and np.isfinite(pv[tj]).all():
                    q = pv[tj]
                    break
            if q is None:
                continue
            x, y = (q * S).astype(int)
            c = tuple(int(0.6 * v + 0.4 * 150) for v in COLS[j])
            cv2.circle(ov, (x, y), DOT_R + 1, c, 3, cv2.LINE_AA)
            cv2.circle(ov, (x, y), DOT_R + 4, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.25, ov, 0.75, 0)

    for j, tj in enumerate(sel):
        if mode == "ghost":
            break
        if tj >= len(p) or not np.isfinite(p[tj]).all():
            continue
        x, y = (p[tj] * S).astype(int)
        if True:
            if tj < len(v) and not v[tj]:
                continue
            cv2.circle(canvas, (x, y), DOT_R, COLS[j], -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), DOT_R, (255, 255, 255), 2, cv2.LINE_AA)

    crop = canvas[cy0 * S:cy1 * S, cx0 * S:cx1 * S]
    out = f"{OUT_DIR}/{out_name}"
    cv2.imwrite(out, crop)
    print("saved", out)
    return crop


tag = f"{DS}_obj{OBJ}_{T_BEFORE}"
t1 = render(T_BEFORE, "trail", f"reentry_{tag}_1_tracked.png")
t2 = render(T_OCC, "ghost", f"reentry_{tag}_2_occluded.png")
t3 = render(T_SNAP, "dots", f"reentry_{tag}_3_snap.png")
t4 = render(T_AFTER, "after", f"reentry_{tag}_4_resumed.png")

# quick preview strip with a timeline bar
hgt = min(t.shape[0] for t in (t1, t2, t3, t4))
tiles = [t[:hgt] for t in (t1, t2, t3, t4)]
gap = np.full((hgt, 10, 3), 255, np.uint8)
strip = np.concatenate(sum(([t, gap] for t in tiles), [])[:-1], axis=1)
bar_h = 26
bar = np.full((bar_h, strip.shape[1], 3), 245, np.uint8)
n_total = T_AFTER - T_BEFORE
x_occ0 = int((T_OCC - 25 - T_BEFORE) / n_total * strip.shape[1])
x_occ1 = int((T_SNAP - 8 - T_BEFORE) / n_total * strip.shape[1])
bar[:, max(0, x_occ0):x_occ1] = (170, 170, 170)
for T in TS:
    x = int((T - T_BEFORE) / n_total * (strip.shape[1] - 4))
    bar[:, x:x + 4] = (60, 120, 230)
strip = np.concatenate([strip, bar], axis=0)
cv2.imwrite(f"{OUT_DIR}/reentry_{tag}_strip.png", strip)
print("saved", f"{OUT_DIR}/reentry_{tag}_strip.png")
