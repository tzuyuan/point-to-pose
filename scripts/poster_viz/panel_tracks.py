"""Panel 2: 2D point tracker — CoTracker-style rainbow trails on a real frame.

Tracks are colored with a rainbow colormap by their spatial position
(like CoTracker demos), drawn as thick anti-aliased trails with an alpha
ramp and dot endpoints with white rings.

Usage: panel_tracks.py <dataset> <frame|auto> [trail_len] [obj <id>]

With `obj <id>` only that object's tracks are drawn, with the caps lifted
so (nearly) every currently-visible point shows.
"""
import sys

import cv2
import numpy as np
from matplotlib import cm

from common import Run, OUT_DIR, upscale

args = sys.argv[1:]
ONLY_OBJ = None
if "obj" in args:
    k = args.index("obj")
    ONLY_OBJ = int(args[k + 1])
    args = args[:k] + args[k + 2:]

DS = args[0] if len(args) > 0 else "ycb3"
ARG_T = args[1] if len(args) > 1 else "auto"
TRAIL = int(args[2]) if len(args) > 2 else 48
STEP = 2
MAX_DOTS_PER_OBJ = 400 if ONLY_OBJ is not None else 40
import os  # noqa: E402
MAX_DOTS_PER_OBJ = int(os.environ.get("MAX_DOTS", MAX_DOTS_PER_OBJ))
ONLY_TRAILS = os.environ.get("ONLY_TRAILS") == "1"
MAX_TRAILS_PER_OBJ = int(os.environ.get("MAX_TRAILS", 200 if ONLY_OBJ is not None else 30))
MAX_STEP_PX = 12 * STEP
MIN_VIS_FRAC = float(os.environ.get("MIN_VIS_FRAC",
                                    0.6 if ONLY_OBJ is not None else 0.8))
MIN_TRAIL_DISP = float(os.environ.get("MIN_TRAIL_DISP",
                                      15.0 if ONLY_OBJ is not None else 25.0))
S = 2
LINE_W = 4 if ONLY_OBJ is not None else 5   # trail thickness at 2x
DOT_R = 7 if ONLY_OBJ is not None else 8

r = Run(DS)


def pick_frames(n=4):
    """Frames with highest median visible-track speed (trailing window)."""
    speed = np.zeros(r.n_frames)
    prev = None
    for t in range(r.n_frames):
        p = r.track2d(t)
        v = r.visibles(t)
        if prev is not None and len(prev) == len(p):
            dp = np.linalg.norm(p - prev, axis=1)
            ok = np.isfinite(dp) & v
            speed[t] = np.median(dp[ok]) if ok.any() else 0
        prev = p
    trail = np.convolve(speed, np.ones(45) / 45, mode="same")
    order = np.argsort(-trail)
    picked = []
    for t in order:
        if t < TRAIL + 5:
            continue
        if all(abs(t - q) > 120 for q in picked):
            picked.append(int(t))
        if len(picked) == n:
            break
    return picked


def rainbow_colors(pts, subset=None):
    """CoTracker-style rainbow: colormap over normalized x+y position.

    Normalization spans `subset` (the drawn tracks) so the full spectrum
    stretches across whatever is actually visualized.
    """
    g = pts[:, 0] / r.W * 0.75 + pts[:, 1] / r.H * 0.25
    ref = g[subset] if subset is not None and len(subset) else g
    lo, hi = np.nanpercentile(ref, 3), np.nanpercentile(ref, 97)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    cols = (cm.gist_rainbow(g)[:, :3] * 255).astype(np.uint8)
    return [tuple(int(c) for c in col[::-1]) for col in cols]  # RGB->BGR


def render(T):
    ids = r.track_obj_ids_voted()[: r.n_tracks(T)]
    n_t = len(ids)
    vis = r.visibles(T)
    val = r.valid(T)

    hist_frames = list(range(max(0, T - TRAIL), T + 1, STEP))
    if hist_frames[-1] != T:
        hist_frames.append(T)
    nh = len(hist_frames)
    hist = np.full((nh, n_t, 2), np.nan, dtype=np.float32)
    hist_vis = np.zeros((nh, n_t), dtype=bool)
    for i, t in enumerate(hist_frames):
        p = r.track2d(t)
        v = r.visibles(t)
        hist[i, : len(p)] = p
        hist_vis[i, : len(v)] = v

    steps = np.linalg.norm(np.diff(hist, axis=0), axis=2)
    smooth = np.nan_to_num(steps, nan=1e9).max(axis=0) < MAX_STEP_PX
    vis_frac = hist_vis.mean(axis=0)
    disp = np.nan_to_num(np.linalg.norm(hist[-1] - hist[0], axis=1))
    cur = hist[-1]

    dots, trails = [], []
    rng = np.random.default_rng(0)
    obj_list = [ONLY_OBJ] if ONLY_OBJ is not None else range(r.n_obj)
    for o in obj_list:
        base = np.where((ids == o) & val & vis & np.isfinite(cur).all(1))[0]
        if len(base) == 0:
            continue
        med = np.median(cur[base], axis=0)
        dist = np.linalg.norm(cur[base] - med, axis=1)
        mad = np.median(dist) + 1e-6
        keep = base[dist < max(4 * mad, 70.0)]
        tr = keep[(disp[keep] >= MIN_TRAIL_DISP) & smooth[keep]
                  & (vis_frac[keep] >= MIN_VIS_FRAC)]
        tr = tr[np.argsort(-disp[tr])][:MAX_TRAILS_PER_OBJ]
        trails += list(tr)
        rest = np.empty(0, dtype=int) if ONLY_TRAILS else np.setdiff1d(keep, tr)
        if len(rest) > MAX_DOTS_PER_OBJ:
            rest = rng.choice(rest, MAX_DOTS_PER_OBJ, replace=False)
        dots += list(rest) + list(tr)

    print(f"{DS} t={T}: dots {len(dots)}, trails {len(trails)}")
    canvas = upscale(r.rgb(T), S)

    # rainbow color per track, anchored at the track's position at trail start
    anchor = np.where(np.isfinite(hist[0]).all(1)[:, None], hist[0], cur)
    drawn = np.unique(np.array(dots + trails, dtype=int))
    colors = rainbow_colors(anchor, subset=drawn)

    # trails: thick AA polylines, alpha ramps up toward the present
    for i in range(nh - 1):
        a = 0.35 + 0.65 * (i + 1) / (nh - 1)
        seg = canvas.copy()
        drew = False
        for j in trails:
            p0, p1 = hist[i, j] * S, hist[i + 1, j] * S
            if steps[i, j] > MAX_STEP_PX or not np.isfinite([p0, p1]).all():
                continue
            cv2.line(seg, tuple(p0.astype(int)), tuple(p1.astype(int)),
                     colors[j], LINE_W, cv2.LINE_AA)
            drew = True
        if drew:
            canvas = cv2.addWeighted(canvas, 1 - a, seg, a, 0)

    # dots: colored fill + white ring (CoTracker look)
    for j in dots:
        p = cur[j] * S
        if not np.isfinite(p).all():
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(canvas, (x, y), DOT_R, colors[j], -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), DOT_R, (255, 255, 255), 2, cv2.LINE_AA)

    out = f"{OUT_DIR}/panel2_tracks_{DS}_t{T}.png"
    cv2.imwrite(out, canvas)
    print("saved", out)


if ARG_T == "auto":
    for T in pick_frames():
        render(T)
else:
    render(int(ARG_T))
