"""Panel 4: multi-hypothesis frame-to-map registration on a real frame.

Projects the object-0 keypoint map bbox under each pose hypothesis:
selected hypothesis in green (solid), rejected in red/gray (dashed-ish).

Usage: panel_hypo.py [scan | <frame>]
"""
import sys

import cv2
import numpy as np

from common import Run, OUT_DIR, upscale

r = Run("ycb3")
S = 2

GREEN = (80, 200, 60)
GRAY = (90, 90, 230)  # BGR red-ish for rejected


def n_clusters(t):
    c = r.d["reg_clusters"][t]
    try:
        return len(c)
    except TypeError:
        return 0


def pose_dist(A, B):
    dt = np.linalg.norm(A[:3, 3] - B[:3, 3])
    cth = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    dr = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    return dt, dr


def map_bbox(t):
    kp = r.keypoints_obj0(t)
    lo = np.percentile(kp, 2, axis=0)
    hi = np.percentile(kp, 98, axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return corners


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_bbox(canvas, T, corners, color, thick, dashed=False):
    pc = (T[:3, :3] @ corners.T).T + T[:3, 3]
    px = r.project(pc) * S
    for a, b in EDGES:
        p0, p1 = px[a], px[b]
        if not np.isfinite([p0, p1]).all():
            continue
        if dashed:
            n = 9
            for k in range(0, n, 2):
                q0 = p0 + (p1 - p0) * k / n
                q1 = p0 + (p1 - p0) * (k + 1) / n
                cv2.line(canvas, tuple(q0.astype(int)), tuple(q1.astype(int)),
                         color, thick, cv2.LINE_AA)
        else:
            cv2.line(canvas, tuple(p0.astype(int)), tuple(p1.astype(int)),
                     color, thick, cv2.LINE_AA)


def render(t):
    cl = r.d["reg_clusters"][t]
    best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
    corners = map_bbox(t)
    canvas = upscale(r.rgb(t), S)

    # top rejected hypotheses first (underneath), strongest alternatives only
    rej = [(i, c) for i, c in enumerate(cl) if i != best]
    rej.sort(key=lambda ic: -ic[1]["ninliers"])
    for i, c in rej[:3]:
        draw_bbox(canvas, c["T"], corners, GRAY, 2, dashed=True)
    Tb = cl[best]["T"]
    draw_bbox(canvas, Tb, corners, GREEN, 3)

    # inlier map points (object frame) under the selected hypothesis
    reg_kp = r.ragged("reg_key_points", t).reshape(-1, 3)
    inl = cl[best]["inliers"]
    pts = reg_kp[inl]
    pc = (Tb[:3, :3] @ pts.T).T + Tb[:3, 3]
    px = r.project(pc) * S
    for q in px:
        if np.isfinite(q).all():
            cv2.circle(canvas, (int(q[0]), int(q[1])), 4, (255, 255, 255), -1,
                       cv2.LINE_AA)
            cv2.circle(canvas, (int(q[0]), int(q[1])), 3, GREEN, -1, cv2.LINE_AA)

    out = f"{OUT_DIR}/panel4_hypotheses_t{t}.png"
    cv2.imwrite(out, canvas)
    print("saved", out, f"({len(cl)} hypotheses, best={best})")


if len(sys.argv) > 1 and sys.argv[1] == "scan":
    # frames with >=3 hypotheses and big spread between best and an alternative
    rows = []
    for t in range(30, r.n_frames):
        if n_clusters(t) < 3:
            continue
        cl = r.d["reg_clusters"][t]
        best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
        if best < 0 or best >= len(cl):
            continue
        Tb = cl[best]["T"]
        spread = max(pose_dist(Tb, c["T"])[1] for i, c in enumerate(cl) if i != best)
        alt_inl = max(c["ninliers"] for i, c in enumerate(cl) if i != best)
        rows.append((t, len(cl), spread, alt_inl))
    rows.sort(key=lambda x: -(x[2] * np.log1p(x[3])))
    for row in rows[:15]:
        print("t=%d n=%d spread=%.1fdeg alt_ninl=%d" % row)
else:
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    render(t)
