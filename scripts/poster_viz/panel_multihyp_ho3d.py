"""Multi-hypothesis registration on HO3D AP (pitcher): phantom-map rendering.

Follows the semantics of scripts/debug_visualization/plot_clustered_registration_stats.py:
each cluster candidate = a subset of correspondences + an absolute pose T
(first-frame coords -> camera). For each top-K cluster we project the FULL
keypoint map under its T — on the pitcher, rival hypotheses are rotations
about the body axis and put the handle in the wrong place; the TSDF check
picks the right one.

Usage:
  panel_multihyp_ho3d.py <seq> scan          # list ambiguous + clean frames
  panel_multihyp_ho3d.py <seq> <t> [<t>...]  # render frames
"""
import pickle
import sys

import cv2
import numpy as np

from common import Run, OUT_DIR, upscale

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP14"
S = 2
TOPK = 3
# hypothesis colors (BGR): selected green, then orange, purple
HYP_COLS = [(98, 211, 60), (0, 150, 255), (200, 80, 180)]

r = Run(f"ho3d_{SEQ}")
with open(f"/home/justin/data/HO3D_V3/evaluation/{SEQ}/meta/0000.pkl", "rb") as f:
    r.K = pickle.load(f)["camMat"]


def clusters_at(t):
    c = r.d["reg_clusters"][t]
    try:
        return list(c)
    except TypeError:
        return []


def best_idx_at(t, n):
    try:
        b = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
        return b if 0 <= b < n else 0
    except (TypeError, ValueError, IndexError):
        return 0


def rot_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def scan():
    amb, clean = [], []
    for t in range(30, r.n_frames):
        cl = clusters_at(t)
        if not cl:
            continue
        n = sorted((int(c["ninliers"]) for c in cl), reverse=True)
        b = best_idx_at(t, len(cl))
        if len(cl) >= 2 and n[0] >= 25:
            order = np.argsort([-int(c["ninliers"]) for c in cl])
            spread = rot_deg(cl[order[0]]["T"], cl[order[1]]["T"])
            ratio = n[1] / n[0]
            if ratio > 0.35 and 20 < spread < 160:
                amb.append((t, n[0], n[1], spread, ratio))
        if n[0] >= 60 and (len(cl) == 1 or n[1] / n[0] < 0.12):
            clean.append((t, n[0], n[1] if len(n) > 1 else 0))
    amb.sort(key=lambda x: -(x[4] * min(x[3], 90)))
    print("AMBIGUOUS (t, n1, n2, rot spread deg, ratio):")
    for a in amb[:15]:
        print("  %4d n1=%3d n2=%3d spread=%5.1f ratio=%.2f" % a)
    print("CLEAN (t, n1, n2):")
    step = max(1, len(clean) // 15)
    for c in clean[::step][:15]:
        print("  %4d n1=%3d n2=%3d" % c)


def render(t):
    cl = clusters_at(t)
    if not cl:
        print(f"t={t}: no clusters")
        return
    order = np.argsort([-int(c["ninliers"]) for c in cl])[:TOPK]
    b = best_idx_at(t, len(cl))
    # selected hypothesis first (green), then strongest alternatives
    order = [b] + [int(j) for j in order if j != b]
    order = order[:TOPK]

    kp = r.keypoints_obj0(t)
    c0 = np.median(kp, axis=0)
    rad = np.linalg.norm(kp - c0, axis=1)
    kp = kp[rad < np.percentile(rad, 94)]
    if len(kp) > 500:
        kp = kp[np.random.default_rng(0).choice(len(kp), 500, replace=False)]

    img = cv2.imread(r.cfg["rgb"].format(data=r.data_dir, t=t))
    canvas = upscale(img, S)

    # draw phantoms back-to-front: weakest hypothesis first
    for rank in range(len(order) - 1, -1, -1):
        c = cl[order[rank]]
        T = np.asarray(c["T"], float)
        pc = (T[:3, :3] @ kp.T).T + T[:3, 3]
        px = r.project(pc) * S
        ov = canvas.copy()
        for q in px:
            if np.isfinite(q).all():
                cv2.circle(ov, (int(q[0]), int(q[1])), 3, HYP_COLS[rank], -1,
                           cv2.LINE_AA)
        alpha = 0.9 if rank == 0 else 0.45
        canvas = cv2.addWeighted(canvas, 1 - alpha, ov, alpha, 0)

    # inlier-count badges
    x0, y0 = 14, 16
    for rank, j in enumerate(order):
        cv2.rectangle(canvas, (x0, y0), (x0 + 26, y0 + 26), HYP_COLS[rank], -1)
        label = f"{int(cl[j]['ninliers'])}" + ("  selected" if rank == 0 else "")
        cv2.putText(canvas, label, (x0 + 36, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 40

    out = f"{OUT_DIR}/multihyp_{SEQ}_t{t}.png"
    cv2.imwrite(out, canvas)
    spreads = [round(rot_deg(cl[order[0]]["T"], cl[j]["T"]), 1)
               for j in order[1:]]
    print(f"saved {out} hyps={[int(cl[j]['ninliers']) for j in order]} "
          f"spread_deg={spreads}")


if len(sys.argv) > 2 and sys.argv[2] == "scan":
    scan()
else:
    for t in [int(a) for a in sys.argv[2:]] or [200]:
        render(t)
