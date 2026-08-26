"""Multi-hypothesis registration explained with POINTS (map keypoints).

One panel per hypothesis: the keypoint map projected under that pose as a
sparse constellation of dots (white-ringed, same style as the other panels),
handle-cluster points drawn larger with a dark edge, plus a check/cross badge
with the inlier count. Also writes a side-by-side pair image.

Usage: panel_multihyp_pts.py <seq> <t> [<t> ...]
"""
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d

from common import Run, OUT_DIR, upscale

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP10"
S = 2
UNC_MAX = float(os.environ.get("UNC_MAX", 0.3))
N_PTS = 140                 # subsampled body points per hypothesis
GREEN = (98, 211, 60)
ORANGE = (0, 150, 255)

r = Run(f"ho3d_{SEQ}")
with open(f"/home/justin/data/HO3D_V3/evaluation/{SEQ}/meta/0000.pkl", "rb") as f:
    m0 = pickle.load(f)
r.K = m0["camMat"]


def badge(canvas, x, y, color, ok, label):
    cv2.rectangle(canvas, (x, y), (x + 30, y + 30), color, -1)
    if ok:
        cv2.polylines(canvas, [np.array([[x+7, y+16], [x+13, y+23], [x+24, y+8]])],
                      False, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        cv2.line(canvas, (x+8, y+8), (x+22, y+22), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(canvas, (x+22, y+8), (x+8, y+22), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 40, y + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


for t in [int(a) for a in sys.argv[2:]] or [1504]:
    cl = list(r.d["reg_clusters"][t])
    best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
    order = np.argsort([-int(c["ninliers"]) for c in cl])
    rival = int(order[0]) if int(order[0]) != best else int(order[1])

    # map + handle cluster (radial outliers, largest coherent cluster)
    kp_all = r.keypoints_obj0(t)
    unc = r.ragged("uncertainties", t)
    n = min(len(kp_all), len(unc))
    keep_pt = (r.visibles(t)[:n] & r.valid(t)[:n] & (unc[:n] < UNC_MAX)
               & np.isfinite(kp_all[:n]).all(1))
    kp = kp_all[:n][keep_pt]
    c0 = np.median(kp, axis=0)
    rad = np.linalg.norm(kp - c0, axis=1)
    kp = kp[rad < np.percentile(rad, 97)]
    print(f"  visible & unc<{UNC_MAX}: {len(kp)} map points")
    cm_ = kp.mean(0)
    _, _, Vt = np.linalg.svd(kp - cm_, full_matrices=False)
    axis = Vt[0]
    radial = np.linalg.norm((kp - cm_) - np.outer((kp - cm_) @ axis, axis), axis=1)
    cand = np.where(radial > 1.25 * np.median(radial))[0]
    hmask = np.zeros(len(kp), bool)
    if len(cand) > 10:
        pc_ = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(kp[cand].astype(np.float64)))
        lb = np.asarray(pc_.cluster_dbscan(eps=0.025, min_points=6))
        if lb.max() >= 0:
            hmask[cand[lb == np.bincount(lb[lb >= 0]).argmax()]] = True

    rng = np.random.default_rng(0)
    body = np.where(~hmask)[0]
    if len(body) > N_PTS:
        body = rng.choice(body, N_PTS, replace=False)
    handle = np.where(hmask)[0]
    if len(handle) > 45:
        handle = rng.choice(handle, 45, replace=False)

    base = upscale(r.rgb(t), S)
    panels = []
    for tag, j, col, ok in [("sel", best, GREEN, True),
                            ("alt", rival, ORANGE, False)]:
        T = np.asarray(cl[j]["T"], float)
        pc = (T[:3, :3] @ kp.T).T + T[:3, 3]
        px = r.project(pc) * S
        im = base.copy()
        for i in body:
            if np.isfinite(px[i]).all():
                x, y = px[i].astype(int)
                cv2.circle(im, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(im, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
        for i in handle:
            if np.isfinite(px[i]).all():
                x, y = px[i].astype(int)
                cv2.circle(im, (x, y), 9, col, -1, cv2.LINE_AA)
                cv2.circle(im, (x, y), 9, (40, 40, 40), 2, cv2.LINE_AA)
        badge(im, 14, 14, col, ok, f"{int(cl[j]['ninliers'])} inliers")
        out = f"{OUT_DIR}/multihyp_pts_{SEQ}_t{t}_{tag}.png"
        cv2.imwrite(out, im)
        panels.append(im)
        print("saved", out)

    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    pair = np.concatenate([panels[0], gap, panels[1]], axis=1)
    cv2.imwrite(f"{OUT_DIR}/multihyp_pts_{SEQ}_t{t}_pair.png", pair)
    print("saved", f"{OUT_DIR}/multihyp_pts_{SEQ}_t{t}_pair.png")

    # single-image OVERLAY of both hypotheses (uniform dot size)
    ov = base.copy()
    draw = np.concatenate([body, handle])
    for j, col in [(rival, ORANGE), (best, GREEN)]:
        T = np.asarray(cl[j]["T"], float)
        pc = (T[:3, :3] @ kp.T).T + T[:3, 3]
        px = r.project(pc) * S
        for i in draw:
            if np.isfinite(px[i]).all():
                x, y = px[i].astype(int)
                cv2.circle(ov, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(ov, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
    badge(ov, 14, 14, GREEN, True, f"{int(cl[best]['ninliers'])} inliers")
    badge(ov, 14, 58, ORANGE, False, f"{int(cl[rival]['ninliers'])} inliers")
    cv2.imwrite(f"{OUT_DIR}/multihyp_pts_{SEQ}_t{t}_overlay.png", ov)
    print("saved", f"{OUT_DIR}/multihyp_pts_{SEQ}_t{t}_overlay.png")
