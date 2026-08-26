"""Find multi-hypothesis frames where GT confirms the selected pose is right
and the strongest rival is genuinely wrong."""
import os
import pickle
import sys

import cv2
import numpy as np

from common import Run

GLCAM = np.diag([1.0, -1.0, -1.0, 1.0])


def gt_pose(seq, i):
    f = f"/home/justin/data/HO3D_V3/evaluation/{seq}/meta/{i:04d}.pkl"
    if not os.path.exists(f):
        return None
    with open(f, "rb") as fh:
        m = pickle.load(fh)
    if m.get("objTrans") is None or m.get("objRot") is None:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(m["objRot"], float).reshape(3))[0]
    T[:3, 3] = np.asarray(m["objTrans"], float).reshape(3)
    return GLCAM @ T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def err(A, B):
    dt = float(np.linalg.norm(A[:3, 3] - B[:3, 3])) * 100
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return dt, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


for seq in sys.argv[1:] or ["AP10", "AP11", "AP12", "AP13", "AP14"]:
    r = Run(f"ho3d_{seq}")
    T0 = gt_pose(seq, 0)
    if T0 is None:
        print(seq, "no GT"); continue
    T0i = inv(T0)
    rows = []
    for t in range(60, r.n_frames):
        cl = r.d["reg_clusters"][t]
        try:
            cl = list(cl)
        except TypeError:
            continue
        if len(cl) < 2:
            continue
        nin = [int(c["ninliers"]) for c in cl]
        order = np.argsort(nin)[::-1]
        b = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
        if b < 0 or b >= len(cl):
            continue
        rv = int(order[0]) if int(order[0]) != b else int(order[1])
        if nin[rv] < 12 or nin[rv] / max(nin[b], 1) < 0.35:
            continue
        Tg = gt_pose(seq, t)
        if Tg is None:
            continue
        Tgr = Tg @ T0i
        et_b, er_b = err(np.asarray(cl[b]["T"], float), Tgr)
        et_r, er_r = err(np.asarray(cl[rv]["T"], float), Tgr)
        if not (et_b < 3.0 and er_b < 12.0):          # selected must be right
            continue
        if er_r < 25.0:                                # rival must be wrong
            continue
        unc = r.ragged("uncertainties", t)
        p3 = r.track3d(t)
        n = min(len(unc), len(p3))
        npts = int((r.visibles(t)[:n] & r.valid(t)[:n] & (unc[:n] < 0.3)).sum())
        if npts < 120:
            continue
        m = r.mask(t, 0)
        area = int(m.sum()) if m is not None else 0
        rows.append((t, nin[b], nin[rv], er_b, er_r, npts, area))
    rows.sort(key=lambda x: -(x[2] / max(x[1], 1) * min(x[4], 90) * x[5]))
    print(f"\n{seq}: {len(rows)} candidates (t, n_sel, n_riv, sel_err, riv_err, npts, mask_area)")
    for row in rows[:8]:
        print("  t=%4d  sel=%3d riv=%3d  selerr=%4.1f°  riverr=%5.1f°  pts=%3d  area=%5d" % row)
