"""Simplified per-hypothesis 3D plot: observed points (one color) + map
keypoints under the hypothesis pose (hypothesis color), same marker size.

Usage: panel_multihyp_3d_simple.py <seq> <t> [n_hyp=2]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP10"
T_FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 1504
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 2

OBS_RGB = "#37474f"
HYP_RGB = ["#3cd362", "#ff9600", "#c850b4"]
SIZE = 22

r = Run(f"ho3d_{SEQ}")
cl = list(r.d["reg_clusters"][T_FRAME])
best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][T_FRAME])[0])
order = list(np.argsort([-int(c["ninliers"]) for c in cl]))
order = ([best] + [j for j in order if j != best])[:TOPK]

# observed + map: SAME subset of points, visible at this frame with low
# tracker uncertainty (obj_key_points is indexed like the track table)
UNC_MAX = float(os.environ.get("UNC_MAX", 0.3))
p3 = r.track3d(T_FRAME)
unc = r.ragged("uncertainties", T_FRAME)
kp_all = r.keypoints_obj0(T_FRAME)
n = min(len(p3), len(unc), len(kp_all))
ok = (r.visibles(T_FRAME)[:n] & r.valid(T_FRAME)[:n] & (unc[:n] < UNC_MAX)
      & np.isfinite(p3[:n]).all(1) & (p3[:n, 2] > 0.05)
      & np.isfinite(kp_all[:n]).all(1))
tgt, kp = p3[:n][ok], kp_all[:n][ok]
rad = np.linalg.norm(kp - np.median(kp, axis=0), axis=1)
keep = rad < np.percentile(rad, 97)
tgt, kp = tgt[keep], kp[keep]
print(f"visible & unc<{UNC_MAX}: {len(kp)} points")


def tf(T, p):
    return (np.asarray(T)[:3, :3] @ p.T).T + np.asarray(T)[:3, 3]


clouds = [tgt] + [tf(cl[j]["T"], kp) for j in order]
allp = np.vstack(clouds)
lo, hi = allp.min(0), allp.max(0)
pad = 0.04 * np.linalg.norm(hi - lo)
lo, hi = lo - pad, hi + pad

for rank, j in enumerate(order):
    kp_t = tf(cl[j]["T"], kp)
    fig = plt.figure(figsize=(6.4, 5.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*kp_t.T, c=HYP_RGB[rank], s=SIZE, alpha=0.85, linewidths=0)
    ax.scatter(*tgt.T, c=OBS_RGB, s=SIZE, alpha=0.95, linewidths=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=-64, azim=-88)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    tag = "sel" if rank == 0 else f"alt{rank}"
    out = f"{OUT_DIR}/multihyp3d_simple_{SEQ}_t{T_FRAME}_{tag}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"saved {out} ({int(cl[j]['ninliers'])} inliers)")
