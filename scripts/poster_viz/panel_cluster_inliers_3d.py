"""3D plot of ONLY the correspondences each cluster actually uses.

Per hypothesis: its inlier map points (under that cluster's pose) + the
observed points they explain, same marker size, plus the correspondence
lines. Shared limits/viewpoint across hypotheses. A "_ctx" variant adds the
remaining visible observations in faint gray for context.

Usage: panel_cluster_inliers_3d.py <seq> <t> [n_hyp=2]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP10"
T_FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 242
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 2

OBS_RGB = "#37474f"
HYP_RGB = ["#3cd362", "#ff9600", "#c850b4"]
SIZE = 60

r = Run(f"ho3d_{SEQ}")
cl = list(r.d["reg_clusters"][T_FRAME])
best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][T_FRAME])[0])
order = list(np.argsort([-int(c["ninliers"]) for c in cl]))
order = ([best] + [j for j in order if j != best])[:TOPK]

src = r.ragged("reg_key_points", T_FRAME).reshape(-1, 3)   # map, object frame
tgt = r.ragged("reg_curr3d", T_FRAME).reshape(-1, 3)       # observed, camera
print(f"correspondence set: {len(src)}")

# context cloud: all visible low-uncertainty observations
p3 = r.track3d(T_FRAME)
unc = r.ragged("uncertainties", T_FRAME)
n = min(len(p3), len(unc))
ctx = p3[:n][r.visibles(T_FRAME)[:n] & r.valid(T_FRAME)[:n] & (unc[:n] < 0.3)
             & np.isfinite(p3[:n]).all(1)]


def tf(T, p):
    return (np.asarray(T)[:3, :3] @ p.T).T + np.asarray(T)[:3, 3]


pts_all = [ctx]
for j in order:
    inl = np.asarray(cl[j]["inliers"], int)
    inl = inl[(inl >= 0) & (inl < len(src))]
    pts_all += [tgt[inl], tf(cl[j]["T"], src[inl])]
allp = np.vstack(pts_all)
lo, hi = allp.min(0), allp.max(0)
pad = 0.05 * np.linalg.norm(hi - lo)
lo, hi = lo - pad, hi + pad

for rank, j in enumerate(order):
    inl = np.asarray(cl[j]["inliers"], int)
    inl = inl[(inl >= 0) & (inl < len(src))]
    obs_i, map_i = tgt[inl], tf(cl[j]["T"], src[inl])
    for ctx_on in (False, True):
        fig = plt.figure(figsize=(6.4, 5.2), dpi=220)
        ax = fig.add_subplot(111, projection="3d")
        if ctx_on:
            ax.scatter(*ctx.T, c=OBS_RGB, s=14, alpha=0.45, linewidths=0)
        for a, b in zip(map_i, obs_i):
            ax.plot(*zip(a, b), c="#9aa3ab", lw=1.0, alpha=0.9)
        ax.scatter(*obs_i.T, c=OBS_RGB, s=SIZE, alpha=0.95, linewidths=0)
        ax.scatter(*map_i.T, c=HYP_RGB[rank], s=SIZE, alpha=0.95,
                   linewidths=0.6, edgecolors="w")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=-64, azim=-88)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        tag = ("sel" if rank == 0 else f"alt{rank}") + ("_ctx" if ctx_on else "")
        out = f"{OUT_DIR}/cluster_inliers3d_{SEQ}_t{T_FRAME}_{tag}.png"
        fig.savefig(out, transparent=True)
        plt.close(fig)
        print(f"saved {out} ({len(inl)} inlier correspondences)")
