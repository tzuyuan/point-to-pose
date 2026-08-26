"""Per-hypothesis 3D figures: observed points vs map keypoints + lines.

One figure per registration hypothesis (mirrors plot_3d_single_cluster in
scripts/debug_visualization/plot_clustered_registration_stats.py):
observations (curr3d, camera frame) are IDENTICAL in every figure; the
keypoint map is transformed by that hypothesis's pose T; lines connect the
hypothesis's inlier correspondences. Shared viewpoint + limits across
figures so the pose difference is the only thing that changes.

Usage: panel_multihyp_3d.py <seq> <t> [topk]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR

args = [a for a in sys.argv[1:] if a != "top"]
TOP_VIEW = "top" in sys.argv[1:]
SEQ = args[0] if len(args) > 0 else "AP10"
T_FRAME = int(args[1]) if len(args) > 1 else 1045
TOPK = int(args[2]) if len(args) > 2 else 3

# hypothesis colors (match the 2D figure): green, orange, purple
HYP_RGB = ["#3cd362", "#ff9600", "#c850b4"]
OBS_RGB = "#37474f"          # observations: dark slate, constant everywhere

r = Run(f"ho3d_{SEQ}")

cl = list(r.d["reg_clusters"][T_FRAME])
best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][T_FRAME])[0])
order = list(np.argsort([-int(c["ninliers"]) for c in cl]))
order = [best] + [j for j in order if j != best]
order = order[:TOPK]

src = r.ragged("reg_key_points", T_FRAME).reshape(-1, 3)   # first-frame coords
tgt = r.ragged("reg_curr3d", T_FRAME).reshape(-1, 3)       # camera frame
full_map = r.keypoints_obj0(T_FRAME)
c0 = np.median(full_map, axis=0)
rad = np.linalg.norm(full_map - c0, axis=1)
full_map = full_map[rad < np.percentile(rad, 94)]
if len(full_map) > 700:
    full_map = full_map[np.random.default_rng(0).choice(len(full_map), 700,
                                                        replace=False)]


def tf(T, p):
    return (np.asarray(T)[:3, :3] @ p.T).T + np.asarray(T)[:3, 3]


# identify the HANDLE in the map: the largest coherent cluster of points
# radially far from the fitted body axis
import open3d as o3d  # noqa: E402

cm_ = full_map.mean(0)
_, _, Vt = np.linalg.svd(full_map - cm_, full_matrices=False)
axis = Vt[0]                                   # pitcher height axis
radial = np.linalg.norm((full_map - cm_)
                        - np.outer((full_map - cm_) @ axis, axis), axis=1)
cand = np.where(radial > 1.25 * np.median(radial))[0]
handle_mask = np.zeros(len(full_map), bool)
if len(cand) > 10:
    pc_ = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(full_map[cand].astype(np.float64)))
    labels = np.asarray(pc_.cluster_dbscan(eps=0.025, min_points=6))
    if labels.max() >= 0:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        handle_mask[cand[labels == biggest]] = True
if handle_mask.sum() < 10:
    handle_mask[radial > np.percentile(radial, 92)] = True
print(f"handle points: {handle_mask.sum()} / {len(full_map)}")


# shared limits over observations + every hypothesis's phantom map
clouds = [tgt] + [tf(cl[j]["T"], full_map) for j in order]
allp = np.vstack(clouds)
lo, hi = allp.min(0), allp.max(0)
pad = 0.04 * np.linalg.norm(hi - lo)
lo, hi = lo - pad, hi + pad

for rank, j in enumerate(order):
    c = cl[j]
    T = np.asarray(c["T"], float)
    map_T = tf(T, full_map)
    src_T = tf(T, src)
    inl = np.asarray(c["inliers"], dtype=int)
    inl = inl[(inl >= 0) & (inl < len(src))]

    fig = plt.figure(figsize=(6.4, 5.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    # phantom map under this hypothesis (its color, light); handle points solid
    ax.scatter(*map_T[~handle_mask].T, c=HYP_RGB[rank], s=9, alpha=0.28,
               linewidths=0)
    ax.scatter(*map_T[handle_mask].T, c=HYP_RGB[rank], s=30, alpha=0.95,
               marker="^", linewidths=0)
    # observations NOT explained by this hypothesis: dark slate
    mask = np.zeros(len(tgt), bool)
    mask[inl] = True
    ax.scatter(*tgt[~mask].T, c=OBS_RGB, s=24, alpha=0.95, linewidths=0)
    # observations CLAIMED by this hypothesis: filled in its color, dark edge
    ax.scatter(*tgt[mask].T, c=HYP_RGB[rank], s=42, alpha=1.0,
               edgecolors=OBS_RGB, linewidths=0.9)
    # correspondence lines (mostly short — the rival aligns too; that is the point)
    for i in inl:
        ax.plot(*zip(src_T[i], tgt[i]), c=HYP_RGB[rank], lw=1.2, alpha=0.9)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    if TOP_VIEW:
        # look along the pitcher's body axis (in camera frame, under the
        # selected pose) so the handle azimuth is the visible difference
        a = np.asarray(cl[order[0]]["T"], float)[:3, :3] @ axis
        a = a / np.linalg.norm(a)
        if a[2] > 0:
            a = -a
        ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(a[2], -1, 1)))),
                     azim=float(np.degrees(np.arctan2(a[1], a[0]))))
    else:
        ax.view_init(elev=-64, azim=-88)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    tag = ("sel" if rank == 0 else f"alt{rank}") + ("_top" if TOP_VIEW else "")
    out = f"{OUT_DIR}/multihyp3d_{SEQ}_t{T_FRAME}_{tag}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"saved {out} (hyp {j}, {int(c['ninliers'])} inliers)")
