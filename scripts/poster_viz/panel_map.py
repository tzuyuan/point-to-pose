"""Panel 3: object-centric 3D keypoint map, colored by the frame each point was added.

Main figure: final map. Optional strip: map growth over time.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR

r = Run("ycb3")
T_FINAL = r.n_frames - 1

kp = r.keypoints_obj0(T_FINAL)
birth = r.kp_birth_frames_obj0(T_FINAL).astype(float)
print("map points:", len(kp), "birth range", birth.min(), birth.max())

# center the map
c = np.median(kp, axis=0)
p = kp - c

# drop far outliers for a tight view
rad = np.linalg.norm(p, axis=1)
keep = rad < np.percentile(rad, 99)
p, birth = p[keep], birth[keep]


def scatter(ax, pts, col, elev=-70, azim=-90, s=14):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, cmap="viridis",
               s=s, alpha=0.9, linewidths=0, rasterized=True)
    ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# --- main: final map, a couple of viewpoints to choose from ---
for elev, azim, tag in [(-70, -90, "v1"), (-60, -60, "v2"), (10, -80, "v3")]:
    fig = plt.figure(figsize=(5, 5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    scatter(ax, p, birth, elev, azim)
    fig.tight_layout(pad=0)
    fig.savefig(f"{OUT_DIR}/panel3_map_final_{tag}.png", transparent=True)
    plt.close(fig)
    print("saved", f"{OUT_DIR}/panel3_map_final_{tag}.png")

# --- growth strip: same view, three time points ---
stages = [60, 600, T_FINAL]
fig = plt.figure(figsize=(12, 4.2), dpi=220)
for i, t in enumerate(stages):
    kp_t = r.keypoints_obj0(t) - c
    b_t = r.kp_birth_frames_obj0(t).astype(float)
    rad_t = np.linalg.norm(kp_t, axis=1)
    k = rad_t < np.percentile(rad, 99)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(kp_t[k][:, 0], kp_t[k][:, 1], kp_t[k][:, 2],
               c=b_t[k], cmap="viridis", vmin=0, vmax=birth.max(),
               s=14, alpha=0.9, linewidths=0)
    ax.set_xlim(p[:, 0].min(), p[:, 0].max())
    ax.set_ylim(p[:, 1].min(), p[:, 1].max())
    ax.set_zlim(p[:, 2].min(), p[:, 2].max())
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2])))
    ax.view_init(elev=-70, azim=-90)
    ax.set_axis_off()
    ax.set_title(f"t = {t}", fontsize=11)
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT_DIR}/panel3_map_growth.png", transparent=True)
plt.close(fig)
print("saved", f"{OUT_DIR}/panel3_map_growth.png")
