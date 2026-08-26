"""Panel 5: pose-graph optimization — keyframe virtual cameras around the
object-centric keypoint map, with observation edges. All from the real run.

Cameras are placed on a fixed-radius shell around the map (directions and
orientations are the real optimized values; radial distance is normalized
for legibility).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from common import Run, OUT_DIR

r = Run("ycb3")
T_FINAL = r.n_frames - 1
N_SHOW = 8

kp = r.keypoints_obj0(T_FINAL)
birth = r.kp_birth_frames_obj0(T_FINAL)
c0 = np.median(kp, axis=0)
rad = np.linalg.norm(kp - c0, axis=1)
keep = rad < np.percentile(rad, 99)
kp, birth = kp[keep], birth[keep]

kf_frames = np.where(r.d["is_key_frame"])[0]


def inv_se3(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


cams = np.array([inv_se3(r.d["obj_pose"][t]) for t in kf_frames])

# fixed-radius shell: keep real direction + orientation, normalize distance
ctr = kp.mean(0)
ext = np.linalg.norm(kp.max(0) - kp.min(0))
SHELL = 1.7 * ext
for i in range(len(cams)):
    v = cams[i, :3, 3] - ctr
    cams[i, :3, 3] = ctr + v / max(np.linalg.norm(v), 1e-9) * SHELL

# pick well-separated keyframes (greedy farthest-point over shell positions)
sel = [0]
pos = cams[:, :3, 3]
while len(sel) < min(N_SHOW, len(cams)):
    d = np.min(np.linalg.norm(pos[:, None] - pos[sel][None], axis=2), axis=1)
    sel.append(int(np.argmax(d)))
sel = sorted(sel)

fig = plt.figure(figsize=(6.4, 5.4), dpi=220)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c="#8d9aa5", s=10, alpha=0.85,
           linewidths=0, rasterized=True)

colors = cm.viridis(np.linspace(0.1, 0.92, len(sel)))
scale = 0.16 * ext

for ci, si in enumerate(sel):
    X = cams[si]
    o = X[:3, 3]
    Rm = X[:3, :3]
    w, h, z = 0.62 * scale, 0.47 * scale, 1.05 * scale
    corners = np.array([[-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]])
    cc = (Rm @ corners.T).T + o
    col = colors[ci]
    for k in range(4):
        ax.plot(*zip(o, cc[k]), c=col, lw=1.8)
        ax.plot(*zip(cc[k], cc[(k + 1) % 4]), c=col, lw=1.8)
    # a few observation edges to keypoints born at this keyframe
    born = np.where(birth == kf_frames[si])[0]
    if len(born) == 0:
        born = np.argsort(np.linalg.norm(kp - ctr, axis=1))[:4]
    elif len(born) > 5:
        born = born[np.linspace(0, len(born) - 1, 5).astype(int)]
    for j in born:
        ax.plot(*zip(o, kp[j]), c=col, lw=0.55, alpha=0.45)

lims = np.array([pos[sel].min(0) - scale, pos[sel].max(0) + scale])
ax.set_xlim(lims[0, 0], lims[1, 0])
ax.set_ylim(lims[0, 1], lims[1, 1])
ax.set_zlim(lims[0, 2], lims[1, 2])
ax.set_box_aspect(lims[1] - lims[0])
ax.view_init(elev=-62, azim=-90)
ax.set_axis_off()
fig.tight_layout(pad=0)
fig.savefig(f"{OUT_DIR}/panel5_graph.png", transparent=True)
print("saved", f"{OUT_DIR}/panel5_graph.png")
