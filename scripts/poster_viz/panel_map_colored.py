"""RGB-colored 3D keypoint map (notebook-style landmark plot).

Each map keypoint is colorized by projecting it into the frame where it was
born (obj_key_point_frames + obj_pose at that frame) and sampling the RGB
image there — so the label texture and the handle geometry both show.

Usage: panel_map_colored.py <seq> [t_map]
"""
import pickle
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP14"
r = Run(f"ho3d_{SEQ}")
T_MAP = int(sys.argv[2]) if len(sys.argv) > 2 else r.n_frames - 1

with open(f"/home/justin/data/HO3D_V3/evaluation/{SEQ}/meta/0000.pkl", "rb") as f:
    r.K = pickle.load(f)["camMat"]

import open3d as o3d  # noqa: E402

kp = r.keypoints_obj0(T_MAP)
birth = r.kp_birth_frames_obj0(T_MAP).astype(int)
c0 = np.median(kp, axis=0)
rad = np.linalg.norm(kp - c0, axis=1)
keep = rad < np.percentile(rad, 95)
kp, birth = kp[keep], birth[keep]
# keep the coherent shell: largest DBSCAN cluster
pc_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kp.astype(np.float64)))
labels_ = np.asarray(pc_.cluster_dbscan(eps=0.03, min_points=8))
if labels_.max() >= 0:
    big = np.bincount(labels_[labels_ >= 0]).argmax()
    kp, birth = kp[labels_ == big], birth[labels_ == big]
print(f"{len(kp)} map points at t={T_MAP}")

# sample RGB at each point's birth frame
cols = np.full((len(kp), 3), 0.55)
cache = {}
for j in range(len(kp)):
    tb = int(np.clip(birth[j], 0, r.n_frames - 1))
    if tb not in cache:
        im = cv2.imread(r.cfg["rgb"].format(data=r.data_dir, t=tb))
        cache[tb] = None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cache[tb]
    if im is None:
        continue
    T = r.d["obj_pose"][tb]
    pc = T[:3, :3] @ kp[j] + T[:3, 3]
    if pc[2] <= 0.05:
        continue
    u = r.K[0, 0] * pc[0] / pc[2] + r.K[0, 2]
    v = r.K[1, 1] * pc[1] / pc[2] + r.K[1, 2]
    ui, vi = int(round(u)), int(round(v))
    if 0 <= vi < im.shape[0] and 0 <= ui < im.shape[1]:
        # small patch median for robustness
        patch = im[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
        cols[j] = np.median(patch.reshape(-1, 3), axis=0) / 255.0

p = kp - kp.mean(0)
lo, hi = p.min(0), p.max(0)

# auto view: look along the label's outward normal (radial direction of the
# bright/low-saturation + red points), sweeping a few azimuthal offsets
v = cols.max(1)
sat = (cols.max(1) - cols.min(1)) / np.maximum(cols.max(1), 1e-6)
is_red = (cols[:, 0] > 0.35) & (cols[:, 0] > cols[:, 1] + 0.1) \
    & (cols[:, 0] > cols[:, 2] + 0.1)
label_m = ((v > 0.45) & (sat < 0.4)) | is_red
print(f"label points: {label_m.sum()}")
# true body axis from HO3D GT: canonical YCB model is height-up (z);
# map frame == camera frame at t=0, so axis = R_gt(frame0) @ z
with open(f"/home/justin/data/HO3D_V3/evaluation/{SEQ}/meta/0000.pkl", "rb") as f:
    m0 = pickle.load(f)
R0, _ = cv2.Rodrigues(np.asarray(m0["objRot"], dtype=float).reshape(3))
glcam = np.diag([1.0, -1.0, -1.0])           # HO3D OpenGL -> OpenCV camera
axis_b = glcam @ (R0 @ np.array([0.0, 0.0, 1.0]))
axis_b = axis_b / np.linalg.norm(axis_b)
cl_ = p[label_m].mean(0)
nr = cl_ - (cl_ @ axis_b) * axis_b
nr = nr / np.linalg.norm(nr)


def rotate_about(vec, ax_, deg):
    th = np.radians(deg)
    return (vec * np.cos(th) + np.cross(ax_, vec) * np.sin(th)
            + ax_ * (ax_ @ vec) * (1 - np.cos(th)))


def view_params(u, up_vec):
    """elev/azim/roll so the camera looks along -u with up_vec vertical."""
    u = u / np.linalg.norm(u)
    elev = float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(u[1], u[0])))
    f = -u
    z = np.array([0.0, 0.0, 1.0])
    pz = z - (z @ f) * f
    if np.linalg.norm(pz) < 1e-6:
        return elev, azim, 0.0
    pz = pz / np.linalg.norm(pz)
    d = up_vec - (up_vec @ f) * f
    d = d / np.linalg.norm(d)
    ang = float(np.degrees(np.arctan2(np.cross(pz, d) @ f, pz @ d)))
    return elev, azim, ang


# orient the body axis so the label's bright side points "up" consistently
axis_s = axis_b if (p[label_m].mean(0) @ axis_b) < 0 else -axis_b

# geometry coloring: turbo colormap along the body axis (height)
from matplotlib import cm as mpl_cm  # noqa: E402
h = p @ axis_s
h = (h - h.min()) / max(np.ptp(h), 1e-6)
cols_geo = mpl_cm.turbo(h)[:, :3]

GEO = len(sys.argv) > 3 and sys.argv[3] == "geo"
if GEO:
    cols = cols_geo

views = []
# notebook-style angle: offset around the axis + tilt above the rim plane
for off in (60, 90, -60, -90):
    for tilt in (25, 40):
        u = rotate_about(nr, axis_b, off)
        u = u / np.linalg.norm(u)
        u = u * np.cos(np.radians(tilt)) - axis_s * np.sin(np.radians(tilt))
        u = u / np.linalg.norm(u)
        e_, a_, r_ = view_params(u, axis_s)
        views.append((e_, a_, r_, f"o{off}_t{tilt}"))
for off, tag in [(0, "n0"), (35, "n35")]:
    u = rotate_about(nr, axis_b, off)
    e_, a_, r_ = view_params(u, axis_s)
    views.append((e_, a_, r_, tag))

for elev, azim, roll, tag in views:
    # handle = coherent cluster of radial outliers (as in panel_multihyp_3d)
    radial_v = p - np.outer(p @ axis_b, axis_b)
    radial_d = np.linalg.norm(radial_v, axis=1)
    hcand = np.where(radial_d > 1.25 * np.median(radial_d))[0]
    hmask = np.zeros(len(p), bool)
    if len(hcand) > 8:
        pc2 = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(p[hcand].astype(np.float64)))
        lb2 = np.asarray(pc2.cluster_dbscan(eps=0.025, min_points=5))
        if lb2.max() >= 0:
            hmask[hcand[lb2 == np.bincount(lb2[lb2 >= 0]).argmax()]] = True

    fig = plt.figure(figsize=(5.6, 5.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[~hmask, 0], p[~hmask, 1], p[~hmask, 2], c=cols[~hmask],
               s=20, alpha=0.95, linewidths=0, rasterized=True)
    ax.scatter(p[hmask, 0], p[hmask, 1], p[hmask, 2], c=cols[hmask],
               s=34, alpha=1.0, linewidths=0.4, edgecolors="#555555",
               rasterized=True)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    out = f"{OUT_DIR}/map_colored_{SEQ}_{'geo_' if GEO else ''}{tag}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("saved", out,
          f"(elev {elev:.0f}, azim {azim:.0f}, roll {roll:.0f})")
