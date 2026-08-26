"""SVD frame-to-map registration visualization (real correspondences).

Two outputs per frame:
  svdreg_img_t{T}.png    on-image: observed points (filled), map keypoints
                         projected under the PREVIOUS pose (rings), lines
                         showing the offset the SVD step closes
  svdreg_3d_t{T}.png     two-cloud diagram: keypoint map <-> observed cloud,
                         correspondence lines (green inlier / gray outlier)

Usage: panel_svdreg.py <frame> [<frame> ...]
"""
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Run, OUT_DIR, upscale

r = Run("ycb3")
S = 2

OBS_C = (98, 211, 60)     # BGR spring green (observed, current frame)
MAP_C = (0, 176, 255)     # BGR amber (map keypoints)
IN_C = (245, 245, 245)    # correspondence line, inlier
OUT_C = (90, 90, 230)     # outlier line (red-ish)


def inv_se3(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def render_image(t):
    kp = r.ragged("reg_key_points", t).reshape(-1, 3)   # object frame
    obs = r.ragged("reg_curr3d", t).reshape(-1, 3)      # camera frame
    inl = r.ragged("reg_inliers", t).astype(bool)
    T_prev = r.d["obj_pose"][t - 1]

    kp_cam = (T_prev[:3, :3] @ kp.T).T + T_prev[:3, 3]
    p_map = r.project(kp_cam) * S
    p_obs = r.project(obs) * S

    canvas = upscale(r.rgb(t), S)
    ok = np.isfinite(p_map).all(1) & np.isfinite(p_obs).all(1)

    # lines first (under the dots): outliers light, inliers bright
    for j in np.where(ok & ~inl)[0]:
        cv2.line(canvas, tuple(p_map[j].astype(int)), tuple(p_obs[j].astype(int)),
                 OUT_C, 1, cv2.LINE_AA)
    for j in np.where(ok & inl)[0]:
        cv2.line(canvas, tuple(p_map[j].astype(int)), tuple(p_obs[j].astype(int)),
                 IN_C, 2, cv2.LINE_AA)
    # map keypoints: amber rings; observed: green filled
    for j in np.where(ok)[0]:
        x, y = p_map[j].astype(int)
        cv2.circle(canvas, (x, y), 7, MAP_C, 2, cv2.LINE_AA)
    for j in np.where(ok)[0]:
        x, y = p_obs[j].astype(int)
        cv2.circle(canvas, (x, y), 5, OBS_C, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # crop tightly around the correspondences and zoom
    pp = np.vstack([p_map[ok], p_obs[ok]]) / S
    cx0, cy0 = np.maximum(pp.min(0) - 45, 0).astype(int)
    cx1, cy1 = pp.max(0) + 45
    w, h = cx1 - cx0, cy1 - cy0
    if w / h > 4 / 3:
        cy0, cy1 = cy0 - (w * 3 / 4 - h) / 2, cy1 + (w * 3 / 4 - h) / 2
    else:
        cx0, cx1 = cx0 - (h * 4 / 3 - w) / 2, cx1 + (h * 4 / 3 - w) / 2
    cx0, cy0 = max(0, int(cx0)), max(0, int(cy0))
    cx1, cy1 = min(r.W, int(cx1)), min(r.H, int(cy1))
    crop = canvas[cy0 * S:cy1 * S, cx0 * S:cx1 * S]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    out = f"{OUT_DIR}/svdreg_img_t{t}.png"
    cv2.imwrite(out, crop)
    print("saved", out, f"({len(kp)} corr, {inl.mean():.2f} inliers)")


def render_3d(t, max_lines=45):
    kp = r.ragged("reg_key_points", t).reshape(-1, 3)
    obs = r.ragged("reg_curr3d", t).reshape(-1, 3)
    inl = r.ragged("reg_inliers", t).astype(bool)
    full_map = r.keypoints_obj0(t)

    # both clouds centered on their own centroids, side by side, in their
    # NATIVE orientations (map: object frame, observations: camera frame) —
    # the crossing lines encode the rotation the SVD step must solve
    c_map = np.median(full_map, axis=0)
    rad = np.linalg.norm(full_map - c_map, axis=1)
    keep = rad < np.percentile(rad, 98)
    full_map = full_map[keep]
    ext = np.linalg.norm(full_map.max(0) - full_map.min(0))
    c_obs = np.median(obs, axis=0)
    off = np.array([1.35 * ext, 0, 0])

    mp0 = full_map - c_map
    kp0 = kp - c_map
    ob0 = obs - c_obs + off

    fig = plt.figure(figsize=(7.5, 4.6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*mp0.T, c="#b9c2c9", s=7, alpha=0.75, linewidths=0)
    ax.scatter(*kp0.T, c="#e8a33d", s=18, alpha=0.95, linewidths=0)
    ax.scatter(*ob0.T, c="#3cd362", s=18, alpha=0.95, linewidths=0)

    sub = np.random.default_rng(0).permutation(len(kp))[:max_lines]
    for j in sub:
        col = "#7bd489" if inl[j] else "#d9a0a0"
        ax.plot(*zip(kp0[j], ob0[j]), c=col, lw=0.8 if inl[j] else 0.5,
                alpha=0.85 if inl[j] else 0.5)

    allp = np.vstack([mp0, ob0])
    lo, hi = allp.min(0), allp.max(0)
    ax.set_box_aspect(hi - lo)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=-58, azim=-72)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    out = f"{OUT_DIR}/svdreg_3d_t{t}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("saved", out)


for t in [int(a) for a in sys.argv[1:]] or [1344]:
    render_image(t)
    render_3d(t)
