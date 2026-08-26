"""Panel 7: pose output — oriented 3D boxes + axis triads on a real frame.

The bbox per object is the oriented bounding box of the t=0 masked depth
point cloud (the same construction the pipeline uses when estimating init
bboxes), expressed in the object frame (== camera frame at t=0), then
projected under the tracked pose.
"""
import sys

import cv2
import numpy as np
import open3d as o3d

from common import Run, OBJ_COLORS_BGR, OUT_DIR, upscale

r = Run("ycb3")
S = 2
LINE_W = 4

# dataset object -> run pose index (verified by rigidity analysis)
D2R = {0: 1, 1: 2, 2: 0}

# corners built as [-,+] product over x,y,z -> 12 edges
OBB_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def obb_from_init_frame(obj):
    """OBB corners (object frame) from the t=0 masked depth cloud."""
    dep = r.depth(0).astype(np.float32) / 1000.0
    m = r.mask(0, obj)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.where(m & (dep > 0.15) & (dep < 2.5))
    z = dep[ys, xs]
    zmed = np.median(z)
    keep = np.abs(z - zmed) < 0.15
    ys, xs, z = ys[keep], xs[keep], z[keep]
    K = r.K
    pts = np.stack([(xs - K[0, 2]) * z / K[0, 0],
                    (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    labels = np.asarray(pcd.cluster_dbscan(eps=0.02, min_points=15))
    if labels.max() >= 0:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        pcd = pcd.select_by_index(np.where(labels == biggest)[0])
    obb = pcd.get_oriented_bounding_box()
    center, R_obb = np.asarray(obb.center), np.asarray(obb.R).copy()
    extent = np.asarray(obb.extent).copy()
    # single-view init only sees the front face: pad the thin (depth) axis
    order = np.argsort(extent)
    extent[order[0]] = max(extent[order[0]], 0.6 * extent[order[1]])
    half = extent / 2
    corners = center + (R_obb @ np.array(
        [[sx, sy, sz] for sx in (-half[0], half[0])
         for sy in (-half[1], half[1]) for sz in (-half[2], half[2])]).T).T
    return corners, center, R_obb, extent


def render(t, boxes):
    canvas = upscale(r.rgb(t), S)
    order = np.argsort([-r.pose(t, D2R[o])[2, 3] for o in range(r.n_obj)])
    for o in order:
        T = r.pose(t, int(D2R[o]))
        col = OBJ_COLORS_BGR[o]
        corners, center, R_obb, extent = boxes[o]
        pc = (T[:3, :3] @ corners.T).T + T[:3, 3]
        px = r.project(pc) * S
        for a, b in OBB_EDGES:
            if np.isfinite([px[a], px[b]]).all():
                cv2.line(canvas, tuple(px[a].astype(int)),
                         tuple(px[b].astype(int)), col, LINE_W, cv2.LINE_AA)
        # axis triad at the box center, along the OBB principal axes
        axlen = 0.62 * extent.max()
        tips = [center + R_obb[:, k] * axlen for k in range(3)]
        pa = (T[:3, :3] @ np.stack([center] + tips).T).T + T[:3, 3]
        qa = r.project(pa) * S
        for k, ac in zip(range(1, 4),
                         [(60, 60, 235), (80, 220, 80), (235, 100, 60)]):
            if np.isfinite([qa[0], qa[k]]).all():
                cv2.arrowedLine(canvas, tuple(qa[0].astype(int)),
                                tuple(qa[k].astype(int)), ac, LINE_W,
                                cv2.LINE_AA, tipLength=0.12)
    out = f"{OUT_DIR}/panel7_output_t{t}.png"
    cv2.imwrite(out, canvas)
    print("saved", out)


boxes = []
for o in range(r.n_obj):
    corners, center, R_obb, extent = obb_from_init_frame(o)
    boxes.append((corners, center, R_obb, extent))
    print(f"obj{o} OBB extent {np.round(extent, 3)}")

frames = [int(a) for a in sys.argv[1:]] or [120, 700, 1500]
for t in frames:
    render(t, boxes)
