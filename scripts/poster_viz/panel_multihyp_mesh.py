"""Multi-hypothesis registration as MESH CONTOURS (clear 6D-pose language).

For each top hypothesis, the reconstructed mesh is rendered at that pose and
its silhouette drawn on the frame: solid green = selected (check mark),
dashed orange = strongest rival (cross). Also writes per-hypothesis images.

Usage: panel_multihyp_mesh.py <seq> <t> [<t> ...]
"""
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from common import Run, OUT_DIR, upscale

SEQ = sys.argv[1] if len(sys.argv) > 1 else "AP10"
S = 2
GREEN = (98, 211, 60)
ORANGE = (0, 150, 255)

r = Run(f"ho3d_{SEQ}")
with open(f"/home/justin/data/HO3D_V3/evaluation/{SEQ}/meta/0000.pkl", "rb") as f:
    r.K = pickle.load(f)["camMat"]

mesh = o3d.io.read_triangle_mesh(
    f"/home/justin/results/eccv_point2pose/final_results/ho3d_all_final/{SEQ}/mesh/pred_mesh_cleaned_{SEQ}.obj")
tri_cl, n_tri, _ = mesh.cluster_connected_triangles()
keep = np.asarray(tri_cl) == np.asarray(n_tri).argmax()
mesh.remove_triangles_by_mask(~keep)
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
print(f"mesh: {len(mesh.vertices)} verts")

ren = rendering.OffscreenRenderer(r.W, r.H)
intr = o3d.camera.PinholeCameraIntrinsic(
    r.W, r.H, r.K[0, 0], r.K[1, 1], r.K[0, 2], r.K[1, 2])


def silhouette(T):
    """Binary mask of the mesh rendered at pose T (object->camera)."""
    m = o3d.geometry.TriangleMesh(mesh)
    m.transform(T)
    ren.scene.clear_geometry()
    mat = rendering.MaterialRecord()
    mat.shader = "unlitSolidColor"
    mat.base_color = [1, 1, 1, 1]
    ren.scene.add_geometry("m", m, mat)
    ren.scene.set_background([0, 0, 0, 1])
    ren.setup_camera(intr, np.eye(4))
    img = np.asarray(ren.render_to_image())
    msk = (img[:, :, 0] > 40).astype(np.uint8)
    # tame reconstruction-noise wobble: close gaps, drop specks, smooth edge
    k = np.ones((9, 9), np.uint8)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k)
    msk = (cv2.GaussianBlur(msk * 255.0, (13, 13), 0) > 127).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(msk)
    if n_lab > 2:
        sizes = [(lab == i).sum() for i in range(1, n_lab)]
        msk = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    return msk


def draw_contour(canvas, mask, color, thick, dashed=False):
    mu = cv2.resize(mask, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if len(c) < 40:
            continue
        if not dashed:
            cv2.polylines(canvas, [c], True, color, thick, cv2.LINE_AA)
        else:
            pts = c[:, 0]
            n = 14  # dash period in samples
            for i in range(0, len(pts) - n // 2, n):
                seg = pts[i:i + n // 2]
                if len(seg) > 1:
                    cv2.polylines(canvas, [seg.reshape(-1, 1, 2)], False,
                                  color, thick, cv2.LINE_AA)


def badge(canvas, x, y, color, ok, label):
    cv2.rectangle(canvas, (x, y), (x + 30, y + 30), color, -1)
    if ok:  # check mark
        cv2.polylines(canvas, [np.array([[x+7, y+16], [x+13, y+23], [x+24, y+8]])],
                      False, (255, 255, 255), 3, cv2.LINE_AA)
    else:   # cross
        cv2.line(canvas, (x+8, y+8), (x+22, y+22), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(canvas, (x+22, y+8), (x+8, y+22), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 40, y + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


for t in [int(a) for a in sys.argv[2:]] or [1504]:
    cl = list(r.d["reg_clusters"][t])
    best = int(np.atleast_1d(r.d["reg_best_cluster_idx"][t])[0])
    order = np.argsort([-int(c["ninliers"]) for c in cl])
    rival = int(order[0]) if int(order[0]) != best else int(order[1])

    m_sel = silhouette(np.asarray(cl[best]["T"], float))
    m_riv = silhouette(np.asarray(cl[rival]["T"], float))

    base = upscale(r.rgb(t), S)

    combo = base.copy()
    draw_contour(combo, m_riv, ORANGE, 3, dashed=True)
    draw_contour(combo, m_sel, GREEN, 4)
    badge(combo, 14, 14, GREEN, True, f"{int(cl[best]['ninliers'])} inliers")
    badge(combo, 14, 58, ORANGE, False, f"{int(cl[rival]['ninliers'])} inliers")
    cv2.imwrite(f"{OUT_DIR}/multihyp_mesh_{SEQ}_t{t}.png", combo)

    for tag, msk, col, ok, j in [("sel", m_sel, GREEN, True, best),
                                 ("alt", m_riv, ORANGE, False, rival)]:
        im = base.copy()
        draw_contour(im, msk, col, 4, dashed=not ok)
        badge(im, 14, 14, col, ok, f"{int(cl[j]['ninliers'])} inliers")
        cv2.imwrite(f"{OUT_DIR}/multihyp_mesh_{SEQ}_t{t}_{tag}.png", im)
    print(f"saved multihyp_mesh_{SEQ}_t{t}[_sel/_alt].png "
          f"(sel {int(cl[best]['ninliers'])} vs rival {int(cl[rival]['ninliers'])})")
