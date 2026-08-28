"""
Interactive viser viewer for the model-tracking map.

Loads the textured mesh and the pre-built map, and gives you a slider over the rendered view
index. For the selected view it shows, all aligned in the object frame:

  * the textured mesh (static),
  * that view's 3D map keypoints (backprojected SuperPoint points, in object frame),
  * the camera frustum + the render image for that view.

This lets you visually confirm that each view's keypoints lie on the correct part of the mesh
surface (i.e. the render -> SuperPoint -> backprojection -> object-frame transform is correct).

Run (in the ``point2pose`` conda env):
    python examples/model_based_tracking/viser_map_viewer.py \
        --config configs/pipeline/model_tracking.yaml

Then open the printed URL (http://localhost:8080 by default).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for sub in ("LightGlue", "tapnet"):
    p = str(Path(__file__).resolve().parents[2] / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from omegaconf import OmegaConf

from point2pose.model_tracking.map_builder import MapBuilder, ModelMap
from point2pose.utils.transform import inverse_SE3

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/pipeline/model_tracking.yaml"


def make_intrinsics(H, W, fov_deg=55.0):
    f = (H / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def load_mesh_trimesh(mesh_path, mesh_scale=1.0):
    m = trimesh.load(mesh_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    if mesh_scale != 1.0:
        m.apply_scale(mesh_scale)
    # center to match the map builder (object frame == centered mesh frame)
    m.apply_translation(-m.vertices.mean(0))
    return m


def load_gaussians_for_viz(gsplat_path):
    """Load a gaussian checkpoint's means + DC color (sh0) for a point-cloud preview, in the
    same object frame the map was built in (no extra centering -- the checkpoint's own frame
    IS the map's object frame, see GsplatRenderer docstring)."""
    import torch

    splats = torch.load(gsplat_path, map_location="cpu")
    means = splats["means"].numpy()
    # sh0 is the DC term of the SH color basis; C0 undoes rgb_to_sh's (rgb - 0.5) / C0.
    C0 = 0.28209479177387814
    colors = (splats["sh0"][:, 0, :].numpy() * C0 + 0.5).clip(0, 1)
    return means, colors


def per_view_slices(model_map: ModelMap):
    """Return list of (start, end) index ranges into map_pts_obj for each view."""
    slices = []
    cursor = 0
    for kps in model_map.view_kps:
        m = kps.shape[0]
        slices.append((cursor, cursor + m))
        cursor += m
    return slices


def build_map(config_path, mesh_override, res):
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.load(str(DEFAULT_CONFIG))
    cfg.setdefault("pipeline", OmegaConf.create({"params": {"device": "cuda"}}))
    if mesh_override:
        cfg.model_tracking.params.mesh_path = mesh_override

    H = W = res
    K = make_intrinsics(H, W)
    mb = MapBuilder(cfg)
    model_map, tracker, track_table, obj = mb.build(K, (H, W))

    is_gsplat = cfg.model_tracking.params.get("renderer", "nvdiffrast") == "gsplat"
    mesh = None
    gaussians = None
    if is_gsplat:
        gaussians = load_gaussians_for_viz(cfg.model_tracking.params.gsplat_path)
    else:
        mesh = load_mesh_trimesh(cfg.model_tracking.params.mesh_path,
                                 float(cfg.model_tracking.params.get("mesh_scale", 1.0)))
    return cfg, model_map, mesh, gaussians, K, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mesh", default=None, help="override mesh_path")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    cfg, model_map, mesh, gaussians, K, H, W = build_map(args.config, args.mesh, args.res)
    slices = per_view_slices(model_map)
    n_views = len(model_map.view_rgbs)
    print(f"[viser] map: {model_map.map_pts_obj.shape[0]} points, {n_views} views")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # Static object visualization (object frame): textured mesh, or a gaussian-splat means
    # point cloud when the map was built with the gsplat renderer (no mesh to show).
    if mesh is not None:
        server.scene.add_mesh_trimesh("/mesh", mesh)
    else:
        means, colors = gaussians
        server.scene.add_point_cloud(
            "/gaussians", points=means.astype(np.float32),
            colors=(colors * 255).astype(np.uint8), point_size=0.002,
        )

    # GUI controls
    with server.gui.add_folder("View"):
        gui_view = server.gui.add_slider("view index", min=0, max=max(0, n_views - 1),
                                         step=1, initial_value=0)
        gui_show_all = server.gui.add_checkbox("show all views' points", initial_value=False)
        gui_pt_size = server.gui.add_slider("point size", min=0.001, max=0.02, step=0.001,
                                            initial_value=0.004)
        gui_info = server.gui.add_markdown("")

    def render_state():
        vi = int(gui_view.value)
        ps = float(gui_pt_size.value)

        # Points for the selected view (green), highlighted.
        s, e = slices[vi]
        pts = model_map.map_pts_obj[s:e]
        valid = model_map.map_valid[s:e]
        pts = pts[valid]
        server.scene.add_point_cloud(
            "/view_points",
            points=pts.astype(np.float32),
            colors=np.tile(np.array([[0, 255, 0]], np.uint8), (len(pts), 1)),
            point_size=ps,
        )

        # Optionally, all other views' points faint (gray).
        if gui_show_all.value:
            all_pts = model_map.map_pts_obj[model_map.map_valid]
            server.scene.add_point_cloud(
                "/all_points",
                points=all_pts.astype(np.float32),
                colors=np.tile(np.array([[120, 120, 120]], np.uint8), (len(all_pts), 1)),
                point_size=ps * 0.6,
            )
        else:
            server.scene.add_point_cloud(
                "/all_points", points=np.zeros((0, 3), np.float32),
                colors=np.zeros((0, 3), np.uint8), point_size=ps,
            )

        # Camera frustum at this view's pose. T_obj2cam maps object->camera; the camera pose
        # in the object frame is its inverse.
        T_obj2cam = model_map.view_poses_obj2cam[vi]
        T_cam2obj = inverse_SE3(T_obj2cam)
        R = T_cam2obj[:3, :3]
        t = T_cam2obj[:3, 3]
        # Use the K the map was actually rendered with (centered standard pinhole).
        Kr = model_map.render_intrinsics
        fov = 2 * np.arctan2(H / 2, Kr[1, 1])
        server.scene.add_camera_frustum(
            "/cam",
            fov=float(fov),
            aspect=W / H,
            scale=0.06,
            wxyz=vtf.SO3.from_matrix(R).wxyz,
            position=t.astype(np.float32),
            image=model_map.view_rgbs[vi],
        )

        gui_info.content = (
            f"**view {vi}** / {n_views - 1}  \n"
            f"points this view: {len(pts)}  \n"
            f"total map points: {int(model_map.map_valid.sum())}"
        )

    gui_view.on_update(lambda _: render_state())
    gui_show_all.on_update(lambda _: render_state())
    gui_pt_size.on_update(lambda _: render_state())
    render_state()

    print(f"[viser] serving at http://localhost:{args.port}  (Ctrl-C to stop)")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
