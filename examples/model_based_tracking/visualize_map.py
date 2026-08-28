"""
Visualization tool for the model-based-tracking map.

Builds the offline map from a textured mesh (rendered views -> SuperPoint -> 3D-object points
-> TAPIR seeding) and produces three diagnostics:

  1. views_montage.png      -- a grid of rendered views with their SuperPoint keypoints.
  2. map_pointcloud.ply     -- the fused 3D map points (object frame), colored by seed view.
  3. tracking_diag_*.png    -- for a rendered "live" frame at a chosen viewpoint, overlays
                               every TAPIR-tracked map point colored by whether it landed near
                               its ground-truth reprojection (green = correct, red = drifted),
                               with the object mask boundary. This makes TAPIR's view-locality
                               behavior (a query only localizes in frames close to its seed view)
                               directly visible.

Run (in the ``point2pose`` conda env):
    python examples/model_based_tracking/visualize_map.py \
        --mesh assets/ycb_006_mustard_bottle/textured_mesh.obj \
        --out debug/model_tracking_viz --num-views 100

The default mesh is the YCB mustard bottle shipped under ``assets/``.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Bundled third-party sources used by the pipeline.
for sub in ("LightGlue", "tapnet"):
    p = str(Path(__file__).resolve().parents[2] / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
from omegaconf import OmegaConf

from point2pose.data_types.frame import Frame
from point2pose.model_tracking.map_builder import MapBuilder
from point2pose.utils.transform import transform_pts


REPO = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG = REPO / "configs/pipeline/model_tracking.yaml"


def _default_cfg():
    """Fallback config when no --config file is given."""
    return OmegaConf.create(
        {
            "pipeline": {"params": {"device": "cuda"}},
            "model_tracking": {
                "params": {
                    "mesh_path": str(REPO / "assets/ycb_006_mustard_bottle/textured_mesh.obj"),
                    "num_views": 20,
                    "max_map_points": 1500,
                    "render_radius": 0.4,
                    "min_depth": 0.05,
                    "max_depth": 2.0,
                    "mesh_scale": 1.0,
                }
            },
            "sampler": {
                "type": "super_point",
                "params": {
                    "num_points": 80,
                    "edge_margin_px": 4,
                    "cell_size": 8,
                    "remove_convex_hull": False,
                    "crop_to_mask": False,
                    "super_point_max_num_keypoints": 1024,
                    "device": "cuda",
                    "debug_level": 0,
                },
            },
            "tracker": {
                "type": "tapir",
                "params": {
                    "resize_height": 256,
                    "resize_width": 256,
                    "device": "cuda",
                    "checkpoint_path": str(
                        REPO / "checkpoints/tapir/causal_bootstapir_checkpoint.pt"
                    ),
                },
            },
        }
    )


def build_cfg(args):
    """
    Load the pipeline config (from --config, or the default model_tracking.yaml if present),
    then apply any explicitly-passed CLI overrides. CLI flags left at their sentinel default
    (None) do not override the config.
    """
    if args.config and os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
        print(f"[viz] loaded config: {args.config}")
    elif DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(str(DEFAULT_CONFIG))
        print(f"[viz] loaded default config: {DEFAULT_CONFIG}")
    else:
        cfg = _default_cfg()
        print("[viz] using built-in default config")

    # Ensure the sub-trees the map builder needs exist.
    cfg.setdefault("pipeline", OmegaConf.create({"params": {"device": "cuda"}}))
    mt = cfg.model_tracking.params

    # CLI overrides (only when explicitly provided).
    if args.mesh is not None:
        mt.mesh_path = args.mesh
    if args.num_views is not None:
        mt.num_views = args.num_views
    if args.num_rotations is not None:
        mt.num_rotations = args.num_rotations
    if args.max_map_points is not None:
        mt.max_map_points = args.max_map_points
    if args.radius is not None:
        mt.render_radius = args.radius
    if args.mesh_scale is not None:
        mt.mesh_scale = args.mesh_scale
    if args.points_per_view is not None:
        cfg.sampler.params.num_points = args.points_per_view
    return cfg


def make_intrinsics(H, W, fov_deg=55.0):
    """Intrinsics for a square render with a given vertical FOV."""
    f = (H / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def viz_views_montage(model_map, out_path, max_tiles=16, cols=4):
    tiles = []
    n = min(max_tiles, len(model_map.view_rgbs))
    for v in range(n):
        im = cv2.cvtColor(model_map.view_rgbs[v], cv2.COLOR_RGB2BGR).copy()
        for (x, y) in model_map.view_kps[v]:
            cv2.circle(im, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(im, f"view {model_map.view_timestamps[v]}: {len(model_map.view_kps[v])} pts",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    if not tiles:
        print("[viz] no views to montage")
        return
    while len(tiles) % cols != 0:
        tiles.append(np.zeros((240, 320, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_path, np.vstack(rows))
    print(f"[viz] wrote {out_path}")


def viz_map_pointcloud(model_map, out_path):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pts = model_map.map_pts_obj[model_map.map_valid]
    pcd.points = o3d.utility.Vector3dVector(pts)
    # color by seed view index
    view_of_pt = np.concatenate(
        [np.full(len(kps), i) for i, kps in enumerate(model_map.view_kps)]
    )[model_map.map_valid]
    if len(view_of_pt):
        t = view_of_pt / max(1, view_of_pt.max())
        colors = np.stack([t, 0.4 * np.ones_like(t), 1 - t], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"[viz] wrote {out_path} ({len(pts)} points)")


def viz_tracking_diag(mb, model_map, obj, tracker, K, H, W, view_idx, out_path,
                      unc_thres=0.7, ok_px=6.0):
    """Render a live frame at ``view_idx``'s pose and overlay tracked points vs GT reproj."""
    rend = mb._make_renderer(K, H, W)
    try:
        T_gt = model_map.view_poses_obj2cam[view_idx]
        rgb, depth = mb._render_view(rend, T_gt)
        frame = Frame(id=10_000 + view_idx, rgb=rgb, depth=depth, intrinsics=K,
                      depth_factor=1.0)
        tracks, unc, vis = tracker.track_once(frame)

        mask = depth > 0
        pc = transform_pts(T_gt, obj.key_points)
        infront = pc[:, 2] > 1e-6
        proj = (K @ pc.T).T
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)

        n_ok = n_drift = 0
        for i in range(len(tracks)):
            if not vis[i] or not infront[i] or unc[i] >= unc_thres:
                continue
            err = np.linalg.norm(tracks[i] - proj[i])
            ok = err <= ok_px
            col = (0, 255, 0) if ok else (0, 0, 255)
            cv2.circle(img, (int(tracks[i, 0]), int(tracks[i, 1])), 2, col, -1)
            if ok:
                n_ok += 1
            else:
                n_drift += 1

        cv2.putText(img, f"live @ view {view_idx} pose", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"green: track near GT ({n_ok})   red: drifted ({n_drift})",
                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(out_path, img)
        print(f"[viz] wrote {out_path}  (ok={n_ok}, drift={n_drift})")
    finally:
        mb._delete_renderer(rend)


def main():
    ap = argparse.ArgumentParser(
        description="Build & visualize the model-tracking map. Reads configs/pipeline/"
                    "model_tracking.yaml by default; CLI flags override individual params."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="pipeline YAML to read (default: configs/pipeline/model_tracking.yaml)")
    ap.add_argument("--out", default=str(REPO / "debug/model_based_tracking"))
    ap.add_argument("--res", type=int, default=256, help="square render resolution (mult. of 8)")
    # Overrides: default None means "use the value from the config".
    ap.add_argument("--mesh", default=None, help="override mesh_path")
    ap.add_argument("--num-views", type=int, default=None, help="override num_views")
    ap.add_argument("--num-rotations", type=int, default=None, help="override in-plane roll aug count")
    ap.add_argument("--points-per-view", type=int, default=None, help="override sampler num_points")
    ap.add_argument("--max-map-points", type=int, default=None, help="override max_map_points")
    ap.add_argument("--radius", type=float, default=None, help="override render_radius")
    ap.add_argument("--mesh-scale", type=float, default=None, help="override mesh_scale")
    ap.add_argument("--diag-views", type=int, nargs="*", default=[0, 5, 10],
                    help="view indices to render as 'live' frames for the tracking diagnostic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    H, W = args.res, args.res
    K = make_intrinsics(H, W)

    cfg = build_cfg(args)
    mb = MapBuilder(cfg)
    model_map, tracker, track_table, obj = mb.build(K, (H, W))
    print(f"[viz] map: {obj.key_points.shape[0]} points across "
          f"{len(model_map.view_rgbs)} views")

    viz_views_montage(model_map, os.path.join(args.out, "views_montage.png"))
    viz_map_pointcloud(model_map, os.path.join(args.out, "map_pointcloud.ply"))
    for vi in args.diag_views:
        if vi < len(model_map.view_poses_obj2cam):
            viz_tracking_diag(
                mb, model_map, obj, tracker, K, H, W, vi,
                os.path.join(args.out, f"tracking_diag_view{vi}.png"),
            )


if __name__ == "__main__":
    main()
