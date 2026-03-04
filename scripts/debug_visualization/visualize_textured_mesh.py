#!/usr/bin/env python3
"""
Visualize textured/colored meshes exported by the SDF pipeline.

Examples:
  python visualize_textured_mesh.py --mesh_path /path/to/pred_mesh_obj_0_textured.glb
  python visualize_textured_mesh.py --mesh_dir /path/to/out/mesh --object_number 0
"""

import argparse
import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_mesh_dir() -> str:
    return os.path.join(_project_root(), "debug", "mesh")


def resolve_mesh_path(
    mesh_path: Optional[str],
    mesh_dir: Optional[str],
    object_number: int,
    prefix: str,
) -> Optional[str]:
    if mesh_path:
        return mesh_path if os.path.exists(mesh_path) else None

    base_dir = mesh_dir if mesh_dir else _default_mesh_dir()
    candidates = [
        os.path.join(base_dir, f"{prefix}_{object_number}_textured.glb"),
        os.path.join(base_dir, f"{prefix}_{object_number}.ply"),
        os.path.join(base_dir, f"{prefix}_{object_number}.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_mesh(path: str) -> Optional[o3d.geometry.TriangleMesh]:
    try:
        mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    except TypeError:
        mesh = o3d.io.read_triangle_mesh(path)
    except Exception:
        return None

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None
    return mesh


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()


def filter_disconnected_components(
    mesh: o3d.geometry.TriangleMesh,
    keep_components: int = 1,
    min_component_triangles: int = 0,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, int]]:
    """
    Filter disconnected triangle components.

    Keeps the largest connected components by triangle count and optionally
    drops components below a minimum triangle threshold.
    """
    if keep_components < 1:
        keep_components = 1
    if min_component_triangles < 0:
        min_component_triangles = 0

    if not hasattr(mesh, "cluster_connected_triangles"):
        return mesh, {
            "num_components": -1,
            "kept_components": -1,
            "removed_triangles": 0,
            "kept_triangles": int(len(mesh.triangles)),
        }

    labels, triangle_counts, _ = mesh.cluster_connected_triangles()
    labels_np = np.asarray(labels, dtype=np.int64)
    counts_np = np.asarray(triangle_counts, dtype=np.int64)

    if counts_np.size == 0 or labels_np.size == 0:
        return mesh, {
            "num_components": int(counts_np.size),
            "kept_components": int(counts_np.size),
            "removed_triangles": 0,
            "kept_triangles": int(len(mesh.triangles)),
        }

    order = np.argsort(-counts_np)
    keep_ids = order[: int(keep_components)]
    if min_component_triangles > 0:
        keep_ids = np.asarray(
            [
                cid
                for cid in keep_ids.tolist()
                if int(counts_np[cid]) >= int(min_component_triangles)
            ],
            dtype=np.int64,
        )
        if keep_ids.size == 0:
            keep_ids = np.asarray([int(order[0])], dtype=np.int64)

    keep_mask = np.isin(labels_np, keep_ids)
    remove_idx = np.flatnonzero(~keep_mask).astype(np.int64)

    if remove_idx.size == 0:
        return mesh, {
            "num_components": int(counts_np.size),
            "kept_components": int(keep_ids.size),
            "removed_triangles": 0,
            "kept_triangles": int(np.count_nonzero(keep_mask)),
        }

    mesh_filtered = copy.deepcopy(mesh)
    mesh_filtered.remove_triangles_by_index(remove_idx.tolist())
    _cleanup_mesh(mesh_filtered)

    return mesh_filtered, {
        "num_components": int(counts_np.size),
        "kept_components": int(keep_ids.size),
        "removed_triangles": int(remove_idx.size),
        "kept_triangles": int(len(mesh_filtered.triangles)),
    }


def filter_bottom_growth(
    mesh: o3d.geometry.TriangleMesh,
    bottom_axis: str = "z",
    bottom_quantile: float = 0.10,
    bottom_radial_percentile: float = 95.0,
    bottom_radial_scale: float = 1.15,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, float]]:
    """
    Remove bottom growth that extends outside the object footprint.

    A vertex is removed if:
    1) it lies in the lower `bottom_quantile` band along `bottom_axis`, and
    2) its distance in the orthogonal plane exceeds a robust body radius.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map.get(str(bottom_axis).lower(), 2)

    q = float(np.clip(bottom_quantile, 0.0, 0.49))
    rp = float(np.clip(bottom_radial_percentile, 50.0, 100.0))
    rs = max(float(bottom_radial_scale), 1e-6)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tri = np.asarray(mesh.triangles, dtype=np.int64)
    if verts.shape[0] == 0 or tri.shape[0] == 0:
        return mesh, {
            "bottom_threshold": 0.0,
            "radial_threshold": 0.0,
            "removed_vertices": 0.0,
            "removed_triangles": 0.0,
            "kept_triangles": float(len(mesh.triangles)),
        }

    h = verts[:, axis_idx]
    h_cut = float(np.quantile(h, q))
    bottom_mask = h <= h_cut

    plane_axes = [0, 1, 2]
    plane_axes.remove(axis_idx)
    plane = verts[:, plane_axes]

    upper_q = min(0.95, q + 0.25)
    body_mask = h >= float(np.quantile(h, upper_q))
    if int(np.count_nonzero(body_mask)) >= 10:
        center = np.median(plane[body_mask], axis=0)
    else:
        center = np.median(plane, axis=0)

    radial = np.linalg.norm(plane - center.reshape(1, 2), axis=1)
    ref = radial[body_mask] if int(np.count_nonzero(body_mask)) >= 10 else radial
    r_base = float(np.percentile(ref, rp))
    r_thresh = rs * r_base

    outlier_vertices = bottom_mask & (radial > r_thresh)
    removed_vertices = int(np.count_nonzero(outlier_vertices))
    if removed_vertices == 0:
        return mesh, {
            "bottom_threshold": h_cut,
            "radial_threshold": r_thresh,
            "removed_vertices": 0.0,
            "removed_triangles": 0.0,
            "kept_triangles": float(len(mesh.triangles)),
        }

    remove_tri_mask = np.any(outlier_vertices[tri], axis=1)
    remove_idx = np.flatnonzero(remove_tri_mask).astype(np.int64)
    removed_triangles = int(remove_idx.size)
    if removed_triangles == 0:
        return mesh, {
            "bottom_threshold": h_cut,
            "radial_threshold": r_thresh,
            "removed_vertices": float(removed_vertices),
            "removed_triangles": 0.0,
            "kept_triangles": float(len(mesh.triangles)),
        }

    mesh_filtered = copy.deepcopy(mesh)
    mesh_filtered.remove_triangles_by_index(remove_idx.tolist())
    _cleanup_mesh(mesh_filtered)

    # Prevent accidental empty output due to too-aggressive thresholds.
    if len(mesh_filtered.triangles) == 0:
        return mesh, {
            "bottom_threshold": h_cut,
            "radial_threshold": r_thresh,
            "removed_vertices": 0.0,
            "removed_triangles": 0.0,
            "kept_triangles": float(len(mesh.triangles)),
        }

    return mesh_filtered, {
        "bottom_threshold": h_cut,
        "radial_threshold": r_thresh,
        "removed_vertices": float(removed_vertices),
        "removed_triangles": float(removed_triangles),
        "kept_triangles": float(len(mesh_filtered.triangles)),
    }


def _has_textures(mesh: o3d.geometry.TriangleMesh) -> bool:
    textures = getattr(mesh, "textures", [])
    return textures is not None and len(textures) > 0


def print_mesh_info(path: str, mesh: o3d.geometry.TriangleMesh) -> None:
    verts = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    ext = hi - lo

    print("=" * 72)
    print(f"Mesh path: {path}")
    print(f"Vertices: {len(verts)}")
    print(f"Triangles: {len(tri)}")
    print(f"Has vertex colors: {mesh.has_vertex_colors()}")
    print(f"Has UVs: {mesh.has_triangle_uvs()}")
    print(f"Has textures: {_has_textures(mesh)}")
    print(
        "Bounds min/max: "
        f"[{lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}] / "
        f"[{hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}]"
    )
    print(f"Extent xyz: [{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]")
    print("=" * 72)


def visualize_mesh(mesh: o3d.geometry.TriangleMesh, mesh_path: str, wireframe: bool) -> None:
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Textured Mesh Viewer - {os.path.basename(mesh_path)}")
    vis.add_geometry(mesh)

    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.mesh_show_wireframe = bool(wireframe)
    render_opt.background_color = np.array([0.05, 0.05, 0.05], dtype=np.float64)

    print("Controls: left-drag rotate, shift+drag pan, wheel zoom, Q to close.")
    vis.run()
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a textured/colored mesh exported by point-to-pose."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Path to a mesh file (.glb/.obj/.ply). If omitted, resolve from --mesh_dir.",
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=None,
        help=f"Mesh directory (default: {_default_mesh_dir()}).",
    )
    parser.add_argument(
        "--object_number",
        "-o",
        type=int,
        default=0,
        help="Object index used when resolving mesh from --mesh_dir.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pred_mesh_obj",
        help="Mesh filename prefix when resolving from --mesh_dir.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Overlay wireframe while rendering.",
    )
    parser.add_argument(
        "--filter_disconnected",
        action="store_true",
        help="Filter out disconnected mesh components before visualization.",
    )
    parser.add_argument(
        "--keep_components",
        type=int,
        default=1,
        help="When --filter_disconnected is set, number of largest components to keep (default: 1).",
    )
    parser.add_argument(
        "--min_component_triangles",
        type=int,
        default=0,
        help="When --filter_disconnected is set, drop kept components below this triangle count.",
    )
    parser.add_argument(
        "--filter_bottom_growth",
        action="store_true",
        help="Filter bottom growth that extends outside object footprint.",
    )
    parser.add_argument(
        "--bottom_axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Axis used as vertical direction for bottom-growth filtering.",
    )
    parser.add_argument(
        "--bottom_quantile",
        type=float,
        default=0.10,
        help="Lower quantile band considered as bottom region (default: 0.10).",
    )
    parser.add_argument(
        "--bottom_radial_percentile",
        type=float,
        default=95.0,
        help="Percentile for robust object radius in horizontal plane (default: 95).",
    )
    parser.add_argument(
        "--bottom_radial_scale",
        type=float,
        default=1.15,
        help="Outlier radius multiplier for bottom-growth filtering (default: 1.15).",
    )
    args = parser.parse_args()

    mesh_path = resolve_mesh_path(
        mesh_path=args.mesh_path,
        mesh_dir=args.mesh_dir,
        object_number=args.object_number,
        prefix=args.prefix,
    )
    if mesh_path is None:
        print("No mesh found.")
        print("Use --mesh_path, or set --mesh_dir/--object_number to an existing export.")
        return

    mesh = load_mesh(mesh_path)
    if mesh is None:
        print(f"Failed to load a valid triangle mesh from: {mesh_path}")
        return

    if args.filter_disconnected:
        mesh, stats = filter_disconnected_components(
            mesh=mesh,
            keep_components=int(args.keep_components),
            min_component_triangles=int(args.min_component_triangles),
        )
        print(
            "Disconnected-component filtering: "
            f"components={stats['num_components']}, "
            f"kept_components={stats['kept_components']}, "
            f"kept_triangles={stats['kept_triangles']}, "
            f"removed_triangles={stats['removed_triangles']}"
        )

    if args.filter_bottom_growth:
        mesh, stats = filter_bottom_growth(
            mesh=mesh,
            bottom_axis=str(args.bottom_axis),
            bottom_quantile=float(args.bottom_quantile),
            bottom_radial_percentile=float(args.bottom_radial_percentile),
            bottom_radial_scale=float(args.bottom_radial_scale),
        )
        print(
            "Bottom-growth filtering: "
            f"bottom_threshold={stats['bottom_threshold']:.6f}, "
            f"radial_threshold={stats['radial_threshold']:.6f}, "
            f"removed_vertices={int(stats['removed_vertices'])}, "
            f"removed_triangles={int(stats['removed_triangles'])}, "
            f"kept_triangles={int(stats['kept_triangles'])}"
        )

    print_mesh_info(mesh_path, mesh)
    visualize_mesh(mesh, mesh_path, wireframe=args.wireframe)


if __name__ == "__main__":
    main()
