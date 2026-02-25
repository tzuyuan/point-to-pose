#!/usr/bin/env python3
"""
Visualize textured/colored meshes exported by the SDF pipeline.

Examples:
  python visualize_textured_mesh.py --mesh_path /path/to/pred_mesh_obj_0_textured.glb
  python visualize_textured_mesh.py --mesh_dir /path/to/out/mesh --object_number 0
"""

import argparse
import os
from typing import Optional

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

    print_mesh_info(mesh_path, mesh)
    visualize_mesh(mesh, mesh_path, wireframe=args.wireframe)


if __name__ == "__main__":
    main()
