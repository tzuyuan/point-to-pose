"""Panel 6: TSDF reconstruction — offscreen renders of reconstructed meshes.

White background, largest-connected-component filtering, three views each.
"""
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from common import OUT_DIR

FINAL = "/home/justin/results/eccv_point2pose/final_results"
MESHES = [
    ("mustard", f"{FINAL}/ycb_multi_track_final/006_mustard_bottle/mesh/pred_mesh_obj_0.ply"),
    ("meat", f"{FINAL}/ycb_multi_track_final/010_potted_meat_can/mesh/pred_mesh_obj_0.ply"),
    ("tomato", f"{FINAL}/ycb_multi_track_final/005_tomato_soup_can/mesh/pred_mesh_obj_0.ply"),
    ("bleach", f"{FINAL}/ycb_multi_track_final/021_bleach_cleanser/mesh/pred_mesh_obj_0.ply"),
    ("sloth", "/home/justin/results/eccv_point2pose/good_mesh/sloth.ply"),
]

W = H = 900
VIEWS = [("a", [0.7, -0.35, -1.0]), ("b", [-0.8, -0.3, -1.0]), ("c", [0.1, -0.9, -0.7])]


def largest_component(mesh):
    tri_cl, n_tri, _ = mesh.cluster_connected_triangles()
    tri_cl = np.asarray(tri_cl)
    n_tri = np.asarray(n_tri)
    keep = tri_cl == n_tri.argmax()
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh


for name, path in MESHES:
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        print(name, "EMPTY"); continue
    mesh = largest_component(mesh)
    mesh.compute_vertex_normals()
    bb = mesh.get_axis_aligned_bounding_box()
    c = bb.get_center()
    ext = np.linalg.norm(bb.get_extent())
    for tag, d in VIEWS:
        ren = rendering.OffscreenRenderer(W, H)
        ren.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        ren.scene.add_geometry("m", mesh, mat)
        ren.scene.scene.set_sun_light([0.4, 0.4, -0.8], [1, 1, 1], 75000)
        ren.scene.scene.enable_sun_light(True)
        d = np.asarray(d, float)
        eye = c + d / np.linalg.norm(d) * ext * 1.5
        ren.setup_camera(35.0, c, eye, [0, -1, 0])
        img = ren.render_to_image()
        out = f"{OUT_DIR}/panel6_mesh_{name}_{tag}.png"
        o3d.io.write_image(out, img)
        print("saved", out)
        del ren
