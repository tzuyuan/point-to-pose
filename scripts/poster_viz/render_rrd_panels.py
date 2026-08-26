"""Render poster panels from the extracted cheetah.rrd data (run with ms env).

Outputs:
    panel5_graph_cheetah.png   keyframe frusta + real keyframe images + colored map
    panel6_mesh_growth.png     TSDF mesh growth strip (4 stages)
    panel6_mesh_cheetah_*.png  final cheetah mesh, 2 views
"""
import glob
import os

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

EX = "/tmp/claude-1000/-home-justin-code-point-to-pose/98025e7b-880c-4111-94de-90330a4d52a2/scratchpad/rrd_extract"
OUT = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"
W = H = 1100


def unpack_colors(raw, n):
    """Rerun colors are u32 0xRRGGBBAA (or already Nx3/Nx4)."""
    a = np.asarray(raw)
    if a.size == 0:
        return np.full((n, 3), 0.6)
    if a.ndim == 2 and a.shape[1] in (3, 4):
        c = a[:, :3].astype(np.float64)
        return c / 255.0 if c.max() > 1.5 else c
    a = a.astype(np.int64).ravel()
    r = (a >> 24) & 255
    g = (a >> 16) & 255
    b = (a >> 8) & 255
    return np.stack([r, g, b], 1) / 255.0


def load_mesh(path):
    d = np.load(path, allow_pickle=True)
    vp = d["vertex_positions"]
    ti = np.asarray(d["triangle_indices"])
    if ti.dtype == object or ti.ndim == 1:
        ti = np.array([[q["x"], q["y"], q["z"]] if isinstance(q, dict) else list(np.ravel(q))[:3]
                       for q in ti])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vp.astype(np.float64)),
        o3d.utility.Vector3iVector(ti.astype(np.int32)))
    vc = unpack_colors(d["vertex_colors"], len(vp))
    m.vertex_colors = o3d.utility.Vector3dVector(vc)
    m.compute_vertex_normals()
    return m


def new_renderer(bg=(1, 1, 1, 1)):
    ren = rendering.OffscreenRenderer(W, H)
    ren.scene.set_background(list(bg))
    ren.scene.scene.set_sun_light([0.4, 0.4, -0.8], [1, 1, 1], 70000)
    ren.scene.scene.enable_sun_light(True)
    return ren


def mat(shader="defaultLit", line_width=None):
    m = rendering.MaterialRecord()
    m.shader = shader
    if line_width:
        m.line_width = line_width
    return m


def render_scene(ren, center, ext, direction, out, fov=32):
    d = np.asarray(direction, float)
    eye = center + d / np.linalg.norm(d) * ext * 1.9
    ren.setup_camera(fov, center, eye, [0, -1, 0])
    o3d.io.write_image(out, ren.render_to_image())
    print("saved", out)


# ================= mesh growth strip =================
mesh_files = sorted(glob.glob(f"{EX}/mesh_f*.npz"))
frames = [int(os.path.basename(f)[6:12]) for f in mesh_files]
picks = [frames[0], 90, 146, frames[-1]]
picks = [min(frames, key=lambda x: abs(x - p)) for p in picks]
final_mesh = load_mesh(f"{EX}/mesh_f{frames[-1]:06d}.npz")
bb = final_mesh.get_axis_aligned_bounding_box()
c, ext = np.asarray(bb.get_center()), np.linalg.norm(bb.get_extent())

tiles = []
for f in picks:
    m = load_mesh(f"{EX}/mesh_f{f:06d}.npz")
    ren = new_renderer()
    ren.scene.add_geometry("m", m, mat())
    render_scene(ren, c, ext, [0.55, -0.3, -1.0], f"{OUT}/_tmp_growth_{f}.png")
    del ren
    tile = cv2.imread(f"{OUT}/_tmp_growth_{f}.png")
    os.remove(f"{OUT}/_tmp_growth_{f}.png")
    tiles.append(tile)
# crop all tiles with one shared content box so scale stays comparable
fg = np.zeros(tiles[0].shape[:2], bool)
for t in tiles:
    fg |= (np.abs(t.astype(int) - t[0, 0].astype(int)).sum(2) > 24)
ys, xs = np.where(fg)
pad = 30
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, fg.shape[0])
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, fg.shape[1])
tiles = [t[y0:y1, x0:x1] for t in tiles]
strip = np.concatenate(tiles, axis=1)
cv2.imwrite(f"{OUT}/panel6_mesh_growth_cheetah.png", strip)
print("saved", f"{OUT}/panel6_mesh_growth_cheetah.png", "stages:", picks)

# final mesh, two views
for tag, d in [("a", [0.55, -0.3, -1.0]), ("b", [-0.7, -0.25, -1.0])]:
    ren = new_renderer()
    ren.scene.add_geometry("m", final_mesh, mat())
    render_scene(ren, c, ext, d, f"{OUT}/panel6_mesh_cheetah_{tag}.png")
    del ren

# ================= pose graph with real keyframe images =================
kf = np.load(f"{EX}/keyframes.npz")
poses, Ks, resos, kf_ids = kf["poses"], kf["Ks"], kf["resolutions"], kf["kf_ids"]

mp_file = sorted(glob.glob(f"{EX}/map_points_f*.npz"))[-1]
mp = np.load(mp_file, allow_pickle=True)
pts = mp["positions"]
cols = unpack_colors(mp["colors"], len(pts))

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
pc.colors = o3d.utility.Vector3dVector(cols)

ctr = pts.mean(0)
ext_m = np.linalg.norm(pts.max(0) - pts.min(0))

VIEW_DIR = np.array([0.15, -0.45, -1.0])
VIEW_DIR = VIEW_DIR / np.linalg.norm(VIEW_DIR)

# prefer far-side keyframes (their image planes face the render camera),
# then greedy farthest-point for spread
pos = poses[:, :3, 3]
dist_ctr = np.linalg.norm(pos - ctr, axis=1)
dirn = (pos - ctr) / dist_ctr[:, None]
far = np.where((dirn @ VIEW_DIR < -0.45) & (dist_ctr > 0.5))[0]
if len(far) < 5:
    far = np.where(dirn @ VIEW_DIR < -0.3)[0]
sel = [int(far[0])]
while len(sel) < min(6, len(far)):
    dmin = np.min(np.linalg.norm(pos[far][:, None] - pos[sel][None], axis=2), axis=1)
    sel.append(int(far[int(np.argmax(dmin))]))
sel = sorted(set(sel))

# real keyframe positions (0.4-0.6m around the map) are already legible
ren = new_renderer(bg=(1, 1, 1, 1))
map_mat = mat("defaultUnlit")
map_mat.point_size = 5.0
ren.scene.add_geometry("map", pc, map_mat)

# observation edges: thin lines from each selected keyframe to sample points
rng = np.random.default_rng(3)
for si in sel:
    o = poses[si][:3, 3]
    # target points on this camera's near side so edges don't pierce the cloud
    dcam = np.linalg.norm(pts - o, axis=1)
    near = np.where(dcam < np.percentile(dcam, 35))[0]
    targets = pts[rng.choice(near, min(5, len(near)), replace=False)]
    lp = np.vstack([o[None], targets])
    edges = [(0, k + 1) for k in range(len(targets))]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(lp),
                              o3d.utility.Vector2iVector(edges))
    ls.colors = o3d.utility.Vector3dVector(np.tile([[0.72, 0.76, 0.8]], (len(edges), 1)))
    ren.scene.add_geometry(f"obs{si}", ls, mat("unlitLine", line_width=1.2))

DEPTH = 0.42 * ext_m           # image plane distance in front of each camera
HW = 0.17 * ext_m              # plane half-width; height from 4:3 aspect
for si in sel:
    T = poses[si].copy()
    z = DEPTH
    hw, hh = HW, HW * 120.0 / 160.0
    corners_local = np.array([[-hw, -hh, z], [hw, -hh, z], [hw, hh, z], [-hw, hh, z]])
    corners = (T[:3, :3] @ corners_local.T).T + T[:3, 3]
    o = T[:3, 3]
    # frustum wireframe
    lines = [(0, k + 1) for k in range(4)] + [(k + 1, (k + 1) % 4 + 1) for k in range(4)]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.vstack([o, corners])),
        o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile([[0.15, 0.35, 0.75]], (len(lines), 1)))
    ren.scene.add_geometry(f"fr{si}", ls, mat("unlitLine", line_width=3))
    # image plane
    img_path = f"{EX}/kf_{kf_ids[si]:02d}.jpg"
    if os.path.exists(img_path):
        img = o3d.io.read_image(img_path)
        plane = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(corners),
            o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
        plane.triangle_uvs = o3d.utility.Vector2dVector(
            np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float64))
        plane.triangle_material_ids = o3d.utility.IntVector([0, 0])
        plane.textures = [img]
        pm = mat("defaultUnlit")
        pm.albedo_img = img
        ren.scene.add_geometry(f"img{si}", plane, pm)

frame_ctr = 0.55 * ctr + 0.45 * pos[sel].mean(0)
render_scene(ren, frame_ctr, 0.60, VIEW_DIR,
             f"{OUT}/panel5_graph_cheetah.png", fov=38)
