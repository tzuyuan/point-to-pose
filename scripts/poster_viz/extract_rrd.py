"""Extract poster-usable data from the cheetah.rrd live-demo recording.

Dumps to <out>/:
    mesh_f{frame:06d}.npz     vertex_positions, triangle_indices, vertex_colors, normals
    keyframes.npz             per-kf: pose (4x4), K (3x3), resolution
    kf_{i:02d}.jpg|png        keyframe images
    map_points_f{frame}.npz   positions + colors at a few times
    trajectory.npz            final trajectory strips
    cam_f{frame:06d}.png      a few live camera frames
    traces_f{frame}.npz       2D track traces (strips, colors) at a few times
"""
import io
import os
import re

import numpy as np

from rerun.server import Server

RRD = "/home/justin/results/eccv_point2pose/rerun_demo/cheetah.rrd"
OUT = "/tmp/claude-1000/-home-justin-code-point-to-pose/98025e7b-880c-4111-94de-90330a4d52a2/scratchpad/rrd_extract"
os.makedirs(OUT, exist_ok=True)



def to_bytes(blob):
    """EncodedImage blob arrives as bytes, or a (possibly nested) list of ints."""
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if isinstance(blob, list):
        if len(blob) == 1 and isinstance(blob[0], (bytes, bytearray, list)):
            return to_bytes(blob[0])
        return bytes(bytearray(blob))
    raise TypeError(type(blob))

srv = Server(host="127.0.0.1", datasets={"demo": [RRD]})
client = srv.client()
ds = client.get_dataset("demo")


def read(entity, component_cols, index="frame"):
    """Return dict of frame -> tuple(values) for non-null rows."""
    cols = [f"{entity}:{c}" for c in component_cols]
    df = ds.filter_contents([entity]).reader(index=index)
    tbl = df.to_arrow_table().to_pylist()
    rows = {}
    for row in tbl:
        vals = [row.get(c) for c in cols]
        if all(v is None for v in vals):
            continue
        rows[row[index]] = vals
    return rows


# ---- meshes over time ----
mesh_rows = read("/world/obj_0/mesh",
                 ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices",
                  "Mesh3D:vertex_colors", "Mesh3D:vertex_normals"])
print("mesh versions at frames:", sorted(mesh_rows))
for f, (vp, ti, vc, vn) in sorted(mesh_rows.items()):
    if vp is None:
        continue
    np.savez_compressed(
        f"{OUT}/mesh_f{f:06d}.npz",
        vertex_positions=np.array([[q["x"], q["y"], q["z"]] if isinstance(q, dict) else q for q in vp], dtype=np.float32),
        triangle_indices=np.array(ti),
        vertex_colors=np.array(vc) if vc is not None else np.array([]),
        vertex_normals=np.array(vn) if vn is not None else np.array([]),
    )
print("saved", len(mesh_rows), "meshes")

# ---- keyframes: pose + pinhole + image ----
sch = ds.schema()
kf_paths = sorted(set(
    str(c.entity_path) for c in sch.component_columns()
    if re.fullmatch(r"/world/obj_0/keyframes/kf_\d+", str(c.entity_path))
), key=lambda p: int(p.rsplit("_", 1)[1]))
poses, Ks, resos, kf_ids, kf_frames = [], [], [], [], []
for p in kf_paths:
    i = int(p.rsplit("_", 1)[1])
    rows = read(p, ["Transform3D:mat3x3", "Transform3D:translation",
                    "Pinhole:image_from_camera", "Pinhole:resolution"])
    if not rows:
        continue
    f, (m3, tr, K, reso) = sorted(rows.items())[-1]
    T = np.eye(4)
    T[:3, :3] = np.array(m3).reshape(3, 3)
    tr = tr[0] if isinstance(tr, list) else tr
    T[:3, 3] = [tr["x"], tr["y"], tr["z"]] if isinstance(tr, dict) else np.array(tr).ravel()[:3]
    poses.append(T)
    Ks.append(np.array(K).reshape(3, 3) if K is not None else np.eye(3))
    resos.append(np.array(reso).ravel() if reso is not None else np.zeros(2))
    kf_ids.append(i)
    kf_frames.append(f)
    img = read(p + "/image", ["EncodedImage:blob", "EncodedImage:media_type"])
    if img:
        _, (blob, mt) = sorted(img.items())[-1]
        mt = mt[0] if isinstance(mt, list) else (mt or "image/jpeg")
        ext = "png" if "png" in str(mt) else "jpg"
        with open(f"{OUT}/kf_{i:02d}.{ext}", "wb") as fh:
            fh.write(to_bytes(blob))
np.savez(f"{OUT}/keyframes.npz", poses=np.array(poses), Ks=np.array(Ks),
         resolutions=np.array(resos), kf_ids=np.array(kf_ids),
         kf_frames=np.array(kf_frames))
print("saved", len(poses), "keyframe poses+images")

# ---- map points (few time samples) ----
mp = read("/world/obj_0/map_points", ["Points3D:positions", "Points3D:colors"])
frames = sorted(mp)
picks = [frames[len(frames)//4], frames[len(frames)//2], frames[-1]]
for f in picks:
    pos, col = mp[f]
    np.savez_compressed(f"{OUT}/map_points_f{f:06d}.npz",
                        positions=np.array([[q["x"], q["y"], q["z"]] if isinstance(q, dict) else q for q in pos], dtype=np.float32),
                        colors=np.array(col) if col is not None else np.array([]))
print("map_points frames saved:", picks, "final count:", len(mp[frames[-1]][0]))

# ---- trajectory ----
tr = read("/world/obj_0/trajectory", ["LineStrips3D:strips", "LineStrips3D:colors"])
if tr:
    f, (strips, cols) = sorted(tr.items())[-1]
    flat = []
    for s in strips:
        pts = [[q["x"], q["y"], q["z"]] if isinstance(q, dict) else list(np.ravel(q))[:3] for q in s]
        flat.append(np.array(pts, dtype=np.float32))
    np.savez_compressed(f"{OUT}/trajectory.npz",
                        n=len(flat), **{f"strip_{k}": s for k, s in enumerate(flat)})
    print("trajectory strips:", len(flat), "at frame", f)

# ---- a few live camera frames + final 2D traces ----
cam = read("/world/obj_0/camera/image", ["EncodedImage:blob", "EncodedImage:media_type"])
cf = sorted(cam)
print("camera image frames:", cf[:3], "...", cf[-3:], f"({len(cf)} total)")
for f in [cf[0], cf[len(cf)//3], cf[2*len(cf)//3], cf[-1]]:
    blob, mt = cam[f]
    mt = mt[0] if isinstance(mt, list) else (mt or "image/jpeg")
    ext = "png" if "png" in str(mt) else "jpg"
    with open(f"{OUT}/cam_f{f:06d}.{ext}", "wb") as fh:
        fh.write(to_bytes(blob))

tra = read("/camframe/obj_0/traces", ["LineStrips2D:strips", "LineStrips2D:colors"])
tf = sorted(tra)
print("traces frames:", len(tf))
for f in [tf[len(tf)//3], tf[2*len(tf)//3], tf[-1]]:
    strips, cols = tra[f]
    flat = [np.array([[q["x"], q["y"]] if isinstance(q, dict) else list(np.ravel(q))[:2] for q in s],
                     dtype=np.float32) for s in strips]
    np.savez_compressed(f"{OUT}/traces_f{f:06d}.npz", n=len(flat),
                        colors=np.array(cols) if cols is not None else np.array([]),
                        **{f"strip_{k}": s for k, s in enumerate(flat)})

srv.shutdown()
print("DONE ->", OUT)
