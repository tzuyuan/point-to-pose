"""
Textured mesh renderer built on nvdiffrast.

We render textured RGB + metric depth from a given camera pose ``T_obj2cam`` (OpenCV
convention: +Z forward into the scene, +Y down, +X right, world/object -> camera). This
avoids Open3D's OBJ/texture reader (which hard-crashes on some PNGs) and its EGL context.

Conventions
-----------
- Mesh (vertices, faces, uvs, texture) are loaded with trimesh + PIL as torch tensors.
- ``T_obj2cam`` maps object-frame points to the camera frame (CV convention).
- The returned depth is metric camera-frame Z (meters), 0 where no geometry -- directly
  compatible with ``convert_pixel_to_world`` (CV backprojection).

Lighting
--------
Shading comes from an environment map (``EnvmapLight``), fixed in OBJECT frame. Because the
render loop keeps the object fixed at the origin and orbits the camera around it (see
``MapBuilder``), object frame doubles as world frame here: the env map stays fixed in space
while the camera moves, exactly like a real light while you turn the object/camera in front of
it -- unlike a light welded to the object frame, which would rotate together with the object
and never match a real photo. If no ``envmap_path`` is given, the mesh is rendered unlit
(diffuse albedo only), which is preferable to a fake fixed light for photometric tracking.
"""

import numpy as np
import torch
import nvdiffrast.torch as dr

from point2pose.model_tracking.envmap_light import EnvmapLight


class NvdiffrastRenderer:
    def __init__(self, mesh_path, K, H, W, device="cuda", mesh_scale=1.0,
                 envmap_path=None, envmap_intensity=1.0, envmap_cube_res=128,
                 envmap_ambient_term=0.25, specular=False, roughness=0.5):
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.K = np.asarray(K, dtype=np.float64)

        self.envlight = None
        if envmap_path:
            self.envlight = EnvmapLight(
                envmap_path, cube_res=envmap_cube_res, device=device,
                intensity=envmap_intensity, ambient_term=envmap_ambient_term,
            )
        self.specular = bool(specular) and self.envlight is not None
        self.roughness = float(roughness)

        v, f, uv, tex, vn, vc = self._load_mesh(mesh_path, mesh_scale)
        self.center = v.mean(0)  # object frame == centered mesh frame
        v = v - self.center

        self.v = torch.tensor(v, dtype=torch.float32, device=device).contiguous()
        self.f = torch.tensor(f, dtype=torch.int32, device=device).contiguous()
        self.vn = torch.tensor(vn, dtype=torch.float32, device=device).contiguous()
        self.uv = (
            torch.tensor(uv, dtype=torch.float32, device=device) if uv is not None else None
        )
        self.tex = (
            torch.tensor(tex, dtype=torch.float32, device=device) / 255.0
            if tex is not None
            else None
        )
        # per-vertex colors (used when there is no UV texture)
        self.vc = (
            torch.tensor(vc, dtype=torch.float32, device=device).contiguous()
            if vc is not None else None
        )

        try:
            self.ctx = dr.RasterizeGLContext()
        except Exception:
            self.ctx = dr.RasterizeCudaContext()

        # OpenGL-style projection matrix from CV intrinsics.
        self.znear, self.zfar = 0.01, 10.0
        self.proj = self._gl_projection(self.K, self.W, self.H, self.znear, self.zfar)
        self.proj_t = torch.tensor(self.proj, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------ #
    def render(self, T_obj2cam):
        """
        Render one view.

        Args:
            T_obj2cam: (4,4) object->camera pose (CV convention).

        Returns:
            rgb: (H,W,3) uint8
            depth: (H,W) float32 metric camera-frame Z (0 where empty)
        """
        T = torch.tensor(T_obj2cam, dtype=torch.float32, device=self.device)

        n = self.v.shape[0]
        v_h = torch.cat([self.v, torch.ones(n, 1, device=self.device)], dim=1)  # (N,4)
        # camera-frame coords (CV): +Z forward
        v_cam = (T @ v_h.T).T  # (N,4)
        z_cam = v_cam[:, 2].contiguous()  # metric depth per vertex

        # clip-space via GL projection (expects CV camera coords with +Z forward)
        v_clip = (self.proj_t @ v_cam.T).T[None].contiguous().float()  # (1,N,4)

        rast, _ = dr.rasterize(self.ctx, v_clip, self.f, resolution=[self.H, self.W])

        # --- depth: interpolate camera-frame Z ---
        z_attr = z_cam[None, :, None].contiguous().float()  # (1,N,1)
        depth, _ = dr.interpolate(z_attr, rast, self.f)  # (1,H,W,1)
        depth = depth[0, ..., 0]
        mask = rast[0, ..., 3] > 0
        depth = torch.where(mask, depth, torch.zeros_like(depth))

        # --- color: UV texture, else per-vertex color, else plain white ---
        # A mesh with no color information renders as a WHITE object; env-map shading below
        # still gives it shape-dependent brightness so SuperPoint has gradients to key on.
        if self.tex is not None and self.uv is not None:
            uv_attr = self.uv[None].contiguous().float()  # (1,N,2)
            texc, _ = dr.interpolate(uv_attr, rast, self.f)  # (1,H,W,2)
            color = dr.texture(self.tex[None], texc, filter_mode="linear")[0]
        elif self.vc is not None:
            vc_attr = self.vc[None].contiguous().float()  # (1,N,3)
            color, _ = dr.interpolate(vc_attr, rast, self.f)  # (1,H,W,3)
            color = color[0]
        else:
            color = torch.ones((self.H, self.W, 3), device=self.device)  # white

        # --- shading: environment-map irradiance, object/world-frame normals ---
        if self.envlight is not None:
            n_attr = self.vn[None].contiguous().float()  # (1,N,3) object-frame normals
            n_px, _ = dr.interpolate(n_attr, rast, self.f)  # (1,H,W,3)
            n_px = n_px[0]
            n_px = n_px / (torch.linalg.norm(n_px, dim=-1, keepdim=True) + 1e-9)

            view_dir = None
            if self.specular:
                # camera position in object frame, from T_obj2cam: cam_pos = -R^T t
                R = T[:3, :3]
                t = T[:3, 3]
                cam_pos_obj = -(R.T @ t)
                pos_attr = self.v[None].contiguous().float()  # (1,N,3) object-frame positions
                pos_px, _ = dr.interpolate(pos_attr, rast, self.f)  # (1,H,W,3)
                view_dir = cam_pos_obj[None, None, :] - pos_px[0]
                view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-9)

            diffuse_irr, spec_irr = self.envlight.shade(n_px, view_dir, self.roughness)
            shaded = color * diffuse_irr
            if spec_irr is not None:
                shaded = shaded + spec_irr
            color = shaded.clamp(0.0, 1.0)

        color = torch.where(mask[..., None], color, torch.zeros_like(color))
        color = dr.antialias(color[None], rast, v_clip, self.f)[0]

        rgb = (color.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        depth_np = depth.cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(rgb), depth_np

    def delete(self):
        self.ctx = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_mesh(mesh_path, mesh_scale):
        import trimesh

        m = trimesh.load(mesh_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        v = np.asarray(m.vertices, dtype=np.float64)
        if mesh_scale != 1.0:
            v = v * float(mesh_scale)
        f = np.asarray(m.faces, dtype=np.int64)

        # Vertex normals for Lambert shading (trimesh computes them; fall back to a constant
        # up-normal if unavailable). Normals are rotation-covariant so no scaling needed.
        try:
            vn = np.asarray(m.vertex_normals, dtype=np.float64)
            if vn.shape != v.shape:
                raise ValueError
        except Exception:
            vn = np.tile(np.array([[0.0, 0.0, 1.0]]), (v.shape[0], 1))

        uv = None
        tex = None
        vc = None  # per-vertex RGB in [0,1] (for meshes with vertex colors instead of a texture)
        vis = getattr(m, "visual", None)
        if vis is not None and getattr(vis, "uv", None) is not None:
            uv = np.asarray(vis.uv, dtype=np.float64)
            mat = getattr(vis, "material", None)
            img = None
            if mat is not None:
                img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
            if img is not None:
                tex = np.asarray(img.convert("RGB"))
                # nvdiffrast samples with v increasing downward in the texture image; OBJ/UV
                # convention has v increasing upward, so flip V.
                uv = uv.copy()
                uv[:, 1] = 1.0 - uv[:, 1]

        # Fall back to vertex colors when there's no UV texture (e.g. .glb/.ply with per-vertex
        # colors). nvdiffrast interpolates them per-fragment just like any other attribute.
        # A UNIFORM vertex color (a single color for the whole mesh, which trimesh assigns as a
        # default when a mesh has no real color) carries no useful appearance -- treat it as
        # "no color" so the renderer falls back to a plain white object instead.
        if tex is None:
            vcol = getattr(vis, "vertex_colors", None) if vis is not None else None
            if vcol is not None and len(vcol) == len(v):
                cand = np.asarray(vcol, dtype=np.float64)[:, :3] / 255.0
                if np.ptp(cand, axis=0).max() > 1e-3:  # non-uniform -> real vertex colors
                    vc = cand
        return v, f, uv, tex, vn, vc

    @staticmethod
    def _gl_projection(K, W, H, znear, zfar):
        """
        Projection matrix from OpenCV intrinsics for a CV camera (+Z forward), producing
        clip coords for nvdiffrast.

        Key convention: nvdiffrast's output array has row 0 at NDC y = -1 (its NDC y is up,
        so the returned image is indexed bottom-up-in-NDC == top-down in array rows only if we
        map correctly). To make CV pixel row v (v=0 at the TOP) land in output row v, we need
        y_ndc(v=0) = -1, i.e. y_ndc = 2v/H - 1 -- the SAME sign structure as x. Using the
        opposite sign flips the image vertically (and, fatally, flips the backprojected 3D
        points relative to the mesh).

        For a point (X, Y, Z)_cam:  u = fx*X/Z + cx,  v = fy*Y/Z + cy
          x_ndc = 2u/W - 1,  y_ndc = 2v/H - 1
        so with w = Z:
          x_c = (2fx/W) X + (2cx/W - 1) Z
          y_c = (2fy/H) Y + (2cy/H - 1) Z
        z_c is chosen monotonic in Z for the depth test.
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        n, fr = znear, zfar
        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = 2 * fx / W
        P[0, 2] = 2 * cx / W - 1
        P[1, 1] = 2 * fy / H
        P[1, 2] = 2 * cy / H - 1
        P[2, 2] = (fr + n) / (fr - n)
        P[2, 3] = -2 * fr * n / (fr - n)
        P[3, 2] = 1.0  # w = Z
        return P
