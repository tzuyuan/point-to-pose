"""
Environment-map (image-based) lighting for the nvdiffrast renderer.

A simplified version of the split-sum IBL used in NVIDIA's nvdiffrec (render/light.py):
an equirectangular HDRI is converted to a cubemap. Diffuse irradiance is a brute-force
cosine-weighted hemisphere integral over a small (downsampled) direction set per output
texel -- box-blurring the base cubemap is NOT enough here because studio/indoor HDRIs have
very peaky light-source texels (hundreds to thousands of times brighter than the rest of
the sphere); averaging without a cosine weight either washes them out or (at low res)
represents them by a handful of lucky/unlucky texels, both of which read as far too dark.
The specular mip chain (for the optional cheap specular term) is still a plain repeated
2x2 box downsample of the base cubemap -- specular only needs "blurrier = rougher", not a
correctly normalized integral. No GGX importance sampling / BRDF LUT -- this trades
physical accuracy for simplicity, which is enough to get "shape reads like a real lit
photo, shading rotates with the camera like a real light would" instead of the previous
single fixed directional light welded to the object frame.

The env map is defined in OBJECT frame. Callers whose render loop keeps the object fixed
at the origin and moves the camera around it (as ``MapBuilder`` does) can treat object
frame as world frame directly -- the light then stays fixed in space while the camera
orbits, exactly like a real studio light while you turn the object/camera around it.
"""

import numpy as np
import torch
import nvdiffrast.torch as dr


def _cube_to_dir(s, x, y):
    """Direction for cube face `s` at NDC-ish coords (x, y) in [-1, 1]. Matches nvdiffrec's
    face layout: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z."""
    one = torch.ones_like(x)
    if s == 0:
        rx, ry, rz = one, -y, -x
    elif s == 1:
        rx, ry, rz = -one, -y, x
    elif s == 2:
        rx, ry, rz = x, one, y
    elif s == 3:
        rx, ry, rz = x, -one, -y
    elif s == 4:
        rx, ry, rz = x, -y, one
    else:
        rx, ry, rz = -x, -y, -one
    return torch.stack((rx, ry, rz), dim=-1)


def _safe_normalize(v, eps=1e-9):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def latlong_to_cubemap(latlong, res, device):
    """Equirectangular HxWx3 float image -> (6,res,res,3) cubemap, by sampling the latlong
    image at each cube texel's direction (u = atan2(x,-z)/2pi + 0.5, v = acos(y)/pi)."""
    latlong_t = torch.as_tensor(latlong, dtype=torch.float32, device=device)[None]  # (1,H,W,3)
    cubemap = torch.zeros(6, res, res, 3, dtype=torch.float32, device=device)
    lin = torch.linspace(-1 + 1.0 / res, 1 - 1.0 / res, res, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    for s in range(6):
        v = _safe_normalize(_cube_to_dir(s, gx, gy))
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], -1.0, 1.0)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)[None]  # (1,res,res,2)
        cubemap[s] = dr.texture(latlong_t, texcoord, filter_mode="linear")[0]
    return cubemap


def _avg_pool_cube(cubemap):
    """2x2 box downsample each of the 6 faces (N,H,W,C) -> (N,H/2,W/2,C)."""
    x = cubemap.permute(0, 3, 1, 2)  # (6,C,H,W)
    x = torch.nn.functional.avg_pool2d(x, 2)
    return x.permute(0, 2, 3, 1).contiguous()


def _cube_directions_and_solid_angles(res, device):
    """All (6*res*res) unit directions of a `res`-cubemap and their per-texel solid angle
    (needed so brighter/larger source texels contribute proportionally -- a cube face texel
    covers less solid angle near its corners than at its center)."""
    lin = torch.linspace(-1 + 1.0 / res, 1 - 1.0 / res, res, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    texel = 2.0 / res
    # Solid angle of a texel on a unit cube face (Reference: standard cubemap texel solid
    # angle formula): d_omega = texel^2 / (x^2+y^2+1)^1.5 * cos-correction already folded in
    # via the (x^2+y^2+1) denominator (distance^2 from cube center to the texel on the face).
    dist2 = gx * gx + gy * gy + 1.0
    solid_angle = (texel * texel) / (dist2 * torch.sqrt(dist2))  # (res,res)
    dirs = torch.stack([_safe_normalize(_cube_to_dir(s, gx, gy)) for s in range(6)], dim=0)
    omega = solid_angle[None].expand(6, res, res).contiguous()
    return dirs, omega  # (6,res,res,3), (6,res,res)


def diffuse_irradiance_cubemap(src_cube, out_res, sample_res=16, device="cuda"):
    """Cosine-weighted hemisphere integral of `src_cube` (a (6,H,W,3) radiance cubemap),
    evaluated at each of `out_res`-cubemap's texel normals. Brute-force O(out_res^2 *
    sample_res^2) -- fine at the small resolutions IBL needs (e.g. out_res=8, sample_res=16).

    Downsamples `src_cube` to `sample_res` first (cheap box filter) so the integral only
    has to sum ~6*sample_res^2 terms per output texel instead of the full source resolution.
    """
    src = src_cube
    while src.shape[1] > sample_res:
        src = _avg_pool_cube(src)
    sample_dirs, sample_omega = _cube_directions_and_solid_angles(src.shape[1], src.device)
    sample_dirs = sample_dirs.reshape(-1, 3)  # (M,3)
    sample_omega = sample_omega.reshape(-1)  # (M,)
    sample_radiance = src.reshape(-1, 3)  # (M,3)

    out_dirs, _ = _cube_directions_and_solid_angles(out_res, device)  # (6,res,res,3)
    n = out_dirs.reshape(-1, 3)  # (P,3)

    # cos(theta) between each output normal and each sample direction, clamped to the
    # (upper) hemisphere; weight by the sample's solid angle and divide by pi so a
    # uniformly-lit sphere (radiance=1 everywhere) integrates to irradiance=1 (energy-
    # preserving normalization, matches the standard diffuse-irradiance convention).
    cos_theta = (n @ sample_dirs.T).clamp(min=0.0)  # (P,M)
    weight = cos_theta * sample_omega[None, :] / np.pi  # (P,M)
    irradiance = weight @ sample_radiance  # (P,3)
    return irradiance.reshape(6, out_res, out_res, 3).contiguous()


class EnvmapLight:
    """Object/world-frame environment light: diffuse irradiance cubemap + a roughness mip
    chain for a (optional, cheap) specular term. Usage mirrors nvdiffrec's EnvironmentLight
    but without the GGX/BRDF-LUT split-sum -- box-blur mips stand in for prefiltering.
    """

    MIN_RES = 8
    DIFFUSE_RES = 8
    DIFFUSE_SAMPLE_RES = 16

    def __init__(self, hdr_path, cube_res=128, device="cuda", intensity=1.0, ambient_term=0.25):
        """
        Args:
            ambient_term: flat term ADDED to the diffuse irradiance everywhere (Phong-style
                ambient + diffuse), as a fraction of the diffuse cubemap's *mean* irradiance.
                A pure directional-cosine integral makes faces pointed away from every bright
                direction in the HDRI go fully black, which real surfaces never do (indirect
                bounce off walls/floor/other objects fills them in). Adding a uniform constant
                (rather than clamping each texel to a floor) keeps the original shading shape
                -- the falloff/contrast between lit and shadowed faces is preserved, just
                lifted -- which is what a real room's ambient bounce actually looks like. 0
                disables it; ~0.2-0.4 reads close to a real photo of a small object indoors.
        """
        self.device = device
        # This precompute must run at full float32 precision regardless of any ambient
        # autocast context (e.g. SAM2 enters a process-wide bf16 autocast at import time) --
        # dr.texture() requires float32 input, and the diffuse integral's dynamic range
        # (HDRI light-source texels can be 1000x the average) is precision-sensitive.
        with torch.autocast(device_type="cuda", enabled=False):
            base = self._load_hdr(hdr_path) * float(intensity)
            base_cube = latlong_to_cubemap(base, cube_res, device)

            # Mip chain: index 0 = sharpest (base), last = most blurred. Used only for the
            # (optional, cheap) specular term -- "blurrier mip = rougher reflection" doesn't
            # need a properly normalized integral, just progressively softer copies.
            self.mips = [base_cube]
            while self.mips[-1].shape[1] > self.MIN_RES:
                self.mips.append(_avg_pool_cube(self.mips[-1]))
            self._spec_base = self.mips[0][None].contiguous().float()
            self._spec_stack = [m[None].contiguous().float() for m in self.mips[1:]]
            self.num_mips = len(self.mips)

            # Diffuse irradiance: a real cosine-weighted hemisphere integral (see
            # `diffuse_irradiance_cubemap`) -- a plain box-blur mip under-represents the
            # bright, spatially-small light sources typical of indoor/studio HDRIs and comes
            # out far too dark; the solid-angle-weighted integral correctly accounts for
            # their contribution.
            diffuse_cube = diffuse_irradiance_cubemap(
                base_cube, self.DIFFUSE_RES, sample_res=self.DIFFUSE_SAMPLE_RES, device=device,
            )
            if ambient_term > 0:
                diffuse_cube = diffuse_cube + float(ambient_term) * diffuse_cube.mean()
            self.diffuse = diffuse_cube[None].contiguous().float()  # (1,6,res,res,3)

    @staticmethod
    def _load_hdr(path):
        import cv2

        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"could not load HDRI: {path}")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img.astype(np.float32))

    def shade(self, normal, view_dir=None, roughness=0.5):
        """
        Args:
            normal: (H,W,3) unit surface normals, OBJECT/WORLD frame (see module docstring).
            view_dir: (H,W,3) unit vectors surface->camera, OBJECT/WORLD frame. Only needed if
                you want the (optional) specular term; diffuse alone ignores it.
            roughness: scalar in [0,1] controlling which mip the (optional) specular term reads.
        Returns:
            diffuse_rgb: (H,W,3) diffuse irradiance (unitless env-map radiance, NOT yet
                multiplied by albedo -- caller does `color = albedo * diffuse_rgb`).
            specular_rgb: (H,W,3) or None if view_dir is None.
        """
        # dr.texture() requires float32 tensors; force it regardless of any ambient autocast
        # (e.g. SAM2 enters a process-wide bf16 autocast at import time -- see __init__).
        with torch.autocast(device_type="cuda", enabled=False):
            n = normal[None].float().contiguous()  # (1,H,W,3)
            diffuse_rgb = dr.texture(
                self.diffuse, n, filter_mode="linear", boundary_mode="cube",
            )[0]

            specular_rgb = None
            if view_dir is not None:
                reflvec = _safe_normalize(
                    2.0 * (normal * view_dir).sum(-1, keepdim=True) * normal - view_dir
                )[None].float().contiguous()
                mip_level = float(np.clip(roughness, 0.0, 1.0)) * (self.num_mips - 1)
                bias = torch.full(reflvec.shape[:3], mip_level, dtype=torch.float32,
                                  device=self.device)
                specular_rgb = dr.texture(
                    self._spec_base, reflvec, mip=self._spec_stack, mip_level_bias=bias,
                    filter_mode="linear-mipmap-linear", boundary_mode="cube",
                )[0]
        return diffuse_rgb, specular_rgb
