"""
Gaussian-splat renderer built on gsplat, as a drop-in replacement for
``NvdiffrastRenderer`` in the offline model-map pipeline.

Loads a trained gaussian checkpoint (``means``/``quats``/``scales``/``opacities``/
``sh0``/``shN``, as saved by kv_tracker's ``reconstruct.py``) instead of a textured mesh,
and renders textured RGB + metric depth from a given camera pose ``T_obj2cam`` (OpenCV
convention: +Z forward into the scene, +Y down, +X right, world/object -> camera).

Conventions
-----------
- The gaussian checkpoint is trained in whatever coordinate frame kv_tracker's
  reconstruction used (metric-scaled world frame centered near the object). That frame is
  treated as the *object frame* here, exactly like ``NvdiffrastRenderer`` treats the
  centered mesh frame as the object frame -- ``MapBuilder`` orbits a virtual camera around
  the origin, so no extra alignment step is required as long as the reconstruction's
  world origin sits at (or near) the object center.
- ``T_obj2cam`` maps object-frame points to the camera frame (CV convention), identical to
  ``NvdiffrastRenderer``. gsplat's ``rasterization``/``rasterization_2dgs`` expect
  ``viewmats`` = world-to-camera, i.e. exactly ``T_obj2cam`` (no extra inversion needed).
- The returned depth is metric camera-frame Z (meters), 0 where no geometry -- directly
  compatible with ``convert_pixel_to_world`` (CV backprojection), matching
  ``NvdiffrastRenderer``'s contract.
"""

import numpy as np
import torch

from gsplat.rendering import rasterization, rasterization_2dgs


class GsplatRenderer:
    """Renders a pretrained gaussian splat checkpoint with the same ``render(T_obj2cam) ->
    (rgb, depth)`` / ``delete()`` interface as ``NvdiffrastRenderer``, so it can be used as
    a ``MapBuilder`` mapper without any other pipeline changes."""

    def __init__(self, splat_path, K, H, W, device="cuda", model="2dgs",
                 sh_degree=None, near_plane=0.01, far_plane=10.0,
                 depth_mode="median"):
        """
        Args:
            splat_path: path to a gaussian checkpoint (.pt dict with means/quats/scales/
                opacities/sh0/shN, as saved by kv_tracker's ``reconstruct.py``).
            K: (3,3) camera intrinsics used to render.
            H, W: render resolution.
            model: "2dgs" (rasterization_2dgs) or "3dgs" (rasterization).
            sh_degree: SH degree to evaluate. None uses the checkpoint's full degree.
            depth_mode: "median" (2DGS's actual surface depth) or "accum" (alpha-
                accumulated expected depth, via render_mode="RGB+ED"). 2DGS only returns a
                median depth; 3DGS only has accumulated depth, so this is ignored for 3DGS.
        """
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.K = np.asarray(K, dtype=np.float64)
        self.model = model
        assert model in ("2dgs", "3dgs"), f"unknown model {model!r}, expected 2dgs or 3dgs"
        self.depth_mode = depth_mode

        splats = torch.load(splat_path, map_location=device)
        self.means = splats["means"].to(device)
        self.quats = splats["quats"].to(device)
        self.scales = torch.exp(splats["scales"]).to(device)
        self.opacities = torch.sigmoid(splats["opacities"]).to(device)
        self.colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)  # (N,K,3)

        max_sh_degree = int(np.sqrt(self.colors.shape[1]) - 1)
        self.sh_degree = max_sh_degree if sh_degree is None else int(sh_degree)

        self.Ks = torch.tensor(self.K, dtype=torch.float32, device=device)[None]  # (1,3,3)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)

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
        viewmat = torch.tensor(T_obj2cam, dtype=torch.float32, device=self.device)[None]  # (1,4,4)

        with torch.no_grad():
            if self.model == "2dgs":
                colors, alphas, _, _, _, median_depth, _ = rasterization_2dgs(
                    means=self.means,
                    quats=self.quats,
                    scales=self.scales,
                    opacities=self.opacities,
                    colors=self.colors,
                    viewmats=viewmat,
                    Ks=self.Ks,
                    width=self.W,
                    height=self.H,
                    sh_degree=self.sh_degree,
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    # median_depth only comes out correct when the accumulated-depth (ED)
                    # pass runs too -- with render_mode="RGB" it's silently wrong (verified:
                    # same call with "RGB" gives a corrupted median_depth full of near/far-
                    # plane outliers, "RGB+ED" gives depths matching the actual camera
                    # distance). So always render RGB+ED regardless of depth_mode.
                    render_mode="RGB+ED",
                )
                mask = alphas[0, ..., 0] > 0
                if self.depth_mode == "median":
                    depth = median_depth[0, ..., 0]
                else:
                    depth = colors[0, ..., 3]
            else:
                colors, alphas, _ = rasterization(
                    means=self.means,
                    quats=self.quats,
                    scales=self.scales,
                    opacities=self.opacities,
                    colors=self.colors,
                    viewmats=viewmat,
                    Ks=self.Ks,
                    width=self.W,
                    height=self.H,
                    sh_degree=self.sh_degree,
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    render_mode="RGB+ED",
                    packed=False,
                )
                mask = alphas[0, ..., 0] > 0
                depth = colors[0, ..., 3]

            depth = torch.where(mask, depth, torch.zeros_like(depth))
            rgb = (colors[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            depth_np = depth.cpu().numpy().astype(np.float32)

        return np.ascontiguousarray(rgb), depth_np

    def delete(self):
        self.means = self.quats = self.scales = self.opacities = self.colors = None
        torch.cuda.empty_cache()
