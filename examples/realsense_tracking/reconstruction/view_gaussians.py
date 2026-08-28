"""
Load a trained gaussians.pt and view it free-viewpoint in viser.

Run:
    python examples/realsense_tracking/reconstruction/view_gaussians.py \
        --checkpoint debug/my_capture/gaussians.pt [--model 2dgs]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import viser

from gsplat.rendering import rasterization, rasterization_2dgs


def rigid_inverse(camtoworlds: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a batch of rigid (rotation+translation) 4x4 transforms --
    avoids torch.linalg.inv_ex, which can hit a 'lazy wrapper should be called at most
    once' error when first invoked from a non-main thread (e.g. viser's callback
    thread)."""
    R = camtoworlds[..., :3, :3]
    t = camtoworlds[..., :3, 3]
    R_t = R.transpose(-1, -2)
    out = torch.eye(4, dtype=camtoworlds.dtype, device=camtoworlds.device).expand_as(camtoworlds).clone()
    out[..., :3, :3] = R_t
    out[..., :3, 3] = -(R_t @ t.unsqueeze(-1)).squeeze(-1)
    return out


def render_splats(model, splats, camtoworlds, Ks, width, height, sh_degree,
                   near_plane=0.001, far_plane=100.0, render_mode="RGB"):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    viewmats = rigid_inverse(camtoworlds.float())

    if model == "2dgs":
        render_colors, render_alphas, _, _, _, render_median, info = rasterization_2dgs(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height, sh_degree=sh_degree,
            near_plane=near_plane, far_plane=far_plane, render_mode=render_mode,
        )
    else:
        render_colors, render_alphas, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height, sh_degree=sh_degree,
            near_plane=near_plane, far_plane=far_plane, render_mode=render_mode, packed=False,
        )
    return render_colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="2dgs", choices=["2dgs", "3dgs"])
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    device = "cuda:0"
    splats = torch.load(args.checkpoint, map_location=device)
    splats = {k: v.to(device) for k, v in splats.items()}
    sh_degree = int(round(math.sqrt(splats["sh0"].shape[1] + splats["shN"].shape[1]) - 1))
    print(f"Loaded {splats['means'].shape[0]} gaussians (sh_degree={sh_degree}) "
          f"from {args.checkpoint}")

    # Warm up gsplat's internal CUDA lazy wrappers (e.g. torch.inverse) on the MAIN
    # thread -- calling them for the first time from viser's callback thread instead
    # raises "lazy wrapper should be called at most once".
    with torch.no_grad():
        dummy_c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        dummy_K = torch.tensor(
            [[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], dtype=torch.float32, device=device
        ).unsqueeze(0)
        render_splats(args.model, splats, dummy_c2w, dummy_K, 64, 64, sh_degree)
    print("Warmed up CUDA kernels.")

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # CUDA context is thread-local; viser fires connect/camera-update callbacks on its
    # own worker threads, and touching CUDA tensors from those threads crashes. So the
    # callback only enqueues a request; the render itself always runs on the MAIN
    # thread, drained in the loop below.
    pending_clients = set()

    def render_for_client(client):
        """MAIN THREAD ONLY."""
        cam = client.camera
        h = args.height
        w = int(round(h * cam.aspect))
        fy = h / (2 * math.tan(cam.fov / 2))
        K = torch.tensor(
            [[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device=device
        ).unsqueeze(0)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = viser.transforms.SO3(cam.wxyz).as_matrix()
        c2w[:3, 3] = cam.position
        camtoworld = torch.from_numpy(c2w).to(device).unsqueeze(0)
        with torch.no_grad():
            render = render_splats(
                args.model, splats, camtoworld, K, w, h, sh_degree,
            )[0].clamp(0, 1).cpu().numpy()
        client.scene.set_background_image((render * 255).astype(np.uint8))

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        @client.camera.on_update
        def _(_) -> None:
            pending_clients.add(client)
        pending_clients.add(client)

    print(f"Viewer running at http://localhost:{server.get_port()} -- Ctrl+C to exit.")
    import time
    while True:
        if pending_clients:
            clients = list(pending_clients)
            pending_clients.clear()
            for client in clients:
                render_for_client(client)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
