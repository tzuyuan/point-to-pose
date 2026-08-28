"""
Live depth via Fast-FoundationStereo instead of the RealSense z16 depth stream.

Runs stereo matching on the camera's left/right IR streams every frame and converts the
predicted disparity to metric depth with the IR baseline, as a drop-in replacement for
``RealSenseModelTracker._grab_raw``'s depth channel. See Fast-FoundationStereo/scripts/
run_demo.py for the reference single-image forward pass this mirrors.
"""

import sys
from pathlib import Path

import numpy as np
import torch

FS_ROOT = Path(__file__).resolve().parents[2] / "Fast-FoundationStereo"
if str(FS_ROOT) not in sys.path:
    sys.path.insert(0, str(FS_ROOT))

from core.utils.utils import InputPadder  # noqa: E402
from Utils import AMP_DTYPE  # noqa: E402


class FoundationStereoDepth:
    """Wraps a Fast-FoundationStereo checkpoint to turn a live IR stereo pair into a metric
    depth map (meters), matching the (H,W) float32 convention ``_grab_raw`` otherwise gets
    from the RealSense depth sensor."""

    def __init__(self, model_dir, device="cuda", valid_iters=8, max_disp=None, hiera=False):
        self.device = device
        self.hiera = bool(hiera)
        self.valid_iters = int(valid_iters)

        self.model = torch.load(model_dir, map_location="cpu", weights_only=False)
        self.model.args.valid_iters = self.valid_iters
        if max_disp is not None:
            self.model.args.max_disp = int(max_disp)
        self.model.to(device).eval()

    @torch.no_grad()
    def infer_depth(self, left_rgb, right_rgb, K, baseline_m):
        """
        Args:
            left_rgb, right_rgb: (H,W,3) uint8, rectified stereo pair (left == the RGB/depth
                frame's own viewpoint, i.e. the RealSense left IR sensor for D4xx cameras).
            K: (3,3) intrinsics of the left (reference) camera, at left_rgb's resolution.
            baseline_m: stereo baseline in meters (left-to-right IR sensor distance).

        Returns:
            depth: (H,W) float32 meters, 0 where disparity is non-positive/invalid.
        """
        H, W = left_rgb.shape[:2]
        img0 = torch.as_tensor(left_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            if self.hiera:
                disp = self.model.run_hierachical(
                    img0, img1, iters=self.valid_iters, test_mode=True, small_ratio=0.5
                )
            else:
                disp = self.model.forward(
                    img0, img1, iters=self.valid_iters, test_mode=True,
                    optimize_build_volume="pytorch1",
                )
        disp = padder.unpad(disp.float())
        # disp_up's exact rank (a [B,1,H,W] vs [B,H,W]) isn't guaranteed across model
        # variants -- mirror run_demo.py and just flatten+reshape rather than index a fixed
        # number of leading dims.
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        fx = float(K[0, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp > 0, fx * baseline_m / disp, 0.0)
        return depth.astype(np.float32)


def reproject_depth(depth_src, K_src, K_dst, R_src2dst, t_src2dst, dst_hw):
    """Reproject a depth map from the source camera (e.g. left IR, where FoundationStereo
    computed it) into a different camera's viewpoint (e.g. color), via RealSense's
    ``get_extrinsics_to`` rotation/translation -- z-buffered so occluded-in-dst points don't
    overwrite nearer ones.

    Args:
        depth_src: (H,W) float32 meters, source camera.
        K_src, K_dst: (3,3) intrinsics of source/destination cameras.
        R_src2dst: (3,3) rotation, src camera frame -> dst camera frame.
        t_src2dst: (3,) translation (meters), src camera frame -> dst camera frame.
        dst_hw: (H,W) of the destination image.

    Returns:
        depth_dst: (H,W) float32 meters, 0 where nothing reprojects there.
    """
    H, W = depth_src.shape
    valid = depth_src > 0
    if not np.any(valid):
        return np.zeros(dst_hw, dtype=np.float32)

    ys, xs = np.nonzero(valid)
    z = depth_src[ys, xs]
    fx, fy, cx, cy = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    x_src = (xs - cx) / fx * z
    y_src = (ys - cy) / fy * z
    pts_src = np.stack([x_src, y_src, z], axis=-1)  # (N,3)

    pts_dst = pts_src @ R_src2dst.T + t_src2dst
    z_dst = pts_dst[:, 2]
    in_front = z_dst > 1e-6
    pts_dst, z_dst = pts_dst[in_front], z_dst[in_front]

    fx_d, fy_d, cx_d, cy_d = K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2]
    u = np.round(fx_d * pts_dst[:, 0] / z_dst + cx_d).astype(np.int64)
    v = np.round(fy_d * pts_dst[:, 1] / z_dst + cy_d).astype(np.int64)
    Hd, Wd = dst_hw
    in_bounds = (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)
    u, v, z_dst = u[in_bounds], v[in_bounds], z_dst[in_bounds]

    # z-buffer: keep the nearest depth per destination pixel (multiple source points can land
    # on the same dst pixel when reprojecting -- e.g. a slanted surface). Paint far-to-near so
    # a nearer point always overwrites a farther one that lands on the same pixel.
    depth_dst = np.zeros(dst_hw, dtype=np.float32)
    order = np.argsort(-z_dst)
    depth_dst[v[order], u[order]] = z_dst[order].astype(np.float32)

    return _fill_forward_warp_holes(depth_dst)


def _fill_forward_warp_holes(depth_dst, win=3):
    """Fill the sub-pixel gaps a forward warp leaves behind (each source pixel lands on at
    most one dst pixel, so a dst pixel with no source landing on it stays 0 even though it's
    genuinely covered by geometry) -- NOT a general hole-fill for real sensor no-return
    regions, which should stay 0.

    For each still-zero pixel, take the MIN nonzero depth in a small window around it (not
    the median/mean): forward-warp holes sit *between* a nearer foreground point and whatever
    is behind it, so nearest-wins is the safe choice -- it can only make a point look a little
    closer than it is, never silently swap in background depth for a foreground pixel."""
    import cv2

    # cv2.erode as a cheap per-window min-of-nonzero: treat holes as a large sentinel so they
    # never win the min (np.inf itself gets silently clamped to float32-max by cv2.erode,
    # which np.isfinite() would then treat as "valid" -- use an explicit finite sentinel well
    # above any real depth instead). BORDER_REPLICATE avoids the implicit 0-padding at the
    # image edges that a min-filter would otherwise treat as the nearest possible depth,
    # eroding valid depth away right at the border.
    SENTINEL = np.float32(1e6)
    inv = np.where(depth_dst > 0, depth_dst, SENTINEL)
    filled_inv = cv2.erode(inv, np.ones((win, win), np.float32), borderType=cv2.BORDER_REPLICATE)
    filled = np.where(filled_inv < SENTINEL, filled_inv, 0.0).astype(np.float32)

    holes = depth_dst == 0
    depth_dst = depth_dst.copy()
    depth_dst[holes] = filled[holes]
    return depth_dst
