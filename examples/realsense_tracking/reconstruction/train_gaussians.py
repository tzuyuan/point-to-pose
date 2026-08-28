"""
Simplified 2DGS/3DGS reconstruction from a capture.py export directory.

Loads all_frames_{rgb,depth,mask}/ + poses (written by ReconstructionExporter, see
capture.py), filters pose-jump frames, fuses a gaussian-init pointcloud (statistical
outlier removal + DBSCAN largest-cluster selection), optionally farthest-point-samples
+ coverage-filters the training views, and trains gaussians + per-view camera pose
corrections (CameraOptModule), dropping views whose loss gets stuck. Headless by
default; pass --viewer for a live viser preview (free-viewpoint render, updates as you
move the camera). No mesh extraction.

Run:
    python examples/realsense_tracking/reconstruction/train_gaussians.py \
        --config configs/reconstruct/export_test.yaml [--viewer]
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from pytorch_msssim import ssim as ssim_fn

from sklearn.neighbors import NearestNeighbors

from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs/reconstruct/export_test.yaml"


# ----------------------------------------------------------------------
# Camera pose optimization (delta translation + 6D rotation per keyframe)
# ----------------------------------------------------------------------
class CameraOptModule(torch.nn.Module):
    """Per-view (translation, 6D rotation) pose correction, world-frame."""

    def __init__(self, n: int):
        super().__init__()
        self.embed_t = torch.nn.Embedding(n, 3)
        self.embed_r = torch.nn.Embedding(n, 6)
        self.register_buffer(
            "identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )
        torch.nn.init.zeros_(self.embed_t.weight)
        torch.nn.init.normal_(self.embed_r.weight, mean=0.0, std=1e-4)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch_dims = camtoworlds.shape[:-2]
        dx = self.embed_t(embed_ids).float()
        drot = self.embed_r(embed_ids).float()
        rot = rotation_6d_to_matrix(drot + self.identity.float().expand(*batch_dims, -1))
        transform = torch.eye(4, dtype=torch.float32, device=dx.device).repeat(*batch_dims, 1, 1)
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(transform, camtoworlds.float())


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def knn(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    x_np = x.float().cpu().numpy()
    model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def depth_edge_mask(depth, max_jump, depth_scale=1.0):
    d = depth.astype(np.float32) / float(depth_scale)
    valid = d > 0
    ok = np.ones(d.shape, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(d, shift=(dy, dx), axis=(0, 1))
        shifted_valid = np.roll(valid, shift=(dy, dx), axis=(0, 1))
        jump = np.abs(d - shifted) > max_jump
        ok &= ~(valid & shifted_valid & jump)
    return ok


def relative_pose_delta(T_a: np.ndarray, T_b: np.ndarray):
    """Translation distance (meters) and rotation angle (degrees) of the relative
    transform between two cam-to-world poses."""
    T_rel = np.linalg.inv(T_a) @ T_b
    dt = float(np.linalg.norm(T_rel[:3, 3]))
    cos_angle = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    dtheta_deg = float(np.degrees(np.arccos(cos_angle)))
    return dt, dtheta_deg


def filter_frames_by_pose_jump(poses: np.ndarray, max_trans: float = 0.05, max_rot_deg: float = 15.0):
    """Flags frames adjacent to a pose discontinuity: if the relative pose between
    frame i and its temporal neighbor i+1 exceeds max_trans meters OR max_rot_deg
    degrees, BOTH frames on either side of that jump are dropped (a jump means at
    least one of the two is wrong, and telling which cheaply is often not possible
    with only local information)."""
    n = poses.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n - 1):
        dt, dtheta_deg = relative_pose_delta(poses[i], poses[i + 1])
        if dt > max_trans or dtheta_deg > max_rot_deg:
            keep[i] = False
            keep[i + 1] = False
    n_dropped = int((~keep).sum())
    print(f"Pose jump filter ({n} frames): {keep.sum()}/{n} kept "
          f"(max_trans={max_trans}m, max_rot={max_rot_deg}deg), {n_dropped} dropped around jumps")
    return keep


def farthest_point_sample_indices(positions: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling over camera positions. Returns n indices
    into `positions`, spread out to maximize view diversity."""
    n = min(n, positions.shape[0])
    chosen = [0]
    dists = np.linalg.norm(positions - positions[0], axis=-1)
    for _ in range(1, n):
        next_idx = int(np.argmax(dists))
        chosen.append(next_idx)
        dists = np.minimum(dists, np.linalg.norm(positions - positions[next_idx], axis=-1))
    return np.array(chosen)


def project_pointcloud_occupancy(points, camtoworld, K, width, height):
    """Projects a metric-scale pointcloud into a camera view and returns a binary
    occupancy mask (True where >=1 point lands) -- a cheap proxy silhouette."""
    w2c = np.linalg.inv(camtoworld)
    points_cam = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    in_front = points_cam[:, 2] > 1e-6
    points_cam = points_cam[in_front]

    pixels = (K @ points_cam.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    px = np.round(pixels[:, 0]).astype(np.int64)
    py = np.round(pixels[:, 1]).astype(np.int64)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[py[valid], px[valid]] = True
    return occupancy


def mask_iou(a, b) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def pointcloud_mask_coverage(points, camtoworld, K, mask) -> float:
    """IoU between the fused init pointcloud's projected occupancy silhouette and
    this view's own stored mask."""
    height, width = mask.shape[:2]
    occupancy = project_pointcloud_occupancy(points, camtoworld, K, width, height)
    return mask_iou(occupancy, mask > 0)


def backproject_masked_depth(
    depth, mask, K, camtoworld, rgb=None, depth_scale=1.0, min_depth=0.05, max_depth=1.0,
    max_depth_jump=0.03,
):
    m = mask
    if max_depth_jump > 0:
        m = m & depth_edge_mask(depth, max_depth_jump, depth_scale=depth_scale)

    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float64), (
            np.empty((0, 3), dtype=np.float64) if rgb is not None else None
        )

    z = depth[ys, xs].astype(np.float64) / float(depth_scale)
    valid = (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), (
            np.empty((0, 3), dtype=np.float64) if rgb is not None else None
        )
    xs, ys, z = xs[valid], ys[valid], z[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)

    R, t = camtoworld[:3, :3], camtoworld[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    colors = None
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.float64) / 255.0
    return pts_world, colors


def load_all_frames(
    results_path: Path, max_depth_jump: float = 0.03,
    min_depth: float = 0.05, max_depth: float = 1.0,
):
    """Loads all_frames_rgb/depth/mask/poses (as saved by ReconstructionExporter),
    backprojecting each frame's own masked depth into a world-frame pointcloud."""
    poses_path = results_path / "all_frames_poses.npy"
    idx_path = results_path / "all_frames_idx.npy"
    K_path = results_path / "intrinsics.npy"
    depth_dir = results_path / "all_frames_depth"
    mask_dir = results_path / "all_frames_mask"
    rgb_dir = results_path / "all_frames_rgb"
    for p in (poses_path, idx_path, K_path, depth_dir, mask_dir, rgb_dir):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run capture.py --export-dir first.")

    K = np.load(K_path)
    all_poses = np.load(poses_path)
    all_idx = np.load(idx_path)

    t0 = time.perf_counter()
    images, masks = [], []
    for fid in all_idx:
        fid = int(fid)
        mask = np.array(Image.open(mask_dir / f"{fid:06d}.png")) > 0
        rgb = np.array(Image.open(rgb_dir / f"{fid:06d}.png").convert("RGB"))
        images.append(rgb)
        masks.append(mask)
    t1 = time.perf_counter()
    print(f"  loaded {len(all_idx)} rgb/mask images in {t1 - t0:.2f}s")

    points, colors = [], []
    for i, fid in enumerate(all_idx):
        fid = int(fid)
        depth = np.array(Image.open(depth_dir / f"{fid:06d}.png")).astype(np.float32)
        mask, rgb = masks[i], images[i]
        pts, cols = backproject_masked_depth(
            depth, mask, K, all_poses[i], rgb=rgb, depth_scale=1000.0,
            min_depth=min_depth, max_depth=max_depth, max_depth_jump=max_depth_jump,
        )
        points.append(pts)
        colors.append(cols)
    t2 = time.perf_counter()
    print(f"  backprojected {len(all_idx)} frames in {t2 - t1:.2f}s")

    return points, colors, np.stack(images), np.stack(masks), all_poses, K, all_idx


def build_init_pointcloud(
    points, colors, keep, device, voxel_size=0.0, outlier_nb_neighbors=20, outlier_std_ratio=2.0,
    cluster_eps=0.02, cluster_min_points=0,
):
    """Fuses per-frame backprojected pointclouds for frames where keep[i] is True,
    then two cleanup passes: statistical outlier removal (isolated flying-pixel
    points), then DBSCAN largest-cluster selection (drops coherent misprojected
    blobs, e.g. background leaking through mask edges, that outlier removal's
    local-density check can't catch). cluster_min_points <= 0 disables DBSCAN."""
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    if o3d is None:
        return None, None

    fused = o3d.geometry.PointCloud()
    n_kept = 0
    for pts, cols, k in zip(points, colors, keep):
        if not k or pts.shape[0] == 0:
            continue
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.colors = o3d.utility.Vector3dVector(cols)
        fused += p
        n_kept += 1
    if n_kept == 0:
        return None, None

    n_before = len(fused.points)
    if voxel_size > 0:
        fused = fused.voxel_down_sample(voxel_size)
    n_after_voxel = len(fused.points)
    print(f"Fused pointcloud: {n_before} points, downsampled to {n_after_voxel} "
          f"(voxel_size={voxel_size})")

    if outlier_nb_neighbors > 0 and n_after_voxel > outlier_nb_neighbors:
        fused, inlier_idx = fused.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
        )
        n_removed = n_after_voxel - len(inlier_idx)
        if n_removed > 0:
            print(f"Removed {n_removed} flying-point outliers from the fused init pointcloud")
    n_after_outlier = len(fused.points)

    if cluster_min_points > 0 and n_after_outlier > cluster_min_points:
        print(f"Running DBSCAN (eps={cluster_eps}, min_points={cluster_min_points}) "
              f"on {n_after_outlier} points -- this can take a while for large clouds...")
        t_dbscan0 = time.perf_counter()
        labels = np.asarray(fused.cluster_dbscan(eps=cluster_eps, min_points=cluster_min_points))
        print(f"DBSCAN done in {time.perf_counter() - t_dbscan0:.2f}s")
        if labels.max() >= 0:  # -1 == unclustered noise; skip if everything is noise
            counts = np.bincount(labels[labels >= 0])
            largest = int(np.argmax(counts))
            n_dropped = int((labels != largest).sum())
            if n_dropped > 0:
                fused = fused.select_by_index(np.where(labels == largest)[0])
                print(f"Kept largest cluster ({int(counts[largest])} points), "
                      f"dropped {n_dropped} points in other clusters/noise")

    fused_points = np.asarray(fused.points)
    fused_colors = np.asarray(fused.colors)
    print(f"Initializing gaussians from {n_kept}/{len(points)} frames "
          f"({n_before} points, downsampled to {n_after_voxel}, "
          f"{fused_points.shape[0]} after outlier/cluster cleanup)")
    return (
        torch.from_numpy(fused_points).float().to(device),
        torch.from_numpy(fused_colors).float().to(device),
    )


class KeyframeDataset(torch.utils.data.Dataset):
    def __init__(
        self, all_images, all_masks, all_poses, K, candidate_idx, n_views: int = 0,
        init_points: np.ndarray = None, view_coverage_thresh: float = 0.0,
    ):
        """
        Args:
            candidate_idx: indices into the all_frames arrays (e.g. frames that
                survived the pose-jump filter).
            n_views: how many to farthest-point-sample from candidate_idx for
                training. <= 0 means use every candidate frame.
            init_points: fused gaussian-init pointcloud (world frame), used only for
                the optional post-FPS coverage check below.
            view_coverage_thresh: if > 0, each sampled view is projected against
                init_points via pointcloud_mask_coverage and dropped if it scores
                below this. 0 disables.
        """
        if n_views <= 0:
            chosen = candidate_idx
        else:
            chosen = candidate_idx[
                farthest_point_sample_indices(all_poses[candidate_idx, :3, 3], n_views)
            ]

        if view_coverage_thresh > 0 and init_points is not None:
            scores = np.array([
                pointcloud_mask_coverage(init_points, all_poses[i], K, all_masks[i])
                for i in chosen
            ])
            view_keep = scores >= view_coverage_thresh
            n_dropped = int((~view_keep).sum())
            print(f"View coverage check (thresh={view_coverage_thresh}): "
                  f"{len(chosen) - n_dropped}/{len(chosen)} kept, {n_dropped} dropped "
                  f"(scores min={scores.min():.3f} max={scores.max():.3f} "
                  f"mean={scores.mean():.3f})")
            if n_dropped > 0:
                dropped_ids = [int(c) for c, k in zip(chosen, view_keep) if not k]
                dropped_scores = [round(float(s), 3) for s, k in zip(scores, view_keep) if not k]
                print(f"  dropped frame ids: {dropped_ids}, scores: {dropped_scores}")
            chosen = chosen[view_keep]
            if len(chosen) == 0:
                raise RuntimeError(
                    "view_coverage_thresh dropped every sampled view -- lower it or "
                    "check that the fused init pointcloud/masks look sane."
                )

        self.source_indices = np.asarray(chosen)

        images = torch.from_numpy(all_images[chosen]).float() / 255.0  # [N, H, W, 3]
        masks = torch.from_numpy(all_masks[chosen])  # [N, H, W] bool
        self.images = images * masks[..., None].float()
        self.masks = masks
        self.camtoworlds = torch.from_numpy(all_poses[chosen]).float()  # [N, 4, 4]
        self.Ks = torch.from_numpy(K).float().unsqueeze(0).repeat(len(chosen), 1, 1)
        self.height, self.width = all_images.shape[1:3]
        print(f"Loaded {len(chosen)} training views (from {len(candidate_idx)} "
              f"candidate frames, {self.width}x{self.height})")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "mask": self.masks[idx],
            "camtoworld": self.camtoworlds[idx],
            "K": self.Ks[idx],
            "image_id": idx,
        }


def create_splats_with_optimizers(
    points, rgbs, init_scale, init_opacity, sh_degree, scene_scale, device, means_lr_mult=1.0,
):
    N = points.shape[0]
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3).float()
    quats = torch.rand((N, 4), dtype=torch.float32)
    opacities = torch.logit(torch.full((N,), init_opacity, dtype=torch.float32))

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3), dtype=torch.float32)
    colors[:, 0, :] = rgb_to_sh(rgbs.cpu()).float()

    params = [
        ("means", torch.nn.Parameter(points.cpu().float()), 1.6e-4 * scene_scale * means_lr_mult),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
    ]
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr}], eps=1e-15)
        for name, _, lr in params
    }
    return splats, optimizers


def compute_pca_alignment(points: np.ndarray) -> np.ndarray:
    """Computes a 4x4 world-to-canonical transform that recenters `points` at their
    centroid and rotates them onto their PCA principal axes -- an object-centric
    frame (helps scene_scale and CameraOptModule's world-frame correction, both of
    which assume the object sits near the origin)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    R = eigvecs[:, ::-1].T  # descending variance -> rows are the new x/y/z axes

    if np.linalg.det(R) < 0:
        R[-1, :] *= -1
    aligned_preview = (R @ centered.T).T
    for axis in range(3):
        if aligned_preview[np.argmax(np.abs(aligned_preview[:, axis])), axis] < 0:
            R[axis, :] *= -1
    if np.linalg.det(R) < 0:
        R[-1, :] *= -1

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centroid
    return T


def apply_alignment_to_gaussians(splats: dict, T_align: np.ndarray) -> dict:
    """Returns a new splats dict with means/quats transformed by T_align (4x4, applied
    as canonical = T_align @ world)."""
    from scipy.spatial.transform import Rotation as ScipyR

    R = T_align[:3, :3]
    t = T_align[:3, 3]
    means = splats["means"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()  # wxyz, gsplat convention

    new_means = (R @ means.T).T + t

    quats_norm = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12)
    r_gauss = ScipyR.from_quat(quats_norm[:, [1, 2, 3, 0]])  # wxyz -> xyzw
    r_align = ScipyR.from_matrix(R)
    r_new = r_align * r_gauss
    new_quats_xyzw = r_new.as_quat()
    new_quats = new_quats_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    out = dict(splats)
    out["means"] = torch.from_numpy(new_means).float()
    out["quats"] = torch.from_numpy(new_quats).float()
    return out


def remove_floater_gaussians(splats: dict, nb_neighbors: int = 20, std_ratio: float = 2.5) -> dict:
    """Statistical-outlier removal on gaussian means to drop floaters left over after
    training -- out-of-view/behind-camera gaussians the rendering loss never sees, so
    densification/pruning can't clean them up on its own. nb_neighbors <= 0 disables."""
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    if o3d is None or nb_neighbors <= 0:
        return splats
    means_np = splats["means"].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_np)
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    keep = np.zeros(means_np.shape[0], dtype=bool)
    keep[inlier_idx] = True
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"Removed {n_dropped}/{means_np.shape[0]} floater gaussians (statistical outlier removal)")
    keep_t = torch.from_numpy(keep).to(splats["means"].device)
    return {k: v[keep_t] for k, v in splats.items()}


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
        render_median = None

    return {"colors": render_colors, "alphas": render_alphas, "median_depth": render_median, "info": info}


def train(cfg, use_viewer=False):
    device = "cuda:0"
    results_path = Path(cfg.results_path)
    t_start = time.perf_counter()

    print("Loading exported frames and backprojecting masked depth...")
    points, colors, all_images, all_masks, all_poses, K, frame_idx = load_all_frames(
        results_path, max_depth_jump=cfg.get("max_depth_jump", 0.03),
        min_depth=cfg.get("min_depth", 0.05), max_depth=cfg.get("max_depth", 1.0),
    )
    n = len(points)
    t_loaded = time.perf_counter()
    print(f"Loaded {n} frames in {t_loaded - t_start:.2f}s")

    pose_jump_max_trans = float(cfg.get("pose_jump_max_trans", 0.0))
    if pose_jump_max_trans > 0:
        keep = filter_frames_by_pose_jump(
            all_poses, max_trans=pose_jump_max_trans,
            max_rot_deg=float(cfg.get("pose_jump_max_rot_deg", 15.0)),
        )
    else:
        keep = np.ones(n, dtype=bool)

    candidate_idx = np.where(keep)[0]

    init_points, init_colors = build_init_pointcloud(
        points, colors, keep, device, voxel_size=float(cfg.get("init_voxel_size", 0.0)),
        cluster_eps=float(cfg.get("init_cluster_eps", 0.02)),
        cluster_min_points=int(cfg.get("init_cluster_min_points", 0)),
    )
    t_fused = time.perf_counter()
    print(f"Fused init pointcloud in {t_fused - t_loaded:.2f}s")

    if init_points is None:
        print("No export data found for gaussian init, falling back to random init")
        scene_scale = float(np.linalg.norm(all_poses[:, :3, 3], axis=-1).max()) + 1e-2
        n_pts = 50_000
        init_points = scene_scale * (torch.rand((n_pts, 3), dtype=torch.float32, device=device) * 2 - 1)
        init_colors = torch.rand((n_pts, 3), dtype=torch.float32, device=device)
        candidate_idx = np.arange(n)
    elif bool(cfg.get("recenter_gaussians", True)):
        # Re-center + PCA-align BEFORE training: scene_scale and CameraOptModule's
        # world-frame pose correction both benefit from the object sitting at the origin.
        T_align_pre = compute_pca_alignment(init_points.detach().cpu().numpy())
        R_pre = torch.from_numpy(T_align_pre[:3, :3]).float().to(device)
        t_pre = torch.from_numpy(T_align_pre[:3, 3]).float().to(device)
        init_points = (R_pre @ init_points.T).T + t_pre
        all_poses = np.einsum("ij,njk->nik", T_align_pre, all_poses)
        print("Pre-training PCA alignment applied to init pointcloud + all camera poses")

    centroid = init_points.mean(dim=0)
    scene_scale = float((init_points - centroid).norm(dim=-1).max().item()) + 1e-2
    print(f"scene_scale = {scene_scale:.4f} (from init pointcloud spread)")

    dataset = KeyframeDataset(
        all_images, all_masks, all_poses, K, candidate_idx,
        n_views=int(cfg.get("n_views", 0)),
        init_points=init_points.detach().cpu().numpy(),
        view_coverage_thresh=float(cfg.get("view_coverage_thresh", 0.0)),
    )

    model = cfg.get("model", "2dgs")
    sh_degree = int(cfg.get("sh_degree", 0))
    max_steps = int(cfg.get("max_steps", 5000))
    ssim_lambda = float(cfg.get("ssim_lambda", 0.0))
    bg_alpha_lambda = float(cfg.get("bg_alpha_lambda", 0.5))

    splats, optimizers = create_splats_with_optimizers(
        init_points, init_colors,
        init_scale=float(cfg.get("init_scale", 1.0)),
        init_opacity=float(cfg.get("init_opacity", 0.1)),
        sh_degree=sh_degree, scene_scale=scene_scale, device=device,
        means_lr_mult=float(cfg.get("means_lr_mult", 1.0)),
    )
    print("Initialized", splats["means"].shape[0], "gaussians")

    key_for_gradient = "gradient_2dgs" if model == "2dgs" else "means2d"
    strategy = DefaultStrategy(verbose=True, key_for_gradient=key_for_gradient)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    pose_adjust = CameraOptModule(len(dataset)).to(device)
    pose_opt_lr = float(cfg.get("pose_opt_lr", 1e-4))
    pose_optimizer = torch.optim.Adam(
        [
            {"params": pose_adjust.embed_t.parameters(), "lr": pose_opt_lr},
            {"params": pose_adjust.embed_r.parameters(),
             "lr": pose_opt_lr * float(cfg.get("pose_opt_rot_lr_mult", 1.0))},
        ],
        weight_decay=float(cfg.get("pose_opt_reg", 0.0)),
    )

    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / max_steps))
        for opt in optimizers.values()
    ]
    schedulers.append(
        torch.optim.lr_scheduler.ExponentialLR(pose_optimizer, gamma=0.01 ** (1.0 / max_steps))
    )

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    trainloader_iter = iter(trainloader)

    server = None
    gui_step = gui_loss = gui_preview_view = gui_preview = None
    frustum_handles = []
    if use_viewer:
        import viser

        # Warm up gsplat's internal CUDA lazy wrappers (e.g. torch.inverse) on the
        # MAIN thread -- calling them for the first time from viser's callback
        # thread instead raises "lazy wrapper should be called at most once".
        with torch.no_grad():
            dummy_c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
            dummy_K = torch.tensor(
                [[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], dtype=torch.float32, device=device
            ).unsqueeze(0)
            render_splats(model, splats, dummy_c2w, dummy_K, 64, 64, sh_degree)

        server = viser.ViserServer()
        server.scene.set_up_direction("+z")

        # Camera frustums for every training view, moved live as pose_adjust corrects them.
        fx0 = dataset.Ks[0, 0, 0].item()
        fov0 = 2 * math.atan(dataset.width / (2 * fx0))
        aspect0 = dataset.width / dataset.height
        for i in range(len(dataset)):
            c2w0 = dataset.camtoworlds[i].numpy()
            img_np = (dataset.images[i].numpy() * 255).astype(np.uint8)
            handle = server.scene.add_camera_frustum(
                f"cameras/frustum_{i:03d}", fov=fov0, aspect=aspect0, scale=0.05,
                color=(180, 180, 180), image=img_np,
                wxyz=viser.transforms.SO3.from_matrix(c2w0[:3, :3]).wxyz,
                position=c2w0[:3, 3],
            )
            frustum_handles.append(handle)

        with server.gui.add_folder("Training"):
            gui_step = server.gui.add_text("step", initial_value="0 / 0")
            gui_loss = server.gui.add_text("loss", initial_value="-")
        with server.gui.add_folder("GT | Render"):
            gui_preview_view = server.gui.add_slider(
                "view", min=0, max=max(len(dataset) - 1, 0), step=1, initial_value=0
            )
            gui_preview = server.gui.add_image(
                np.zeros((dataset.height, dataset.width * 2, 3), dtype=np.uint8), label="gt | render"
            )
        print(f"[viewer] viser running at http://localhost:{server.get_port()}")

        # CUDA context is thread-local; viser fires connect/camera-update/slider callbacks
        # on its own worker threads, and touching CUDA tensors from those threads crashes
        # (either "lazy wrapper should be called at most once" or a CUDAGuard assert).
        # So callbacks only enqueue a request; the render itself always runs on the MAIN
        # thread (drained once per training-loop iteration, and in a dedicated loop once
        # training is done).
        pending_clients = set()
        pending_preview = False

        def render_preview_view(view_idx: int) -> np.ndarray:
            """Side-by-side GT | current-render for one training view. MAIN THREAD ONLY."""
            gt = (dataset.images[view_idx].numpy() * 255).astype(np.uint8)
            camtoworld = dataset.camtoworlds[view_idx : view_idx + 1].to(device)
            K = dataset.Ks[view_idx : view_idx + 1].to(device)
            with torch.no_grad():
                image_id = torch.tensor([view_idx], device=device)
                camtoworld_adj = pose_adjust(camtoworld, image_id)
                render = render_splats(
                    model, splats, camtoworld_adj, K, dataset.width, dataset.height, sh_degree,
                )["colors"][0].clamp(0, 1).cpu().numpy()
            return np.concatenate([gt, (render * 255).astype(np.uint8)], axis=1)

        @gui_preview_view.on_update
        def _(_) -> None:
            nonlocal pending_preview
            pending_preview = True

        def render_for_client(client) -> None:
            """Free-viewpoint render from this client's current camera. MAIN THREAD ONLY."""
            cam = client.camera
            aspect = cam.aspect
            h = dataset.height
            w = int(round(h * aspect))
            fov = cam.fov
            fy = h / (2 * math.tan(fov / 2))
            fx = fy  # square pixels
            K = torch.tensor(
                [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device=device
            ).unsqueeze(0)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = viser.transforms.SO3(cam.wxyz).as_matrix()
            c2w[:3, 3] = cam.position
            camtoworld = torch.from_numpy(c2w).to(device).unsqueeze(0)
            with torch.no_grad():
                render = render_splats(
                    model, splats, camtoworld, K, w, h, sh_degree,
                )["colors"][0].clamp(0, 1).cpu().numpy()
            client.scene.set_background_image((render * 255).astype(np.uint8))

        def drain_render_queue() -> None:
            """Call once per training-loop iteration (and in the post-training wait
            loop) from the MAIN thread to actually perform any pending renders."""
            nonlocal pending_preview
            if pending_preview:
                pending_preview = False
                gui_preview.image = render_preview_view(gui_preview_view.value)
            if pending_clients:
                clients = list(pending_clients)
                pending_clients.clear()
                for client in clients:
                    render_for_client(client)

        @server.on_client_connect
        def _(client: "viser.ClientHandle") -> None:
            @client.camera.on_update
            def _(_) -> None:
                pending_clients.add(client)
            pending_clients.add(client)

    drop_stuck_views_every = int(cfg.get("drop_stuck_views_every", 0))
    drop_stuck_views_l1_thresh = float(cfg.get("drop_stuck_views_l1_thresh", 0.08))
    drop_stuck_views_min_delta = float(cfg.get("drop_stuck_views_min_delta", 1e-3))
    drop_stuck_views_min_samples = int(cfg.get("drop_stuck_views_min_samples", 2))
    active_views = torch.ones(len(dataset), dtype=torch.bool)
    best_view_l1 = torch.full((len(dataset),), float("inf"))
    best_view_l1_at_check = torch.full((len(dataset),), float("inf"))
    view_samples_since_check = torch.zeros(len(dataset), dtype=torch.int64)

    for step in range(max_steps):
        if (
            drop_stuck_views_every > 0
            and step > 0
            and step % drop_stuck_views_every == 0
        ):
            finite_now = torch.isfinite(best_view_l1)
            finite_prev = torch.isfinite(best_view_l1_at_check)
            improvement = best_view_l1_at_check - best_view_l1
            stuck = (
                active_views
                & finite_now
                & finite_prev
                & (view_samples_since_check >= drop_stuck_views_min_samples)
                & (best_view_l1 >= drop_stuck_views_l1_thresh)
                & (improvement < drop_stuck_views_min_delta)
            )
            active_count = int(active_views.sum().item())
            if active_count - int(stuck.sum().item()) <= 0 and active_count > 0:
                worst_active = torch.where(active_views, best_view_l1, torch.tensor(-1.0)).argmax()
                stuck[worst_active] = False
            if stuck.any():
                drop_ids = torch.nonzero(stuck, as_tuple=False).flatten()
                active_views[drop_ids] = False
                l1_str = ", ".join(
                    f"dataset={i} frame={int(frame_idx[dataset.source_indices[i]])} "
                    f"l1={best_view_l1[i]:.4f}"
                    for i in drop_ids.tolist()
                )
                print(f"step {step}: dropped {len(drop_ids)} stuck high-L1 view(s): {l1_str}; "
                      f"{int(active_views.sum().item())}/{len(dataset)} active")
            best_view_l1_at_check.copy_(best_view_l1)
            view_samples_since_check.zero_()

        if not active_views.any():
            print("All training views were dropped; stopping early.")
            break

        for _ in range(max(1, len(dataset) * 2)):
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            image_ids_cpu = data["image_id"].view(-1)
            if bool(active_views[int(image_ids_cpu[0])]):
                break
        else:
            active_ids = torch.nonzero(active_views, as_tuple=False).flatten()
            forced_idx = int(active_ids[torch.randint(len(active_ids), (1,))].item())
            data = dataset[forced_idx]
            data = {
                k: v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v])
                for k, v in data.items()
            }

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device)
        masks = data["mask"].to(device)
        image_ids = data["image_id"].to(device)

        camtoworlds = pose_adjust(camtoworlds, image_ids)

        sh_degree_to_use = min(step // 1000, sh_degree)

        out = render_splats(model, splats, camtoworlds, Ks, dataset.width, dataset.height, sh_degree_to_use)
        render_colors, render_alphas, info = out["colors"], out["alphas"], out["info"]

        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        rendered = render_colors * masks[..., None]
        target = pixels * masks[..., None]

        l1loss = F.l1_loss(rendered, target)
        with torch.no_grad():
            mask_denom = masks.float().sum(dim=(1, 2)).clamp_min(1.0) * rendered.shape[-1]
            per_view_l1 = (rendered - target).abs().sum(dim=(1, 2, 3)) / mask_denom
            image_ids_cpu = image_ids.detach().cpu().long()
            best_view_l1[image_ids_cpu] = torch.minimum(best_view_l1[image_ids_cpu], per_view_l1.cpu())
            view_samples_since_check[image_ids_cpu] += 1

        ssimloss = 1.0 - ssim_fn(
            rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), data_range=1.0
        )
        loss = (1 - ssim_lambda) * l1loss + ssim_lambda * ssimloss

        if bg_alpha_lambda > 0:
            bg_alpha = render_alphas[..., 0] * (~masks).float()
            loss = loss + bg_alpha_lambda * bg_alpha.mean()

        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        for sch in schedulers:
            sch.step()

        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        if step % 100 == 0 or step == max_steps - 1:
            print(f"step {step}/{max_steps} loss={loss.item():.4f} "
                  f"n_gaussians={splats['means'].shape[0]}")
            if server is not None:
                gui_step.value = f"{step} / {max_steps}"
                gui_loss.value = f"{loss.item():.4f}"
                with torch.no_grad():
                    all_ids = torch.arange(len(dataset), device=device)
                    all_c2w = dataset.camtoworlds.to(device)
                    updated = pose_adjust(all_c2w, all_ids).float().cpu().numpy()
                for i, h in enumerate(frustum_handles):
                    h.wxyz = viser.transforms.SO3.from_matrix(updated[i, :3, :3]).wxyz
                    h.position = updated[i, :3, 3]
                gui_preview.image = render_preview_view(gui_preview_view.value)
                for client in server.get_clients().values():
                    pending_clients.add(client)
                drain_render_queue()
        if server is not None:
            drain_render_queue()

    print("Training done.")

    splats = remove_floater_gaussians(
        splats, nb_neighbors=int(cfg.get("floater_outlier_nb_neighbors", 20)),
        std_ratio=float(cfg.get("floater_outlier_std_ratio", 2.5)),
    )

    if bool(cfg.get("recenter_gaussians", True)):
        # Re-align AGAIN post-training: pose_opt/densification can drift the
        # gaussians' centroid/orientation, so recompute from the final means.
        means_np = splats["means"].detach().cpu().numpy()
        T_align = compute_pca_alignment(means_np)
        splats = apply_alignment_to_gaussians(splats, T_align)
        print("Post-training PCA re-alignment applied to final gaussians")

    ckpt_path = results_path / "gaussians.pt"
    torch.save({k: v.detach().cpu() for k, v in splats.items()}, ckpt_path)
    print(f"Saved gaussians to {ckpt_path}")

    if server is not None:
        print(f"[viewer] training done, viser still running at "
              f"http://localhost:{server.get_port()} -- Ctrl+C to exit.")
        while True:
            drain_render_queue()
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--results_path", default=None,
                    help="overrides results_path from --config")
    ap.add_argument("--viewer", action="store_true",
                    help="live viser preview: camera frustums + current render")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.results_path is not None:
        cfg.results_path = args.results_path
    train(cfg, use_viewer=args.viewer)


if __name__ == "__main__":
    main()
