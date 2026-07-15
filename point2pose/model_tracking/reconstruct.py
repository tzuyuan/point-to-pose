"""2D/3D Gaussian Splatting reconstruction from point2pose's ReconstructionExporter output.

Loads full-resolution frame RGB/depth/mask/poses saved by ``ReconstructionExporter``
(see point2pose/pipeline/components/reconstruction_exporter.py), initializes gaussians
from the backprojected pointcloud, and jointly optimizes gaussians + per-view camera
pose corrections. Ported from ../kv_tracker/reconstruct.py -- unlike kv_tracker's
monocular pi3 poses, point2pose's poses come from RealSense depth backprojection and
are already metric, so no scale_align.py pass is needed here.

Usage:
    python -m point2pose.model_tracking.reconstruct --results_path debug/export_test
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
from sklearn.neighbors import NearestNeighbors

from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy

# GsplatViewer/GsplatRenderTabState ship as example code in kv_tracker's thirdparty/gsplat,
# not as part of the installed gsplat package -- vendored locally (Apache-2.0) since
# point-to-pose does not carry a gsplat source checkout. See ../kv_tracker/reconstruct.py
# for why the 2DGS viewer/state classes are reused for both models.
sys.path.insert(0, str(Path(__file__).parent / "gsplat_viewer"))
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, apply_float_colormap

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ----------------------------------------------------------------------
# Camera pose optimization (delta translation + 6D rotation per keyframe)
# ----------------------------------------------------------------------
class CameraOptModule(torch.nn.Module):
    """Per-view (translation, 6D rotation) pose correction, world-frame.

    Translation and rotation are separate embeddings so they can have separate
    learning rates -- dx is added directly (lr tied to scene_scale like the gaussian
    means), while drot goes through rotation_6d_to_matrix's nonlinear Gram-Schmidt
    normalize, whose gradient scale has no inherent relationship to dx's.

    Correction is composed as T_correction @ camtoworlds (left-multiply, world/
    object-frame axes) rather than camtoworlds @ T_correction (each camera's own local
    frame): SVD/PnP registration errors tend to share a common cause (an off object-
    frame origin/axes) that shows up as correlated drift across frames, which is
    easier to fit along a shared set of world axes than a per-camera local frame.
    """

    def __init__(self, n: int):
        super().__init__()
        self.embed_t = torch.nn.Embedding(n, 3)
        self.embed_r = torch.nn.Embedding(n, 6)
        self.register_buffer(
            "identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )
        # NOT zeros_: at drot==0, Gram-Schmidt normalize() sits at a gradient
        # singularity (rotation gets zero gradient for the first step(s)). A tiny
        # random init breaks that so rotation trains from step 0 too.
        torch.nn.init.zeros_(self.embed_t.weight)
        torch.nn.init.normal_(self.embed_r.weight, mean=0.0, std=1e-4)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch_dims = camtoworlds.shape[:-2]
        dx = self.embed_t(embed_ids).float()  # (..., 3), world-frame translation
        drot = self.embed_r(embed_ids).float()  # (..., 6), world-frame rotation
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


def depth_edge_mask(depth, max_jump, depth_scale=1.0):
    """Marks pixels False where they differ from a 4-connected neighbor by more than
    max_jump (meters) -- cheap flying-pixel / depth-discontinuity filter for the
    noisy stereo depth right at object/background mask edges."""
    d = depth.astype(np.float32) / float(depth_scale)
    valid = d > 0
    ok = np.ones(d.shape, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(d, shift=(dy, dx), axis=(0, 1))
        shifted_valid = np.roll(valid, shift=(dy, dx), axis=(0, 1))
        jump = np.abs(d - shifted) > max_jump
        ok &= ~(valid & shifted_valid & jump)
    return ok


def backproject_masked_depth(
    depth, mask, K, camtoworld, rgb=None, depth_scale=1.0, min_depth=0.05, max_depth=1.0,
    max_depth_jump=0.03,
):
    """Backprojects one frame's masked depth pixels into world-frame 3D points, using
    the frame's own pose. max_depth_jump > 0 additionally drops pixels whose depth
    differs from a neighbor by more than this many meters (see depth_edge_mask); 0
    disables. Pass rgb to get matching per-point colors back.

    Returns (N,3) world-frame points (empty if no valid pixels), and (N,3) float
    [0,1] colors if rgb was given (else None)."""
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
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)  # (N,3)

    R, t = camtoworld[:3, :3], camtoworld[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    colors = None
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.float64) / 255.0
    return pts_world, colors


def project_pointcloud_occupancy(
    points: np.ndarray, camtoworld: np.ndarray, K: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Projects a metric-scale pointcloud into a camera view and returns a binary
    occupancy mask (True where >=1 point lands), used as a cheap proxy silhouette to
    sanity-check a pose against its stored image mask."""
    w2c = np.linalg.inv(camtoworld)
    points_cam = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T  # [N, 3]
    in_front = points_cam[:, 2] > 1e-6
    points_cam = points_cam[in_front]

    pixels = (K @ points_cam.T).T  # [N, 3]
    pixels = pixels[:, :2] / pixels[:, 2:3]
    px = np.round(pixels[:, 0]).astype(np.int64)
    py = np.round(pixels[:, 1]).astype(np.int64)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[py[valid], px[valid]] = True
    return occupancy


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def pointcloud_mask_coverage(
    points: np.ndarray, camtoworld: np.ndarray, K: np.ndarray, mask: np.ndarray
) -> float:
    """IoU between the fused init pointcloud's projected occupancy silhouette and this
    view's own stored mask -- stricter than a one-sided "fraction of points inside the
    mask" score since it also penalizes the silhouette covering area the mask doesn't."""
    height, width = mask.shape[:2]
    occupancy = project_pointcloud_occupancy(points, camtoworld, K, width, height)
    return mask_iou(occupancy, mask > 0)


def compute_pca_alignment(points: np.ndarray) -> np.ndarray:
    """Computes a 4x4 world-to-canonical transform that recenters `points` at their
    centroid and rotates them onto their PCA principal axes (largest-variance axis ->
    +x, etc.) -- an object-centric frame for downstream consumers (grasping, viewers
    that orbit the origin)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    R = eigvecs[:, ::-1].T  # descending variance -> rows are the new x/y/z axes

    # Fix handedness (eigh doesn't guarantee a proper rotation), then pick a sign per
    # axis so its largest-magnitude component is positive (consistent run to run).
    if np.linalg.det(R) < 0:
        R[-1, :] *= -1
    aligned_preview = (R @ centered.T).T
    for axis in range(3):
        if aligned_preview[np.argmax(np.abs(aligned_preview[:, axis])), axis] < 0:
            R[axis, :] *= -1
    if np.linalg.det(R) < 0:  # re-check after sign flips
        R[-1, :] *= -1

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centroid
    return T


def apply_alignment_to_gaussians(splats: dict, T_align: np.ndarray) -> dict:
    """Returns a new splats dict with means/quats transformed by T_align (4x4, applied
    as canonical = T_align @ world). Other params are unaffected by rotation."""
    R = T_align[:3, :3]
    t = T_align[:3, 3]
    means = splats["means"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()  # wxyz, gsplat convention

    new_means = (R @ means.T).T + t

    # scipy uses xyzw; gsplat uses wxyz -- convert, compose, convert back.
    from scipy.spatial.transform import Rotation as ScipyR

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


def relative_pose_delta(T_a: np.ndarray, T_b: np.ndarray):
    """Translation distance (meters) and rotation angle (degrees) of the relative
    transform between two cam-to-world poses."""
    T_rel = np.linalg.inv(T_a) @ T_b
    dt = float(np.linalg.norm(T_rel[:3, 3]))
    # rotation angle from the trace of the relative rotation matrix
    cos_angle = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    dtheta_deg = float(np.degrees(np.arccos(cos_angle)))
    return dt, dtheta_deg


def filter_frames_by_pose_jump(
    results_path: Path,
    poses: np.ndarray,
    frame_idx: np.ndarray,
    max_trans: float = 0.05,
    max_rot_deg: float = 15.0,
):
    """Flags frames adjacent to a pose discontinuity: if the relative pose between
    frame i and its temporal neighbor i+1 exceeds max_trans meters OR max_rot_deg
    degrees, BOTH frames on either side of that jump are dropped (conservative -- a
    jump means at least one of the two is wrong, and telling which cheaply is often
    not possible with only local information). O(N), no pointcloud projection needed.

    Returns: keep (N,) bool, deltas (N-1, 2) array of (dt, dtheta_deg) per consecutive
        pair, for logging/inspection.
    """
    n = poses.shape[0]
    out_dir = results_path / "view_filter"
    out_dir.mkdir(exist_ok=True)

    keep = np.ones(n, dtype=bool)
    deltas = np.zeros((max(n - 1, 0), 2), dtype=np.float32)
    for i in range(n - 1):
        dt, dtheta_deg = relative_pose_delta(poses[i], poses[i + 1])
        deltas[i] = (dt, dtheta_deg)
        if dt > max_trans or dtheta_deg > max_rot_deg:
            keep[i] = False
            keep[i + 1] = False

    np.save(out_dir / "pose_jump_deltas.npy", deltas)
    n_dropped = int((~keep).sum())
    print(
        f"Pose jump filter ({n} frames): {keep.sum()}/{n} kept "
        f"(max_trans={max_trans}m, max_rot={max_rot_deg}deg), "
        f"{n_dropped} dropped around jumps. Log saved to {out_dir}/pose_jump_deltas.npy"
    )
    return keep, deltas


def load_all_frames(
    results_path: Path, max_depth_jump: float = 0.03,
    min_depth: float = 0.05, max_depth: float = 1.0,
):
    """Loads all_frames_rgb/depth/mask/poses (as saved by ReconstructionExporter --
    every tracked frame), backprojecting each frame's own masked depth into a
    world-frame pointcloud with backproject_masked_depth.

    depth_scale=1000.0 assumes all_frames_depth is raw uint16 mm (RealSense
    depth_factor=1000, as realsense_tracking_rerun.py always sets).

    Returns: (points_list, colors_list, images (N,H,W,3) uint8, masks (N,H,W) bool,
              poses (N,4,4), K, frame_idx (N,) -- original all_frames indices)
    """
    poses_path = results_path / "all_frames_poses.npy"
    idx_path = results_path / "all_frames_idx.npy"
    K_path = results_path / "intrinsics.npy"
    depth_dir = results_path / "all_frames_depth"
    mask_dir = results_path / "all_frames_mask"
    rgb_dir = results_path / "all_frames_rgb"
    for p in (poses_path, idx_path, K_path, depth_dir, mask_dir, rgb_dir):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run the RealSense demo with --export-dir first."
            )

    K = np.load(K_path)
    all_poses = np.load(poses_path)
    all_idx = np.load(idx_path)

    images, masks = [], []
    for fid in all_idx:
        fid = int(fid)
        mask = np.array(Image.open(mask_dir / f"{fid:06d}.png")) > 0
        rgb = np.array(Image.open(rgb_dir / f"{fid:06d}.png").convert("RGB"))
        images.append(rgb)
        masks.append(mask)

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

    return points, colors, np.stack(images), np.stack(masks), all_poses, K, all_idx


def build_init_pointcloud(
    points, colors, keep, device, voxel_size: float = 0.0,
    outlier_nb_neighbors: int = 20, outlier_std_ratio: float = 2.0,
    cluster_eps: float = 0.02, cluster_min_points: int = 20,
):
    """Fuses the per-frame backprojected pointclouds (from load_all_frames) for frames
    where keep[i] is True into a single gaussian-init pointcloud, then two cleanup
    passes: statistical outlier removal (isolated flying-pixel points), then DBSCAN
    largest-cluster selection (drops coherent misprojected blobs, e.g. background
    leaking through mask edges, that outlier removal's local-density check can't
    catch). Set outlier_nb_neighbors <= 0 / cluster_min_points <= 0 to disable either."""
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

    if outlier_nb_neighbors > 0 and n_after_voxel > outlier_nb_neighbors:
        fused, inlier_idx = fused.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
        )
        n_removed = n_after_voxel - len(inlier_idx)
        if n_removed > 0:
            print(f"Removed {n_removed} flying-point outliers from the fused init pointcloud")
    n_after_outlier = len(fused.points)

    if cluster_min_points > 0 and n_after_outlier > cluster_min_points:
        labels = np.asarray(fused.cluster_dbscan(eps=cluster_eps, min_points=cluster_min_points))
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
    print(
        f"Initializing gaussians from {n_kept}/{len(points)} frames "
        f"({n_before} points, downsampled to {n_after_voxel}, "
        f"{fused_points.shape[0]} after outlier/cluster cleanup)"
    )
    return (
        torch.from_numpy(fused_points).float().to(device),
        torch.from_numpy(fused_colors).float().to(device),
    )


# ----------------------------------------------------------------------
# Dataset -- training views are farthest-point sampled (by camera position) from
# whichever frames survived the pairwise pose/mask check.
# ----------------------------------------------------------------------
class KeyframeDataset(torch.utils.data.Dataset):
    def __init__(
        self, all_images, all_masks, all_poses, K, candidate_idx, n_views: int = 40,
        init_points: np.ndarray = None, view_coverage_thresh: float = 0.0,
    ):
        """
        Args:
            candidate_idx: indices into the all_frames arrays that survived the
                pose-jump filter (or np.arange(M) if no filtering was done).
            n_views: how many to farthest-point-sample from candidate_idx for
                training. <= 0 means use every candidate frame.
            init_points: fused gaussian-init pointcloud (world frame), used only for
                the optional post-FPS coverage check below.
            view_coverage_thresh: if > 0, each farthest-point-sampled view is
                projected against init_points via pointcloud_mask_coverage and
                dropped (no replacement) if it scores below this. Catches views whose
                pose/mask disagrees with the cleaned-up pointcloud despite passing
                the coarser pose-jump filter. 0 disables.
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
            if n_dropped > 0:
                print(
                    f"View coverage check: dropped {n_dropped}/{len(chosen)} sampled "
                    f"views scoring below {view_coverage_thresh} against the fused init "
                    f"pointcloud (scores: {np.round(scores, 3).tolist()})"
                )
            chosen = chosen[view_keep]

        # Mask GT RGB at load time so self.images is the masked GT everywhere it's
        # used (training target, preview GT|Render comparison).
        images = torch.from_numpy(all_images[chosen]).float() / 255.0  # [N, H, W, 3]
        masks = torch.from_numpy(all_masks[chosen])  # [N, H, W] bool
        self.images = images * masks[..., None].float()
        self.masks = masks
        self.camtoworlds = torch.from_numpy(all_poses[chosen]).float()  # [N, 4, 4]
        self.Ks = torch.from_numpy(K).float().unsqueeze(0).repeat(len(chosen), 1, 1)
        self.height, self.width = all_images.shape[1:3]

        print(
            f"Loaded {len(chosen)} training views (farthest-point sampled from "
            f"{len(candidate_idx)} candidate frames, {self.width}x{self.height})"
        )

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


# ----------------------------------------------------------------------
# Gaussian initialization
# ----------------------------------------------------------------------
def create_splats_with_optimizers(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scale: float,
    init_opacity: float,
    sh_degree: int,
    scene_scale: float,
    device: str,
    means_lr_mult: float = 1.0,
):
    # dtype=torch.float32 is explicit throughout: an earlier step (SAM2 autocast, etc.)
    # can leave torch's default dtype stamped as bfloat16, which any factory call
    # without dtype= would silently inherit.
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
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr}],
            eps=1e-15,
        )
        for name, _, lr in params
    }
    return splats, optimizers


# ----------------------------------------------------------------------
# Viser scene helpers
# ----------------------------------------------------------------------
def add_camera_frustums(server: viser.ViserServer, dataset: KeyframeDataset, prefix: str, color):
    fx = dataset.Ks[0, 0, 0].item()
    fov = 2 * math.atan(dataset.width / (2 * fx))
    aspect = dataset.width / dataset.height

    for i in range(len(dataset)):
        c2w = dataset.camtoworlds[i].numpy()
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        wxyz = rotmat_to_wxyz(R)
        img_np = (dataset.images[i].numpy() * 255).astype(np.uint8)
        server.scene.add_camera_frustum(
            f"{prefix}/frustum_{i:03d}",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            color=color,
            image=img_np,
            wxyz=wxyz,
            position=t,
        )


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


# ----------------------------------------------------------------------
# Shared rendering (used by both the training loop and the viewer callback).
# Returns a dict with keys: colors, alphas, median_depth (None for 3dgs), info.
# ----------------------------------------------------------------------
def render_splats(
    model: str,
    splats: torch.nn.ParameterDict,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    near_plane: float = 0.001,
    far_plane: float = 100.0,
    render_mode: str = "RGB",
):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    # .float(): camtoworlds can arrive as bfloat16 (see dtype note above); linalg.inv
    # doesn't support it.
    viewmats = torch.linalg.inv_ex(camtoworlds.float()).inverse

    if model == "2dgs":
        render_colors, render_alphas, _, _, _, render_median, info = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
        )
    else:
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            packed=False,  # DefaultStrategy.step_post_backward defaults to packed=False
        )
        render_median = None

    return {
        "colors": render_colors,
        "alphas": render_alphas,
        "median_depth": render_median,
        "info": info,
    }


# ----------------------------------------------------------------------
# Mesh extraction via TSDF fusion over the keyframe views
# ----------------------------------------------------------------------
def extract_mesh(
    model: str,
    splats: torch.nn.ParameterDict,
    dataset: "KeyframeDataset",
    device: str,
    sh_degree: int,
    out_path: Path,
    pose_adjust: "CameraOptModule" = None,
    voxel_size: float = 0.0015,
    sdf_trunc: float = None,
    depth_trunc: float = 3.0,
):
    """Fuses per-view renders of `splats` into a TSDF volume for vertex-colored
    mesh extraction."""
    if o3d is None:
        print("open3d not available, skipping mesh extraction")
        return None

    if sdf_trunc is None:
        sdf_trunc = voxel_size * 4

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    K = dataset.Ks[0].numpy()
    width, height = dataset.width, dataset.height
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    Ks_render = torch.from_numpy(K).float().to(device)[None].repeat(len(dataset), 1, 1)

    with torch.no_grad():
        all_ids = torch.arange(len(dataset), device=device)
        camtoworlds = dataset.camtoworlds.to(device).float()
        if pose_adjust is not None:
            camtoworlds = pose_adjust(camtoworlds, all_ids).float()

        for i in range(len(dataset)):
            out = render_splats(
                model,
                splats,
                camtoworlds[i : i + 1],
                Ks_render[i : i + 1],
                width,
                height,
                sh_degree,
                render_mode="RGB+ED",
            )
            render_colors = out["colors"]
            rgb = (render_colors[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            if out["median_depth"] is not None:
                depth = out["median_depth"][0, ..., 0].cpu().numpy().astype(np.float32)
            else:
                depth = render_colors[0, ..., 3].cpu().numpy().astype(np.float32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            w2c = np.linalg.inv(camtoworlds[i].cpu().numpy())
            volume.integrate(rgbd, intrinsic, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Saved mesh to {out_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
    return mesh


def _organized_normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Computes per-pixel camera-frame surface normals from a depth map's own
    image-space gradients (cross product of the local dx/dy 3D tangent vectors) --
    exploits the depth map's pixel-grid structure instead of unorganized-pointcloud
    KNN, and gets a camera-facing orientation for free.

    Returns an (H,W,3) array (unit normals; garbage/zero for zero-depth or
    ill-defined pixels -- callers must combine with a validity mask)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    Z = depth
    pts = np.stack([X, Y, Z], axis=-1)  # (H,W,3), camera frame

    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, :-1] = pts[:, 1:] - pts[:, :-1]
    dy[:-1, :] = pts[1:, :] - pts[:-1, :]

    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norm, 1e-8, None)

    # Orient toward the camera (a point at depth Z looking back at the origin has
    # view direction -pts; a correctly-oriented outward normal satisfies dot(n, -pts) > 0).
    facing = np.sum(normals * -pts, axis=-1, keepdims=True)
    normals = np.where(facing < 0, -normals, normals)
    return normals


def extract_mesh_poisson(
    model: str,
    splats: torch.nn.ParameterDict,
    dataset: "KeyframeDataset",
    device: str,
    sh_degree: int,
    out_path: Path,
    pose_adjust: "CameraOptModule" = None,
    depth_trunc: float = 3.0,
    poisson_depth: int = 9,
    density_trim_quantile: float = 0.02,
    pool_voxel_size: float = 0.001,
):
    """Alternative to extract_mesh's TSDF fusion: renders depth from every training
    view, backprojects each into a point+normal+color cloud (normals via
    _organized_normals_from_depth), pools all views' points together without
    volumetric averaging (TSDF fusion was smearing/distorting the mesh), then runs
    Poisson surface reconstruction. pool_voxel_size downsamples the pooled cloud
    first (also denoises normals by averaging within each voxel); 0 disables.
    density_trim_quantile drops the lowest-density fraction of reconstructed
    vertices (standard Poisson cleanup for hallucinated surface at sparse regions)."""
    if o3d is None:
        print("open3d not available, skipping mesh extraction")
        return None

    K = dataset.Ks[0].numpy()
    width, height = dataset.width, dataset.height

    all_points, all_normals, all_colors = [], [], []
    with torch.no_grad():
        all_ids = torch.arange(len(dataset), device=device)
        camtoworlds = dataset.camtoworlds.to(device).float()
        if pose_adjust is not None:
            camtoworlds = pose_adjust(camtoworlds, all_ids).float()

        Ks_render = torch.from_numpy(K).float().to(device)[None].repeat(len(dataset), 1, 1)
        for i in range(len(dataset)):
            out = render_splats(
                model, splats, camtoworlds[i : i + 1], Ks_render[i : i + 1],
                width, height, sh_degree, render_mode="RGB+ED",
            )
            render_colors, render_alphas = out["colors"], out["alphas"]
            rgb = render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
            depth = (
                out["median_depth"][0, ..., 0].cpu().numpy().astype(np.float32)
                if out["median_depth"] is not None
                else render_colors[0, ..., 3].cpu().numpy().astype(np.float32)
            )
            valid = (render_alphas[0, ..., 0].cpu().numpy() > 0.5) & (depth > 0) & (depth < depth_trunc)
            if not np.any(valid):
                continue

            normals_cam = _organized_normals_from_depth(depth, K)

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            X = (xs - cx) * depth / fx
            Y = (ys - cy) * depth / fy
            pts_cam = np.stack([X, Y, depth], axis=-1)[valid]  # (M,3)
            normals_cam_v = normals_cam[valid]

            c2w = camtoworlds[i].cpu().numpy()
            R, t = c2w[:3, :3], c2w[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            normals_world = (R @ normals_cam_v.T).T  # rotation only, no translation

            all_points.append(pts_world)
            all_normals.append(normals_world)
            all_colors.append(rgb[valid])

    if not all_points:
        print("No valid depth pixels across any view, skipping Poisson reconstruction")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate(all_normals, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
    n_before_downsample = len(pcd.points)
    print(f"Pooled {n_before_downsample} points (with normals) from {len(dataset)} views for Poisson")

    if pool_voxel_size > 0:
        pcd = pcd.voxel_down_sample(pool_voxel_size)
        print(f"Downsampled to {len(pcd.points)} points (voxel_size={pool_voxel_size})")

    return _run_poisson_and_save(pcd, out_path, poisson_depth, density_trim_quantile, "depth-render")


def _run_poisson_and_save(pcd, out_path: Path, poisson_depth: int, density_trim_quantile: float, label: str):
    """Shared Poisson-reconstruct + density-trim + save tail, used by both
    extract_mesh_poisson (depth-render-derived cloud) and extract_mesh_means_knn
    (gaussian-means-derived cloud)."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    if density_trim_quantile > 0:
        densities = np.asarray(densities)
        thresh = np.quantile(densities, density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Saved {label} Poisson mesh to {out_path} "
          f"({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
    return mesh


def extract_mesh_means_knn(
    splats: torch.nn.ParameterDict,
    out_path: Path,
    device: str,
    poisson_depth: int = 9,
    density_trim_quantile: float = 0.02,
    knn_neighbors: int = 30,
    opacity_thresh: float = 0.1,
):
    """Alternative to extract_mesh (TSDF) and extract_mesh_poisson (depth-render
    clouds): skips rendering entirely and treats the gaussians' own means directly as
    the point cloud (densification already spreads them over the surface).
    Low-opacity gaussians are dropped first via opacity_thresh. Normals come from
    Open3D's KNN local-plane-fit + a tangent-plane consistency pass, since an
    unorganized cloud has no pixel-grid structure to get orientation for free from."""
    if o3d is None:
        print("open3d not available, skipping mesh extraction")
        return None

    means = splats["means"].detach().cpu().numpy()
    opacities = torch.sigmoid(splats["opacities"]).detach().cpu().numpy()
    # sh0 is the DC SH term, not a plain color -- undo rgb_to_sh (same as
    # viser_map_viewer.load_gaussians_for_viz).
    C0 = 0.28209479177387814
    sh0 = splats["sh0"][:, 0, :].detach().cpu().numpy()
    colors = np.clip(sh0 * C0 + 0.5, 0.0, 1.0)

    keep = opacities > opacity_thresh
    means, colors = means[keep], colors[keep]
    print(f"Using {means.shape[0]}/{len(keep)} gaussian means (opacity > {opacity_thresh}) as the point cloud")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_neighbors)
    )
    pcd.orient_normals_consistent_tangent_plane(k=knn_neighbors)

    return _run_poisson_and_save(pcd, out_path, poisson_depth, density_trim_quantile, "means+KNN")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # --config is parsed first so its values can become parser defaults via
    # set_defaults() before parse_args() -- an explicit CLI flag always overrides it.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=None, type=str,
        help="Path to a YAML file of default CLI args (e.g. configs/reconstruct/"
        "*.yaml). Flags passed explicitly on the command line override it.",
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--results_path", default=None, type=str)
    parser.add_argument(
        "--max_depth_jump", default=0.03, type=float,
        help="Drop depth pixels differing from a 4-connected neighbor by more than "
        "this many meters before backprojecting (see depth_edge_mask). 0 disables.",
    )
    parser.add_argument(
        "--min_depth", default=0.05, type=float,
        help="Discard depth pixels closer than this (meters) before backprojecting.",
    )
    parser.add_argument(
        "--max_depth", default=1.0, type=float,
        help="Discard depth pixels farther than this (meters) before backprojecting.",
    )
    parser.add_argument("--model", default="2dgs", choices=["2dgs", "3dgs"])
    parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--init_scale", default=1.0, type=float)
    parser.add_argument("--init_opacity", default=0.5, type=float)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--ssim_lambda", default=0.2, type=float)
    parser.add_argument(
        "--bg_alpha_lambda", default=0.01, type=float,
        help="Weight for penalizing accumulated alpha outside the mask -- catches "
        "stray gaussians the L1/SSIM terms can't see behind the masked background. "
        "0 disables.",
    )
    parser.add_argument(
        "--pose_opt_lr", default=1.0e-4, type=float,
        help="Learning rate for the translation (dx) pose-correction embedding.",
    )
    parser.add_argument(
        "--pose_opt_rot_lr_mult", default=10.0, type=float,
        help="Rotation (drot) pose-correction embedding's lr = --pose_opt_lr * this "
        "(separate lr since dx/drot go through different nonlinearities).",
    )
    parser.add_argument("--pose_opt_reg", default=1e-7, type=float)
    parser.add_argument("--no_pose_opt", default=False, action="store_true")
    parser.add_argument(
        "--pose_warmup_steps", default=1000, type=int,
        help="Phase 1: for this many steps, only pose correction is optimized -- "
        "gaussians are frozen and densification disabled. 0 disables.",
    )
    parser.add_argument("--pose_opt_lr_phase2_mult", default=2.0, type=float)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--viewer_every", default=10, type=int)
    parser.add_argument("--disable_viewer", default=False, action="store_true")
    parser.add_argument("--extract_mesh", default=False, action="store_true")
    parser.add_argument(
        "--mesh_method", default="poisson", choices=["poisson", "means_knn", "tsdf"],
        help="poisson (default): pool every view's rendered depth+normals and run "
        "Poisson surface reconstruction (see extract_mesh_poisson). means_knn: "
        "Poisson-reconstruct directly from the gaussians' means, no rendering (see "
        "extract_mesh_means_knn). tsdf: volumetric-fusion mesh (see extract_mesh).",
    )
    parser.add_argument("--mesh_voxel_size", default=0.005, type=float)
    parser.add_argument("--mesh_sdf_trunc", default=0.01, type=float)
    parser.add_argument(
        "--poisson_depth", default=9, type=int,
        help="Octree depth for Poisson surface reconstruction. Higher = finer "
        "detail but slower/more memory.",
    )
    parser.add_argument(
        "--poisson_density_trim_quantile", default=0.02, type=float,
        help="Drop the lowest-density fraction of Poisson-reconstructed vertices "
        "(usually hallucinated surface at sparse/noisy edges). 0 disables.",
    )
    parser.add_argument(
        "--poisson_pool_voxel_size", default=0.001, type=float,
        help="Voxel-downsample the pooled multi-view point+normal cloud before "
        "Poisson (see extract_mesh_poisson). 0 disables.",
    )
    parser.add_argument(
        "--means_knn_neighbors", default=30, type=int,
        help="(--mesh_method means_knn) Neighbors for KNN normal estimation + "
        "tangent-plane orientation consistency.",
    )
    parser.add_argument(
        "--means_opacity_thresh", default=0.1, type=float,
        help="(--mesh_method means_knn) Drop gaussians with opacity below this "
        "before reconstructing from their means.",
    )
    parser.add_argument(
        "--recenter_gaussians", default=True,
        action=argparse.BooleanOptionalAction,
        help="Before saving, recenter+PCA-align the gaussians (and mesh) onto an "
        "object-centric frame (centroid at origin, PCA principal axes). Camera "
        "poses (optimized_kf_poses.npy) stay in the original frame regardless. The "
        "alignment transform is saved to pca_alignment.npy.",
    )
    parser.add_argument(
        "--init_voxel_size", default=0.002, type=float,
        help="Voxel size (meters) to downsample the init pointcloud before "
        "creating gaussians. 0 disables.",
    )
    parser.add_argument(
        "--init_outlier_nb_neighbors", default=20, type=int,
        help="Statistical outlier removal on the fused init pointcloud: drops "
        "points beyond --init_outlier_std_ratio std devs from their neighbors. "
        "0 disables.",
    )
    parser.add_argument("--init_outlier_std_ratio", default=2.0, type=float)
    parser.add_argument(
        "--init_cluster_min_points", default=20, type=int,
        help="After outlier removal, DBSCAN-cluster the fused init pointcloud and "
        "keep only the largest cluster (drops coherent misprojected blobs, e.g. "
        "background leaking through mask edges). 0 disables.",
    )
    parser.add_argument(
        "--init_cluster_eps", default=0.02, type=float,
        help="DBSCAN neighborhood radius (meters) for --init_cluster_min_points.",
    )
    parser.add_argument("--means_lr_mult", default=3.0, type=float)
    parser.add_argument(
        "--n_views", default=40, type=int,
        help="Number of views to farthest-point sample (by camera position) "
        "from the exported per-frame pool.",
    )
    parser.add_argument(
        "--view_coverage_thresh", default=0.0, type=float,
        help="After farthest-point sampling, drop sampled views scoring below this "
        "on the fused-pointcloud/mask IoU check (see KeyframeDataset). 0 disables.",
    )
    parser.add_argument(
        "--pose_jump_max_trans", default=0.05, type=float,
        help="Drop both frames flanking a temporally-adjacent pose jump larger than "
        "this many meters (see filter_frames_by_pose_jump). 0 disables.",
    )
    parser.add_argument(
        "--pose_jump_max_rot_deg", default=15.0, type=float,
        help="Rotation-angle counterpart to --pose_jump_max_trans (degrees); either "
        "threshold being exceeded triggers the drop.",
    )
    if config_args.config is not None:
        import yaml

        with open(config_args.config) as f:
            config_defaults = yaml.safe_load(f) or {}
        unknown = set(config_defaults) - {a.dest for a in parser._actions}
        if unknown:
            raise ValueError(
                f"{config_args.config} has unrecognized key(s) {sorted(unknown)} -- "
                "check for typos against reconstruct.py's --help."
            )
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(remaining_argv)
    if args.results_path is None:
        parser.error("--results_path is required (either on the command line or in --config)")

    device = "cuda:0"
    results_path = Path(args.results_path)

    print("Loading exported frames and backprojecting masked depth...")
    points, colors, all_images, all_masks, all_poses, K, frame_idx = load_all_frames(
        results_path, max_depth_jump=args.max_depth_jump,
        min_depth=args.min_depth, max_depth=args.max_depth,
    )

    n = len(points)
    keep = np.ones(n, dtype=bool)
    if args.pose_jump_max_trans > 0:
        keep, _ = filter_frames_by_pose_jump(
            results_path, all_poses, frame_idx,
            max_trans=args.pose_jump_max_trans, max_rot_deg=args.pose_jump_max_rot_deg,
        )
    candidate_idx = np.where(keep)[0]

    init_points, init_colors = build_init_pointcloud(
        points, colors, keep, device, voxel_size=args.init_voxel_size,
        outlier_nb_neighbors=args.init_outlier_nb_neighbors,
        outlier_std_ratio=args.init_outlier_std_ratio,
        cluster_eps=args.init_cluster_eps,
        cluster_min_points=args.init_cluster_min_points,
    )
    if init_points is None:
        print("No export data found for gaussian init, falling back to random init")
        scene_scale = float(np.linalg.norm(all_poses[:, :3, 3], axis=-1).max()) + 1e-2
        n_pts = 50_000
        init_points = scene_scale * (torch.rand((n_pts, 3), dtype=torch.float32, device=device) * 2 - 1)
        init_colors = torch.rand((n_pts, 3), dtype=torch.float32, device=device)
        candidate_idx = np.arange(n)
    elif args.recenter_gaussians:
        # Re-center + PCA-align BEFORE training, not just at save time: the live
        # tracker's object-frame origin can sit far off-center, and scene_scale /
        # pose_opt's world-frame correction (see CameraOptModule) both benefit from
        # the object actually sitting at the origin during training.
        T_align_pre = compute_pca_alignment(init_points.detach().cpu().numpy())
        R_pre, t_pre = T_align_pre[:3, :3], T_align_pre[:3, 3]
        R_pre_t = torch.from_numpy(R_pre).float().to(device)
        t_pre_t = torch.from_numpy(t_pre).float().to(device)
        init_points = (R_pre_t @ init_points.T).T + t_pre_t
        all_poses = np.einsum("ij,njk->nik", T_align_pre, all_poses)
        print("Pre-training PCA alignment applied to init pointcloud + all camera poses")

    centroid = init_points.mean(dim=0)
    scene_scale = float((init_points - centroid).norm(dim=-1).max().item()) + 1e-2
    print(f"scene_scale = {scene_scale:.4f} (from init pointcloud spread)")

    dataset = KeyframeDataset(
        all_images, all_masks, all_poses, K, candidate_idx, n_views=args.n_views,
        init_points=init_points.detach().cpu().numpy(),
        view_coverage_thresh=args.view_coverage_thresh,
    )

    splats, optimizers = create_splats_with_optimizers(
        init_points,
        init_colors,
        init_scale=args.init_scale,
        init_opacity=args.init_opacity,
        sh_degree=args.sh_degree,
        scene_scale=scene_scale,
        device=device,
        means_lr_mult=args.means_lr_mult,
    )
    print("Initialized", splats["means"].shape[0], "gaussians")

    key_for_gradient = "gradient_2dgs" if args.model == "2dgs" else "means2d"
    refine_start_iter = max(500, args.pose_warmup_steps)
    strategy = DefaultStrategy(
        verbose=True, key_for_gradient=key_for_gradient, refine_start_iter=refine_start_iter
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    pose_opt = not args.no_pose_opt
    if pose_opt:
        pose_adjust = CameraOptModule(len(dataset)).to(device)
        pose_optimizer = torch.optim.Adam(
            [
                {"params": pose_adjust.embed_t.parameters(), "lr": args.pose_opt_lr},
                {"params": pose_adjust.embed_r.parameters(),
                 "lr": args.pose_opt_lr * args.pose_opt_rot_lr_mult},
            ],
            weight_decay=args.pose_opt_reg,
        )

    if args.pose_warmup_steps > 0:
        if not pose_opt:
            raise ValueError("--pose_warmup_steps > 0 requires pose optimization (don't pass --no_pose_opt)")
        print(f"Phase 1: freezing gaussians for {args.pose_warmup_steps} steps (pose-only warmup)")
        for p in splats.values():
            p.requires_grad_(False)

    # ------------------------------------------------------------
    # Viser scene setup
    # ------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    # point2pose poses are Z-up, right-handed (see point2pose/utils/rerun_viz.py
    # rr_init: RIGHT_HAND_Z_UP), unlike kv_tracker's Y-down convention.
    server.scene.set_up_direction("+z")
    add_camera_frustums(server, dataset, "cameras/initial", color=(180, 180, 180))
    opt_frustum_handles = []
    fx = dataset.Ks[0, 0, 0].item()
    fov = 2 * math.atan(dataset.width / (2 * fx))
    aspect = dataset.width / dataset.height
    for i in range(len(dataset)):
        c2w = dataset.camtoworlds[i].numpy()
        h = server.scene.add_camera_frustum(
            f"cameras/optimized/frustum_{i:03d}",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            color=(255, 140, 0),
            wxyz=rotmat_to_wxyz(c2w[:3, :3]),
            position=c2w[:3, 3],
        )
        opt_frustum_handles.append(h)

    gui_step = server.gui.add_text("Step", initial_value="0 / " + str(args.max_steps))
    gui_loss = server.gui.add_text("Loss", initial_value="-")

    gui_view_slider = server.gui.add_slider(
        "Preview view", min=0, max=len(dataset) - 1, step=1, initial_value=0
    )

    current_sh_degree = [0]  # mutable box; updated by the training loop each step

    def render_preview_view(view_idx: int) -> np.ndarray:
        gt_np = (dataset.images[view_idx].numpy() * 255).astype(np.uint8)
        c2w = dataset.camtoworlds[view_idx : view_idx + 1].to(device)
        K = dataset.Ks[view_idx : view_idx + 1].to(device)
        with torch.no_grad():
            if pose_opt:
                view_id = torch.tensor([view_idx], device=device)
                c2w = pose_adjust(c2w, view_id)
            out = render_splats(
                args.model, splats, c2w, K, dataset.width, dataset.height, current_sh_degree[0]
            )
            render_np = (out["colors"][0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return np.concatenate([gt_np, render_np], axis=1)

    gui_preview_view = server.gui.add_image(
        render_preview_view(0), label="GT | Render (selected view)"
    )

    @gui_view_slider.on_update
    def _(_) -> None:
        gui_preview_view.image = render_preview_view(gui_view_slider.value)

    viewer = None
    if not args.disable_viewer:

        def viewer_render_fn(camera_state: CameraState, render_tab_state: GsplatRenderTabState):
            if render_tab_state.preview_render:
                width = render_tab_state.render_width
                height = render_tab_state.render_height
            else:
                width = render_tab_state.viewer_width
                height = render_tab_state.viewer_height
            c2w = torch.from_numpy(camera_state.c2w).float().to(device)
            K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)

            with torch.no_grad():
                out = render_splats(
                    args.model,
                    splats,
                    c2w[None],
                    K[None],
                    width,
                    height,
                    min(render_tab_state.max_sh_degree, args.sh_degree),
                    near_plane=render_tab_state.near_plane,
                    far_plane=render_tab_state.far_plane,
                    render_mode="RGB+ED",
                )
                render_colors, render_alphas, render_median, info = (
                    out["colors"],
                    out["alphas"],
                    out["median_depth"],
                    out["info"],
                )
                render_tab_state.total_gs_count = len(splats["means"])
                render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

                if render_tab_state.render_mode == "depth(accumulated)":
                    depth = render_colors[0, ..., 3]
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
                    return apply_float_colormap(depth_norm[..., None], render_tab_state.colormap).cpu().numpy()
                elif render_tab_state.render_mode == "depth(expected)":
                    depth = (
                        render_median[0, ..., 0] if render_median is not None else render_colors[0, ..., 3]
                    )
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
                    return apply_float_colormap(depth_norm[..., None], render_tab_state.colormap).cpu().numpy()
                elif render_tab_state.render_mode == "alpha":
                    return apply_float_colormap(render_alphas[0], render_tab_state.colormap).cpu().numpy()
                else:
                    return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()

        viewer = GsplatViewer(
            server=server,
            render_fn=viewer_render_fn,
            output_dir=results_path,
            mode="training",
        )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
        )
    ]
    if pose_opt:
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(
                pose_optimizer, gamma=0.01 ** (1.0 / args.max_steps)
            )
        )

    trainloader_iter = iter(trainloader)
    for step in range(args.max_steps):
        if args.pose_warmup_steps > 0 and step == args.pose_warmup_steps:
            print(f"\nPhase 2: unfreezing gaussians at step {step}, "
                  f"dropping pose lr by {args.pose_opt_lr_phase2_mult}x")
            for p in splats.values():
                p.requires_grad_(True)
            for g in pose_optimizer.param_groups:
                g["lr"] *= args.pose_opt_lr_phase2_mult

        if viewer is not None:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device)  # [1, H, W, 3]
        masks = data["mask"].to(device)  # [1, H, W]
        image_ids = data["image_id"].to(device)  # [1]

        if pose_opt:
            camtoworlds = pose_adjust(camtoworlds, image_ids)

        sh_degree_to_use = min(step // 1000, args.sh_degree)
        current_sh_degree[0] = sh_degree_to_use

        out = render_splats(
            args.model,
            splats,
            camtoworlds,
            Ks,
            dataset.width,
            dataset.height,
            sh_degree_to_use,
        )
        render_colors, render_alphas, info = out["colors"], out["alphas"], out["info"]

        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        rendered = render_colors * masks[..., None]
        target = pixels * masks[..., None]

        l1loss = F.l1_loss(rendered, target)
        ssimloss = 1.0 - ssim_fn(
            rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), data_range=1.0
        )
        loss = (1 - args.ssim_lambda) * l1loss + args.ssim_lambda * ssimloss

        if args.bg_alpha_lambda > 0:
            bg_alpha = render_alphas[..., 0] * (~masks).float()
            bg_alpha_loss = bg_alpha.mean()
            loss = loss + args.bg_alpha_lambda * bg_alpha_loss

        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_opt:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        for sch in schedulers:
            sch.step()

        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        if viewer is not None:
            viewer.lock.release()
            num_train_rays_per_step = dataset.width * dataset.height
            num_train_steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
            viewer.render_tab_state.num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            viewer.update(step, num_train_rays_per_step)

        if step % args.viewer_every == 0 or step == args.max_steps - 1:
            gui_step.value = f"{step} / {args.max_steps}"
            gui_loss.value = f"{loss.item():.4f}"

            if pose_opt:
                with torch.no_grad():
                    all_ids = torch.arange(len(dataset), device=device)
                    all_c2w = dataset.camtoworlds.to(device)
                    updated = pose_adjust(all_c2w, all_ids).float().cpu().numpy()
                for i, h in enumerate(opt_frustum_handles):
                    h.wxyz = rotmat_to_wxyz(updated[i, :3, :3])
                    h.position = updated[i, :3, 3]

            gui_preview_view.image = render_preview_view(gui_view_slider.value)

            print(f"step {step}/{args.max_steps} loss={loss.item():.4f} "
                  f"n_gaussians={splats['means'].shape[0]}", end="\r")

    print("\nTraining done.")

    if pose_opt:
        with torch.no_grad():
            all_ids = torch.arange(len(dataset), device=device)
            all_c2w = dataset.camtoworlds.to(device)
            optimized_poses = pose_adjust(all_c2w, all_ids).float().cpu().numpy()
        # NOT re-aligned to the PCA frame below -- for debugging/inspection only;
        # only the gaussians/mesh get the object-centric frame.
        np.save(results_path / "optimized_kf_poses.npy", optimized_poses)
        print(f"Saved optimized poses to {results_path / 'optimized_kf_poses.npy'}")

    mesh = None
    if args.extract_mesh:
        # Must happen before re-aligning splats below: extract_mesh(_poisson) renders
        # `splats` from `dataset.camtoworlds`, valid together only in the original frame.
        mesh_out_path = results_path / "mesh_raw.ply" if args.recenter_gaussians else results_path / "mesh.ply"
        if args.mesh_method == "poisson":
            mesh = extract_mesh_poisson(
                args.model,
                splats,
                dataset,
                device,
                args.sh_degree,
                mesh_out_path,
                pose_adjust=pose_adjust if pose_opt else None,
                poisson_depth=args.poisson_depth,
                density_trim_quantile=args.poisson_density_trim_quantile,
                pool_voxel_size=args.poisson_pool_voxel_size,
            )
        elif args.mesh_method == "means_knn":
            mesh = extract_mesh_means_knn(
                splats,
                mesh_out_path,
                device,
                poisson_depth=args.poisson_depth,
                density_trim_quantile=args.poisson_density_trim_quantile,
                knn_neighbors=args.means_knn_neighbors,
                opacity_thresh=args.means_opacity_thresh,
            )
        else:
            mesh = extract_mesh(
                args.model,
                splats,
                dataset,
                device,
                args.sh_degree,
                mesh_out_path,
                pose_adjust=pose_adjust if pose_opt else None,
                voxel_size=args.mesh_voxel_size,
                sdf_trunc=args.mesh_sdf_trunc,
            )

    if args.recenter_gaussians:
        # Re-align AGAIN post-training: pose_opt/densification can drift the
        # gaussians' centroid/orientation, so recompute from the final means rather
        # than assuming the pre-training alignment still holds exactly.
        means_np = splats["means"].detach().cpu().numpy()
        T_align = compute_pca_alignment(means_np)
        aligned_splats = apply_alignment_to_gaussians(splats, T_align)
        np.save(results_path / "pca_alignment.npy", T_align)
        print(f"Computed PCA alignment (recentered + axis-aligned), saved to "
              f"{results_path / 'pca_alignment.npy'}")

        if mesh is not None:
            R, t = T_align[:3, :3], T_align[:3, 3]
            verts = np.asarray(mesh.vertices)
            mesh.vertices = o3d.utility.Vector3dVector((R @ verts.T).T + t)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(results_path / "mesh.ply"), mesh)
            print(f"Saved PCA-aligned mesh to {results_path / 'mesh.ply'}")
    else:
        aligned_splats = splats

    ckpt_path = results_path / "gaussians.pt"
    torch.save({k: v.detach().cpu() for k, v in aligned_splats.items()}, ckpt_path)
    print(f"Saved gaussians to {ckpt_path}")

    if viewer is not None:
        viewer.complete()
        print(f"Viewer running at http://localhost:{args.port} -- Ctrl+C to exit.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
