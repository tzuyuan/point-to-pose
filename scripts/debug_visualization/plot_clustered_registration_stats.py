#!/usr/bin/env python3
"""
Analyze + visualize clustered registration results saved in meta_data.npz.

This script is intended for runs using the clustered register module
`svd_cluster_ransac_register.py`, which returns `stats["clusters"]` as a list of
candidate clusters. The pipeline logger stores this raw list under `reg_clusters`
(an object array in the NPZ).

What you get:
  - Dataset-wide plots: #clusters per frame, coverage, cluster size histogram
  - Frame-level plots: candidate summary (ninliers/mean_res/score) + 3D plots
    that color correspondences by cluster assignment (derived from inlier indices).
  - (Optional) GT comparison: per-cluster pose error vs GT (GT normalized to the
    first frame by right-multiplying inv(gt_pose_first), as used in the GT
    reconstruction notebook.

Example:
  python scripts/debug_visualization/plot_clustered_registration_stats.py \
    --meta_data_path /path/to/meta_data/meta_data.npz \
    --frame 120 \
    --video_dir /path/to/HO3D_V3/evaluation/MPM10
"""

from __future__ import annotations

import os
import sys
import argparse
import glob
import pickle
from typing import Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    import torch
except ImportError:
    torch = None

# Add project root to path for imports (same pattern as plot_registration_stats.py)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional dataset readers for GT pose
try:
    from point2pose.io.sources.dataset.datareader import Ho3dReader, YcbineoatReader
except Exception:
    Ho3dReader = None
    YcbineoatReader = None

try:
    # For pose distance metric matching the register's `_pose_dist`
    from point2pose.utils.lie import log_SE3 as _log_SE3
except Exception:
    _log_SE3 = None

try:
    from point2pose.data_types.frame import Frame
    from point2pose.utils.camera import (
        project_points_to_image,
        extract_cropped_point_cloud,
    )
    from point2pose.utils.transform import inverse_SE3
except Exception:
    Frame = None
    project_points_to_image = None
    extract_cropped_point_cloud = None
    inverse_SE3 = None


def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def _rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation formula without requiring OpenCV."""
    r = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    k = r / theta
    K = _skew(k)
    return np.eye(3, dtype=float) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class _SimpleHo3dGTReader:
    """Lightweight HO3D GT pose reader (avoids cv2/imageio/trimesh deps)."""

    def __init__(self, video_dir: str):
        self.video_dir = video_dir
        self.color_files = sorted(
            glob.glob(os.path.join(self.video_dir, "rgb", "*.jpg"))
        )
        if not self.color_files:
            raise FileNotFoundError(f"No RGB jpgs found under: {self.video_dir}/rgb")
        self.id_strs = [
            os.path.splitext(os.path.basename(p))[0] for p in self.color_files
        ]
        # HO3D: OpenGL cam -> OpenCV cam conversion
        self.glcam_in_cvcam = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=float,
        )

    def __len__(self):
        return len(self.color_files)

    def get_gt_pose(self, i: int) -> Optional[np.ndarray]:
        try:
            i = int(i)
            if i < 0 or i >= len(self.color_files):
                return None
            id_str = self.id_strs[i]
            meta_file = os.path.join(self.video_dir, "meta", f"{id_str}.pkl")
            if not os.path.exists(meta_file):
                # fallback: derive from rgb path
                meta_file = (
                    self.color_files[i]
                    .replace(".jpg", ".pkl")
                    .replace(f"{os.sep}rgb{os.sep}", f"{os.sep}meta{os.sep}")
                )
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            if meta.get("objTrans", None) is None or meta.get("objRot", None) is None:
                return None
            t = np.asarray(meta["objTrans"], dtype=float).reshape(3)
            rvec = np.asarray(meta["objRot"], dtype=float).reshape(3)
            R = _rotvec_to_R(rvec)
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = t
            return self.glcam_in_cvcam @ T
        except Exception:
            return None


class _SimpleYcbineoatGTReader:
    """Lightweight YCBInEOAT GT pose reader."""

    def __init__(self, video_dir: str):
        self.video_dir = video_dir
        self.gt_pose_files = sorted(
            glob.glob(os.path.join(self.video_dir, "annotated_poses", "*"))
        )
        if not self.gt_pose_files:
            raise FileNotFoundError(
                f"No GT pose files found under: {self.video_dir}/annotated_poses"
            )

    def __len__(self):
        return len(self.gt_pose_files)

    def get_gt_pose(self, i: int) -> Optional[np.ndarray]:
        try:
            pose = np.loadtxt(self.gt_pose_files[int(i)]).reshape(4, 4)
            return np.asarray(pose, dtype=float)
        except Exception:
            return None


def find_meta_data_path(
    register_folder: str,
    meta_data_path_override: Optional[str] = None,
    results_dir: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """Find meta_data.npz file in expected locations.

    Mirrors the search logic in `plot_registration_stats.py` but kept minimal here.
    """
    if meta_data_path_override is not None:
        if os.path.exists(meta_data_path_override):
            return meta_data_path_override

    meta_data_paths: list[str] = []

    if results_dir and video_name:
        meta_data_paths.append(
            os.path.join(results_dir, video_name, "meta_data", "meta_data.npz")
        )
    if register_folder:
        meta_data_paths.extend(
            [
                os.path.join(register_folder, "meta_data.npz"),
                os.path.join(os.path.dirname(register_folder), "meta_data.npz"),
                os.path.join(register_folder, "meta_data", "meta_data.npz"),
                os.path.join(
                    os.path.dirname(register_folder), "meta_data", "meta_data.npz"
                ),
            ]
        )

    # Common debug locations
    meta_data_paths.extend(
        [
            os.path.join(
                project_root, "debug", "pipeline", "meta_data", "meta_data.npz"
            ),
            os.path.join(
                os.getcwd(), "debug", "pipeline", "meta_data", "meta_data.npz"
            ),
            os.path.join(os.getcwd(), "meta_data", "meta_data.npz"),
        ]
    )

    # Backward-compat typo
    meta_data_paths.extend(
        [
            os.path.join(register_folder, "meata_data.npz"),
            os.path.join(os.path.dirname(register_folder), "meata_data.npz"),
        ]
    )

    for p in meta_data_paths:
        if p and os.path.exists(p):
            return p
    return None


def unpack_ragged(name: str, store: dict, dim: int = -1) -> list[np.ndarray]:
    """Unpack ragged array data from NPZ storage format (DataLogger packing)."""
    data_key = f"{name}_data"
    offsets_key = f"{name}_offsets"
    lengths_key = f"{name}_lengths"

    if data_key not in store or offsets_key not in store or lengths_key not in store:
        return []

    data = store[data_key]
    offsets = store[offsets_key]
    lengths = store[lengths_key]

    out: list[np.ndarray] = []
    for off, L in zip(offsets, lengths):
        flat = data[off : off + L]

        if L == 0 or len(flat) == 0:
            if dim == 3:
                out.append(np.empty((0, 3)))
            elif dim == 2:
                out.append(np.empty((0, 2)))
            else:
                out.append(np.empty((0,)))
            continue

        if dim == 3:
            if len(flat) < 3 or len(flat) % 3 != 0:
                out.append(np.empty((0, 3)))
                continue
            out.append(flat.reshape(-1, 3))
        elif dim == 2:
            if len(flat) < 2 or len(flat) % 2 != 0:
                out.append(np.empty((0, 2)))
                continue
            out.append(flat.reshape(-1, 2))
        else:
            out.append(np.asarray(flat))

    return out


def _as_py(obj: Any) -> Any:
    """Convert common numpy wrapper types to plain Python objects."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return _as_py(obj.item())
        return obj.tolist()
    return obj


def _clusters_for_frame(store: dict, frame_idx: int) -> list[dict]:
    if "reg_clusters" not in store:
        return []
    arr = store["reg_clusters"]
    if not isinstance(arr, np.ndarray):
        # unexpected, but handle list-of-frames
        v = arr[frame_idx]
        v = _as_py(v)
        return v if isinstance(v, list) else []

    v = arr[frame_idx]
    v = _as_py(v)
    if v is None:
        return []
    if isinstance(v, list):
        # ensure each is a dict
        return [c for c in v if isinstance(c, dict)]
    if isinstance(v, dict):
        return [v]
    return []


def _transform_pts(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t[None, :]


def _inverse_SE3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def _rot_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=float)
    cos = (np.trace(R) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Return (translation_error, rotation_error_deg) for T_est vs T_gt."""
    T_err = np.asarray(T_est, dtype=float) @ _inverse_SE3(np.asarray(T_gt, dtype=float))
    t_err = T_err[:3, 3]
    R_err = T_err[:3, :3]
    return float(np.linalg.norm(t_err)), _rot_angle_deg(R_err)


def _pose_dist(Ta: np.ndarray, Tb: np.ndarray) -> float:
    """Match register's pose distance: ||log_SE3(Ta @ inv(Tb))|| (Frobenius)."""
    Ta = np.asarray(Ta, dtype=float)
    Tb = np.asarray(Tb, dtype=float)
    if _log_SE3 is not None:
        return float(np.linalg.norm(_log_SE3(Ta @ _inverse_SE3(Tb))))
    # Fallback: combine translation + rotation (radians) if log_SE3 import fails
    t_err, r_err_deg = _pose_error(Ta, Tb)
    r_err = np.deg2rad(r_err_deg)
    return float(np.sqrt(t_err * t_err + r_err * r_err))


def _infer_dataset(video_dir: str) -> str:
    if not video_dir:
        return "auto"
    if os.path.isdir(os.path.join(video_dir, "annotated_poses")):
        return "ycbineoat"
    rgb_dir = os.path.join(video_dir, "rgb")
    if os.path.isdir(rgb_dir):
        has_jpg = any(fn.lower().endswith(".jpg") for fn in os.listdir(rgb_dir))
        has_png = any(fn.lower().endswith(".png") for fn in os.listdir(rgb_dir))
        if has_jpg and not has_png:
            return "ho3d"
        if has_png and not has_jpg:
            return "ycbineoat"
    return "auto"


def _build_reader(dataset: str, video_dir: str, ho3d_root: Optional[str]) -> Any:
    dataset = (dataset or "auto").lower()
    if dataset == "auto":
        dataset = _infer_dataset(video_dir)

    if dataset == "ho3d":
        # Prefer the full reader if available, but fall back to a lightweight reader.
        if Ho3dReader is not None:
            root = ho3d_root if ho3d_root is not None else os.path.dirname(video_dir)
            try:
                return Ho3dReader(video_dir, root)
            except Exception:
                pass
        try:
            return _SimpleHo3dGTReader(video_dir)
        except Exception:
            return None
    if dataset == "ycbineoat":
        if YcbineoatReader is not None:
            try:
                return YcbineoatReader(video_dir)
            except Exception:
                pass
        try:
            return _SimpleYcbineoatGTReader(video_dir)
        except Exception:
            return None
    return None


def _get_gt_pose(reader: Any, frame_id: int) -> Optional[np.ndarray]:
    """Robustly get GT pose for a given frame id/index."""
    if reader is None:
        return None
    try:
        pose = reader.get_gt_pose(int(frame_id))
        if pose is not None:
            return np.asarray(pose, dtype=float)
    except Exception:
        pass

    # Try map by id_strs if present
    id_strs = getattr(reader, "id_strs", None)
    if isinstance(id_strs, list) and len(id_strs) > 0:
        candidates = []
        candidates.append(str(int(frame_id)))
        candidates.append(f"{int(frame_id):05d}")
        candidates.append(f"{int(frame_id):06d}")
        for s in candidates:
            if s in id_strs:
                idx = int(id_strs.index(s))
                try:
                    pose = reader.get_gt_pose(idx)
                    if pose is not None:
                        return np.asarray(pose, dtype=float)
                except Exception:
                    return None
    return None


def _get_reference_gt_pose(
    reader: Any, frame_ids: np.ndarray, preferred_ref: int = 0
) -> tuple[Optional[np.ndarray], Optional[int]]:
    ref = _get_gt_pose(reader, preferred_ref)
    if ref is not None:
        return ref, int(preferred_ref)
    for fid in frame_ids.tolist():
        ref = _get_gt_pose(reader, int(fid))
        if ref is not None:
            return ref, int(fid)
    return None, None


def plot_clusters_vs_gt_pose(
    clusters: list[dict],
    best_idx: int,
    T_gt: np.ndarray,
    *,
    title: str,
) -> Optional[plt.Figure]:
    if not clusters or T_gt is None:
        return None

    ninliers = np.array(
        [int(c.get("ninliers", len(c.get("inliers", [])))) for c in clusters]
    )
    mean_res = np.array(
        [float(c.get("mean_res", np.nan)) for c in clusters], dtype=float
    )
    score = np.array([float(c.get("score", np.nan)) for c in clusters], dtype=float)

    trans_err = np.full((len(clusters),), np.nan, dtype=float)
    rot_err = np.full((len(clusters),), np.nan, dtype=float)
    for j, c in enumerate(clusters):
        Tj = np.asarray(c.get("T", np.eye(4)), dtype=float)
        te, re = _pose_error(Tj, T_gt)
        trans_err[j] = te
        rot_err[j] = re

    reproj_errors = np.array(
        [float(c.get("reproj_error", np.nan)) for c in clusters], dtype=float
    )
    has_reproj = np.isfinite(reproj_errors).any()

    x = np.arange(len(clusters))
    n_plots = 6 if has_reproj else 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)

    axs[0].bar(x, ninliers, color="tab:blue", alpha=0.8)
    axs[0].set_ylabel("ninliers")

    axs[1].plot(x, mean_res, "o-", color="tab:orange", alpha=0.9)
    axs[1].set_ylabel("mean_res")

    axs[2].plot(x, score, "o-", color="tab:green", alpha=0.9)
    axs[2].set_ylabel("score")

    axs[3].plot(x, trans_err, "o-", color="tab:purple", alpha=0.9)
    axs[3].set_ylabel("trans err (m)")

    axs[4].plot(x, rot_err, "o-", color="tab:red", alpha=0.9)
    axs[4].set_ylabel("rot err (deg)")
    if has_reproj:
        axs[4].set_ylabel("rot err (deg)")
    else:
        axs[4].set_xlabel("cluster candidate index")

    if has_reproj:
        axs[5].plot(x, reproj_errors, "o-", color="tab:pink", alpha=0.9)
        axs[5].set_ylabel("reproj_error")
        axs[5].set_xlabel("cluster candidate index")

    if 0 <= best_idx < len(clusters):
        for ax in axs:
            ax.axvline(best_idx, color="k", linestyle="--", alpha=0.6, linewidth=1.2)

    best_gt = int(np.nanargmin(trans_err)) if np.isfinite(trans_err).any() else -1
    if 0 <= best_gt < len(clusters):
        for ax in axs:
            ax.axvline(
                best_gt, color="magenta", linestyle=":", alpha=0.8, linewidth=1.4
            )

    axs[0].set_title(
        f"{title}\nblack=selected best_idx ({best_idx}), magenta=min trans error ({best_gt})"
    )
    fig.tight_layout()
    return fig


def _set_axes_equal(ax):
    # https://stackoverflow.com/a/31364297
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    # Matplotlib >= 3.3: make the 3D box itself have equal aspect.
    # This improves "axis equal" visually compared to just setting limits.
    try:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _best_cluster_idx(
    store: dict, frame_idx: int, clusters: list[dict], inliers: np.ndarray
) -> int:
    if "reg_best_cluster_idx" in store:
        try:
            return int(np.asarray(store["reg_best_cluster_idx"])[frame_idx])
        except Exception:
            pass

    # Fallback: choose cluster with max overlap with logged inlier set
    if inliers is None or inliers.size == 0 or not clusters:
        return -1

    inl_idx = np.where(inliers.astype(bool))[0]
    if inl_idx.size == 0:
        return -1

    best = -1
    best_overlap = -1
    inl_set = set(inl_idx.tolist())
    for j, c in enumerate(clusters):
        ci = np.asarray(c.get("inliers", []), dtype=int)
        overlap = sum((int(x) in inl_set) for x in ci.tolist())
        if overlap > best_overlap:
            best_overlap = overlap
            best = j
    return best


def plot_cluster_summary_for_frame(
    clusters: list[dict],
    best_idx: int,
    *,
    title: Optional[str] = None,
    d_prev: Optional[np.ndarray] = None,
    d_gt: Optional[np.ndarray] = None,
    t_err_gt: Optional[np.ndarray] = None,
    r_err_gt: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
):
    if not clusters:
        return None

    ninliers = np.array(
        [int(c.get("ninliers", len(c.get("inliers", [])))) for c in clusters]
    )
    mean_res = np.array(
        [float(c.get("mean_res", np.nan)) for c in clusters], dtype=float
    )
    score = np.array([float(c.get("score", np.nan)) for c in clusters], dtype=float)

    x = np.arange(len(clusters))
    rows = (
        3
        + (1 if d_prev is not None else 0)
        + (1 if d_gt is not None else 0)
        + (1 if t_err_gt is not None else 0)
        + (1 if r_err_gt is not None else 0)
        + (1 if reproj_errors is not None else 0)
    )
    fig, axs = plt.subplots(rows, 1, figsize=(10, 2.3 * rows), sharex=True)
    if rows == 1:
        axs = [axs]

    axs[0].bar(x, ninliers, color="tab:blue", alpha=0.8)
    axs[0].set_ylabel("ninliers")

    axs[1].plot(x, mean_res, "o-", color="tab:orange", alpha=0.9)
    axs[1].set_ylabel("mean_res (cluster)")

    axs[2].plot(x, score, "o-", color="tab:green", alpha=0.9)
    axs[2].set_ylabel("score")
    axs[2].set_xlabel("cluster candidate index")

    row = 3
    best_gt_idx = -1
    best_prev_idx = -1
    best_tgt_idx = -1
    best_rgt_idx = -1
    best_reproj_idx = -1
    if d_prev is not None:
        d_prev = np.asarray(d_prev, dtype=float).reshape(-1)
        axs[row].plot(x, d_prev, "o-", color="tab:purple", alpha=0.9)
        axs[row].set_ylabel("d(prev)")
        if np.isfinite(d_prev).any():
            best_prev_idx = int(np.nanargmin(d_prev))
        row += 1

    if d_gt is not None:
        d_gt = np.asarray(d_gt, dtype=float).reshape(-1)
        axs[row].plot(x, d_gt, "o-", color="tab:red", alpha=0.9)
        axs[row].set_ylabel("d(GT)")
        axs[row].set_xlabel("cluster candidate index")
        if np.isfinite(d_gt).any():
            best_gt_idx = int(np.nanargmin(d_gt))
        row += 1

    if t_err_gt is not None:
        t_err_gt = np.asarray(t_err_gt, dtype=float).reshape(-1)
        axs[row].plot(x, t_err_gt, "o-", color="tab:cyan", alpha=0.9)
        axs[row].set_ylabel("t_err(GT)")
        if np.isfinite(t_err_gt).any():
            best_tgt_idx = int(np.nanargmin(t_err_gt))
        row += 1

    if r_err_gt is not None:
        r_err_gt = np.asarray(r_err_gt, dtype=float).reshape(-1)
        axs[row].plot(x, r_err_gt, "o-", color="tab:brown", alpha=0.9)
        axs[row].set_ylabel("r_err(GT) deg")
        axs[row].set_xlabel("cluster candidate index")
        if np.isfinite(r_err_gt).any():
            best_rgt_idx = int(np.nanargmin(r_err_gt))
        row += 1

    if reproj_errors is not None:
        reproj_errors = np.asarray(reproj_errors, dtype=float).reshape(-1)
        axs[row].plot(x, reproj_errors, "o-", color="tab:pink", alpha=0.9)
        axs[row].set_ylabel("reproj_error")
        axs[row].set_xlabel("cluster candidate index")
        if np.isfinite(reproj_errors).any():
            best_reproj_idx = int(np.nanargmin(reproj_errors))
        row += 1

    if 0 <= best_idx < len(clusters):
        for ax in axs:
            ax.axvline(best_idx, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        extra = []
        if best_prev_idx >= 0:
            extra.append(f"min d(prev)={best_prev_idx}")
        if best_gt_idx >= 0:
            extra.append(f"min d(GT)={best_gt_idx}")
        if best_tgt_idx >= 0:
            extra.append(f"min t_err(GT)={best_tgt_idx}")
        if best_rgt_idx >= 0:
            extra.append(f"min r_err(GT)={best_rgt_idx}")
        if best_reproj_idx >= 0:
            extra.append(f"min reproj_error={best_reproj_idx}")
        extra_str = f" | {'; '.join(extra)}" if extra else ""
        axs[0].set_title(
            (title or f"Cluster candidates (best_idx={best_idx})") + extra_str
        )

    fig.tight_layout()
    return fig


def plot_3d_clustered_correspondences(
    src: np.ndarray,
    tgt: np.ndarray,
    clusters: list[dict],
    best_idx: int,
    *,
    title: str,
    T_gt: Optional[np.ndarray] = None,
    key_points: Optional[np.ndarray] = None,
    max_lines: int = 200,
):
    if src.size == 0 or tgt.size == 0 or src.shape != tgt.shape:
        return None

    N = src.shape[0]

    # label assignment from candidate inlier sets
    labels = np.full((N,), -1, dtype=int)
    for j, c in enumerate(clusters):
        idx = np.asarray(c.get("inliers", []), dtype=int)
        idx = idx[(0 <= idx) & (idx < N)]
        labels[idx] = j

    num_clusters = int(max(labels.max() + 1, 0))
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((N, 4), dtype=float)
    for j in range(num_clusters):
        colors[labels == j] = cmap(j % 20)
    colors[labels < 0] = np.array([0.6, 0.6, 0.6, 0.35])

    # Use best cluster transform to visualize transformed source points
    if 0 <= best_idx < len(clusters) and isinstance(clusters[best_idx], dict):
        T_best = np.asarray(clusters[best_idx].get("T", np.eye(4)), dtype=float)
        inl_best = np.asarray(clusters[best_idx].get("inliers", []), dtype=int)
    else:
        T_best = np.eye(4)
        inl_best = np.array([], dtype=int)

    src_T = _transform_pts(T_best, src)
    src_GT = _transform_pts(T_gt, src) if T_gt is not None else None
    key_T = (
        _transform_pts(T_best, np.asarray(key_points, dtype=float))
        if key_points is not None and np.asarray(key_points).size
        else None
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Background: all keypoints (grey) transformed by best cluster
    if key_T is not None and key_T.size:
        ax.scatter(
            key_T[:, 0],
            key_T[:, 1],
            key_T[:, 2],
            c="0.55",
            s=10,
            alpha=0.25,
            marker=".",
            label="all keypoints (grey)",
        )

    # Background: all source correspondences (light grey) transformed by best cluster
    ax.scatter(
        src_T[:, 0],
        src_T[:, 1],
        src_T[:, 2],
        c="0.80",
        s=10,
        alpha=0.25,
        marker="^",
        label="source (all, light grey)",
    )

    ax.scatter(
        tgt[:, 0],
        tgt[:, 1],
        tgt[:, 2],
        c=colors,
        s=14,
        alpha=0.9,
        marker="o",
        label="target (curr3d)",
    )
    ax.scatter(
        src_T[:, 0],
        src_T[:, 1],
        src_T[:, 2],
        c=colors,
        s=14,
        alpha=0.9,
        marker="^",
        label="source (colored by cluster label)",
    )
    if src_GT is not None and src_GT.size:
        ax.scatter(
            src_GT[:, 0],
            src_GT[:, 1],
            src_GT[:, 2],
            c="k",
            s=10,
            alpha=0.35,
            marker="x",
            label="source (transformed by GT)",
        )

    # Draw correspondence lines (best cluster inliers only, sub-sampled)
    if inl_best.size:
        inl_best = inl_best[(0 <= inl_best) & (inl_best < N)]
        if inl_best.size > max_lines:
            inl_best = np.random.choice(inl_best, size=max_lines, replace=False)
        for i in inl_best.tolist():
            ax.plot(
                [src_T[i, 0], tgt[i, 0]],
                [src_T[i, 1], tgt[i, 1]],
                [src_T[i, 2], tgt[i, 2]],
                color=colors[i],
                alpha=0.55,
                linewidth=0.8,
            )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    _set_axes_equal(ax)
    fig.tight_layout()
    return fig


def plot_3d_single_cluster(
    src: np.ndarray,
    tgt: np.ndarray,
    cluster: dict,
    cluster_idx: int,
    *,
    title: str,
    T_gt: Optional[np.ndarray] = None,
    key_points: Optional[np.ndarray] = None,
    max_lines: int = 300,
):
    if src.size == 0 or tgt.size == 0 or src.shape != tgt.shape:
        return None

    idx = np.asarray(cluster.get("inliers", []), dtype=int)
    idx = idx[(0 <= idx) & (idx < src.shape[0])]
    if idx.size == 0:
        return None

    T = np.asarray(cluster.get("T", np.eye(4)), dtype=float)
    src_T = _transform_pts(T, src[idx])
    src_GT = _transform_pts(T_gt, src[idx]) if T_gt is not None else None
    tgt_i = tgt[idx]

    # Background context: all keypoints + all non-inlier source correspondences (transformed by this cluster)
    key_T = (
        _transform_pts(T, np.asarray(key_points, dtype=float))
        if key_points is not None and np.asarray(key_points).size
        else None
    )
    mask_inl = np.zeros((src.shape[0],), dtype=bool)
    mask_inl[idx] = True
    src_rest = src[~mask_inl]
    src_rest_T = _transform_pts(T, src_rest) if src_rest.size else np.empty((0, 3))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if key_T is not None and key_T.size:
        ax.scatter(
            key_T[:, 0],
            key_T[:, 1],
            key_T[:, 2],
            c="0.55",
            s=10,
            alpha=0.25,
            marker=".",
            label="all keypoints (grey)",
        )
    if src_rest_T.size:
        ax.scatter(
            src_rest_T[:, 0],
            src_rest_T[:, 1],
            src_rest_T[:, 2],
            c="0.75",
            s=12,
            alpha=0.25,
            marker="^",
            label="source (non-inliers, light grey)",
        )

    ax.scatter(
        tgt_i[:, 0],
        tgt_i[:, 1],
        tgt_i[:, 2],
        c="tab:blue",
        s=16,
        alpha=0.9,
        label="tgt",
    )
    ax.scatter(
        src_T[:, 0],
        src_T[:, 1],
        src_T[:, 2],
        c="tab:orange",
        s=16,
        alpha=0.9,
        label="src (cluster T)",
    )
    if src_GT is not None and src_GT.size:
        ax.scatter(
            src_GT[:, 0],
            src_GT[:, 1],
            src_GT[:, 2],
            c="tab:green",
            s=14,
            alpha=0.7,
            marker="x",
            label="src (GT T)",
        )

    if idx.size > max_lines:
        pick = np.random.choice(np.arange(idx.size), size=max_lines, replace=False)
        src_T = src_T[pick]
        tgt_i = tgt_i[pick]

    for a, b in zip(src_T, tgt_i):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color="k",
            alpha=0.35,
            linewidth=0.7,
        )

    ninl = int(cluster.get("ninliers", idx.size))
    mean_res = cluster.get("mean_res", np.nan)
    score = cluster.get("score", np.nan)
    ax.set_title(
        f"{title}\ncluster={cluster_idx} ninliers={ninl} mean_res={mean_res:.6f} score={score:.3f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    _set_axes_equal(ax)
    fig.tight_layout()
    return fig


def _load_frame_from_reader(reader: Any, frame_idx: int) -> Optional[Any]:
    """Load a Frame object from a dataset reader."""
    if reader is None or Frame is None:
        return None
    try:
        if isinstance(reader, Ho3dReader):
            rgb = cv2.imread(reader.color_files[frame_idx])
            if rgb is None:
                return None
            H, W = rgb.shape[:2]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = reader.get_depth(frame_idx)
            mask = reader.get_mask(frame_idx)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if torch is not None:
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            else:
                mask_tensor = np.expand_dims(np.expand_dims(mask, 0), 0)
            return Frame(
                id=frame_idx,
                rgb=rgb,
                depth=depth,
                mask=mask_tensor,
                intrinsics=reader.K,
                depth_factor=1.0,
            )
        elif isinstance(reader, YcbineoatReader):
            rgb = reader.get_color(frame_idx)
            depth = reader.get_depth(frame_idx)
            mask = reader.get_mask(frame_idx)
            H, W = rgb.shape[:2]
            if torch is not None:
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            else:
                mask_tensor = np.expand_dims(np.expand_dims(mask, 0), 0)
            return Frame(
                id=frame_idx,
                rgb=rgb,
                depth=depth,
                mask=mask_tensor,
                intrinsics=reader.K,
                depth_factor=1.0,
            )
    except Exception as e:
        print(f"Warning: Failed to load frame {frame_idx}: {e}")
        return None
    return None


def _compute_projection_consistency_iou(src_pcd, T_src2dst, frame_dst, obj_id=0):
    # Project points
    pts_dst_2d, pts_dst_3d = project_points_to_image(
        src_pcd, frame_dst.intrinsics, T_src2dst
    )

    # Get frame properties
    H, W = frame_dst.depth.shape
    if frame_dst.mask is not None:
        if torch is not None and isinstance(frame_dst.mask, torch.Tensor):
            dst_mask = frame_dst.mask[obj_id, 0].cpu().numpy() > 0
        else:
            dst_mask = np.asarray(frame_dst.mask[obj_id, 0]) > 0
    else:
        # If no mask, assume full image (or empty if strict?)
        # Let's assume full image is the target mask if none provided,
        # but usually we want IOU with object mask.
        dst_mask = np.ones((H, W), dtype=bool)

    # Convert to integer coordinates
    u = np.round(pts_dst_2d[:, 0]).astype(int)
    v = np.round(pts_dst_2d[:, 1]).astype(int)

    # Filter out-of-bounds
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]

    # Create projected mask
    proj_mask = np.zeros((H, W), dtype=bool)
    proj_mask[v_valid, u_valid] = True

    # Compute IoU
    intersection = np.logical_and(proj_mask, dst_mask).sum()
    union = np.logical_or(proj_mask, dst_mask).sum()

    return float(intersection / union) if union > 0 else 0.0


def _compute_per_point_reprojection_errors(
    src_pcd: np.ndarray,
    T_src2dst: np.ndarray,
    frame_dst: Any,
    obj_id: int = 0,
    min_depth: float = 0.01,
    max_depth: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-point reprojection errors (similar to compute_projection_consistency but returns per-point data).

    Args:
        src_pcd: (N, 3) source points in source frame
        T_src2dst: (4, 4) transform from source to destination frame
        frame_dst: Destination Frame object
        obj_id: Object ID
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)

    Returns:
        pts_2d: (N, 2) projected 2D coordinates
        depth_errors: (N,) per-point depth errors (NaN for invalid points)
        valid_mask: (N,) boolean mask indicating valid points
        pts_3d_dst: (N, 3) 3D points in destination frame
    """
    if project_points_to_image is None or frame_dst is None:
        N = src_pcd.shape[0] if src_pcd.size > 0 else 0
        return (
            np.empty((N, 2)),
            np.full(N, np.nan),
            np.zeros(N, dtype=bool),
            np.empty((N, 3)),
        )

    # Project points
    pts_dst_2d, pts_dst_3d = project_points_to_image(
        src_pcd, frame_dst.intrinsics, T_src2dst
    )

    # Get frame properties
    H, W = frame_dst.depth.shape
    if frame_dst.mask is not None:
        if torch is not None and isinstance(frame_dst.mask, torch.Tensor):
            dst_mask = frame_dst.mask[obj_id, 0].cpu().numpy()
        else:
            dst_mask = np.asarray(frame_dst.mask[obj_id, 0])
    else:
        dst_mask = np.ones((H, W), dtype=bool)
    depth_image = frame_dst.depth
    depth_factor = frame_dst.depth_factor if frame_dst.depth_factor is not None else 1.0

    N = len(pts_dst_2d)

    # Convert 2D coordinates to integer pixel indices
    u_coords = np.round(pts_dst_2d[:, 0]).astype(int)
    v_coords = np.round(pts_dst_2d[:, 1]).astype(int)

    # Check bounds
    in_bounds = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)

    # Get mask values for valid points
    mask_values = np.zeros(N, dtype=bool)
    valid_indices = np.where(in_bounds)[0]
    if len(valid_indices) > 0:
        mask_values[valid_indices] = (
            dst_mask[v_coords[valid_indices], u_coords[valid_indices]] > 0
        )

    # Get measured depths
    z_measured = np.full(N, np.nan, dtype=float)
    if len(valid_indices) > 0:
        z_measured[valid_indices] = (
            depth_image[v_coords[valid_indices], u_coords[valid_indices]] / depth_factor
        )

    # Get projected depths
    z_projected = pts_dst_3d[:, 2]

    # Create validity mask: inside mask, valid depth, in range
    valid_depth = (z_measured > 0) & np.isfinite(z_measured)
    in_depth_range = (
        (min_depth <= z_measured)
        & (z_measured <= max_depth)
        & (min_depth <= z_projected)
        & (z_projected <= max_depth)
    )

    valid = mask_values & valid_depth & in_depth_range

    # Compute depth errors (NaN for invalid points)
    depth_errors = np.full(N, np.nan, dtype=float)
    depth_errors[valid] = np.abs(z_projected[valid] - z_measured[valid])

    return pts_dst_2d, depth_errors, valid, pts_dst_3d


def plot_reprojection_2d(
    frame_dst: Any,
    pts_2d: np.ndarray,
    depth_errors: np.ndarray,
    valid_mask: np.ndarray,
    cluster_idx: int,
    *,
    title: str,
    max_points: int = 5000,
) -> Optional[plt.Figure]:
    """
    Visualize reprojected points on 2D image with error coloring.

    Args:
        frame_dst: Destination Frame object (for RGB image)
        pts_2d: (N, 2) projected 2D coordinates
        depth_errors: (N,) per-point depth errors
        valid_mask: (N,) boolean mask indicating valid points
        cluster_idx: Cluster index for title
        title: Plot title
        max_points: Maximum number of points to plot (for performance)

    Returns:
        matplotlib Figure or None
    """
    if frame_dst is None or frame_dst.rgb is None:
        return None

    rgb = frame_dst.rgb.copy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

    # Subsample points if too many
    N = len(pts_2d)
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        pts_2d = pts_2d[indices]
        depth_errors = depth_errors[indices]
        valid_mask = valid_mask[indices]

    # Filter to valid points only
    valid_pts = pts_2d[valid_mask]
    valid_errors = depth_errors[valid_mask]

    if len(valid_pts) == 0:
        return None

    # Create figure with two subplots: image with points, error histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Image with reprojected points colored by error
    ax1.imshow(rgb)
    if len(valid_pts) > 0:
        # Normalize errors for colormap
        if np.isfinite(valid_errors).any():
            error_min = np.nanmin(valid_errors)
            error_max = np.nanmax(valid_errors)
            if error_max > error_min:
                normalized_errors = (valid_errors - error_min) / (error_max - error_min)
            else:
                normalized_errors = np.zeros_like(valid_errors)
        else:
            normalized_errors = np.zeros_like(valid_errors)

        # Use colormap: green (low error) to red (high error)
        scatter = ax1.scatter(
            valid_pts[:, 0],
            valid_pts[:, 1],
            c=normalized_errors,
            cmap="RdYlGn_r",  # Red-Yellow-Green reversed (green=good, red=bad)
            s=8,
            alpha=0.6,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(scatter, ax=ax1, label="Normalized depth error")
    ax1.set_title(f"{title}\nReprojected points (green=low error, red=high error)")
    ax1.set_xlabel("u (pixels)")
    ax1.set_ylabel("v (pixels)")

    # Plot 2: Error histogram
    if np.isfinite(valid_errors).any():
        ax2.hist(valid_errors, bins=50, alpha=0.7, color="tab:blue", edgecolor="black")
        ax2.axvline(
            np.nanmean(valid_errors),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.nanmean(valid_errors):.4f}m",
        )
        ax2.axvline(
            np.nanmedian(valid_errors),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.nanmedian(valid_errors):.4f}m",
        )
        ax2.set_xlabel("Depth error (meters)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Depth error distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No valid errors",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Depth error distribution")

    fig.tight_layout()
    return fig


def plot_reprojection_error_summary(
    clusters: list[dict],
    reproj_errors_per_cluster: list[dict],
    best_idx: int,
    *,
    title: str,
) -> Optional[plt.Figure]:
    """
    Plot summary of reprojection errors across all clusters.

    Args:
        clusters: List of cluster dictionaries
        reproj_errors_per_cluster: List of dicts with keys: 'mean_error', 'median_error', 'std_error', 'n_valid'
        best_idx: Index of best cluster
        title: Plot title

    Returns:
        matplotlib Figure or None
    """
    if not clusters or not reproj_errors_per_cluster:
        return None

    n_clusters = len(clusters)
    x = np.arange(n_clusters)

    mean_errors = [e.get("mean_error", np.nan) for e in reproj_errors_per_cluster]
    median_errors = [e.get("median_error", np.nan) for e in reproj_errors_per_cluster]
    ious = [e.get("iou", 0.0) for e in reproj_errors_per_cluster]
    n_valid = [e.get("n_valid", 0) for e in reproj_errors_per_cluster]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Mean errors
    axs[0, 0].bar(x, mean_errors, color="tab:blue", alpha=0.8)
    if 0 <= best_idx < n_clusters:
        axs[0, 0].axvline(
            best_idx, color="red", linestyle="--", linewidth=2, label="best"
        )
    axs[0, 0].set_ylabel("Mean depth error (m)")
    axs[0, 0].set_title("Mean reprojection error per cluster")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Median errors
    axs[0, 1].bar(x, median_errors, color="tab:orange", alpha=0.8)
    if 0 <= best_idx < n_clusters:
        axs[0, 1].axvline(
            best_idx, color="red", linestyle="--", linewidth=2, label="best"
        )
    axs[0, 1].set_ylabel("Median depth error (m)")
    axs[0, 1].set_title("Median reprojection error per cluster")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # IOU
    axs[1, 0].bar(x, ious, color="tab:green", alpha=0.8)
    if 0 <= best_idx < n_clusters:
        axs[1, 0].axvline(
            best_idx, color="red", linestyle="--", linewidth=2, label="best"
        )
    axs[1, 0].set_ylabel("IOU")
    axs[1, 0].set_xlabel("Cluster index")
    axs[1, 0].set_title("Reprojection IOU per cluster")
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Number of valid points
    axs[1, 1].bar(x, n_valid, color="tab:purple", alpha=0.8)
    if 0 <= best_idx < n_clusters:
        axs[1, 1].axvline(
            best_idx, color="red", linestyle="--", linewidth=2, label="best"
        )
    axs[1, 1].set_ylabel("Number of valid points")
    axs[1, 1].set_xlabel("Cluster index")
    axs[1, 1].set_title("Valid points per cluster")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--meta_data_path", type=str, default=None, help="Path to meta_data.npz"
    )
    ap.add_argument(
        "--register_folder",
        type=str,
        default="",
        help="Optional register/debug folder (used to search for meta_data.npz)",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ho3d_single",
        help="Optional results dir",
    )
    ap.add_argument("--video_name", type=str, default=None, help="Optional video name")
    ap.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to visualize (default: -1 = skip)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional: save figures to this directory (if omitted, figures are not saved).",
    )
    ap.add_argument(
        "--no_show",
        action="store_true",
        help="Disable interactive pop-up windows (headless mode).",
    )
    ap.add_argument(
        "--no_plot_all_clusters",
        action="store_true",
        help="Disable per-cluster 3D visualization for the selected frame.",
    )
    ap.add_argument(
        "--max_lines",
        type=int,
        default=200,
        help="Max correspondence lines in 3D plots",
    )
    ap.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Optional dataset sequence dir for GT comparison (contains rgb/ and meta/).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "ho3d", "ycbineoat"],
        help="Dataset type for GT loading (default: auto).",
    )
    ap.add_argument(
        "--ho3d_root",
        type=str,
        default="/home/justin/data/HO3D_V3/",
        help=(
            "HO3D root directory. If --video_dir is not provided but --video_name is, "
            "the script will try to auto-set video_dir = ho3d_root/video_name "
            "(and also ho3d_root/evaluation/video_name if applicable)."
        ),
    )
    ap.add_argument(
        "--gt_mode",
        type=str,
        default="f2m",
        choices=["f2m", "f2f"],
        help="How to compute GT transform for comparison: f2m = ref(first)->current, f2f = prev->current.",
    )
    ap.add_argument(
        "--gt_ref_frame",
        type=int,
        default=0,
        help="Reference frame id/index for GT normalization (default: 0 / first frame).",
    )
    ap.add_argument(
        "--reproj_target",
        type=str,
        default="prev",
        choices=["prev", "keyframe"],
        help="Target frame for reprojection visualization: 'prev' (previous frame) or 'keyframe' (last keyframe).",
    )
    args = ap.parse_args()

    # Auto-configure video_dir from ho3d_root + video_name
    if args.video_dir is None and args.video_name:
        candidates: list[str] = []

        # For HO3D dataset
        if args.ho3d_root and args.dataset in ("auto", "ho3d"):
            root = os.path.normpath(args.ho3d_root)
            base = os.path.basename(root)

            # Try direct path first
            candidates.append(os.path.join(root, args.video_name))

            # If root is not evaluation/train, try subdirectories
            if base not in ("evaluation", "train"):
                candidates.append(os.path.join(root, "evaluation", args.video_name))
                candidates.append(os.path.join(root, "train", args.video_name))
            # If root is evaluation/train, try parent directory
            elif base in ("evaluation", "train"):
                parent = os.path.dirname(root)
                candidates.append(os.path.join(parent, args.video_name))
                candidates.append(os.path.join(parent, "evaluation", args.video_name))
                candidates.append(os.path.join(parent, "train", args.video_name))

        # For YCBInEOAT dataset
        if args.dataset in ("auto", "ycbineoat"):
            # Try common YCBInEOAT root locations
            ycbineoat_roots = []
            if args.ho3d_root:
                # Try parent directory of ho3d_root
                parent = os.path.dirname(os.path.normpath(args.ho3d_root))
                ycbineoat_roots.append(parent)
                ycbineoat_roots.append(os.path.join(parent, "YCBInEOAT"))

            # Common YCBInEOAT locations
            ycbineoat_roots.extend(
                [
                    "/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCBInEOAT",
                    os.path.expanduser("~/data/YCBInEOAT"),
                    os.path.expanduser("~/datasets/YCBInEOAT"),
                ]
            )

            for ycb_root in ycbineoat_roots:
                if os.path.isdir(ycb_root):
                    candidates.append(os.path.join(ycb_root, args.video_name))
                    break

        # Try to find the video directory
        for c in candidates:
            if os.path.isdir(c):
                # Verify it looks like a valid video directory (has rgb/ subdirectory)
                rgb_dir = os.path.join(c, "rgb")
                if os.path.isdir(rgb_dir):
                    args.video_dir = c
                    print(f"Auto-detected video_dir: {args.video_dir}")
                    break

        if args.video_dir is None and args.video_name:
            print(
                f"Warning: Could not auto-detect video_dir for video_name={args.video_name}. "
                f"Please provide --video_dir explicitly."
            )

    do_show = not args.no_show
    plot_all_clusters = not args.no_plot_all_clusters

    meta_path = args.meta_data_path
    if meta_path is None:
        meta_path = find_meta_data_path(
            register_folder=args.register_folder,
            meta_data_path_override=None,
            results_dir=args.results_dir,
            video_name=args.video_name,
        )
    if meta_path is None or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Could not find meta_data.npz. Provide --meta_data_path, or pass --register_folder/--results_dir + --video_name."
        )

    store = dict(np.load(meta_path, allow_pickle=True))

    frame_ids = store.get("frame_id", None)
    if isinstance(frame_ids, np.ndarray):
        n_frames = int(frame_ids.shape[0])
    else:
        # fall back on ragged lengths
        n_frames = len(unpack_ragged("reg_curr3d", store, dim=3))
        frame_ids = np.arange(n_frames, dtype=int)

    reg_curr3d = unpack_ragged("reg_curr3d", store, dim=3)
    reg_key_points = unpack_ragged("reg_key_points", store, dim=3)
    reg_prev3d = unpack_ragged("reg_prev3d", store, dim=3)
    reg_inliers = unpack_ragged("reg_inliers", store, dim=-1)
    obj_key_points = unpack_ragged("obj_key_points", store, dim=3)

    # Keyframes
    is_key_frame = None
    if "is_key_frame" in store:
        is_key_frame = store["is_key_frame"]
    elif "is_key_frame_data" in store:
        is_key_frame = store["is_key_frame_data"]

    if "reg_clusters" not in store:
        raise KeyError(
            "meta_data.npz does not contain `reg_clusters`. Re-run with the updated pipeline logger."
        )

    save_dir = args.output_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figs: list[plt.Figure] = []

    # ---------------- dataset-level summary ----------------
    num_clusters = np.zeros((n_frames,), dtype=int)
    coverage = np.zeros((n_frames,), dtype=float)
    best_ninliers = np.full((n_frames,), -1, dtype=int)

    for i in range(n_frames):
        clusters = _clusters_for_frame(store, i)
        num_clusters[i] = len(clusters)

        tgt = reg_curr3d[i] if i < len(reg_curr3d) else np.empty((0, 3))
        N = int(tgt.shape[0])

        if N == 0 or not clusters:
            coverage[i] = 0.0
            best_ninliers[i] = -1
            continue

        # coverage = fraction assigned to any cluster
        labels = np.full((N,), -1, dtype=int)
        for j, c in enumerate(clusters):
            idx = np.asarray(c.get("inliers", []), dtype=int)
            idx = idx[(0 <= idx) & (idx < N)]
            labels[idx] = j
        coverage[i] = float(np.mean(labels >= 0)) if N > 0 else 0.0

        # best cluster size (using logged best idx if available)
        inl = reg_inliers[i] if i < len(reg_inliers) else np.empty((0,))
        best_idx = _best_cluster_idx(store, i, clusters, inl)
        if 0 <= best_idx < len(clusters):
            best_ninliers[i] = int(
                clusters[best_idx].get("ninliers", np.sum(labels == best_idx))
            )

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(num_clusters, color="tab:blue")
    axs[0].set_ylabel("#clusters")
    axs[1].plot(coverage, color="tab:orange")
    axs[1].set_ylabel("coverage")
    axs[2].plot(best_ninliers, color="tab:green")
    axs[2].set_ylabel("best ninliers")
    axs[2].set_xlabel("frame index")
    fig.suptitle("Clustered registration summary")
    fig.tight_layout()
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "cluster_summary_over_time.png"), dpi=200)

    # ---------------- frame-level visualization ----------------
    if args.frame >= 0:
        i = int(args.frame)
        if not (0 <= i < n_frames):
            raise ValueError(f"--frame {i} out of range [0, {n_frames})")

        frame_id = int(np.asarray(frame_ids)[i])
        clusters = _clusters_for_frame(store, i)
        tgt = reg_curr3d[i] if i < len(reg_curr3d) else np.empty((0, 3))

        # src depends on reg mode (f2m uses reg_key_points, f2f uses reg_prev3d)
        src = reg_key_points[i] if i < len(reg_key_points) else np.empty((0, 3))
        if src.size == 0 and i < len(reg_prev3d):
            src = reg_prev3d[i]

        key_all = obj_key_points[i] if i < len(obj_key_points) else None

        inl = reg_inliers[i] if i < len(reg_inliers) else np.empty((0,))
        best_idx = _best_cluster_idx(store, i, clusters, inl)

        T_gt = None
        gt_reader = None
        gt_ref_pose = None
        gt_ref_frame = None
        if args.video_dir:
            gt_reader = _build_reader(args.dataset, args.video_dir, args.ho3d_root)
            if gt_reader is None:
                print(
                    "Warning: Could not construct GT reader. "
                    "Check --video_dir/--dataset and dependencies."
                )
            else:
                gt_ref_pose, gt_ref_frame = _get_reference_gt_pose(
                    gt_reader,
                    np.asarray(frame_ids),
                    preferred_ref=int(args.gt_ref_frame),
                )
                if gt_ref_pose is None:
                    print(
                        "Warning: No valid GT pose found for any frame; skipping GT plots."
                    )
                else:
                    gt_pose_cur = _get_gt_pose(gt_reader, frame_id)
                    if gt_pose_cur is None:
                        print(
                            f"Warning: GT pose not found for frame_id={frame_id}; skipping GT plots."
                        )
                    else:
                        if args.gt_mode == "f2m":
                            # GT transform from reference(first) camera to current camera:
                            # T_c(t)_c(ref) = T_c(t)_obj @ inv(T_c(ref)_obj)
                            T_gt = gt_pose_cur @ _inverse_SE3(gt_ref_pose)
                        else:
                            # GT transform from previous camera to current camera
                            if i == 0:
                                T_gt = None
                            else:
                                prev_frame_id = int(np.asarray(frame_ids)[i - 1])
                                gt_pose_prev = _get_gt_pose(gt_reader, prev_frame_id)
                                if gt_pose_prev is not None:
                                    T_gt = gt_pose_cur @ _inverse_SE3(gt_pose_prev)
                                else:
                                    T_gt = None

        # pose distance to previous estimated pose (matches register selection metric)
        d_prev = None
        prev_T = None
        if i > 0 and "obj_pose" in store:
            try:
                prev_T = np.asarray(store["obj_pose"][i - 1], dtype=float)
                if prev_T.shape != (4, 4):
                    prev_T = None
            except Exception:
                prev_T = None
        if prev_T is not None and clusters:
            d_prev = np.array(
                [
                    _pose_dist(np.asarray(c.get("T", np.eye(4))), prev_T)
                    for c in clusters
                ],
                dtype=float,
            )

        d_gt = None
        t_err_gt = None
        r_err_gt = None
        if T_gt is not None and clusters:
            # Match notebook convention for GT:
            # transform_kf_to_ref = reference_gt_pose @ inverse_SE3(gt_pose_frame)
            # Our T_gt above is ref->frame, so invert it to get frame->ref.
            T_gt_frame_to_ref = _inverse_SE3(T_gt)

            d_gt_list = []
            t_list = []
            r_list = []
            for c in clusters:
                T_cand = np.asarray(c.get("T", np.eye(4)), dtype=float)
                T_cand_frame_to_ref = _inverse_SE3(T_cand)
                d_gt_list.append(_pose_dist(T_cand_frame_to_ref, T_gt_frame_to_ref))
                te, re = _pose_error(T_cand_frame_to_ref, T_gt_frame_to_ref)
                t_list.append(te)
                r_list.append(re)

            d_gt = np.asarray(d_gt_list, dtype=float)
            t_err_gt = np.asarray(t_list, dtype=float)
            r_err_gt = np.asarray(r_list, dtype=float)

        # Extract reprojection errors from clusters
        reproj_errors = None
        if clusters:
            reproj_errors_list = [
                float(c.get("reproj_error", np.nan)) for c in clusters
            ]
            if any(np.isfinite(reproj_errors_list)):
                reproj_errors = np.asarray(reproj_errors_list, dtype=float)

        fig1 = plot_cluster_summary_for_frame(
            clusters,
            best_idx,
            title=f"Cluster candidates - log_idx={i} frame_id={frame_id} (best_idx={best_idx})",
            d_prev=d_prev,
            d_gt=d_gt,
            t_err_gt=t_err_gt,
            r_err_gt=r_err_gt,
            reproj_errors=reproj_errors,
        )
        if fig1 is not None:
            figs.append(fig1)
            if save_dir is not None:
                fig1.savefig(
                    os.path.join(save_dir, f"frame_{i:06d}_cluster_candidates.png"),
                    dpi=200,
                )

        fig2 = plot_3d_clustered_correspondences(
            src,
            tgt,
            clusters,
            best_idx,
            title=(
                f"Clustered correspondences - log_idx={i} frame_id={frame_id} "
                f"(best_idx={best_idx}, #clusters={len(clusters)})"
            ),
            T_gt=T_gt,
            key_points=key_all,
            max_lines=args.max_lines,
        )
        if fig2 is not None:
            figs.append(fig2)
            if save_dir is not None:
                fig2.savefig(
                    os.path.join(save_dir, f"frame_{i:06d}_clusters_3d.png"), dpi=200
                )

        if T_gt is not None and clusters:
            fig_gt = plot_clusters_vs_gt_pose(
                clusters,
                best_idx,
                T_gt,
                title=(
                    f"Per-cluster vs GT - log_idx={i} frame_id={frame_id} "
                    f"(GT ref frame_id={gt_ref_frame}, mode={args.gt_mode})"
                ),
            )
            if fig_gt is not None:
                figs.append(fig_gt)
                if save_dir is not None:
                    fig_gt.savefig(
                        os.path.join(save_dir, f"frame_{i:06d}_clusters_vs_gt.png"),
                        dpi=200,
                    )

        if (
            plot_all_clusters
            and clusters
            and src.size
            and tgt.size
            and src.shape == tgt.shape
        ):
            for j, c in enumerate(clusters):
                figc = plot_3d_single_cluster(
                    src,
                    tgt,
                    c,
                    j,
                    title=f"Cluster-only correspondences - log_idx={i} frame_id={frame_id}",
                    T_gt=T_gt,
                    key_points=key_all,
                    max_lines=max(300, args.max_lines),
                )
                if figc is not None:
                    figs.append(figc)
                    if save_dir is not None:
                        figc.savefig(
                            os.path.join(
                                save_dir, f"frame_{i:06d}_cluster_{j:02d}_3d.png"
                            ),
                            dpi=200,
                        )

        # ---------------- Reprojection visualization ----------------
        # Determine reprojection target frame
        target_frame_idx = -1
        target_T = None
        target_desc = "prev"

        if args.reproj_target == "prev":
            if i > 0 and prev_T is not None:
                target_frame_idx = i - 1
                target_T = prev_T
                target_desc = "prev"
        elif args.reproj_target == "keyframe":
            # Find last keyframe index < i
            if is_key_frame is not None and len(is_key_frame) > 0:
                # Convert to indices
                kf_indices = np.where(is_key_frame)[0]
                valid_kf = kf_indices[kf_indices < i]
                if len(valid_kf) > 0:
                    kf_idx = valid_kf[-1]
                    target_frame_idx = int(kf_idx)
                    # Get pose for keyframe
                    if "obj_pose" in store:
                        try:
                            target_T = np.asarray(
                                store["obj_pose"][target_frame_idx], dtype=float
                            )
                        except Exception:
                            target_T = None
                    target_desc = f"keyframe_{kf_idx}"

        if (
            args.video_dir
            and gt_reader is not None
            and extract_cropped_point_cloud is not None
            and inverse_SE3 is not None
            and target_T is not None
            and target_frame_idx >= 0
            and clusters
        ):
            # Load current and target frames
            cur_frame = _load_frame_from_reader(gt_reader, frame_id)

            target_frame = None
            if 0 <= target_frame_idx < len(frame_ids):
                target_frame_id_val = int(np.asarray(frame_ids)[target_frame_idx])
                target_frame = _load_frame_from_reader(gt_reader, target_frame_id_val)

            if cur_frame is not None and target_frame is not None:
                # Extract full point cloud from current frame
                src_pcd_full = extract_cropped_point_cloud(cur_frame, obj_id=0)
                if src_pcd_full.size > 0:
                    reproj_errors_per_cluster = []
                    all_reproj_figs = []

                    # Compute reprojection for each cluster
                    for j, c in enumerate(clusters):
                        cluster_T = np.asarray(c.get("T", np.eye(4)), dtype=float)
                        # T_cur2target = target_T @ inverse_SE3(cluster_T)
                        T_cur2target = target_T @ inverse_SE3(cluster_T)

                        # Compute per-point reprojection errors
                        (
                            pts_2d,
                            depth_errors,
                            valid_mask,
                            pts_3d_dst,
                        ) = _compute_per_point_reprojection_errors(
                            src_pcd_full.copy(),
                            T_cur2target,
                            target_frame,
                            obj_id=0,
                        )

                        # Compute IOU
                        iou = _compute_projection_consistency_iou(
                            src_pcd_full.copy(),
                            T_cur2target,
                            target_frame,
                            obj_id=0,
                        )

                        # Compute statistics
                        valid_errors = depth_errors[valid_mask]
                        if np.isfinite(valid_errors).any() and len(valid_errors) > 0:
                            mean_error = float(np.nanmean(valid_errors))
                            median_error = float(np.nanmedian(valid_errors))
                            std_error = float(np.nanstd(valid_errors))
                            n_valid = int(np.sum(valid_mask))
                        else:
                            mean_error = np.nan
                            median_error = np.nan
                            std_error = np.nan
                            n_valid = 0

                        reproj_errors_per_cluster.append(
                            {
                                "mean_error": mean_error,
                                "median_error": median_error,
                                "std_error": std_error,
                                "n_valid": n_valid,
                                "iou": iou,
                            }
                        )

                        # Visualize reprojection for this cluster
                        fig_reproj = plot_reprojection_2d(
                            target_frame,
                            pts_2d,
                            depth_errors,
                            valid_mask,
                            j,
                            title=(
                                f"Reprojection to {target_desc} - log_idx={i} frame_id={frame_id} "
                                f"cluster={j} (mean_err={mean_error:.4f}m, n_valid={n_valid}, iou={iou:.3f})"
                            ),
                        )
                        if fig_reproj is not None:
                            all_reproj_figs.append((j, fig_reproj))
                            # Show immediately (pop out) instead of adding to figs list
                            if do_show:
                                plt.show(block=False)
                                plt.pause(0.1)  # Small pause to ensure window appears
                            if save_dir is not None:
                                fig_reproj.savefig(
                                    os.path.join(
                                        save_dir,
                                        f"frame_{i:06d}_cluster_{j:02d}_reprojection_{target_desc}.png",
                                    ),
                                    dpi=200,
                                )
                            # Don't add to figs list - we show it immediately above
                            # If user wants to keep it open, they can, otherwise it will close
                            if not do_show:
                                plt.close(fig_reproj)

                    # Summary plot across all clusters
                    if reproj_errors_per_cluster:
                        fig_summary = plot_reprojection_error_summary(
                            clusters,
                            reproj_errors_per_cluster,
                            best_idx,
                            title=(
                                f"Reprojection error summary ({target_desc}) - log_idx={i} "
                                f"frame_id={frame_id}"
                            ),
                        )
                        if fig_summary is not None:
                            # Show immediately (pop out) instead of adding to figs list
                            if do_show:
                                plt.show(block=False)
                                plt.pause(0.1)  # Small pause to ensure window appears
                            if save_dir is not None:
                                fig_summary.savefig(
                                    os.path.join(
                                        save_dir,
                                        f"frame_{i:06d}_reprojection_summary_{target_desc}.png",
                                    ),
                                    dpi=200,
                                )
                            # Don't add to figs list - we show it immediately above
                            if not do_show:
                                plt.close(fig_summary)

    if do_show and figs:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


if __name__ == "__main__":
    main()
