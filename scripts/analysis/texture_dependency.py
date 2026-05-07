"""Texture-dependency rebuttal analysis for HO3D pose tracking.

Reads per-video meta_data.npz produced by the inference pipeline (under
`<results-root>/<video>/meta_data/meta_data.npz`) and the corresponding HO3D
dataset (rgb, masks, GT poses, intrinsics, mesh) from `<data-root>`. Computes:

  - GT 2D track per point (project canonical 3D anchor through GT pose at frame t)
  - e2d_{t,i} = ||predicted_2d_{t,i} - gt_2d_{t,i}||
  - per-frame pose error (ADD, ADD-S, translation, rotation)
  - per-point texture_std on a patch around the predicted 2D at sample frame
  - per-frame tracker-independent texturelessness tx_grad (mean Sobel magnitude
    inside object mask) and tx_std (std of greyscale inside mask)

Emits:
  per_frame_<video>.csv, per_point_<video>.csv, summary.csv, calibration
  histograms, and figures P1-P7. Output dir defaults to
  /home/justin/results/eccv_point2pose/texture_dependency_analysis.

The script never writes inside `<results-root>`.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib

if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from point2pose.utils.evaluation import add_err, adi_err, compute_auc  # noqa: E402
from point2pose.utils.transform import inverse_SE3  # noqa: E402

DEFAULT_RESULTS_ROOT = "/home/justin/results/eccv_point2pose/final_results/ho3d_all_final"
DEFAULT_DATA_ROOT = "/home/justin/data/HO3D_V3"
DEFAULT_OUT_DIR = "/home/justin/results/eccv_point2pose/texture_dependency_analysis"

GLCAM_IN_CVCAM = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


@dataclass
class HO3DPaths:
    rgb_dir: Path
    mask_dir: Path
    meta_dir: Path
    model_path: Path


def video_to_object(video_name: str) -> str:
    prefix_map = {
        "AP": "019_pitcher_base",
        "MPM": "010_potted_meat_can",
        "SB": "021_bleach_cleanser",
        "SM": "006_mustard_bottle",
    }
    for k, v in prefix_map.items():
        if video_name.startswith(k):
            return v
    raise ValueError(f"Unknown HO3D video prefix: {video_name}")


def find_ho3d_paths(data_root: Path, video_name: str) -> Optional[HO3DPaths]:
    """Locate dataset folders for a given HO3D video. Returns None if missing."""
    for split in ("evaluation", "train"):
        rgb_dir = data_root / split / video_name / "rgb"
        if rgb_dir.exists():
            mask_dir = data_root / "masks" / video_name
            meta_dir = data_root / split / video_name / "meta"
            obj_name = video_to_object(video_name)
            return HO3DPaths(
                rgb_dir=rgb_dir,
                mask_dir=mask_dir,
                meta_dir=meta_dir,
                model_path=data_root / "models" / obj_name / "textured_simple.obj",
            )
    return None


def load_intrinsics_and_mesh(paths: HO3DPaths):
    import pickle
    import trimesh

    rgb_files = sorted(paths.rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise RuntimeError(f"No RGB files in {paths.rgb_dir}")
    meta_file = str(rgb_files[0]).replace(".jpg", ".pkl").replace("rgb", "meta")
    with open(meta_file, "rb") as f:
        K = pickle.load(f)["camMat"].astype(np.float64)
    mesh = trimesh.load(str(paths.model_path), force="mesh")
    return K, np.asarray(mesh.vertices, dtype=np.float64), rgb_files


def get_gt_pose(meta_dir: Path, frame_id_str: str) -> Optional[np.ndarray]:
    import pickle

    p = meta_dir / f"{frame_id_str}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        meta = pickle.load(f)
    if meta.get("objTrans") is None:
        return None
    T = np.eye(4)
    T[:3, 3] = meta["objTrans"]
    T[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
    return GLCAM_IN_CVCAM @ T


def load_meta(npz_path: Path):
    return np.load(str(npz_path), allow_pickle=True)


def ragged(npz, name: str, t: int) -> np.ndarray:
    """Read row t of a logged array.

    Handles two storage schemes used by point2pose's DataLogger:
      1. Ragged: <name>_data, <name>_offsets, <name>_lengths.
      2. Object array: npz[<name>] is shape (T,) dtype=object, each entry is a 1D array.
    """
    offsets_key = f"{name}_offsets"
    if offsets_key in npz.files:
        offs = int(npz[offsets_key][t])
        ln = int(npz[f"{name}_lengths"][t])
        return npz[f"{name}_data"][offs : offs + ln]
    if name in npz.files:
        arr = npz[name][t]
        if arr is None:
            return np.empty(0)
        return np.asarray(arr)
    raise KeyError(f"{name} not in npz archive")


def project_points(X: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Project (N,3) world/object points through 4x4 T and (3,3) K -> (N,2)."""
    if X.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    Xh = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    Xc = (T @ Xh.T).T[:, :3]
    proj = (K @ Xc.T).T
    z = proj[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    return proj[:, :2] / z_safe


def recover_sample_frames(npz, n_frames: int) -> np.ndarray:
    """sample_frame[g] = first frame at which global index g appears.

    Relies on the pipeline's monotonic append-only growth of the track table.
    """
    sizes = np.array(
        [int(npz["track2d_lengths"][t]) // 2 for t in range(n_frames)],
        dtype=np.int64,
    )
    max_n = int(sizes.max()) if sizes.size else 0
    sample_frame = np.full(max_n, -1, dtype=np.int64)
    prev = 0
    for t in range(n_frames):
        cur = sizes[t]
        if cur > prev:
            sample_frame[prev:cur] = t
            prev = cur
    return sample_frame


def compute_canonical_anchors(
    npz, sample_frame: np.ndarray, gt_poses: list[Optional[np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """X_can[g] = inv(T_gt(s_g)) @ track3d[s_g, g]. Returns (X_can, valid_anchor)."""
    n_pts = sample_frame.size
    X_can = np.full((n_pts, 3), np.nan, dtype=np.float64)
    valid_anchor = np.zeros(n_pts, dtype=bool)

    by_frame: dict[int, list[int]] = {}
    for g, s in enumerate(sample_frame):
        if s < 0:
            continue
        by_frame.setdefault(int(s), []).append(g)

    for s, indices in by_frame.items():
        T_gt_s = gt_poses[s]
        if T_gt_s is None:
            continue
        T_inv = inverse_SE3(T_gt_s)
        track3d_s = ragged(npz, "track3d", s).reshape(-1, 3)
        # Sample-frame snapshot has track3d[s, g] for g in [0, sizes[s]).
        # We only anchor points whose g < len(track3d_s).
        idx = np.array(indices, dtype=np.int64)
        idx = idx[idx < track3d_s.shape[0]]
        if idx.size == 0:
            continue
        Xc = track3d_s[idx]
        # Filter NaN/inf rows.
        good = np.isfinite(Xc).all(axis=1)
        idx = idx[good]
        Xc = Xc[good]
        if idx.size == 0:
            continue
        Xh = np.concatenate([Xc, np.ones((Xc.shape[0], 1))], axis=1)
        X_can[idx] = (T_inv @ Xh.T).T[:, :3]
        valid_anchor[idx] = True
    return X_can, valid_anchor


def compute_texture_std_at_sample(
    rgb: np.ndarray, xy: np.ndarray, half: int = 5
) -> np.ndarray:
    """Std-dev of greyscale patch (2*half+1)x(2*half+1) around each xy. NaN out-of-frame."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = grey.shape
    out = np.full(xy.shape[0], np.nan, dtype=np.float32)
    xs = xy[:, 0].round().astype(np.int64)
    ys = xy[:, 1].round().astype(np.int64)
    for i in range(xy.shape[0]):
        x, y = xs[i], ys[i]
        if x - half < 0 or y - half < 0 or x + half >= W or y + half >= H:
            continue
        patch = grey[y - half : y + half + 1, x - half : x + half + 1]
        out[i] = float(patch.std())
    return out


def compute_tx_grad_in_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """tx_grad = mean(|Sobel_x|+|Sobel_y|) inside mask; tx_std = std(grey) inside mask."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(sx) + np.abs(sy)
    m = mask > 0
    if m.sum() < 16:
        return float("nan"), float("nan")
    return float(grad[m].mean()), float(grey[m].std())


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(target_hw, dtype=np.uint8)
    m = cv2.imread(str(path), -1)
    if m is None:
        return np.zeros(target_hw, dtype=np.uint8)
    if m.ndim == 3:
        m = (m.sum(axis=-1) > 0).astype(np.uint8)
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def process_video(
    video_name: str,
    results_root: Path,
    data_root: Path,
    out_dir: Path,
    unc_thres: float,
    do_per_point_texture: bool = True,
):
    npz_path = results_root / video_name / "meta_data" / "meta_data.npz"
    if not npz_path.exists():
        print(f"[skip] {video_name}: no meta_data.npz")
        return None

    paths = find_ho3d_paths(data_root, video_name)
    if paths is None:
        print(f"[skip] {video_name}: no HO3D dataset folder")
        return None

    print(f"[run] {video_name}")
    npz = load_meta(npz_path)
    K, mesh_vertices, rgb_files = load_intrinsics_and_mesh(paths)
    n_frames_meta = len(npz["frame_id"])
    n_frames = min(n_frames_meta, len(rgb_files))
    if n_frames_meta != len(rgb_files):
        warnings.warn(
            f"{video_name}: frame count mismatch (meta={n_frames_meta}, rgb={len(rgb_files)}); using {n_frames}"
        )

    # Frame ids as zero-padded strings (HO3D convention).
    rgb_id_strs = [Path(p).stem for p in rgb_files[:n_frames]]
    H, W = read_rgb(rgb_files[0]).shape[:2]

    # Load all GT poses (may have None entries).
    gt_poses: list[Optional[np.ndarray]] = []
    for s in rgb_id_strs:
        gt_poses.append(get_gt_pose(paths.meta_dir, s))

    # Recover sample frame and canonical 3D anchor per point.
    sample_frame = recover_sample_frames(npz, n_frames)
    X_can, valid_anchor = compute_canonical_anchors(npz, sample_frame, gt_poses)
    n_pts = sample_frame.size

    # Aligned predicted poses: pred_aligned[t] = pred[t] @ inv(pred[0]) @ gt[0]
    # (matches run_ho3d_single.py normalization). pred[0] is identity per pipeline init.
    obj_pose = np.asarray(npz["obj_pose"])[:n_frames]
    # Find the first GT-valid frame for alignment (usually 0).
    first_gt_t = next((t for t in range(n_frames) if gt_poses[t] is not None), None)
    if first_gt_t is None:
        print(f"[skip] {video_name}: no GT poses found")
        return None
    align = inverse_SE3(obj_pose[first_gt_t]) @ gt_poses[first_gt_t]
    pred_poses_aligned = np.einsum("nij,jk->nik", obj_pose, align)

    # Per-point texture (one read of the sample-frame RGB per sample frame).
    texture_std = np.full(n_pts, np.nan, dtype=np.float32)
    if do_per_point_texture:
        by_sample: dict[int, list[int]] = {}
        for g in range(n_pts):
            s = int(sample_frame[g])
            if s < 0:
                continue
            by_sample.setdefault(s, []).append(g)
        for s, gs in by_sample.items():
            track2d_s = ragged(npz, "track2d", s).reshape(-1, 2)
            gs = np.array(gs, dtype=np.int64)
            gs = gs[gs < track2d_s.shape[0]]
            if gs.size == 0:
                continue
            xy = track2d_s[gs]
            rgb_s = read_rgb(rgb_files[s])
            tex = compute_texture_std_at_sample(rgb_s, xy, half=5)
            texture_std[gs] = tex

    # Per-frame pass: compute everything that depends on per-frame state.
    rows = []
    e2d_per_point_accum = np.full(n_pts, np.nan, dtype=np.float64)
    e2d_count = np.zeros(n_pts, dtype=np.int64)
    e2d_sum = np.zeros(n_pts, dtype=np.float64)

    valid_anchor_filt = valid_anchor.copy()

    # Cache aligned mesh transform for faster ADD: use vertices once.
    for t in range(n_frames):
        track2d_t = ragged(npz, "track2d", t).reshape(-1, 2)
        unc_t = ragged(npz, "uncertainties", t)
        vis_t = ragged(npz, "visibles", t).astype(bool)
        valid_t = ragged(npz, "valid", t).astype(bool)
        n_t = track2d_t.shape[0]

        # Read mask + RGB once per frame for tx_grad.
        # HO3D mask filenames use 5-digit zero-padding (e.g. 00000.png) while
        # rgb filenames use 4-digit (0000.jpg). Convert to int and pad to 5.
        mask_idx = int(rgb_id_strs[t])
        mask_path = paths.mask_dir / f"{mask_idx:05d}.png"
        mask = read_mask(mask_path, (H, W))
        rgb_t = read_rgb(rgb_files[t])
        tx_grad, tx_std = compute_tx_grad_in_mask(rgb_t, mask)
        mask_area = float(mask.sum())

        T_gt_t = gt_poses[t]
        T_pred_t = pred_poses_aligned[t]

        if T_gt_t is None:
            add_t = np.nan
            adi_t = np.nan
            trans_err = np.nan
            rot_err = np.nan
            mean_e2d = np.nan
            median_e2d = np.nan
            n_eval = 0
        else:
            add_t = float(add_err(T_pred_t, T_gt_t, mesh_vertices))
            adi_t = float(adi_err(T_pred_t, T_gt_t, mesh_vertices))
            trans_err = float(np.linalg.norm(T_pred_t[:3, 3] - T_gt_t[:3, 3]))
            R_rel = T_pred_t[:3, :3] @ T_gt_t[:3, :3].T
            cos = (np.trace(R_rel) - 1.0) / 2.0
            cos = float(np.clip(cos, -1.0, 1.0))
            rot_err = float(np.degrees(np.arccos(cos)))

            # Project canonical anchors with GT pose.
            keep = valid_anchor_filt[:n_t]
            if keep.any():
                gs_eval = np.where(keep)[0]
                gt2d_eval = project_points(X_can[gs_eval], K, T_gt_t)
                pred2d_eval = track2d_t[gs_eval]
                # Mask: visible AND valid AND inside image.
                in_img = (
                    (gt2d_eval[:, 0] >= 0)
                    & (gt2d_eval[:, 0] < W)
                    & (gt2d_eval[:, 1] >= 0)
                    & (gt2d_eval[:, 1] < H)
                )
                m_eval = (
                    vis_t[gs_eval]
                    & valid_t[gs_eval]
                    & in_img
                    & np.isfinite(gt2d_eval).all(axis=1)
                    & np.isfinite(pred2d_eval).all(axis=1)
                )
                if m_eval.any():
                    diffs = pred2d_eval[m_eval] - gt2d_eval[m_eval]
                    e2d = np.linalg.norm(diffs, axis=1)
                    mean_e2d = float(e2d.mean())
                    median_e2d = float(np.median(e2d))
                    n_eval = int(m_eval.sum())
                    g_used = gs_eval[m_eval]
                    e2d_sum[g_used] += e2d
                    e2d_count[g_used] += 1
                else:
                    mean_e2d = np.nan
                    median_e2d = np.nan
                    n_eval = 0
            else:
                mean_e2d = np.nan
                median_e2d = np.nan
                n_eval = 0

        # Uncertainty-based counts (regardless of GT availability).
        if n_t > 0:
            n_unc_010 = int(((unc_t < 0.1) & vis_t).sum())
            n_unc_020 = int(((unc_t < 0.2) & vis_t).sum())
            n_unc_030 = int(((unc_t < 0.3) & vis_t).sum())
            n_unc_050 = int(((unc_t < 0.5) & vis_t).sum())
            n_visible = int(vis_t.sum())
        else:
            n_unc_010 = n_unc_020 = n_unc_030 = n_unc_050 = n_visible = 0

        rows.append(
            dict(
                video=video_name,
                frame=t,
                frame_id=rgb_id_strs[t],
                has_gt=T_gt_t is not None,
                n_pts_total=n_t,
                n_visible=n_visible,
                n_unc_010=n_unc_010,
                n_unc_020=n_unc_020,
                n_unc_030=n_unc_030,
                n_unc_050=n_unc_050,
                add_cm=add_t * 100.0 if not np.isnan(add_t) else np.nan,
                adi_cm=adi_t * 100.0 if not np.isnan(adi_t) else np.nan,
                trans_err_m=trans_err,
                rot_err_deg=rot_err,
                mean_e2d_px=mean_e2d,
                median_e2d_px=median_e2d,
                n_eval_points=n_eval,
                tx_grad=tx_grad,
                tx_std=tx_std,
                mask_area_px=mask_area,
            )
        )

    e2d_count_safe = np.where(e2d_count > 0, e2d_count, 1)
    e2d_per_point_accum = e2d_sum / e2d_count_safe
    e2d_per_point_accum[e2d_count == 0] = np.nan

    df_frame = pd.DataFrame(rows)
    df_point = pd.DataFrame(
        dict(
            video=video_name,
            global_index=np.arange(n_pts),
            sample_frame=sample_frame,
            valid_anchor=valid_anchor,
            texture_std=texture_std,
            mean_e2d_over_lifetime=e2d_per_point_accum,
            n_frames_evaluated=e2d_count,
        )
    )

    # Sequence-level summary.
    add_errs = df_frame.loc[df_frame["has_gt"], "add_cm"].dropna() / 100.0
    adi_errs = df_frame.loc[df_frame["has_gt"], "adi_cm"].dropna() / 100.0
    add_auc = compute_auc(add_errs.values) * 100.0 if len(add_errs) else np.nan
    adi_auc = compute_auc(adi_errs.values) * 100.0 if len(adi_errs) else np.nan
    summary = dict(
        video=video_name,
        n_frames=n_frames,
        n_frames_with_gt=int(df_frame["has_gt"].sum()),
        n_points_total=n_pts,
        n_points_anchored=int(valid_anchor.sum()),
        mean_add_cm=float(df_frame["add_cm"].mean()),
        mean_adi_cm=float(df_frame["adi_cm"].mean()),
        add_auc=add_auc,
        adi_auc=adi_auc,
        mean_trans_err_m=float(df_frame["trans_err_m"].mean()),
        mean_rot_err_deg=float(df_frame["rot_err_deg"].mean()),
        mean_e2d_px=float(df_frame["mean_e2d_px"].mean()),
        median_e2d_px=float(df_frame["median_e2d_px"].mean()),
        tx_grad_mean=float(df_frame["tx_grad"].mean()),
        tx_std_mean=float(df_frame["tx_std"].mean()),
        mean_n_unc_020=float(df_frame["n_unc_020"].mean()),
        mean_n_visible=float(df_frame["n_visible"].mean()),
        mean_texture_std_per_point=float(np.nanmean(texture_std)),
    )

    # Save per-video CSVs.
    out_dir.mkdir(parents=True, exist_ok=True)
    df_frame.to_csv(out_dir / f"per_frame_{video_name}.csv", index=False)
    df_point.to_csv(out_dir / f"per_point_{video_name}.csv", index=False)

    # Sanity: round-trip projection at sample frame should match predicted track2d.
    rt_errs = []
    for s in np.unique(sample_frame[sample_frame >= 0]):
        s = int(s)
        if gt_poses[s] is None:
            continue
        track2d_s = ragged(npz, "track2d", s).reshape(-1, 2)
        idx = np.where(sample_frame == s)[0]
        idx = idx[(idx < track2d_s.shape[0]) & valid_anchor[idx]]
        if idx.size == 0:
            continue
        gt2d = project_points(X_can[idx], K, gt_poses[s])
        rt = np.linalg.norm(track2d_s[idx] - gt2d, axis=1)
        rt_errs.append(rt[np.isfinite(rt)])
    if rt_errs:
        rt_all = np.concatenate(rt_errs)
        print(
            f"  round-trip projection error: max={rt_all.max():.2e}px, "
            f"median={np.median(rt_all):.2e}px"
        )
        # Note: ~1 px round-trip is expected (depth quantization + log/append ordering
        # makes the recovered sample frame s+1 instead of s, so track3d there is the
        # tracker's prediction back-projected with frame-s+1 depth, not the original
        # sampled pixel). e2d magnitudes are typically far larger.
        if rt_all.max() > 5.0:
            warnings.warn(
                f"{video_name}: round-trip error exceeds 5 px (max={rt_all.max():.3f})"
            )

    return summary, df_frame, df_point


def bootstrap_ci(x, y, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    if n_obs < 4:
        return (np.nan, np.nan, np.nan)
    base = fn(x, y)[0]
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            boots[i] = fn(x[idx], y[idx])[0]
        except Exception:
            boots[i] = np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return base, lo, hi


def annotate_corr(ax, x, y, label_prefix=""):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        ax.text(
            0.05, 0.95, f"{label_prefix}N={mask.sum()} (insufficient)",
            transform=ax.transAxes, va="top", fontsize=8,
        )
        return
    r, _ = pearsonr(x[mask], y[mask])
    rho, _ = spearmanr(x[mask], y[mask])
    rho_b, lo, hi = bootstrap_ci(x[mask], y[mask], spearmanr)
    ax.text(
        0.05, 0.95,
        f"{label_prefix}N={mask.sum()}\n"
        f"Pearson r={r:.2f}\n"
        f"Spearman ρ={rho:.2f} [{lo:.2f}, {hi:.2f}]",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def render_plots(
    df_frame_all: pd.DataFrame,
    df_point_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
    unc_thres: float,
    ptx_thresholds: list[float],
    tx_low: Optional[float],
    tx_high: Optional[float],
    cited_videos: list[str],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # P1: per-frame mean e2d vs ADD, colored by sequence.
    fig, ax = plt.subplots(figsize=(7, 5))
    for v, sub in df_frame_all.groupby("video"):
        ax.scatter(
            sub["mean_e2d_px"], sub["add_cm"], s=8, alpha=0.5, label=v
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean per-frame 2D tracking error e2d (px, log)")
    ax.set_ylabel("ADD (cm)")
    ax.set_title("P1: Per-frame 2D tracking error vs pose error")
    annotate_corr(ax, df_frame_all["mean_e2d_px"].values, df_frame_all["add_cm"].values, "pooled ")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "P1_e2d_vs_add.png", dpi=150)
    plt.close(fig)

    # P2: count of confident points (unc<thres) vs ADD per frame.
    col_unc = f"n_unc_0{int(unc_thres*100):02d}"
    if col_unc not in df_frame_all.columns:
        col_unc = "n_unc_020"
    fig, ax = plt.subplots(figsize=(7, 5))
    for v, sub in df_frame_all.groupby("video"):
        ax.scatter(sub[col_unc], sub["add_cm"], s=8, alpha=0.5, label=v)
    ax.set_xlabel(f"#points with uncertainty < {unc_thres} (per frame)")
    ax.set_ylabel("ADD (cm)")
    ax.set_title("P2: Confident-point count vs pose error")
    annotate_corr(ax, df_frame_all[col_unc].values.astype(float), df_frame_all["add_cm"].values, "pooled ")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "P2_nunc_vs_add.png", dpi=150)
    plt.close(fig)

    # P3: per-point texture_std bands vs mean e2d over lifetime.
    df_pt = df_point_all.dropna(subset=["texture_std", "mean_e2d_over_lifetime"]).copy()
    edges = [0.0] + sorted(ptx_thresholds) + [np.inf]
    labels = [f"<{edges[1]:.0f}"] + [
        f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(1, len(edges) - 2)
    ] + [f"≥{edges[-2]:.0f}"]
    df_pt["band"] = pd.cut(df_pt["texture_std"], bins=edges, labels=labels, right=False)
    band_stats = df_pt.groupby("band", observed=False).agg(
        mean_e2d=("mean_e2d_over_lifetime", "mean"),
        median_e2d=("mean_e2d_over_lifetime", "median"),
        n_points=("mean_e2d_over_lifetime", "count"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    band_stats["mean_e2d"].plot(kind="bar", ax=ax, color="steelblue")
    for i, n in enumerate(band_stats["n_points"]):
        ax.text(i, band_stats["mean_e2d"].iloc[i], f"N={int(n)}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Per-point texture_std band (greyscale 11x11 patch std)")
    ax.set_ylabel("Mean lifetime e2d (px)")
    ax.set_title("P3: Per-point texture vs 2D tracking error")
    fig.tight_layout()
    fig.savefig(out_dir / "P3_texture_band_e2d.png", dpi=150)
    plt.close(fig)
    band_stats.to_csv(out_dir / "P3_band_stats.csv")

    # P4: time series for cited videos.
    for v in cited_videos:
        sub = df_frame_all[df_frame_all["video"] == v].sort_values("frame")
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(sub["frame"], sub["mean_e2d_px"], color="C0")
        axes[0].set_ylabel("mean e2d (px)")
        axes[0].set_yscale("log")
        axes[1].plot(sub["frame"], sub[col_unc], color="C1")
        axes[1].set_ylabel(f"#pts unc<{unc_thres}")
        axes[2].plot(sub["frame"], sub["add_cm"], color="C2")
        axes[2].set_ylabel("ADD (cm)")
        axes[3].plot(sub["frame"], sub["trans_err_m"] * 100, color="C3")
        axes[3].set_ylabel("trans err (cm)")
        axes[3].set_xlabel("frame")
        axes[0].set_title(f"P4: per-frame coupling, {v}")
        fig.tight_layout()
        fig.savefig(out_dir / f"P4_timeseries_{v}.png", dpi=150)
        plt.close(fig)

    # P5 requires per-point uncertainty cache (rendered separately by render_p5_unc_sweep).

    # P6: sequence-level scatter of tx_grad vs mean e2d and vs ADD AUC.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sx = summary_df["tx_grad_mean"].values.astype(float)
    sy1 = summary_df["mean_e2d_px"].values.astype(float)
    sy2 = summary_df["add_auc"].values.astype(float)
    for ax, y, ylabel, fname in [
        (axes[0], sy1, "Mean e2d (px)", "e2d"),
        (axes[1], sy2, "ADD AUC", "auc"),
    ]:
        ax.scatter(sx, y, s=40)
        for i, v in enumerate(summary_df["video"]):
            color = "red" if v in cited_videos else "black"
            ax.annotate(v, (sx[i], y[i]), fontsize=7, color=color)
        ax.set_xlabel("tx_grad (mean Sobel magnitude in mask)")
        ax.set_ylabel(ylabel)
        annotate_corr(ax, sx, y)
    axes[0].set_title("P6a: texturelessness vs e2d (sequence-level)")
    axes[1].set_title("P6b: texturelessness vs pose AUC (sequence-level)")
    fig.tight_layout()
    fig.savefig(out_dir / "P6_tx_grad_vs_metrics.png", dpi=150)
    plt.close(fig)

    # P7: per-frame tx_grad coupling for cited videos.
    for v in cited_videos:
        sub = df_frame_all[df_frame_all["video"] == v].sort_values("frame")
        if sub.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(9, 4))
        smoothed_tx = sub["tx_grad"].rolling(15, min_periods=1, center=True).mean()
        smoothed_e2d = sub["mean_e2d_px"].rolling(15, min_periods=1, center=True).mean()
        smoothed_add = sub["add_cm"].rolling(15, min_periods=1, center=True).mean()
        ax1.plot(sub["frame"], smoothed_tx, color="C0", label="tx_grad")
        ax1.set_ylabel("tx_grad", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(sub["frame"], smoothed_e2d, color="C3", label="e2d (px)")
        ax2.plot(sub["frame"], smoothed_add * 5, color="C2",
                 label="ADD*5 (cm)", linestyle="--")
        ax2.set_ylabel("e2d (px) / scaled ADD", color="C3")
        ax1.set_xlabel("frame")
        ax1.set_title(f"P7: per-frame coupling, {v} (15-frame moving avg)")
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"P7_tx_coupling_{v}.png", dpi=150)
        plt.close(fig)

    # Calibration histogram for tx_grad.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary_df["tx_grad_mean"].dropna(), bins=20, color="steelblue", edgecolor="black")
    if tx_low is not None:
        ax.axvline(tx_low, color="red", linestyle="--", label=f"tx_low={tx_low}")
    if tx_high is not None:
        ax.axvline(tx_high, color="green", linestyle="--", label=f"tx_high={tx_high}")
    for _, row in summary_df.iterrows():
        if row["video"] in cited_videos:
            ax.axvline(row["tx_grad_mean"], color="black", linestyle=":", alpha=0.6)
            ax.text(row["tx_grad_mean"], 0.5, row["video"], rotation=90,
                    fontsize=7, color="black")
    ax.set_xlabel("Mean tx_grad per sequence")
    ax.set_ylabel("# sequences")
    ax.set_title("Calibration: distribution of tx_grad_mean across sequences")
    if tx_low is not None or tx_high is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_tx_grad_histogram.png", dpi=150)
    plt.close(fig)

    # Banded summary table.
    if tx_low is not None and tx_high is not None:
        bands = []
        for _, row in summary_df.iterrows():
            t = row["tx_grad_mean"]
            if np.isnan(t):
                bands.append("nan")
            elif t < tx_low:
                bands.append("textureless")
            elif t < tx_high:
                bands.append("mid")
            else:
                bands.append("textured")
        summary_df = summary_df.copy()
        summary_df["band"] = bands
        band_table = summary_df.groupby("band").agg(
            n_seq=("video", "count"),
            mean_tx_grad=("tx_grad_mean", "mean"),
            mean_e2d=("mean_e2d_px", "mean"),
            mean_add_cm=("mean_add_cm", "mean"),
            add_auc=("add_auc", "mean"),
            mean_n_unc_020=("mean_n_unc_020", "mean"),
        )
        band_table.to_csv(out_dir / "band_table.csv")
        print("Band table:\n", band_table)
        # Cited-failures explicit rows.
        cited_rows = summary_df[summary_df["video"].isin(cited_videos)][
            ["video", "tx_grad_mean", "mean_e2d_px", "mean_add_cm", "add_auc",
             "mean_n_unc_020", "band"]
        ]
        cited_rows.to_csv(out_dir / "cited_failures_rows.csv", index=False)
        print("Cited failure rows:\n", cited_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--videos", type=str, nargs="*", default=None,
                        help="Specific video names to process (default: all under results-root)")
    parser.add_argument("--unc-thres", type=float, default=0.2)
    parser.add_argument("--ptx-thresholds", type=float, nargs="*",
                        default=[10, 25, 50],
                        help="Per-point texture_std thresholds for P3 bands")
    parser.add_argument("--tx-low", type=float, default=None,
                        help="Sequence-level tx_grad threshold for 'textureless' band")
    parser.add_argument("--tx-high", type=float, default=None,
                        help="Sequence-level tx_grad threshold for 'textured' band")
    parser.add_argument("--cited-videos", type=str, nargs="*", default=["AP12", "SB13"])
    parser.add_argument("--no-per-point-texture", action="store_true",
                        help="Skip per-point texture (faster).")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.videos:
        videos = args.videos
    else:
        videos = sorted(p.name for p in results_root.iterdir()
                        if (p / "meta_data" / "meta_data.npz").exists())
    print(f"Videos to process ({len(videos)}): {videos}")

    summaries = []
    df_frames = []
    df_points = []
    for v in videos:
        try:
            r = process_video(
                video_name=v,
                results_root=results_root,
                data_root=data_root,
                out_dir=out_dir,
                unc_thres=args.unc_thres,
                do_per_point_texture=not args.no_per_point_texture,
            )
        except Exception as e:
            warnings.warn(f"{v}: failed -- {e}")
            import traceback; traceback.print_exc()
            continue
        if r is None:
            continue
        s, df_f, df_p = r
        summaries.append(s)
        df_frames.append(df_f)
        df_points.append(df_p)

    if not summaries:
        print("No videos processed; aborting.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote summary.csv with {len(summary_df)} rows")

    df_frame_all = pd.concat(df_frames, ignore_index=True)
    df_point_all = pd.concat(df_points, ignore_index=True)
    df_frame_all.to_csv(out_dir / "per_frame_all.csv", index=False)
    df_point_all.to_csv(out_dir / "per_point_all.csv", index=False)

    render_plots(
        df_frame_all=df_frame_all,
        df_point_all=df_point_all,
        summary_df=summary_df,
        out_dir=out_dir,
        unc_thres=args.unc_thres,
        ptx_thresholds=args.ptx_thresholds,
        tx_low=args.tx_low,
        tx_high=args.tx_high,
        cited_videos=args.cited_videos,
    )

    # Headline correlations to print for the rebuttal.
    print("\n=== Headline correlations ===")
    sx = summary_df["tx_grad_mean"].values
    for col, label in [
        ("mean_e2d_px", "tx_grad vs mean e2d (sequence-level)"),
        ("add_auc", "tx_grad vs ADD AUC (sequence-level)"),
        ("mean_add_cm", "tx_grad vs mean ADD (sequence-level)"),
        ("mean_n_unc_020", "tx_grad vs mean #pts unc<0.2 (sequence-level)"),
    ]:
        sy = summary_df[col].values.astype(float)
        m = np.isfinite(sx) & np.isfinite(sy)
        if m.sum() < 4:
            print(f"  [{label}] insufficient N={m.sum()}")
            continue
        r, _ = pearsonr(sx[m], sy[m])
        rho, _ = spearmanr(sx[m], sy[m])
        rho_b, lo, hi = bootstrap_ci(sx[m], sy[m], spearmanr)
        print(f"  {label}: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f} [{lo:.2f},{hi:.2f}]")

    # Frame-level pooled correlations.
    fx = df_frame_all["mean_e2d_px"].values
    fy = df_frame_all["add_cm"].values
    m = np.isfinite(fx) & np.isfinite(fy)
    r, _ = pearsonr(fx[m], fy[m])
    rho, _ = spearmanr(fx[m], fy[m])
    print(f"  frame-level pooled e2d vs ADD: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f}")

    fx = df_frame_all[f"n_unc_020"].values.astype(float)
    fy = df_frame_all["add_cm"].values
    m = np.isfinite(fx) & np.isfinite(fy)
    r, _ = pearsonr(fx[m], fy[m])
    rho, _ = spearmanr(fx[m], fy[m])
    print(f"  frame-level pooled n_unc<0.2 vs ADD: N={m.sum()}, Pearson={r:.3f}, Spearman={rho:.3f}")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
