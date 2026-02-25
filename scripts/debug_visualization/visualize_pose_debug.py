import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from point2pose.io.sources.dataset.datareader import (
        Ho3dReader,
        YcbineoatReader,
        YCBInIsaacReader,
    )
except Exception:
    Ho3dReader = None
    YcbineoatReader = None
    YCBInIsaacReader = None

from point2pose.utils.transform import inverse_SE3


def _rotmat_to_quat_xyzw(Rm: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w] (scalar-last).
    """
    Rm = np.asarray(Rm, dtype=float).reshape(3, 3)
    tr = float(np.trace(Rm))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0  # S=4*qw
        qw = 0.25 * S
        qx = (Rm[2, 1] - Rm[1, 2]) / S
        qy = (Rm[0, 2] - Rm[2, 0]) / S
        qz = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        # Find the largest diagonal element and proceed accordingly
        if (Rm[0, 0] > Rm[1, 1]) and (Rm[0, 0] > Rm[2, 2]):
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0  # S=4*qx
            qw = (Rm[2, 1] - Rm[1, 2]) / S
            qx = 0.25 * S
            qy = (Rm[0, 1] + Rm[1, 0]) / S
            qz = (Rm[0, 2] + Rm[2, 0]) / S
        elif Rm[1, 1] > Rm[2, 2]:
            S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2.0  # S=4*qy
            qw = (Rm[0, 2] - Rm[2, 0]) / S
            qx = (Rm[0, 1] + Rm[1, 0]) / S
            qy = 0.25 * S
            qz = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2.0  # S=4*qz
            qw = (Rm[1, 0] - Rm[0, 1]) / S
            qx = (Rm[0, 2] + Rm[2, 0]) / S
            qy = (Rm[1, 2] + Rm[2, 1]) / S
            qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=float)


def _rotmat_to_rotvec(Rm: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to rotation vector (axis * angle), angle in radians.
    """
    Rm = np.asarray(Rm, dtype=float).reshape(3, 3)
    cos_theta = (np.trace(Rm) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3, dtype=float)
    sin_theta = float(np.sin(theta))
    # Guard for numerical issues near pi
    if abs(sin_theta) < 1e-12:
        # Fallback: derive axis from diagonal
        axis = np.sqrt(np.maximum(np.diag(Rm) + 1.0, 0.0)) / np.sqrt(2.0)
        axis = np.where(np.isfinite(axis), axis, 0.0)
        # Fix signs using off-diagonals
        axis[0] = np.copysign(axis[0], Rm[2, 1] - Rm[1, 2])
        axis[1] = np.copysign(axis[1], Rm[0, 2] - Rm[2, 0])
        axis[2] = np.copysign(axis[2], Rm[1, 0] - Rm[0, 1])
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3, dtype=float)
        axis = axis / n
        return axis * theta
    axis = np.array(
        [Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]],
        dtype=float,
    ) / (2.0 * sin_theta)
    return axis * theta


def _rotmat_to_euler_xyz(Rm: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to roll, pitch, yaw using an XYZ intrinsic (roll-pitch-yaw) convention.

    Returns [roll, pitch, yaw] in radians.
    """
    Rm = np.asarray(Rm, dtype=float).reshape(3, 3)
    # Convention equivalent to R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = np.sqrt(Rm[0, 0] * Rm[0, 0] + Rm[1, 0] * Rm[1, 0])
    singular = sy < 1e-12
    if not singular:
        roll = np.arctan2(Rm[2, 1], Rm[2, 2])
        pitch = np.arctan2(-Rm[2, 0], sy)
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        # Gimbal lock: yaw set to 0
        roll = np.arctan2(-Rm[1, 2], Rm[1, 1])
        pitch = np.arctan2(-Rm[2, 0], sy)
        yaw = 0.0
    return np.array([float(roll), float(pitch), float(yaw)], dtype=float)


# Use a small numpy-only Rotation fallback here.
# Rationale: these debug scripts are often run in minimal environments where SciPy/OpenCV
# may be missing or binary-incompatible with the installed NumPy.
class _RotationFallback:
    def __init__(self, mat: np.ndarray):
        self._mat = np.asarray(mat, dtype=float).reshape(3, 3)

    def as_quat(self) -> np.ndarray:
        return _rotmat_to_quat_xyzw(self._mat)

    def as_rotvec(self) -> np.ndarray:
        return _rotmat_to_rotvec(self._mat)

    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        seq = (seq or "").lower()
        if seq != "xyz":
            raise ValueError(
                f"Rotation fallback only supports as_euler('xyz'), got {seq!r}"
            )
        e = _rotmat_to_euler_xyz(self._mat)
        if degrees:
            return np.degrees(e)
        return e


class R:
    @staticmethod
    def from_matrix(mat):
        return _RotationFallback(mat)


def _set_plot_style():
    """
    Set a nicer default matplotlib style, but fall back gracefully if unavailable.
    """
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        try:
            plt.style.use("seaborn")
        except Exception:
            # Keep matplotlib defaults
            pass


# ---------------------- IO Helpers ---------------------- #
def load_logs(log_path):
    """
    Load the .npz file generated by DataLogger.
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    try:
        data = np.load(log_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading log file: {e}")
        sys.exit(1)


def ensure_numeric(arr, name):
    """
    Convert ragged/object pose arrays into a proper (N, 4, 4) float64 array.
    Replace None / invalid entries with identity.
    """
    if arr.dtype != object:
        return arr.astype(np.float64)

    print(f"Warning: {name} is object array, attempting to stack...")
    cleaned = []
    for i, item in enumerate(arr):
        if item is None:
            cleaned.append(np.eye(4))
        elif isinstance(item, np.ndarray) and item.shape == ():
            # 0-d array; unwrap and check
            val = item.item()
            if val is None or not isinstance(val, np.ndarray) or val.shape != (4, 4):
                cleaned.append(np.eye(4))
            else:
                cleaned.append(val.astype(float))
        elif np.shape(item) != (4, 4):
            cleaned.append(np.eye(4))
        else:
            cleaned.append(item.astype(float))

    return np.array(cleaned, dtype=np.float64)


def unpack_ragged(data, name):
    """
    Reconstruct a list of arrays from the packed ragged format (data/offsets/lengths).
    Returns a list of numpy arrays (or None if keys missing).
    Fallback: if 'name' exists directly, return that.
    """
    # 1. Check if exists directly (e.g. object array or fixed shape)
    if name in data:
        return data[name]

    # 2. Check for packed ragged format
    key_data = f"{name}_data"
    key_offsets = f"{name}_offsets"
    key_lengths = f"{name}_lengths"

    if key_data not in data or key_offsets not in data or key_lengths not in data:
        return None

    all_data = data[key_data]
    offsets = data[key_offsets]
    lengths = data[key_lengths]

    reconstructed = []
    for off, length in zip(offsets, lengths):
        # Ensure indices are integers
        off = int(off)
        length = int(length)
        reconstructed.append(all_data[off : off + length])

    return reconstructed


# ---------------------- Pose / Error Helpers ---------------------- #
def se3_to_xyz_quat(poses):
    """
    Convert (N, 4, 4) poses to (N, 7) [x, y, z, qx, qy, qz, qw]
    """
    poses = np.asarray(poses)
    N = poses.shape[0]
    xyz = poses[:, :3, 3]

    quats = []
    for i in range(N):
        r = R.from_matrix(poses[i, :3, :3])
        quats.append(r.as_quat())

    quats = np.array(quats)
    return np.hstack([xyz, quats])


def se3_to_xyz_rpy(poses):
    """
    Convert (N, 4, 4) poses to (N, 6) [x, y, z, roll, pitch, yaw]
    Uses 'xyz' Euler angle convention (intrinsic rotations).
    """
    poses = np.asarray(poses)
    N = poses.shape[0]
    xyz = poses[:, :3, 3]

    rpy = []
    for i in range(N):
        r = R.from_matrix(poses[i, :3, :3])
        # Use 'xyz' convention (intrinsic rotations)
        euler = r.as_euler("xyz", degrees=False)
        rpy.append(euler)

    rpy = np.array(rpy)
    return np.hstack([xyz, rpy])


def compute_relative_errors(pred_poses, gt_poses):
    """
    Compute per-frame translation and rotation errors.
    Errors are between pred_poses[i] and gt_poses[i], as:
      rel = inv(gt) @ pred
      t_err = ||rel.t||
      r_err = angle(rel.R) in radians
    Returns:
      t_errs: (N,)
      r_errs: (N,)
    """
    pred_poses = np.asarray(pred_poses)
    gt_poses = np.asarray(gt_poses)

    min_len = min(len(pred_poses), len(gt_poses))
    pred_poses = pred_poses[:min_len]
    gt_poses = gt_poses[:min_len]

    t_errs = np.zeros(min_len)
    r_errs = np.zeros(min_len)

    for i in range(min_len):
        rel = np.linalg.inv(gt_poses[i]) @ pred_poses[i]
        t_errs[i] = np.linalg.norm(rel[:3, 3])

        r_mat = rel[:3, :3]
        r = R.from_matrix(r_mat)
        r_errs[i] = np.linalg.norm(r.as_rotvec())

    return t_errs, r_errs


def summarize_error_stats(name, t_err, r_err):
    """
    Print a compact summary of translation / rotation statistics.
    """
    if t_err is None or r_err is None:
        return

    t_err = np.asarray(t_err)
    r_err_deg = np.degrees(np.asarray(r_err))

    def _safe_stats(x):
        if x.size == 0:
            return dict(mean=np.nan, median=np.nan, rmse=np.nan, p90=np.nan, max=np.nan)
        return dict(
            mean=float(np.mean(x)),
            median=float(np.median(x)),
            rmse=float(np.sqrt(np.mean(x**2))),
            p90=float(np.percentile(x, 90)),
            max=float(np.max(x)),
        )

    t_stats = _safe_stats(t_err)
    r_stats = _safe_stats(r_err_deg)

    print(f"\n[{name}] Translation Error (m):")
    print(
        "  mean={mean:.4f}, median={median:.4f}, rmse={rmse:.4f}, "
        "p90={p90:.4f}, max={max:.4f}".format(**t_stats)
    )

    print(f"[{name}] Rotation Error (deg):")
    print(
        "  mean={mean:.2f}, median={median:.2f}, rmse={rmse:.2f}, "
        "p90={p90:.2f}, max={max:.2f}".format(**r_stats)
    )


# ---------------------- Residual Analysis ---------------------- #
def compute_residual_stats(reg_residuals, reg_inliers):
    """
    Compute per-frame mean residual (of inliers) and inlier count.
    Returns:
      mean_residuals: (N,) array, NaN where no data
      inlier_counts: (N,) array, 0 where no data
    """
    N = len(reg_residuals)
    mean_residuals = np.full(N, np.nan)
    inlier_counts = np.zeros(N, dtype=int)

    for i in range(N):
        res = reg_residuals[i]
        # handle case where reg_inliers might be None or shorter
        inl = None
        if reg_inliers is not None and i < len(reg_inliers):
            inl = reg_inliers[i]

        if (
            res is None
            or (isinstance(res, np.ndarray) and res.size == 0)
            or (isinstance(res, list) and len(res) == 0)
        ):
            continue

        res = np.array(res)

        # Check if inliers is valid
        mask = None
        if inl is not None:
            inl = np.array(inl)
            # Basic check: size match
            if inl.size == res.size:
                mask = inl.astype(bool)

        if mask is not None and np.any(mask):
            valid_res = res[mask]
            count = np.sum(mask)
        else:
            # If mask exists but all false -> count=0
            # If mask is None -> use all
            if mask is not None:
                valid_res = []
                count = 0
            else:
                valid_res = res
                count = len(res)

        if len(valid_res) > 0:
            mean_residuals[i] = np.mean(valid_res)
        inlier_counts[i] = count

    return mean_residuals, inlier_counts


def _as_py(obj):
    """
    Lightweight conversion of numpy/object wrappers to plain Python objects.
    Mirrors the helper in plot_clustered_registration_stats, but kept minimal.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return _as_py(obj.item())
        return obj.tolist()
    return obj


def _extract_best_cluster_metric_series(data, metric_key: str):
    """
    Extract per-frame scalar metric for the *selected* registration cluster.

    Requires:
      - data['reg_clusters']: object array of per-frame cluster dicts
      - data['reg_best_cluster_idx']: chosen cluster index per frame

    Returns:
      series: (N,) float array with NaN where unavailable, or None if required keys missing.
    """
    if "reg_clusters" not in data or "reg_best_cluster_idx" not in data:
        return None

    clusters_arr = data["reg_clusters"]
    try:
        N = len(clusters_arr)
    except TypeError:
        return None

    try:
        best_idx_raw = data["reg_best_cluster_idx"]
        # Handle None values in best_idx array
        best_idx_arr = []
        for val in best_idx_raw:
            if val is None:
                best_idx_arr.append(-1)  # Use -1 as sentinel for "no cluster"
            else:
                try:
                    best_idx_arr.append(int(val))
                except (ValueError, TypeError):
                    best_idx_arr.append(-1)
        best_idx_arr = np.asarray(best_idx_arr, dtype=int)
    except Exception:
        return None

    series = np.full(N, np.nan, dtype=float)
    sample_keys_checked = set()  # Track what keys we've seen in clusters
    for i in range(N):
        clusters_i = _as_py(clusters_arr[i])
        if clusters_i is None:
            continue
        if isinstance(clusters_i, dict):
            clusters_i = [clusters_i]
        if not isinstance(clusters_i, list) or not clusters_i:
            continue

        if i >= len(best_idx_arr):
            continue
        idx = int(best_idx_arr[i])
        if idx < 0:  # Sentinel value means no cluster selected
            continue
        if not (0 <= idx < len(clusters_i)):
            continue

        c = clusters_i[idx]
        if not isinstance(c, dict):
            continue

        # Debug: sample first few frames to see what keys exist
        if i < 3 and len(sample_keys_checked) == 0:
            sample_keys_checked.update(c.keys())

        val = c.get(metric_key, None)
        if val is None:
            continue
        try:
            series[i] = float(val)
        except Exception:
            continue

    if not np.isfinite(series).any():
        if len(sample_keys_checked) > 0:
            print(
                f"  Debug: {metric_key} not found in clusters. Sample cluster keys: {sorted(sample_keys_checked)}"
            )
        return None
    return series


def _extract_best_cluster_reproj_error_series(data):
    """
    Extract per-frame reprojection error for the *selected* registration cluster.

    Uses:
      - data['reg_clusters']: object array of per-frame cluster dicts
      - data['reg_best_cluster_idx']: index of chosen cluster per frame (if present)

    Returns:
      best_reproj: (N,) float array, NaN where unavailable, or None if keys missing.
    """
    if "reg_clusters" not in data:
        return None

    clusters_arr = data["reg_clusters"]
    try:
        N = len(clusters_arr)
    except TypeError:
        return None

    best_reproj = np.full(N, np.nan, dtype=float)

    best_idx_arr = None
    if "reg_best_cluster_idx" in data:
        try:
            best_idx_arr = np.asarray(data["reg_best_cluster_idx"]).astype(int)
        except Exception:
            best_idx_arr = None

    if best_idx_arr is None:
        # We currently only support the explicit best-cluster index path,
        # to avoid duplicating the inlier-overlap logic. Fall back to None.
        return None

    for i in range(N):
        # Get clusters for this frame and convert to plain Python objects
        clusters_i = _as_py(clusters_arr[i])
        if clusters_i is None:
            continue
        if isinstance(clusters_i, dict):
            clusters_i = [clusters_i]
        if not isinstance(clusters_i, list) or not clusters_i:
            continue

        if i >= len(best_idx_arr):
            continue
        idx = int(best_idx_arr[i])
        if not (0 <= idx < len(clusters_i)):
            continue

        c = clusters_i[idx]
        if not isinstance(c, dict):
            continue

        val = c.get("reproj_error", None)
        if val is None:
            continue
        try:
            best_reproj[i] = float(val)
        except Exception:
            continue

    if not np.isfinite(best_reproj).any():
        return None
    return best_reproj


def _extract_reproj_error_series(data, name="reproj_error"):
    """
    Extract a 1D per-frame reprojection error series from the logs.

    Handles both:
      - direct scalar array: data['reproj_error'] -> (N,)
      - ragged format:      data['reproj_error'] is a list/array of per-frame arrays
        (we then take the mean per frame).
    Returns (priority order):
      1) best-cluster reproj_error per frame (if cluster logs exist)
      2) generic per-frame series under `name`
    """
    # 1) Prefer per-frame best-cluster reprojection errors if available
    best_cluster_series = _extract_best_cluster_reproj_error_series(data)
    if best_cluster_series is not None:
        return best_cluster_series

    # 2) Fallback: generic series named `name`
    reproj = unpack_ragged(data, name)
    if reproj is None:
        return None

    # Case 1: direct numeric 1D array
    if isinstance(reproj, np.ndarray) and reproj.dtype != object and reproj.ndim == 1:
        return reproj.astype(float)

    # Case 2: object / list-of-arrays -> take mean per frame
    try:
        N = len(reproj)
    except TypeError:
        return None

    series = np.full(N, np.nan, dtype=float)
    for i, v in enumerate(reproj):
        if v is None:
            continue
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            continue
        series[i] = float(np.nanmean(arr))

    # If everything is NaN, treat as missing
    if not np.isfinite(series).any():
        return None
    return series


def plot_reprojection_error_vs_pose_error(
    reproj_err, gl_t_err=None, gl_r_err=None, save_prefix=None
):
    """
    Scatter plot: reprojection error vs. global pose error.

    X-axis: per-frame reprojection error (e.g. mean depth reprojection error)
    Y-axis: global translation error (m) and, if available, rotation error (deg).
    """
    if reproj_err is None or gl_t_err is None:
        print("No reprojection error or global translation error available; skipping.")
        return

    reproj_err = np.asarray(reproj_err, dtype=float)
    gl_t_err = np.asarray(gl_t_err, dtype=float)

    n = min(len(reproj_err), len(gl_t_err))
    if gl_r_err is not None:
        gl_r_err = np.asarray(gl_r_err, dtype=float)
        n = min(n, len(gl_r_err))

    if n == 0:
        print("Reprojection / error arrays are empty; skipping reprojection plot.")
        return

    x = reproj_err[:n]
    y_t = gl_t_err[:n]
    mask = np.isfinite(x) & np.isfinite(y_t)
    if not np.any(mask):
        print("No finite reprojection+translation error pairs; skipping plot.")
        return

    x = x[mask]
    y_t = y_t[mask]

    y_r = None
    if gl_r_err is not None:
        y_r = np.degrees(gl_r_err[:n])[mask]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y_t, s=10, alpha=0.6, color="tab:blue", label="Trans. Err (m)")

    if y_r is not None and np.isfinite(y_r).any():
        plt.scatter(
            x,
            y_r,
            s=10,
            alpha=0.6,
            color="tab:orange",
            label="Rot. Err (deg)",
        )

    # Simple correlation diagnostics
    try:
        corr_t = np.corrcoef(x, y_t)[0, 1]
        print(
            "[Reproj] Corr(reproj_err, global translation error) = {:.3f}".format(
                float(corr_t)
            )
        )
        if y_r is not None and np.isfinite(y_r).any():
            corr_r = np.corrcoef(x, y_r)[0, 1]
            print(
                "[Reproj] Corr(reproj_err, global rotation error) = {:.3f}".format(
                    float(corr_r)
                )
            )
    except Exception as e:
        print(f"[Reproj] Correlation computation failed: {e}")

    plt.xlabel("Reprojection Error")
    plt.ylabel("Pose Error")
    plt.title("Reprojection Error vs Global Pose Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.show()


def plot_reproj_over_3d_dist_vs_pose_error(
    best_reproj_err,
    best_3d_dist,
    gl_t_err=None,
    gl_r_err=None,
    metric_label="reproj_error",
    save_prefix=None,
):
    """
    Scatter plot using ONLY the selected cluster per frame:

      y = metric / 3d_dist  (where metric is reproj_error or mean_res)
      x = pose error (translation error in meters; optional 2nd panel for rotation error in deg)
    """
    if best_reproj_err is None or best_3d_dist is None or gl_t_err is None:
        print(
            "No best-cluster reproj/3d_dist or global translation error; skipping ratio plot."
        )
        return

    best_reproj_err = np.asarray(best_reproj_err, dtype=float)
    best_3d_dist = np.asarray(best_3d_dist, dtype=float)
    gl_t_err = np.asarray(gl_t_err, dtype=float)

    n = min(len(best_reproj_err), len(best_3d_dist), len(gl_t_err))
    if gl_r_err is not None:
        gl_r_err = np.asarray(gl_r_err, dtype=float)
        n = min(n, len(gl_r_err))

    if n == 0:
        print("Ratio plot arrays are empty; skipping.")
        return

    reproj = best_reproj_err[:n]
    d3 = best_3d_dist[:n]
    x_t = gl_t_err[:n]

    with np.errstate(divide="ignore", invalid="ignore"):
        y = reproj / d3

    mask = np.isfinite(x_t) & np.isfinite(y) & (d3 > 0)
    if not np.any(mask):
        print(f"No finite (error, reproj/3d_dist) pairs; skipping ratio plot.")
        print(
            f"  Debug: n={n}, finite(x_t)={np.sum(np.isfinite(x_t))}, finite(y)={np.sum(np.isfinite(y))}, (d3>0)={np.sum(d3 > 0)}"
        )
        print(
            f"  Debug: finite(reproj)={np.sum(np.isfinite(reproj))}, finite(d3)={np.sum(np.isfinite(d3))}"
        )
        return

    x_t = x_t[mask]
    y = y[mask]

    # Optional rotation error panel
    x_r = None
    if gl_r_err is not None:
        x_r = np.degrees(gl_r_err[:n])[mask]

    if x_r is None:
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.scatter(x_t, y, s=10, alpha=0.6, color="tab:purple")
        ax.set_xlabel("Global Translation Error (m)")
        ax.set_ylabel(f"{metric_label} / 3d_dist")
        ax.set_title(f"Selected Cluster: {metric_label}/3d_dist vs Translation Error")
        ax.grid(True, linestyle="--", alpha=0.6)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.scatter(x_t, y, s=10, alpha=0.6, color="tab:purple")
        ax1.set_xlabel("Global Translation Error (m)")
        ax1.set_ylabel(f"{metric_label} / 3d_dist")
        ax1.set_title("... vs Translation Error")
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.scatter(x_r, y, s=10, alpha=0.6, color="tab:purple")
        ax2.set_xlabel("Global Rotation Error (deg)")
        ax2.set_ylabel(f"{metric_label} / 3d_dist")
        ax2.set_title("... vs Rotation Error")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

    # Simple correlation diagnostics (translation / rotation)
    try:
        if x_t.size > 1:
            corr_t = np.corrcoef(x_t, y)[0, 1]
            print(
                f"[Reproj/3D] Corr(global translation error, {metric_label}/3d_dist) = {float(corr_t):.3f}"
            )
        if x_r is not None and x_r.size > 1:
            corr_r = np.corrcoef(x_r, y)[0, 1]
            print(
                f"[Reproj/3D] Corr(global rotation error, {metric_label}/3d_dist) = {float(corr_r):.3f}"
            )
    except Exception as e:
        print(f"[Reproj/3D] Correlation computation failed: {e}")

    print(f"[Reproj/3D] Plotting figure with {len(x_t)} points...")
    plt.show()
    print("[Reproj/3D] Figure displayed.")


def plot_selected_cluster_metric_vs_frames(
    metric_values,
    gl_t_err=None,
    gl_r_err=None,
    metric_label="3d_dist",
    save_prefix=None,
):
    """
    Line plot using ONLY the selected cluster per frame:

      y = metric (either 3d_dist or reproj_error)
      x = frame index

    Optionally overlays translation and rotation errors on secondary y-axes.
    """
    if metric_values is None:
        print(f"No best-cluster {metric_label}; skipping plot.")
        return

    metric_values = np.asarray(metric_values, dtype=float)
    n = len(metric_values)

    if n == 0:
        print("Metric array is empty; skipping plot.")
        return

    frames = np.arange(n)
    mask = np.isfinite(metric_values)

    if not np.any(mask):
        print(f"No finite {metric_label} values; skipping plot.")
        print(f"  Debug: n={n}, finite(values)={np.sum(mask)}")
        return

    frames_valid = frames[mask]
    y_valid = metric_values[mask]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Determine units based on metric type
    if metric_label == "3d_dist":
        ylabel = "3d_dist"
        yunit = ""
    elif metric_label == "reproj_error":
        ylabel = "Reprojection Error"
        yunit = " (m)"
    else:
        ylabel = metric_label
        yunit = ""

    # Main plot: metric over frames
    ax.plot(
        frames_valid,
        y_valid,
        ".-",
        color="tab:purple",
        alpha=0.7,
        markersize=3,
        linewidth=1.0,
        label=metric_label,
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(f"{ylabel}{yunit}", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Overlay translation error if available
    ax_t = None
    if gl_t_err is not None:
        gl_t_err = np.asarray(gl_t_err, dtype=float)
        n_err = min(n, len(gl_t_err))
        ax_t = ax.twinx()
        ax_t.plot(
            frames[:n_err],
            gl_t_err[:n_err],
            "-",
            color="tab:blue",
            alpha=0.6,
            linewidth=1.0,
            label="Translation Error",
        )
        ax_t.set_ylabel("Translation Error (m)", color="tab:blue")
        ax_t.tick_params(axis="y", labelcolor="tab:blue")
        # Offset the right spine to make room for rotation error axis if needed
        if gl_r_err is not None:
            ax_t.spines["right"].set_position(("outward", 60))

    # Overlay rotation error if available
    ax_r = None
    if gl_r_err is not None:
        gl_r_err = np.asarray(gl_r_err, dtype=float)
        n_err = min(n, len(gl_r_err))
        ax_r = ax.twinx()
        if gl_t_err is not None:
            # Offset further right for rotation error
            ax_r.spines["right"].set_position(("outward", 60))
        ax_r.plot(
            frames[:n_err],
            np.degrees(gl_r_err[:n_err]),
            "--",
            color="tab:orange",
            alpha=0.6,
            linewidth=1.0,
            label="Rotation Error",
        )
        ax_r.set_ylabel("Rotation Error (deg)", color="tab:orange")
        ax_r.tick_params(axis="y", labelcolor="tab:orange")

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    if ax_t is not None:
        h_t, l_t = ax_t.get_legend_handles_labels()
        handles.extend(h_t)
        labels.extend(l_t)
    if ax_r is not None:
        h_r, l_r = ax_r.get_legend_handles_labels()
        handles.extend(h_r)
        labels.extend(l_r)
    ax.legend(handles, labels, loc="upper left")

    title_suffix = (
        " (with GT Errors)" if (gl_t_err is not None or gl_r_err is not None) else ""
    )
    ax.set_title(f"Selected Cluster: {metric_label} Over Frames{title_suffix}")

    print(
        f"[Reproj/3D] Plotting figure with {len(frames_valid)} points over {n} frames..."
    )
    plt.tight_layout()
    plt.show()
    print("[Reproj/3D] Figure displayed.")


def plot_reproj_over_3d_dist_vs_error(
    best_reproj_err,
    best_3d_dist,
    err,
    *,
    err_label: str,
):
    """
    Scatter plot using ONLY the selected cluster per frame:

      x = err (some per-frame "error" scalar series)
      y = reproj_error / 3d_dist
    """
    if best_reproj_err is None or best_3d_dist is None or err is None:
        print(
            "Missing best-cluster reproj/3d_dist or error series; skipping ratio plot."
        )
        return

    best_reproj_err = np.asarray(best_reproj_err, dtype=float)
    best_3d_dist = np.asarray(best_3d_dist, dtype=float)
    err = np.asarray(err, dtype=float)

    n = min(len(best_reproj_err), len(best_3d_dist), len(err))
    if n == 0:
        print("Ratio plot arrays are empty; skipping.")
        return

    reproj = best_reproj_err[:n]
    d3 = best_3d_dist[:n]
    x = err[:n]

    with np.errstate(divide="ignore", invalid="ignore"):
        y = reproj / d3

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(d3) & (d3 > 0)
    if not np.any(mask):
        print("No finite (error, reproj/3d_dist) pairs; skipping ratio plot.")
        return

    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.scatter(x, y, s=10, alpha=0.6, color="tab:purple")
    ax.set_xlabel(err_label)
    ax.set_ylabel("reproj_error / 3d_dist")
    ax.set_title("Selected Cluster: reproj_error/3d_dist vs Error")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    try:
        if x.size > 1:
            corr = np.corrcoef(x, y)[0, 1]
            print(
                f"[Reproj/3D] Corr({err_label}, reproj_error/3d_dist) = {float(corr):.3f}"
            )
    except Exception as e:
        print(f"[Reproj/3D] Correlation computation failed: {e}")

    print(f"[Reproj/3D] Plotting {x.size} points (selected cluster only).")
    plt.show()


def plot_residual_analysis(
    mean_res,
    inlier_counts,
    fe_t_err=None,
    lo_t_err=None,
    gl_t_err=None,
    fe_r_err=None,
    lo_r_err=None,
    gl_r_err=None,
    kf_indices=None,
    save_prefix=None,
):
    """
    Plot registration residuals with extra context and GT errors (if available).

    Layout:
      - Top:  mean residual over time (+ optional global translation error on 2nd y-axis)
      - Mid:  inlier counts over time
      - 3rd:  scatter of residual vs. global translation / rotation error
      - 4th:  histogram of residuals (and optionally global translation error)
    """
    frames = np.arange(len(mean_res))
    # Use independent x-axes so that time-based plots and value-based plots
    # (residual vs. error, histograms) are all readable.
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    ax_res, ax_inl, ax_scatter, ax_hist = axes

    valid_mask = ~np.isnan(mean_res)

    # ----- Summary stats -----
    if np.any(valid_mask):
        vals = mean_res[valid_mask]
        print(
            "[Residual] mean={:.4f}, median={:.4f}, p90={:.4f}, max={:.4f}".format(
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.percentile(vals, 90)),
                float(np.max(vals)),
            )
        )

    # ----- 1) Residual (and optionally translation error) -----
    ax_res.plot(
        frames[valid_mask],
        mean_res[valid_mask],
        ".-",
        label="Mean Residual",
        color="tab:red",
        linewidth=1.0,
        markersize=3,
    )

    # Overlay global translation and rotation errors on secondary/tertiary y-axes if available
    twin_handles = []
    twin_labels = []
    if gl_t_err is not None:
        n = min(len(gl_t_err), len(frames))
        ax_res_t = ax_res.twinx()
        ax_res_t.plot(
            frames[:n],
            gl_t_err[:n],
            "-",
            label="Global Trans. Err (m)",
            color="tab:blue",
            linewidth=1.0,
            alpha=0.8,
        )
        ax_res_t.set_ylabel("Trans. Error (m)", color="tab:blue")
        ax_res_t.tick_params(axis="y", labelcolor="tab:blue")

        # collect handles for joint legend
        twin_handles, twin_labels = ax_res_t.get_legend_handles_labels()

        # Add rotation error on a third y-axis (offset to the right)
        if gl_r_err is not None:
            n_r = min(len(gl_r_err), len(frames))
            ax_res_r = ax_res.twinx()
            # Offset the right spine to make room for rotation error axis
            ax_res_r.spines["right"].set_position(("outward", 60))
            ax_res_r.plot(
                frames[:n_r],
                np.degrees(gl_r_err[:n_r]),
                "--",
                label="Global Rot. Err (deg)",
                color="tab:orange",
                linewidth=1.0,
                alpha=0.8,
            )
            ax_res_r.set_ylabel("Rot. Error (deg)", color="tab:orange")
            ax_res_r.tick_params(axis="y", labelcolor="tab:orange")

            # Add to legend
            r_handles, r_labels = ax_res_r.get_legend_handles_labels()
            twin_handles.extend(r_handles)
            twin_labels.extend(r_labels)

    # Keyframe markers
    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_res.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)

    ax_res.set_ylabel("Mean Residual")
    title = "Registration Residual vs. Global Error"
    if gl_t_err is not None and gl_r_err is not None:
        title = "Registration Residual vs. Global Translation & Rotation Error"
    ax_res.set_title(title)
    ax_res.grid(True, linestyle="--", alpha=0.6)

    # Build combined legend
    base_h, base_l = ax_res.get_legend_handles_labels()
    all_h = base_h + twin_handles
    all_l = base_l + twin_labels
    if all_h:
        ax_res.legend(all_h, all_l, loc="upper left")

    # ----- 2) Inlier count over time -----
    ax_inl.plot(
        frames,
        inlier_counts,
        ".-",
        label="Inlier Count",
        color="tab:green",
        linewidth=1.0,
        markersize=3,
    )
    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_inl.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)
    ax_inl.set_ylabel("# Inliers")
    ax_inl.set_title("Registration Inlier Count")
    ax_inl.grid(True, linestyle="--", alpha=0.6)
    ax_inl.legend(loc="upper left")

    # ----- 3) Scatter: residual vs error (global) -----
    if gl_t_err is not None:
        n = min(len(gl_t_err), len(mean_res))
        valid = valid_mask[:n]
        x_res = mean_res[:n][valid]

        # Use global translation + rotation for analysis
        y_t = gl_t_err[:n][valid]
        y_r = None
        if gl_r_err is not None:
            y_r = np.degrees(gl_r_err[:n])[valid]

        ax_scatter.scatter(
            x_res,
            y_t,
            s=8,
            alpha=0.6,
            label="Trans. Err (m)",
            color="tab:blue",
        )
        if y_r is not None:
            ax_scatter.scatter(
                x_res,
                y_r,
                s=8,
                alpha=0.6,
                label="Rot. Err (deg)",
                color="tab:orange",
            )

        # Correlation analysis
        if x_res.size > 1:
            try:
                corr_t = np.corrcoef(x_res, y_t)[0, 1]
                print(
                    "[Analysis] Corr(residual, global translation error) = {:.3f}".format(
                        float(corr_t)
                    )
                )
                if y_r is not None:
                    corr_r = np.corrcoef(x_res, y_r)[0, 1]
                    print(
                        "[Analysis] Corr(residual, global rotation error) = {:.3f}".format(
                            float(corr_r)
                        )
                    )
            except Exception as e:
                print(f"[Analysis] Correlation computation failed: {e}")

        ax_scatter.set_xlabel("Mean Residual")
        ax_scatter.set_ylabel("Error")
        ax_scatter.set_title("Residual vs. Global GT Error")
        ax_scatter.grid(True, linestyle="--", alpha=0.6)
        ax_scatter.legend(loc="upper left")
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No GT errors available\n(skipping residual-vs-error analysis)",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
        )
        ax_scatter.grid(True, linestyle="--", alpha=0.6)

    # ----- 4) Histogram of residuals (+ optional global translation error) -----
    valid_vals = mean_res[valid_mask]
    if valid_vals.size > 0:
        ax_hist.hist(
            valid_vals,
            bins=40,
            alpha=0.75,
            color="tab:red",
            label="Mean Residual",
        )
        if gl_t_err is not None:
            n_hist = min(len(gl_t_err), len(valid_vals))
            ax_hist.hist(
                gl_t_err[:n_hist],
                bins=40,
                alpha=0.4,
                color="tab:blue",
                label="Global Trans. Err (m)",
            )
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(
            "Distribution of Residuals"
            + (" & Global Error" if gl_t_err is not None else "")
        )
        ax_hist.grid(True, linestyle="--", alpha=0.6)
        ax_hist.legend(loc="upper right")
    else:
        ax_hist.text(
            0.5,
            0.5,
            "No valid residuals to plot",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_hist.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_prefix:
        out_file = f"{save_prefix}_residuals_and_errors.png"
        plt.savefig(out_file, dpi=180)
        print(f"Saved residual+error plot to {out_file}")

    plt.show()


# ---------------------- Slope and Threshold Analysis ---------------------- #
def compute_slopes(values, valid_mask=None, window_size=5):
    """
    Compute the slope (derivative) of a time series using a rolling window of past frames.
    Uses linear regression on the past N frames to compute a more robust slope estimate.
    Returns slopes with same length as input (first values are NaN until window is filled).

    Args:
        values: (N,) array of values
        valid_mask: (N,) boolean array indicating valid values (optional)
        window_size: Number of past frames to use for slope computation (default: 5)

    Returns:
        slopes: (N,) array of slopes, NaN where invalid or insufficient history
    """
    values = np.asarray(values)
    slopes = np.full(len(values), np.nan)

    if valid_mask is None:
        valid_mask = ~np.isnan(values)
    else:
        valid_mask = valid_mask & ~np.isnan(values)

    if np.sum(valid_mask) < window_size:
        return slopes

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < window_size:
        return slopes

    # For each valid index, use the past window_size frames (including current) to compute slope
    for i in range(len(valid_indices)):
        idx_curr = valid_indices[i]

        # Find indices in the window (past window_size frames including current)
        window_start = max(0, i - window_size + 1)
        window_indices = valid_indices[window_start : i + 1]

        if len(window_indices) < 2:
            continue

        # Extract values and frame indices for linear regression
        window_values = values[window_indices]
        window_frames = window_indices.astype(float)

        # Compute slope using linear regression (least squares)
        # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(window_frames)
        sum_x = np.sum(window_frames)
        sum_y = np.sum(window_values)
        sum_xy = np.sum(window_frames * window_values)
        sum_x2 = np.sum(window_frames * window_frames)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) > 1e-10:  # Avoid division by zero
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            slopes[idx_curr] = slope

    return slopes


def compute_frame_to_frame_changes(values, valid_mask=None):
    """
    Compute frame-to-frame changes (jumps/drops).
    Returns changes with same length as input (first value is NaN).

    Args:
        values: (N,) array of values
        valid_mask: (N,) boolean array indicating valid values (optional)

    Returns:
        changes: (N,) array of absolute changes, NaN where invalid or at boundaries
    """
    values = np.asarray(values)
    changes = np.full(len(values), np.nan)

    if valid_mask is None:
        valid_mask = ~np.isnan(values)
    else:
        valid_mask = valid_mask & ~np.isnan(values)

    if np.sum(valid_mask) < 2:
        return changes

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return changes

    for i in range(1, len(valid_indices)):
        idx_curr = valid_indices[i]
        idx_prev = valid_indices[i - 1]
        if idx_prev >= 0 and idx_curr < len(values):
            changes[idx_curr] = values[idx_curr] - values[idx_prev]

    return changes


def compute_inlier_drop_ratios(inlier_counts, valid_mask=None):
    """
    Compute inlier drop ratios: 1.0 - (curr / prev) for consecutive frames.
    Returns ratios with same length as input (first value is NaN).

    Args:
        inlier_counts: (N,) array of inlier counts
        valid_mask: (N,) boolean array indicating valid values (optional)

    Returns:
        drop_ratios: (N,) array of drop ratios, NaN where invalid or at boundaries
    """
    inlier_counts = np.asarray(inlier_counts, dtype=float)
    drop_ratios = np.full(len(inlier_counts), np.nan)

    if valid_mask is None:
        valid_mask = (inlier_counts > 0) & ~np.isnan(inlier_counts)
    else:
        valid_mask = valid_mask & (inlier_counts > 0) & ~np.isnan(inlier_counts)

    if np.sum(valid_mask) < 2:
        return drop_ratios

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return drop_ratios

    for i in range(1, len(valid_indices)):
        idx_curr = valid_indices[i]
        idx_prev = valid_indices[i - 1]
        if idx_prev >= 0 and idx_curr < len(inlier_counts):
            prev_count = inlier_counts[idx_prev]
            curr_count = inlier_counts[idx_curr]
            if prev_count > 0:
                drop_ratios[idx_curr] = 1.0 - (curr_count / prev_count)

    return drop_ratios


def plot_slope_and_threshold_analysis(
    mean_res,
    inlier_counts,
    kf_indices=None,
    save_prefix=None,
    reg_residual_thres=0.07,
    residual_jump_threshold=0.02,
    inlier_drop_ratio=0.5,
    high_residual_threshold=0.01,
    min_inlier_count=5,
    slope_window_size=10,
):
    """
    Plot slopes and threshold analysis to help decide threshold values.

    Layout (all in one figure):
      - Rows 1-2: Residual (top) and Residual Slope (bottom, shared x-axis)
      - Rows 3-4: Inlier Count (top) and Inlier Slope (bottom, shared x-axis)
      - Row 5: Residual Jumps
      - Row 6: Inlier Drop Ratios
      - Rows 7-8: Distributions (2x2 grid)

    Args:
        slope_window_size: Number of past frames to use for slope computation (default: 5)
    """
    frames = np.arange(len(mean_res))
    valid_mask = ~np.isnan(mean_res)

    # Compute slopes using rolling window
    residual_slopes = compute_slopes(
        mean_res, valid_mask, window_size=slope_window_size
    )
    inlier_slopes = compute_slopes(
        inlier_counts, valid_mask=None, window_size=slope_window_size
    )

    # Compute frame-to-frame changes
    residual_jumps = compute_frame_to_frame_changes(mean_res, valid_mask)
    inlier_drop_ratios = compute_inlier_drop_ratios(inlier_counts, valid_mask=None)

    # Create figure with GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 20))
    # Main grid: 6 rows for time series, then 2 rows for distributions (will split into 2x2)
    gs_main = GridSpec(
        8,
        2,
        figure=fig,
        hspace=0.35,
        wspace=0.3,
        height_ratios=[1, 0.8, 1, 0.8, 1, 1, 1, 1],
        width_ratios=[1, 1],
    )

    # Time series plots (span full width)
    ax_res = fig.add_subplot(gs_main[0, :])
    ax_res_slope = fig.add_subplot(gs_main[1, :], sharex=ax_res)
    ax_inl = fig.add_subplot(gs_main[2, :])
    ax_inl_slope = fig.add_subplot(gs_main[3, :], sharex=ax_inl)
    ax_jumps = fig.add_subplot(gs_main[4, :])
    ax_drops = fig.add_subplot(gs_main[5, :])

    # Distribution plots (2x2 grid in rows 6-7)
    ax_dist1 = fig.add_subplot(gs_main[6, 0])
    ax_dist2 = fig.add_subplot(gs_main[6, 1])
    ax_dist3 = fig.add_subplot(gs_main[7, 0])
    ax_dist4 = fig.add_subplot(gs_main[7, 1])

    # ----- 1) Residual and Residual Slope (paired) -----
    # Plot residual
    ax_res.plot(
        frames[valid_mask],
        mean_res[valid_mask],
        ".-",
        label="Mean Residual",
        color="tab:red",
        linewidth=1.0,
        markersize=3,
    )
    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_res.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)
    ax_res.set_ylabel("Mean Residual")
    ax_res.set_title("Residual Over Time")
    ax_res.grid(True, linestyle="--", alpha=0.6)
    ax_res.legend(loc="upper left")
    plt.setp(
        ax_res.get_xticklabels(), visible=False
    )  # Hide x-axis labels for shared axis

    # Plot residual slope
    valid_slope_mask = ~np.isnan(residual_slopes)
    if np.any(valid_slope_mask):
        ax_res_slope.plot(
            frames[valid_slope_mask],
            residual_slopes[valid_slope_mask],
            ".-",
            label="Residual Slope",
            color="tab:orange",
            linewidth=1.0,
            markersize=3,
        )
        ax_res_slope.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

        # Statistics
        slope_vals = residual_slopes[valid_slope_mask]
        print("\n[Residual Slope] Statistics:")
        print(
            "  mean={:.6f}, median={:.6f}, std={:.6f}, "
            "p90={:.6f}, max={:.6f}, min={:.6f}".format(
                float(np.mean(slope_vals)),
                float(np.median(slope_vals)),
                float(np.std(slope_vals)),
                float(np.percentile(slope_vals, 90)),
                float(np.max(slope_vals)),
                float(np.min(slope_vals)),
            )
        )

    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_res_slope.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)

    ax_res_slope.set_ylabel("Residual Slope\n(change per frame)")
    ax_res_slope.set_xlabel("Frame")
    ax_res_slope.grid(True, linestyle="--", alpha=0.6)
    ax_res_slope.legend(loc="upper left")

    # ----- 2) Inlier Count and Inlier Slope (paired) -----
    # Plot inlier count
    ax_inl.plot(
        frames,
        inlier_counts,
        ".-",
        label="Inlier Count",
        color="tab:green",
        linewidth=1.0,
        markersize=3,
    )
    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_inl.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)
    ax_inl.set_ylabel("# Inliers")
    ax_inl.set_title("Inlier Count Over Time")
    ax_inl.grid(True, linestyle="--", alpha=0.6)
    ax_inl.legend(loc="upper left")
    plt.setp(
        ax_inl.get_xticklabels(), visible=False
    )  # Hide x-axis labels for shared axis

    # Plot inlier slope
    valid_inl_slope_mask = ~np.isnan(inlier_slopes)
    if np.any(valid_inl_slope_mask):
        ax_inl_slope.plot(
            frames[valid_inl_slope_mask],
            inlier_slopes[valid_inl_slope_mask],
            ".-",
            label="Inlier Count Slope",
            color="tab:cyan",
            linewidth=1.0,
            markersize=3,
        )
        ax_inl_slope.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

        # Statistics
        inl_slope_vals = inlier_slopes[valid_inl_slope_mask]
        print("\n[Inlier Count Slope] Statistics:")
        print(
            "  mean={:.2f}, median={:.2f}, std={:.2f}, "
            "p90={:.2f}, max={:.2f}, min={:.2f}".format(
                float(np.mean(inl_slope_vals)),
                float(np.median(inl_slope_vals)),
                float(np.std(inl_slope_vals)),
                float(np.percentile(inl_slope_vals, 90)),
                float(np.max(inl_slope_vals)),
                float(np.min(inl_slope_vals)),
            )
        )

    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_inl_slope.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)

    ax_inl_slope.set_ylabel("Inlier Count Slope\n(change per frame)")
    ax_inl_slope.set_xlabel("Frame")
    ax_inl_slope.grid(True, linestyle="--", alpha=0.6)
    ax_inl_slope.legend(loc="upper left")

    # ----- 3) Residual Jumps -----
    valid_jump_mask = ~np.isnan(residual_jumps)
    if np.any(valid_jump_mask):
        jump_vals = residual_jumps[valid_jump_mask]
        # Plot absolute jumps
        abs_jumps = np.abs(jump_vals)

        ax_jumps.plot(
            frames[valid_jump_mask],
            jump_vals,
            ".-",
            label="Residual Jump",
            color="tab:orange",
            linewidth=1.0,
            markersize=3,
            alpha=0.7,
        )
        ax_jumps.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

        # Statistics
        print("\n[Residual Jumps] Statistics:")
        print(
            "  mean={:.6f}, median={:.6f}, std={:.6f}, "
            "p90={:.6f}, max={:.6f}, min={:.6f}".format(
                float(np.mean(jump_vals)),
                float(np.median(jump_vals)),
                float(np.std(jump_vals)),
                float(np.percentile(jump_vals, 90)),
                float(np.max(jump_vals)),
                float(np.min(jump_vals)),
            )
        )
        print(
            f"  Frames above threshold ({residual_jump_threshold:.4f}): "
            f"{np.sum(abs_jumps > residual_jump_threshold)} / {len(abs_jumps)} "
            f"({100.0 * np.sum(abs_jumps > residual_jump_threshold) / len(abs_jumps):.1f}%)"
        )

    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_jumps.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)

    ax_jumps.set_ylabel("Residual Jump\n(frame-to-frame change)")
    ax_jumps.set_xlabel("Frame")
    ax_jumps.set_title("Residual Jumps Over Time")
    ax_jumps.grid(True, linestyle="--", alpha=0.6)
    ax_jumps.legend(loc="upper left")

    # ----- 4) Inlier Drop Ratios -----
    valid_drop_mask = ~np.isnan(inlier_drop_ratios)
    if np.any(valid_drop_mask):
        drop_vals = inlier_drop_ratios[valid_drop_mask]

        ax_drops.plot(
            frames[valid_drop_mask],
            drop_vals,
            ".-",
            label="Inlier Drop Ratio",
            color="tab:purple",
            linewidth=1.0,
            markersize=3,
            alpha=0.7,
        )
        ax_drops.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

        # Statistics
        print("\n[Inlier Drop Ratios] Statistics:")
        print(
            "  mean={:.4f}, median={:.4f}, std={:.4f}, "
            "p90={:.4f}, max={:.4f}, min={:.4f}".format(
                float(np.mean(drop_vals)),
                float(np.median(drop_vals)),
                float(np.std(drop_vals)),
                float(np.percentile(drop_vals, 90)),
                float(np.max(drop_vals)),
                float(np.min(drop_vals)),
            )
        )
        print(
            f"  Frames above threshold ({inlier_drop_ratio:.2%}): "
            f"{np.sum(drop_vals > inlier_drop_ratio)} / {len(drop_vals)} "
            f"({100.0 * np.sum(drop_vals > inlier_drop_ratio) / len(drop_vals):.1f}%)"
        )

    if kf_indices is not None:
        for k in kf_indices:
            if 0 <= k < len(frames):
                ax_drops.axvline(x=k, color="k", alpha=0.15, linewidth=0.8)

    ax_drops.set_ylabel("Inlier Drop Ratio\n(1 - curr/prev)")
    ax_drops.set_xlabel("Frame")
    ax_drops.set_title("Inlier Drop Ratios Over Time")
    ax_drops.grid(True, linestyle="--", alpha=0.6)
    ax_drops.legend(loc="upper left")

    # ----- 5) Distribution Analysis (in same figure) -----

    # Distribution 1: Residuals
    valid_res = mean_res[valid_mask]
    if valid_res.size > 0:
        ax_dist1.hist(valid_res, bins=50, alpha=0.7, color="tab:red", edgecolor="black")
        ax_dist1.axvline(
            x=reg_residual_thres,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"reg_residual_thres ({reg_residual_thres:.4f})",
        )
        ax_dist1.axvline(
            x=high_residual_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"high_residual_threshold ({high_residual_threshold:.4f})",
        )
        ax_dist1.set_xlabel("Mean Residual")
        ax_dist1.set_ylabel("Count")
        ax_dist1.set_title("Distribution of Residuals")
        ax_dist1.legend(loc="upper right", fontsize=8)
        ax_dist1.grid(True, linestyle="--", alpha=0.3)

        # Print threshold statistics
        print("\n[Residual Threshold Analysis]:")
        print(
            f"  Frames with residual > {reg_residual_thres:.4f}: "
            f"{np.sum(valid_res > reg_residual_thres)} / {len(valid_res)} "
            f"({100.0 * np.sum(valid_res > reg_residual_thres) / len(valid_res):.1f}%)"
        )
        print(
            f"  Frames with residual > {high_residual_threshold:.4f}: "
            f"{np.sum(valid_res > high_residual_threshold)} / {len(valid_res)} "
            f"({100.0 * np.sum(valid_res > high_residual_threshold) / len(valid_res):.1f}%)"
        )

    # Distribution 2: Inlier counts
    valid_inliers = inlier_counts[inlier_counts > 0]
    if valid_inliers.size > 0:
        ax_dist2.hist(
            valid_inliers, bins=50, alpha=0.7, color="tab:green", edgecolor="black"
        )
        ax_dist2.axvline(
            x=min_inlier_count,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"min_inlier_count ({min_inlier_count})",
        )
        ax_dist2.set_xlabel("Inlier Count")
        ax_dist2.set_ylabel("Count")
        ax_dist2.set_title("Distribution of Inlier Counts")
        ax_dist2.legend(loc="upper right", fontsize=8)
        ax_dist2.grid(True, linestyle="--", alpha=0.3)

        # Print threshold statistics
        print("\n[Inlier Count Threshold Analysis]:")
        print(
            f"  Frames with inlier_count < {min_inlier_count}: "
            f"{np.sum(inlier_counts < min_inlier_count)} / {len(inlier_counts)} "
            f"({100.0 * np.sum(inlier_counts < min_inlier_count) / len(inlier_counts):.1f}%)"
        )

    # Distribution 3: Residual jumps (absolute)
    valid_abs_jumps = (
        np.abs(residual_jumps[valid_jump_mask])
        if np.any(valid_jump_mask)
        else np.array([])
    )
    if valid_abs_jumps.size > 0:
        ax_dist3.hist(
            valid_abs_jumps, bins=50, alpha=0.7, color="tab:orange", edgecolor="black"
        )
        ax_dist3.axvline(
            x=residual_jump_threshold,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"residual_jump_threshold ({residual_jump_threshold:.4f})",
        )
        ax_dist3.set_xlabel("Absolute Residual Jump")
        ax_dist3.set_ylabel("Count")
        ax_dist3.set_title("Distribution of Residual Jumps (abs)")
        ax_dist3.legend(loc="upper right", fontsize=8)
        ax_dist3.grid(True, linestyle="--", alpha=0.3)

    # Distribution 4: Inlier drop ratios
    valid_drops = (
        inlier_drop_ratios[valid_drop_mask] if np.any(valid_drop_mask) else np.array([])
    )
    if valid_drops.size > 0:
        ax_dist4.hist(
            valid_drops, bins=50, alpha=0.7, color="tab:purple", edgecolor="black"
        )
        ax_dist4.axvline(
            x=inlier_drop_ratio,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"inlier_drop_ratio ({inlier_drop_ratio:.2%})",
        )
        ax_dist4.set_xlabel("Inlier Drop Ratio")
        ax_dist4.set_ylabel("Count")
        ax_dist4.set_title("Distribution of Inlier Drop Ratios")
        ax_dist4.legend(loc="upper right", fontsize=8)
        ax_dist4.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Slope and Threshold Analysis", fontsize=16, y=0.995)

    if save_prefix:
        out_file = f"{save_prefix}_slope_and_threshold_analysis.png"
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        print(f"Saved slope+threshold analysis plot to {out_file}")

    plt.show()


# ---------------------- Plotting: Full Trajectory ---------------------- #
def plot_full_trajectory(
    frontend, local, global_, gt, keyframe_indices, save_prefix=None
):
    """
    Plot x, y, z trajectories and global errors over time.
    Still useful for sanity checks, but optional.
    """
    N = len(global_)
    frames = np.arange(N)

    fe_xyzq = se3_to_xyz_quat(frontend)
    lo_xyzq = se3_to_xyz_quat(local)
    gl_xyzq = se3_to_xyz_quat(global_)
    gt_xyzq = se3_to_xyz_quat(gt) if gt is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ["x", "y", "z"]

    # Filter keyframe indices to valid range and convert to numpy array
    if keyframe_indices is not None and len(keyframe_indices) > 0:
        keyframe_indices_arr = np.asarray(keyframe_indices, dtype=int)
        valid_kf_indices = keyframe_indices_arr[
            (keyframe_indices_arr >= 0) & (keyframe_indices_arr < N)
        ]
    else:
        keyframe_indices_arr = np.array([], dtype=int)
        valid_kf_indices = np.array([], dtype=int)

    for i in range(3):
        ax = axes[i]
        ax.plot(frames, fe_xyzq[:, i], label="Frontend", alpha=0.6, linestyle="--")
        ax.plot(frames, lo_xyzq[:, i], label="Local", alpha=0.6, linestyle=":")
        # Plot global optimization only at keyframes
        if len(valid_kf_indices) > 0:
            # Only add label on first subplot to avoid duplicate legend entries
            label = "Global (KF)" if i == 0 else ""
            ax.scatter(
                valid_kf_indices,
                gl_xyzq[valid_kf_indices, i],
                label=label,
                s=50,
                zorder=5,
                linewidths=1.5,
                edgecolors="darkorange",
                color="tab:orange",
                alpha=0.9,
            )
        if gt_xyzq is not None:
            ax.plot(frames, gt_xyzq[:N, i], label="GT", alpha=0.4)

        if keyframe_indices is not None:
            for kf_idx in keyframe_indices:
                if 0 <= kf_idx < N:
                    ax.axvline(x=kf_idx, alpha=0.2, linestyle="-", linewidth=1)

        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True)
        if i == 0:
            ax.legend()
            ax.set_title("Translation Trajectory Comparison")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    if save_prefix:
        out_file = f"{save_prefix}_trajectory.png"
        plt.savefig(out_file)
        print(f"Saved trajectory plot to {out_file}")
    plt.show()

    if gt is None:
        return

    # Errors over all frames
    fe_t_err, fe_r_err = compute_relative_errors(frontend, gt)
    lo_t_err, lo_r_err = compute_relative_errors(local, gt)
    gl_t_err, gl_r_err = compute_relative_errors(global_, gt)

    frames_err = np.arange(len(gl_t_err))

    # Translation error
    plt.figure(figsize=(10, 4))
    plt.plot(frames_err, fe_t_err, label="Frontend Err", alpha=0.5)
    plt.plot(frames_err, lo_t_err, label="Local Err", alpha=0.5)
    plt.plot(frames_err, gl_t_err, label="Global Err", linewidth=2)

    for kf_idx in keyframe_indices:
        if kf_idx < len(frames_err):
            plt.axvline(x=kf_idx, alpha=0.2)

    plt.legend()
    plt.title("Translation Error w.r.t GT (All Frames)")
    plt.ylabel("Error (m)")
    plt.xlabel("Frame")
    plt.grid(True)
    if save_prefix:
        out_file = f"{save_prefix}_error_trans_all.png"
        plt.savefig(out_file)
        print(f"Saved translation error plot to {out_file}")
    plt.show()

    # Rotation error
    plt.figure(figsize=(10, 4))
    plt.plot(frames_err, np.degrees(fe_r_err), label="Frontend Err", alpha=0.5)
    plt.plot(frames_err, np.degrees(lo_r_err), label="Local Err", alpha=0.5)
    plt.plot(frames_err, np.degrees(gl_r_err), label="Global Err", linewidth=2)

    for kf_idx in keyframe_indices:
        if kf_idx < len(frames_err):
            plt.axvline(x=kf_idx, alpha=0.2)

    plt.legend()
    plt.title("Rotation Error w.r.t GT (All Frames)")
    plt.ylabel("Error (deg)")
    plt.xlabel("Frame")
    plt.grid(True)
    if save_prefix:
        out_file = f"{save_prefix}_error_rot_all.png"
        plt.savefig(out_file)
        print(f"Saved rotation error plot to {out_file}")
    plt.show()


# ---------------------- Plotting: Keyframe-Only ---------------------- #
def plot_keyframe_errors(
    fe_t_err,
    fe_r_err,
    lo_t_err,
    lo_r_err,
    gl_t_err,
    gl_r_err,
    kf_indices,
    save_prefix=None,
):
    """
    Visualize errors ONLY at keyframes, with translation and rotation
    in a single figure (two stacked bar plots sharing x).

    For each keyframe index k:
      - show three bars: Frontend, Local, Global (KF)
    """
    if len(kf_indices) == 0:
        print("No keyframes found, skipping keyframe error plots.")
        return

    # keep only keyframes that lie within the error arrays
    max_len = min(
        len(fe_t_err),
        len(lo_t_err),
        len(gl_t_err),
        len(fe_r_err),
        len(lo_r_err),
        len(gl_r_err),
    )
    kf_indices = [k for k in kf_indices if k < max_len]

    if len(kf_indices) == 0:
        print("Keyframe indices are all out of bounds for error arrays.")
        return

    # gather per-keyframe errors
    fe_t_kf = fe_t_err[kf_indices]
    lo_t_kf = lo_t_err[kf_indices]
    gl_t_kf = gl_t_err[kf_indices]

    fe_r_kf = np.degrees(fe_r_err[kf_indices])
    lo_r_kf = np.degrees(lo_r_err[kf_indices])
    gl_r_kf = np.degrees(gl_r_err[kf_indices])

    x = np.arange(len(kf_indices))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax_t, ax_r = axes

    # --- Translation (top) ---
    ax_t.bar(x - width, fe_t_kf, width, label="Frontend")
    ax_t.bar(x, lo_t_kf, width, label="Local")
    ax_t.bar(x + width, gl_t_kf, width, label="Global (KF)")
    ax_t.set_ylabel("Translation Error (m)")
    ax_t.set_title("Pose Error at Keyframes (FE → Local → KF)")
    ax_t.grid(axis="y", linestyle="--", alpha=0.4)
    ax_t.legend(loc="upper left")

    # --- Rotation (bottom) ---
    ax_r.bar(x - width, fe_r_kf, width, label="Frontend")
    ax_r.bar(x, lo_r_kf, width, label="Local")
    ax_r.bar(x + width, gl_r_kf, width, label="Global (KF)")
    ax_r.set_ylabel("Rotation Error (deg)")
    ax_r.set_xlabel("Frame index (Keyframes)")
    ax_r.grid(axis="y", linestyle="--", alpha=0.4)

    # x-ticks = actual KF indices
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(kf_indices, rotation=45)

    plt.tight_layout()
    if save_prefix:
        out_file = f"{save_prefix}_kf_error_trans_rot.png"
        plt.savefig(out_file, dpi=200)
        print(f"Saved keyframe translation+rotation error plot to {out_file}")
    plt.show()


def analyze_keyframe_improvement(
    fe_t_err, lo_t_err, gl_t_err, fe_r_err, lo_r_err, gl_r_err, kf_indices
):
    """
    Print simple statistics showing how errors change FE -> Local -> Global on keyframes.
    Includes FE -> Local, Local -> Global, and FE -> Global comparisons.
    """
    if (
        fe_t_err is None
        or lo_t_err is None
        or gl_t_err is None
        or fe_r_err is None
        or lo_r_err is None
        or gl_r_err is None
        or len(kf_indices) == 0
    ):
        return

    kf_indices = np.asarray(kf_indices, dtype=int)
    max_len = min(
        len(fe_t_err),
        len(lo_t_err),
        len(gl_t_err),
        len(fe_r_err),
        len(lo_r_err),
        len(gl_r_err),
    )
    kf_indices = kf_indices[kf_indices < max_len]
    if kf_indices.size == 0:
        return

    fe_t = fe_t_err[kf_indices]
    lo_t = lo_t_err[kf_indices]
    gl_t = gl_t_err[kf_indices]
    fe_r = fe_r_err[kf_indices]
    lo_r = lo_r_err[kf_indices]
    gl_r = gl_r_err[kf_indices]

    def _improvement_rate(before, after):
        better = np.sum(after < before)
        worse = np.sum(after > before)
        same = np.sum(after == before)
        total = float(before.size)
        return dict(
            better=int(better),
            worse=int(worse),
            same=int(same),
            better_pct=100.0 * better / total if total > 0 else np.nan,
            worse_pct=100.0 * worse / total if total > 0 else np.nan,
        )

    print("\n=== Keyframe Improvement Analysis ===")
    t_fe_to_lo = _improvement_rate(fe_t, lo_t)
    t_lo_to_gl = _improvement_rate(lo_t, gl_t)
    t_fe_to_gl = _improvement_rate(fe_t, gl_t)
    r_fe_to_lo = _improvement_rate(fe_r, lo_r)
    r_lo_to_gl = _improvement_rate(lo_r, gl_r)
    r_fe_to_gl = _improvement_rate(fe_r, gl_r)

    print("Translation FE → Local:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**t_fe_to_lo)
    )
    print("Translation Local → Global:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**t_lo_to_gl)
    )
    print("Translation FE → Global:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**t_fe_to_gl)
    )
    print("Rotation FE → Local:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**r_fe_to_lo)
    )
    print("Rotation Local → Global:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**r_lo_to_gl)
    )
    print("Rotation FE → Global:")
    print(
        "  better={better} ({better_pct:.1f}%), "
        "worse={worse} ({worse_pct:.1f}%), same={same}".format(**r_fe_to_gl)
    )


def plot_pose_components_vs_gt(
    pose_frontend,
    pose_local,
    pose_global,
    gt_poses,
    kf_indices=None,
    save_prefix=None,
):
    """
    Plot x, y, z, roll, pitch, yaw components for frontend, local, and global
    with respect to ground truth.

    Frontend and local are plotted as solid lines, global as dots.
    Ground truth is plotted as solid reference lines.

    Args:
        pose_frontend: (N, 4, 4) array of frontend poses
        pose_local: (N, 4, 4) array of local optimization poses
        pose_global: (N, 4, 4) array of global optimization poses
        gt_poses: (N, 4, 4) array of ground truth poses
        kf_indices: Optional array of keyframe indices to mark
        save_prefix: Optional prefix for saving the plot
    """
    if gt_poses is None:
        print("Warning: No GT poses available, skipping pose components plot.")
        return

    # Convert all poses to xyz + rpy
    fe_xyzrpy = se3_to_xyz_rpy(pose_frontend)
    lo_xyzrpy = se3_to_xyz_rpy(pose_local)
    gl_xyzrpy = se3_to_xyz_rpy(pose_global)
    gt_xyzrpy = se3_to_xyz_rpy(gt_poses)

    # Align lengths
    min_len = min(len(fe_xyzrpy), len(lo_xyzrpy), len(gl_xyzrpy), len(gt_xyzrpy))
    fe_xyzrpy = fe_xyzrpy[:min_len]
    lo_xyzrpy = lo_xyzrpy[:min_len]
    gl_xyzrpy = gl_xyzrpy[:min_len]
    gt_xyzrpy = gt_xyzrpy[:min_len]

    frames = np.arange(min_len)

    # Filter keyframe indices to valid range and convert to numpy array
    if kf_indices is not None and len(kf_indices) > 0:
        kf_indices_arr = np.asarray(kf_indices, dtype=int)
        valid_kf_indices = kf_indices_arr[
            (kf_indices_arr >= 0) & (kf_indices_arr < min_len)
        ]
    else:
        kf_indices_arr = np.array([], dtype=int)
        valid_kf_indices = np.array([], dtype=int)

    # Component names and labels
    component_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    component_labels = [
        "x (m)",
        "y (m)",
        "z (m)",
        "roll (rad)",
        "pitch (rad)",
        "yaw (rad)",
    ]

    # Create figure with 6 vertical subplots (6 rows, 1 column)
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    # Define better colors (avoid pure red and black)
    color_gt = "#2C3E50"  # Dark slate gray
    color_frontend = "#3498DB"  # Nice blue
    color_local = "#27AE60"  # Nice green
    color_global = "#E74C3C"  # Muted red/coral

    for i, (name, label) in enumerate(zip(component_names, component_labels)):
        ax = axes[i]

        # Plot ground truth as solid reference line
        ax.plot(
            frames,
            gt_xyzrpy[:, i],
            label="GT",
            color=color_gt,
            linewidth=2.0,
            linestyle="-",
            alpha=0.8,
        )

        # Plot frontend as solid line
        ax.plot(
            frames,
            fe_xyzrpy[:, i],
            label="Frontend",
            color=color_frontend,
            linewidth=1.8,
            linestyle="-",
            alpha=0.75,
        )

        # Plot local optimization as solid line
        ax.plot(
            frames,
            lo_xyzrpy[:, i],
            label="Local",
            color=color_local,
            linewidth=1.8,
            linestyle="-",
            alpha=0.75,
        )

        # Plot global optimization as dots only at keyframes
        if len(valid_kf_indices) > 0:
            # Only add label on first subplot to avoid duplicate legend entries
            label_global = "Global" if i == 0 else ""
            ax.scatter(
                valid_kf_indices,
                gl_xyzrpy[valid_kf_indices, i],
                label=label_global,
                color=color_global,
                s=40,
                alpha=0.85,
                marker="o",
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
            )

        # Mark keyframes if provided
        if kf_indices is not None:
            for k in kf_indices:
                if 0 <= k < len(frames):
                    ax.axvline(
                        x=k, color="gray", alpha=0.2, linewidth=0.8, linestyle="-"
                    )

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{name.upper()} Component", fontsize=12, pad=5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    # Set x-label on bottom subplot only
    axes[-1].set_xlabel("Frame", fontsize=11)

    plt.suptitle(
        "Pose Components: Frontend, Local, Global vs. Ground Truth",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    if save_prefix:
        out_file = f"{save_prefix}_pose_components_vs_gt.png"
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        print(f"Saved pose components plot to {out_file}")

    plt.show()


def filter_keyframes_with_gt(kf_indices, gt_poses):
    """
    Return only the subset of keyframe indices that have valid GT.
    GT entries that are identity (0) or None are treated as missing.
    """
    valid_kf = []
    for k in kf_indices:
        if k < 0 or k >= len(gt_poses):
            continue
        G = gt_poses[k]

        # GT may be None, identity, or zeros -> treat as missing
        if G is None:
            continue
        if not isinstance(G, np.ndarray) or G.shape != (4, 4):
            continue
        if np.allclose(G, np.eye(4)):
            # identity might be valid GT OR might mean missing
            # but you said missing GT comes as identity → skip it
            continue

        valid_kf.append(k)

    return np.array(valid_kf, dtype=int)


# ---------------------- Dense Recovery Analysis ---------------------- #
def plot_dense_recovery_analysis(
    data,
    pose_frontend,
    pose_global,
    gt_poses=None,
    save_prefix=None,
):
    """
    Plot dense recovery analysis showing before/after stats including pose errors.

    Layout:
      - Row 1: Translation error before/after dense recovery
      - Row 2: Rotation error before/after dense recovery
      - Row 3: Residual comparison before/after
      - Row 4: Inlier count comparison before/after
    """
    # Extract dense recovery data
    if "dense_recovery_triggered" not in data:
        print(
            "Warning: 'dense_recovery_triggered' not found in logs. "
            "This could mean:\n"
            "  1. The logs were generated before dense recovery logging was added\n"
            "  2. Dense recovery is not enabled in the pipeline configuration\n"
            "  3. The pipeline hasn't been run yet with the new logging code\n"
            "Skipping dense recovery analysis."
        )
        # Print available keys for debugging
        available_keys = list(data.keys())
        dense_keys = [k for k in available_keys if "dense" in k.lower()]
        if dense_keys:
            print(f"Found dense-related keys: {dense_keys}")
        else:
            print(
                f"No dense-related keys found. Available keys (first 20): {sorted(available_keys)[:20]}..."
            )
        return

    # Extract dense recovery trigger flags
    triggered_raw = data["dense_recovery_triggered"]
    if isinstance(triggered_raw, np.ndarray):
        if triggered_raw.dtype == object:
            dense_triggered = np.array(
                [bool(x) if x is not None else False for x in triggered_raw]
            )
        else:
            dense_triggered = triggered_raw.astype(bool)
    else:
        dense_triggered = np.array(
            [bool(x) if x is not None else False for x in triggered_raw]
        )

    # Extract pose arrays - try direct first, then unpack_ragged
    dense_pose_before = None
    dense_pose_after = None
    if "dense_recovery_pose_before" in data:
        try:
            dense_pose_before = ensure_numeric(
                data["dense_recovery_pose_before"], "dense_recovery_pose_before"
            )
        except Exception:
            dense_pose_before = unpack_ragged(data, "dense_recovery_pose_before")
    else:
        dense_pose_before = unpack_ragged(data, "dense_recovery_pose_before")

    if "dense_recovery_pose_after" in data:
        try:
            dense_pose_after = ensure_numeric(
                data["dense_recovery_pose_after"], "dense_recovery_pose_after"
            )
        except Exception:
            dense_pose_after = unpack_ragged(data, "dense_recovery_pose_after")
    else:
        dense_pose_after = unpack_ragged(data, "dense_recovery_pose_after")

    # Get frames where dense recovery was triggered
    recovery_frames = np.where(dense_triggered)[0]

    if len(recovery_frames) == 0:
        print("No dense recovery events found in logs.")
        return

    print(
        f"\nFound {len(recovery_frames)} frames with dense recovery triggered: {recovery_frames}"
    )

    # Extract residuals and inliers
    dense_residuals_before = unpack_ragged(data, "dense_recovery_residuals_before")
    dense_residuals_after = unpack_ragged(data, "dense_recovery_residuals_after")
    dense_inliers_before = unpack_ragged(data, "dense_recovery_inliers_before")
    dense_inliers_after = unpack_ragged(data, "dense_recovery_inliers_after")

    # Compute stats for recovery frames
    N = len(dense_triggered)
    mean_res_before = np.full(N, np.nan)
    mean_res_after = np.full(N, np.nan)
    inlier_count_before = np.zeros(N, dtype=int)
    inlier_count_after = np.zeros(N, dtype=int)

    for i in recovery_frames:
        if dense_residuals_before is not None and i < len(dense_residuals_before):
            res_before = dense_residuals_before[i]
            if res_before is not None and len(res_before) > 0:
                inl_before = None
                if dense_inliers_before is not None and i < len(dense_inliers_before):
                    inl_before = np.asarray(dense_inliers_before[i], dtype=bool)
                    if inl_before.size == res_before.size:
                        mean_res_before[i] = (
                            np.mean(res_before[inl_before])
                            if np.any(inl_before)
                            else np.mean(res_before)
                        )
                        inlier_count_before[i] = np.sum(inl_before)
                    else:
                        mean_res_before[i] = np.mean(res_before)
                        inlier_count_before[i] = len(res_before)
                else:
                    mean_res_before[i] = np.mean(res_before)
                    inlier_count_before[i] = len(res_before)

        if dense_residuals_after is not None and i < len(dense_residuals_after):
            res_after = dense_residuals_after[i]
            if res_after is not None and len(res_after) > 0:
                inl_after = None
                if dense_inliers_after is not None and i < len(dense_inliers_after):
                    inl_after = np.asarray(dense_inliers_after[i], dtype=bool)
                    if inl_after.size == res_after.size:
                        mean_res_after[i] = (
                            np.mean(res_after[inl_after])
                            if np.any(inl_after)
                            else np.mean(res_after)
                        )
                        inlier_count_after[i] = np.sum(inl_after)
                    else:
                        mean_res_after[i] = np.mean(res_after)
                        inlier_count_after[i] = len(res_after)
                else:
                    mean_res_after[i] = np.mean(res_after)
                    inlier_count_after[i] = len(res_after)

    # Compute pose errors if GT available
    t_err_before = None
    t_err_after = None
    r_err_before = None
    r_err_after = None

    if gt_poses is not None:
        t_err_before = np.full(N, np.nan)
        t_err_after = np.full(N, np.nan)
        r_err_before = np.full(N, np.nan)
        r_err_after = np.full(N, np.nan)

        for i in recovery_frames:
            if i >= len(gt_poses):
                continue

            gt = gt_poses[i]
            if gt is None or np.allclose(gt, np.eye(4)):
                continue

            # Before dense recovery
            pose_b = None
            if dense_pose_before is not None:
                if isinstance(dense_pose_before, (list, np.ndarray)) and i < len(
                    dense_pose_before
                ):
                    pose_b = dense_pose_before[i]
                    if isinstance(pose_b, np.ndarray) and pose_b.shape == (4, 4):
                        rel_b = np.linalg.inv(gt) @ pose_b
                        t_err_before[i] = np.linalg.norm(rel_b[:3, 3])
                        r_mat_b = rel_b[:3, :3]
                        r_b = R.from_matrix(r_mat_b)
                        r_err_before[i] = np.linalg.norm(r_b.as_rotvec())

            # After dense recovery
            pose_a = None
            if dense_pose_after is not None:
                if isinstance(dense_pose_after, (list, np.ndarray)) and i < len(
                    dense_pose_after
                ):
                    pose_a = dense_pose_after[i]
                    if isinstance(pose_a, np.ndarray) and pose_a.shape == (4, 4):
                        rel_a = np.linalg.inv(gt) @ pose_a
                        t_err_after[i] = np.linalg.norm(rel_a[:3, 3])
                        r_mat_a = rel_a[:3, :3]
                        r_a = R.from_matrix(r_mat_a)
                        r_err_after[i] = np.linalg.norm(r_a.as_rotvec())

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Translation Error
    ax_t = axes[0]
    if t_err_before is not None and t_err_after is not None:
        valid_mask = ~(
            np.isnan(t_err_before[recovery_frames])
            | np.isnan(t_err_after[recovery_frames])
        )
        valid_frames = recovery_frames[valid_mask]
        if len(valid_frames) > 0:
            x = np.arange(len(valid_frames))
            ax_t.bar(
                x - 0.2,
                t_err_before[valid_frames],
                0.4,
                label="Before Dense Recovery",
                color="tab:red",
                alpha=0.7,
            )
            ax_t.bar(
                x + 0.2,
                t_err_after[valid_frames],
                0.4,
                label="After Dense Recovery",
                color="tab:green",
                alpha=0.7,
            )
            ax_t.set_ylabel("Translation Error (m)")
            ax_t.set_title("Translation Error: Before vs. After Dense Recovery")
            ax_t.set_xticks(x)
            ax_t.set_xticklabels(valid_frames, rotation=45)
            ax_t.legend()
            ax_t.grid(True, linestyle="--", alpha=0.6)

            # Print improvement stats
            improvements = t_err_after[valid_frames] < t_err_before[valid_frames]
            print(f"\n[Dense Recovery] Translation Error:")
            print(
                f"  Improved: {np.sum(improvements)}/{len(valid_frames)} ({100*np.sum(improvements)/len(valid_frames):.1f}%)"
            )
            if len(valid_frames) > 0:
                print(f"  Mean before: {np.mean(t_err_before[valid_frames]):.4f} m")
                print(f"  Mean after: {np.mean(t_err_after[valid_frames]):.4f} m")
                print(
                    f"  Mean improvement: {np.mean(t_err_before[valid_frames] - t_err_after[valid_frames]):.4f} m"
                )
        else:
            ax_t.text(
                0.5,
                0.5,
                "No valid GT data for dense recovery frames",
                ha="center",
                va="center",
                transform=ax_t.transAxes,
            )
    else:
        ax_t.text(
            0.5,
            0.5,
            "No GT available for pose error computation",
            ha="center",
            va="center",
            transform=ax_t.transAxes,
        )
    ax_t.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Rotation Error
    ax_r = axes[1]
    if r_err_before is not None and r_err_after is not None:
        valid_mask = ~(
            np.isnan(r_err_before[recovery_frames])
            | np.isnan(r_err_after[recovery_frames])
        )
        valid_frames = recovery_frames[valid_mask]
        if len(valid_frames) > 0:
            x = np.arange(len(valid_frames))
            ax_r.bar(
                x - 0.2,
                np.degrees(r_err_before[valid_frames]),
                0.4,
                label="Before Dense Recovery",
                color="tab:red",
                alpha=0.7,
            )
            ax_r.bar(
                x + 0.2,
                np.degrees(r_err_after[valid_frames]),
                0.4,
                label="After Dense Recovery",
                color="tab:green",
                alpha=0.7,
            )
            ax_r.set_ylabel("Rotation Error (deg)")
            ax_r.set_title("Rotation Error: Before vs. After Dense Recovery")
            ax_r.set_xticks(x)
            ax_r.set_xticklabels(valid_frames, rotation=45)
            ax_r.legend()
            ax_r.grid(True, linestyle="--", alpha=0.6)

            # Print improvement stats
            improvements = r_err_after[valid_frames] < r_err_before[valid_frames]
            print(f"\n[Dense Recovery] Rotation Error:")
            print(
                f"  Improved: {np.sum(improvements)}/{len(valid_frames)} ({100*np.sum(improvements)/len(valid_frames):.1f}%)"
            )
            if len(valid_frames) > 0:
                print(
                    f"  Mean before: {np.mean(np.degrees(r_err_before[valid_frames])):.2f} deg"
                )
                print(
                    f"  Mean after: {np.mean(np.degrees(r_err_after[valid_frames])):.2f} deg"
                )
                print(
                    f"  Mean improvement: {np.mean(np.degrees(r_err_before[valid_frames] - r_err_after[valid_frames])):.2f} deg"
                )
        else:
            ax_r.text(
                0.5,
                0.5,
                "No valid GT data for dense recovery frames",
                ha="center",
                va="center",
                transform=ax_r.transAxes,
            )
    else:
        ax_r.text(
            0.5,
            0.5,
            "No GT available for pose error computation",
            ha="center",
            va="center",
            transform=ax_r.transAxes,
        )
    ax_r.grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Residual Comparison
    ax_res = axes[2]
    valid_mask = ~(
        np.isnan(mean_res_before[recovery_frames])
        | np.isnan(mean_res_after[recovery_frames])
    )
    valid_frames = recovery_frames[valid_mask]
    if len(valid_frames) > 0:
        x = np.arange(len(valid_frames))
        ax_res.bar(
            x - 0.2,
            mean_res_before[valid_frames],
            0.4,
            label="Before Dense Recovery",
            color="tab:red",
            alpha=0.7,
        )
        ax_res.bar(
            x + 0.2,
            mean_res_after[valid_frames],
            0.4,
            label="After Dense Recovery",
            color="tab:green",
            alpha=0.7,
        )
        ax_res.set_ylabel("Mean Residual")
        ax_res.set_title("Registration Residual: Before vs. After Dense Recovery")
        ax_res.set_xticks(x)
        ax_res.set_xticklabels(valid_frames, rotation=45)
        ax_res.legend()
        ax_res.grid(True, linestyle="--", alpha=0.6)

        # Print improvement stats
        improvements = mean_res_after[valid_frames] < mean_res_before[valid_frames]
        print(f"\n[Dense Recovery] Residual:")
        print(
            f"  Improved: {np.sum(improvements)}/{len(valid_frames)} ({100*np.sum(improvements)/len(valid_frames):.1f}%)"
        )
        if len(valid_frames) > 0:
            print(f"  Mean before: {np.mean(mean_res_before[valid_frames]):.6f}")
            print(f"  Mean after: {np.mean(mean_res_after[valid_frames]):.6f}")
            print(
                f"  Mean improvement: {np.mean(mean_res_before[valid_frames] - mean_res_after[valid_frames]):.6f}"
            )
    else:
        ax_res.text(
            0.5,
            0.5,
            "No valid residual data for dense recovery frames",
            ha="center",
            va="center",
            transform=ax_res.transAxes,
        )
    ax_res.grid(True, linestyle="--", alpha=0.6)

    # Plot 4: Inlier Count Comparison
    ax_inl = axes[3]
    valid_mask = (inlier_count_before[recovery_frames] > 0) | (
        inlier_count_after[recovery_frames] > 0
    )
    valid_frames = recovery_frames[valid_mask]
    if len(valid_frames) > 0:
        x = np.arange(len(valid_frames))
        ax_inl.bar(
            x - 0.2,
            inlier_count_before[valid_frames],
            0.4,
            label="Before Dense Recovery",
            color="tab:red",
            alpha=0.7,
        )
        ax_inl.bar(
            x + 0.2,
            inlier_count_after[valid_frames],
            0.4,
            label="After Dense Recovery",
            color="tab:green",
            alpha=0.7,
        )
        ax_inl.set_ylabel("# Inliers")
        ax_inl.set_xlabel("Frame Index")
        ax_inl.set_title("Inlier Count: Before vs. After Dense Recovery")
        ax_inl.set_xticks(x)
        ax_inl.set_xticklabels(valid_frames, rotation=45)
        ax_inl.legend()
        ax_inl.grid(True, linestyle="--", alpha=0.6)

        # Print improvement stats
        improvements = (
            inlier_count_after[valid_frames] > inlier_count_before[valid_frames]
        )
        print(f"\n[Dense Recovery] Inlier Count:")
        print(
            f"  Improved: {np.sum(improvements)}/{len(valid_frames)} ({100*np.sum(improvements)/len(valid_frames):.1f}%)"
        )
        if len(valid_frames) > 0:
            print(f"  Mean before: {np.mean(inlier_count_before[valid_frames]):.1f}")
            print(f"  Mean after: {np.mean(inlier_count_after[valid_frames]):.1f}")
            print(
                f"  Mean improvement: {np.mean(inlier_count_after[valid_frames] - inlier_count_before[valid_frames]):.1f}"
            )
    else:
        ax_inl.text(
            0.5,
            0.5,
            "No valid inlier data for dense recovery frames",
            ha="center",
            va="center",
            transform=ax_inl.transAxes,
        )
    ax_inl.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Dense Recovery Analysis: Before vs. After", fontsize=16, y=0.995)
    plt.tight_layout()

    if save_prefix:
        out_file = f"{save_prefix}_dense_recovery_analysis.png"
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        print(f"Saved dense recovery analysis plot to {out_file}")

    plt.show()


# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize pose debug for pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to meta_data.npz (optional if results_dir and video_name are provided)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Results directory (e.g., /path/to/results/ho3d_single). Used with video_name to find meta_data in new structure.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ho3d", "ycbineoat", "ycbinisaac"],
        default="ho3d",
        help="Dataset type: 'ho3d', 'ycbineoat', or 'ycbinisaac' (default: ho3d)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/justin/data/HO3D_V3/evaluation",
        help="Path to dataset root (HO3D root or YCBInEoat root containing video folders)",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        help="Video name for GT loading (also used to find meta_data if results_dir is provided)",
    )
    parser.add_argument(
        "--align_gt",
        action="store_true",
        default=True,
        help="Align GT to first frame prediction",
    )
    parser.add_argument(
        "--save_prefix", type=str, help="Prefix for saving output plots (optional)"
    )
    parser.add_argument(
        "--plot_full_trajectory",
        action="store_true",
        help="Also plot full trajectory & per-frame error curves",
    )

    args = parser.parse_args()

    _set_plot_style()

    # Determine log_path: use provided path or construct from results_dir and video_name
    log_path = args.log_path
    if log_path is None:
        if args.results_dir and args.video_name:
            log_path = os.path.join(
                args.results_dir, args.video_name, "meta_data", "meta_data.npz"
            )
            print(f"Constructed log_path from results_dir and video_name: {log_path}")
        else:
            print(
                "Error: Either --log_path must be provided, or both --results_dir and --video_name must be provided"
            )
            sys.exit(1)

    # 1. Load Logs
    print(f"Loading logs from {log_path}...")
    data = load_logs(log_path)

    # Extract poses
    if "pose_frontend" not in data:
        print(
            "Error: 'pose_frontend' not found in logs. "
            "Make sure you updated ModularPipeline._log_step to log it."
        )
        sys.exit(1)

    pose_frontend = ensure_numeric(data["pose_frontend"], "pose_frontend")
    pose_local = ensure_numeric(data["pose_local"], "pose_local")
    pose_global = ensure_numeric(data["obj_pose"], "obj_pose")

    print(f"pose_frontend shape: {pose_frontend.shape}, dtype: {pose_frontend.dtype}")
    print(f"pose_local shape: {pose_local.shape}, dtype: {pose_local.dtype}")
    print(f"pose_global shape: {pose_global.shape}, dtype: {pose_global.dtype}")

    # Keyframes
    if "is_key_frame" in data:
        is_key_frame = data["is_key_frame"]
    elif "is_key_frame_data" in data:
        is_key_frame = data["is_key_frame_data"]
    else:
        print("Warning: is_key_frame info not found, assuming no keyframes.")
        is_key_frame = np.zeros(len(pose_global), dtype=bool)

    kf_indices = np.where(is_key_frame)[0]
    print(f"Found {len(kf_indices)} keyframes at indices: {kf_indices}")

    for k in kf_indices:
        T_lo = pose_local[k]
        T_gl = pose_global[k]
        rel = np.linalg.inv(T_lo) @ T_gl

        t = np.linalg.norm(rel[:3, 3])
        r = R.from_matrix(rel[:3, :3])
        ang = np.linalg.norm(r.as_rotvec())
        print(f"KF {k}: Local→Global Δt={t:.4f} m, ΔR={np.degrees(ang):.2f} deg")

    # 2. Load GT if provided
    gt_poses = None
    if args.data_path and args.video_name:
        missing_reader = (
            (args.dataset == "ho3d" and Ho3dReader is None)
            or (args.dataset == "ycbineoat" and YcbineoatReader is None)
            or (args.dataset == "ycbinisaac" and YCBInIsaacReader is None)
        )
        if missing_reader:
            print(
                "Warning: Dataset readers could not be imported (missing optional deps like cv2). "
                "Skipping GT load."
            )
            gt_poses = None
        elif not os.path.exists(args.data_path):
            print(
                f"Warning: Data path {args.data_path} does not exist. Skipping GT load."
            )
        else:
            print(
                f"Loading GT for {args.video_name} from {args.data_path} (dataset: {args.dataset})..."
            )

            # Determine video path based on dataset type
            if args.dataset == "ho3d":
                # typical HO3D layout: root/evaluation/VIDEO_NAME or root/VIDEO_NAME
                video_path = os.path.join(args.data_path, "evaluation", args.video_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(args.data_path, args.video_name)
            elif args.dataset == "ycbineoat":
                # YCBInEoat layout: root/VIDEO_NAME
                video_path = os.path.join(args.data_path, args.video_name)
            elif args.dataset == "ycbinisaac":
                # YCBInIsaac layout: root/VIDEO_NAME
                video_path = os.path.join(args.data_path, args.video_name)
            else:
                print(f"Error: Unknown dataset type: {args.dataset}")
                video_path = None

            if video_path is None or not os.path.exists(video_path):
                print(
                    f"Warning: Video path {video_path} does not exist. Skipping GT load."
                )
            else:
                try:
                    # Initialize reader based on dataset type
                    if args.dataset == "ho3d":
                        reader = Ho3dReader(video_path, args.data_path)
                        reader_name = "Ho3dReader"
                    elif args.dataset == "ycbineoat":
                        reader = YcbineoatReader(video_path)
                        reader_name = "YcbineoatReader"
                    elif args.dataset == "ycbinisaac":
                        reader = YCBInIsaacReader(video_path)
                        reader_name = "YCBInIsaacReader"
                    else:
                        raise ValueError(f"Unknown dataset type: {args.dataset}")

                    if len(reader) == 0:
                        print(
                            f"Warning: {reader_name} found no frames. Skipping GT load."
                        )
                    else:
                        frame_ids = data["frame_id"]
                        gt_list = []
                        gt_valid = []

                        for fid in frame_ids:
                            idx = int(fid)
                            if idx < 0 or idx >= len(reader):
                                # no GT available
                                gt_list.append(np.eye(4))
                                gt_valid.append(False)
                            else:
                                pose = reader.get_gt_pose(idx)
                                if pose is None:
                                    # no GT available for this frame
                                    gt_list.append(np.eye(4))
                                    gt_valid.append(False)
                                else:
                                    gt_list.append(pose)
                                    gt_valid.append(True)

                        gt_poses = np.array(gt_list)
                        gt_valid = np.array(gt_valid, dtype=bool)

                        if args.align_gt and gt_valid.any():
                            # Align using the FIRST frame that actually has GT
                            first_valid = int(np.where(gt_valid)[0][0])
                            T_align = pose_global[first_valid] @ inverse_SE3(
                                gt_poses[first_valid]
                            )

                            aligned_list = []
                            for G, v in zip(gt_poses, gt_valid):
                                if v:
                                    aligned_list.append(G @ T_align)
                                else:
                                    # keep as identity placeholder, still marked invalid
                                    aligned_list.append(G)
                            gt_poses = np.array(aligned_list)

                            print(
                                f"Aligned GT to first valid frame (idx={first_valid})."
                            )
                except Exception as e:
                    print(f"Failed to load GT: {e}")
                    import traceback

                    traceback.print_exc()
                    gt_poses = None

    if gt_poses is None:
        print("No GT poses loaded; will only visualize trajectories (no errors).")

    # 3. Compute errors (if GT exists)
    fe_t_err = lo_t_err = gl_t_err = fe_r_err = lo_r_err = gl_r_err = None
    if gt_poses is not None:
        fe_t_err, fe_r_err = compute_relative_errors(pose_frontend, gt_poses)
        lo_t_err, lo_r_err = compute_relative_errors(pose_local, gt_poses)
        gl_t_err, gl_r_err = compute_relative_errors(pose_global, gt_poses)

        print("\n========== Global Error Summary ==========")
        summarize_error_stats("Frontend", fe_t_err, fe_r_err)
        summarize_error_stats("Local", lo_t_err, lo_r_err)
        summarize_error_stats("Global (KF)", gl_t_err, gl_r_err)

        # Optional: reprojection error vs pose error plot (if reprojection logs exist)
        reproj_err_series = _extract_reproj_error_series(data, name="reproj_error")
        if reproj_err_series is not None:
            print("\n========== Reprojection Error vs Pose Error ==========")
            plot_reprojection_error_vs_pose_error(
                reproj_err_series,
                gl_t_err=gl_t_err,
                gl_r_err=gl_r_err,
                save_prefix=args.save_prefix,
            )
        else:
            # Help debug when the figure does not appear
            reproj_like_keys = [k for k in data.keys() if "reproj" in str(k).lower()]
            if reproj_like_keys:
                print(
                    "Reprojection figure skipped: no usable 'reproj_error' series found.\n"
                    f"Found reprojection-like keys in NPZ instead: {reproj_like_keys}"
                )
            else:
                print(
                    "Reprojection figure skipped: NPZ does not contain a 'reproj_error' "
                    "series or any reprojection-like keys."
                )

        # Selected-cluster: reproj_error/3d_dist vs global error (requires clustered logs)
        print("\n========== Checking for selected-cluster metrics ==========")
        if "reg_clusters" in data and "reg_best_cluster_idx" in data:
            # Debug: inspect first few clusters to see what keys they have
            clusters_arr = data["reg_clusters"]
            try:
                best_idx_raw = data["reg_best_cluster_idx"]
                # Handle None values in best_idx array
                best_idx_arr = []
                for val in best_idx_raw:
                    if val is None:
                        best_idx_arr.append(-1)  # Use -1 as sentinel for "no cluster"
                    else:
                        try:
                            best_idx_arr.append(int(val))
                        except (ValueError, TypeError):
                            best_idx_arr.append(-1)
                best_idx_arr = np.asarray(best_idx_arr, dtype=int)
            except Exception as e:
                print(f"  Warning: Could not parse reg_best_cluster_idx: {e}")
                best_idx_arr = None

            if best_idx_arr is not None:
                sample_keys = set()
                for i in range(min(5, len(clusters_arr))):
                    clusters_i = _as_py(clusters_arr[i])
                    if clusters_i is None:
                        continue
                    if isinstance(clusters_i, dict):
                        clusters_i = [clusters_i]
                    if isinstance(clusters_i, list) and clusters_i:
                        idx = (
                            int(best_idx_arr[i])
                            if i < len(best_idx_arr) and best_idx_arr[i] >= 0
                            else 0
                        )
                        if 0 <= idx < len(clusters_i):
                            c = clusters_i[idx]
                            if isinstance(c, dict):
                                sample_keys.update(c.keys())
                if sample_keys:
                    print(f"  Sample cluster keys found: {sorted(sample_keys)}")

        # Try reproj_error first, fall back to mean_res if not available
        # Extract 3d_dist (always plot this)
        best_3d = _extract_best_cluster_metric_series(data, "3d_dist")

        # Try to extract reproj_error for optional reproj mode
        best_reproj = _extract_best_cluster_metric_series(data, "reproj_error")
        has_reproj = best_reproj is not None

        # Default: plot 3d_dist, but allow switching to reproj_error if available
        plot_metric = best_3d
        metric_label = "3d_dist"
        use_reproj_mode = False

        # If reproj_error is available, use it instead
        if has_reproj:
            print("  reproj_error found in clusters, using reproj_error mode...")
            plot_metric = best_reproj
            metric_label = "reproj_error"
            use_reproj_mode = True
        elif best_3d is not None:
            print("  Using 3d_dist mode (reproj_error not available)...")

        if plot_metric is not None:
            print(
                f"\n========== Selected Cluster: {metric_label} Over Frames =========="
            )
            print(
                f"  Extracted {metric_label}: {np.sum(np.isfinite(plot_metric))}/{len(plot_metric)} finite values"
            )
            plot_selected_cluster_metric_vs_frames(
                plot_metric,
                gl_t_err=gl_t_err,
                gl_r_err=gl_r_err,
                metric_label=metric_label,
                save_prefix=args.save_prefix,
            )
        else:
            # Keep this quiet unless cluster keys exist (avoid noise for non-cluster runs)
            if "reg_clusters" in data:
                has_best_idx = "reg_best_cluster_idx" in data
                print(
                    f"Selected-cluster reproj/3d_dist plot skipped: "
                    f"best_reproj={'None' if best_reproj is None else 'OK'}, "
                    f"best_3d={'None' if best_3d is None else 'OK'}, "
                    f"has reg_best_cluster_idx={has_best_idx}"
                )

    # 4. Residual Analysis (always try; overlays errors if present)
    print("Extracting registration residuals...")
    reg_residuals = unpack_ragged(data, "reg_residuals")

    if reg_residuals is not None:
        reg_inliers = unpack_ragged(data, "reg_inliers")
        mean_res, inlier_counts = compute_residual_stats(reg_residuals, reg_inliers)
        plot_residual_analysis(
            mean_res,
            inlier_counts,
            fe_t_err=fe_t_err,
            lo_t_err=lo_t_err,
            gl_t_err=gl_t_err,
            fe_r_err=fe_r_err,
            lo_r_err=lo_r_err,
            gl_r_err=gl_r_err,
            kf_indices=kf_indices,
            save_prefix=args.save_prefix,
        )

        # If GT wasn't loaded, we still want the requested plot to appear.
        # Fallback "error" series = mean registration residual (inliers).
        if gt_poses is None:
            best_reproj = _extract_best_cluster_metric_series(data, "reproj_error")
            best_3d = _extract_best_cluster_metric_series(data, "3d_dist")
            if best_reproj is not None and best_3d is not None:
                print(
                    "\n========== Selected Cluster: (reproj_error/3d_dist) vs Mean Residual (fallback) =========="
                )
                plot_reproj_over_3d_dist_vs_error(
                    best_reproj,
                    best_3d,
                    mean_res,
                    err_label="Mean Residual (inliers)",
                )
            else:
                if "reg_clusters" in data:
                    print(
                        "Selected-cluster reproj/3d_dist plot skipped (fallback): need 'reg_best_cluster_idx' "
                        "and per-cluster 'reproj_error' + '3d_dist' in 'reg_clusters'."
                    )

        # 4b. Slope and Threshold Analysis
        print("\n" + "=" * 60)
        print("Slope and Threshold Analysis")
        print("=" * 60)
        plot_slope_and_threshold_analysis(
            mean_res,
            inlier_counts,
            kf_indices=kf_indices,
            save_prefix=args.save_prefix,
            reg_residual_thres=0.07,  # Default from front_end.py
            residual_jump_threshold=0.02,  # Default from front_end.py
            inlier_drop_ratio=0.5,  # Default from front_end.py
            high_residual_threshold=0.01,  # Hard-coded in front_end.py
            min_inlier_count=5,  # Hard-coded in front_end.py
        )
    else:
        print(
            "Warning: 'reg_residuals' not found in logs (checked ragged & direct), skipping residual plot."
        )

    # 5. Additional GT-based plots
    if gt_poses is not None:
        # 5a. Keyframe-only error plots
        print("Filtering KF indices that have valid GT...")
        kf_indices_valid = filter_keyframes_with_gt(kf_indices, gt_poses)
        print(f"Valid keyframes with GT: {kf_indices_valid}")

        # Improvement statistics on valid KFs
        analyze_keyframe_improvement(
            fe_t_err,
            lo_t_err,
            gl_t_err,
            fe_r_err,
            lo_r_err,
            gl_r_err,
            kf_indices_valid,
        )

        plot_keyframe_errors(
            fe_t_err,
            fe_r_err,
            lo_t_err,
            lo_r_err,
            gl_t_err,
            gl_r_err,
            kf_indices_valid,
            save_prefix=args.save_prefix,
        )

        # 5b. Pose components vs GT
        print("\n" + "=" * 60)
        print("Pose Components vs Ground Truth")
        print("=" * 60)
        plot_pose_components_vs_gt(
            pose_frontend,
            pose_local,
            pose_global,
            gt_poses,
            kf_indices=kf_indices,
            save_prefix=args.save_prefix,
        )

        # 5c. Optional global / per-frame plots
        if args.plot_full_trajectory:
            plot_full_trajectory(
                pose_frontend,
                pose_local,
                pose_global,
                gt_poses,
                kf_indices,
                save_prefix=args.save_prefix,
            )
    else:
        # No GT; you can still inspect raw poses and keyframe distribution
        if args.plot_full_trajectory:
            plot_full_trajectory(
                pose_frontend,
                pose_local,
                pose_global,
                None,
                kf_indices,
                args.save_prefix,
            )

    # 6. Dense Recovery Analysis
    print("\n" + "=" * 60)
    print("Dense Recovery Analysis")
    print("=" * 60)
    plot_dense_recovery_analysis(
        data,
        pose_frontend,
        pose_global,
        gt_poses=gt_poses,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()
