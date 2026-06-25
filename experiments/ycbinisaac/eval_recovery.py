"""
Recovery-after-occlusion metrics for the SH(K) ablation.

Given per-frame predicted and GT poses, an `is_mask_visible` array, and the
object mesh, compute:
  - occlusion events: contiguous False runs of length >= min_occlusion_len
  - re-emergence frames: first True frame after each gap
  - recovery rate at horizons N: fraction of events where ADD-S <= tau
    within N frames after re-emergence
  - post-occlusion ADD/ADD-S at horizons N (mean over windows [t_r, t_r + N])

Can be used as a library (`compute_recovery_metrics`) or as a CLI that reads
TUM-format pose files saved by the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.utils.evaluation import add_err, adi_err
from point2pose.utils.transform import inverse_SE3


@dataclass
class OcclusionEvent:
    start: int           # first invisible frame
    length: int          # number of invisible frames
    re_emerge: int       # first visible frame after the gap (= start + length)


@dataclass
class RecoveryReport:
    seq_name: str
    method: str
    n_events: int
    horizons: list                # e.g. [5, 15]
    tau_m: float
    recovery_rate: dict           # horizon -> fraction in [0, 1]
    post_occ_adds_mean: dict      # horizon -> meters (mean over frames in windows)
    post_occ_add_mean: dict       # horizon -> meters
    per_event: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def find_occlusion_events(visible: np.ndarray, min_occlusion_len: int) -> list:
    """
    visible: (T,) bool array (True = visible / mask present)
    Returns events with `length >= min_occlusion_len` that have a re-emergence
    inside the sequence (i.e. the gap is followed by a visible frame).
    """
    visible = np.asarray(visible).astype(bool).reshape(-1)
    T = visible.shape[0]
    events = []
    i = 0
    while i < T:
        if not visible[i]:
            j = i
            while j < T and not visible[j]:
                j += 1
            length = j - i
            if length >= min_occlusion_len and j < T:
                events.append(OcclusionEvent(start=i, length=length, re_emerge=j))
            i = j
        else:
            i += 1
    return events


def _adds_at(pred_pose, gt_pose, mesh_pts, symmetric):
    if pred_pose is None or gt_pose is None:
        return np.inf
    if symmetric:
        return adi_err(pred_pose, gt_pose, mesh_pts)
    return add_err(pred_pose, gt_pose, mesh_pts)


def compute_recovery_metrics(
    pred_poses,
    gt_poses,
    visible: np.ndarray,
    mesh_pts: np.ndarray,
    *,
    seq_name: str = "",
    method: str = "",
    min_occlusion_len: int = 10,
    horizons=(5, 15),
    tau_m: float = 0.02,
    align_first_frame: bool = True,
    use_adi_for_recovery: bool = True,
) -> RecoveryReport:
    """
    pred_poses, gt_poses: list/array of (4,4) per frame, length T. Use None
        when GT is missing for a frame.
    visible: (T,) bool, mask visibility per frame.
    mesh_pts: (M,3) object vertices.
    tau_m: ADD-S threshold for "recovered" (meters), e.g. 0.02 (2 cm).
    horizons: list of N values (frames after re-emergence).
    align_first_frame: if True, align prediction to GT at the first valid GT
        frame (matches the in-script eval in run_ycbinisaac_single.py).
    use_adi_for_recovery: use ADD-S (symmetric) for the τ test; ADD reported
        alongside.
    """
    T = len(pred_poses)
    assert len(gt_poses) == T, "pred/gt length mismatch"
    assert visible.shape[0] == T, "visible length mismatch"

    pred_arr = [np.asarray(p, dtype=np.float64) if p is not None else None
                for p in pred_poses]
    gt_arr = [np.asarray(p, dtype=np.float64) if p is not None else None
              for p in gt_poses]

    if align_first_frame:
        first = next((i for i in range(T)
                      if pred_arr[i] is not None and gt_arr[i] is not None),
                     None)
        if first is not None:
            T_align = inverse_SE3(pred_arr[first]) @ gt_arr[first]
            pred_arr = [p @ T_align if p is not None else None for p in pred_arr]

    events = find_occlusion_events(visible, min_occlusion_len)

    horizons = list(horizons)
    recovery_rate = {N: 0.0 for N in horizons}
    add_window = {N: [] for N in horizons}
    adds_window = {N: [] for N in horizons}
    per_event = []

    if len(events) == 0:
        return RecoveryReport(
            seq_name=seq_name, method=method, n_events=0,
            horizons=horizons, tau_m=tau_m,
            recovery_rate=recovery_rate,
            post_occ_adds_mean={N: float("nan") for N in horizons},
            post_occ_add_mean={N: float("nan") for N in horizons},
            per_event=[],
        )

    for ev in events:
        ev_record = {"start": ev.start, "length": ev.length,
                     "re_emerge": ev.re_emerge}
        for N in horizons:
            window_end = min(T, ev.re_emerge + N + 1)
            recovered = False
            for t in range(ev.re_emerge, window_end):
                if pred_arr[t] is None or gt_arr[t] is None:
                    continue
                if not visible[t]:
                    continue
                e_adds = adi_err(pred_arr[t], gt_arr[t], mesh_pts)
                e_add = add_err(pred_arr[t], gt_arr[t], mesh_pts)
                adds_window[N].append(e_adds)
                add_window[N].append(e_add)
                if use_adi_for_recovery and e_adds <= tau_m:
                    recovered = True
                elif (not use_adi_for_recovery) and e_add <= tau_m:
                    recovered = True
            if recovered:
                recovery_rate[N] += 1
            ev_record[f"recovered@{N}"] = bool(recovered)
        per_event.append(ev_record)

    n_events = len(events)
    recovery_rate = {N: recovery_rate[N] / n_events for N in horizons}
    post_occ_adds_mean = {
        N: float(np.mean(adds_window[N])) if adds_window[N] else float("nan")
        for N in horizons
    }
    post_occ_add_mean = {
        N: float(np.mean(add_window[N])) if add_window[N] else float("nan")
        for N in horizons
    }

    return RecoveryReport(
        seq_name=seq_name, method=method,
        n_events=n_events, horizons=horizons, tau_m=tau_m,
        recovery_rate=recovery_rate,
        post_occ_adds_mean=post_occ_adds_mean,
        post_occ_add_mean=post_occ_add_mean,
        per_event=per_event,
    )


# ---------------------------------------------------------------------------
# CLI: post-hoc evaluation reading TUM pose files saved by the pipeline.
# ---------------------------------------------------------------------------

def _tum_line_to_pose(line: str) -> np.ndarray:
    parts = line.strip().split()
    if len(parts) != 8:
        raise ValueError(f"Bad TUM line: {line}")
    tx, ty, tz = map(float, parts[1:4])
    qx, qy, qz, qw = map(float, parts[4:8])
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 0.0 if n == 0 else 2.0 / n
    R = np.eye(3)
    R[0, 0] = 1 - s * (qy * qy + qz * qz)
    R[0, 1] = s * (qx * qy - qz * qw)
    R[0, 2] = s * (qx * qz + qy * qw)
    R[1, 0] = s * (qx * qy + qz * qw)
    R[1, 1] = 1 - s * (qx * qx + qz * qz)
    R[1, 2] = s * (qy * qz - qx * qw)
    R[2, 0] = s * (qx * qz - qy * qw)
    R[2, 1] = s * (qy * qz + qx * qw)
    R[2, 2] = 1 - s * (qx * qx + qy * qy)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = (tx, ty, tz)
    return M


def _load_tum(path: str) -> list:
    poses = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            poses.append(_tum_line_to_pose(line))
    return poses


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="YCBMultiTrack root containing the sequence folder")
    ap.add_argument("--video_name", required=True)
    ap.add_argument("--obj_name", default=None,
                    help="Object name; default = single object found in masks/")
    ap.add_argument("--pose_file", required=True,
                    help="Path to obj_X_pose.txt (TUM format)")
    ap.add_argument("--model_path", required=True,
                    help="YCB model root (containing models/{obj_name}/textured_simple.obj)")
    ap.add_argument("--method", default="",
                    help="Method label for the report (e.g. full, sh10, sh1)")
    ap.add_argument("--min_occlusion_len", type=int, default=10)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 15])
    ap.add_argument("--tau_m", type=float, default=0.02)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    import trimesh

    from point2pose.io.sources.dataset.datareader import YCBInIsaacReader

    video_path = os.path.join(args.data_path, args.video_name)
    reader = YCBInIsaacReader(video_path)
    object_names = reader.get_object_names()
    if args.obj_name is None:
        if len(object_names) != 1:
            raise SystemExit(
                f"Found {len(object_names)} objects, please pass --obj_name. "
                f"Available: {object_names}")
        obj_name = object_names[0]
    else:
        obj_name = args.obj_name

    # Mask visibility
    canonical = reader.videoname_to_object.get(obj_name, obj_name)
    candidates = [
        os.path.join(video_path, "is_mask_visible", obj_name, "is_mask_visible.npy"),
        os.path.join(video_path, "is_mask_visible", canonical, "is_mask_visible.npy"),
    ]
    vis_path = next((c for c in candidates if os.path.exists(c)), None)
    if vis_path is None:
        raise SystemExit(f"Cannot find is_mask_visible.npy in {candidates}")
    visible = np.load(vis_path).astype(bool).reshape(-1)

    # Predicted poses (TUM)
    pred_poses = _load_tum(args.pose_file)
    # GT poses
    gt_poses = []
    for i in range(len(reader)):
        gt_map = reader.get_gt_poses(i)
        gt_poses.append(gt_map.get(obj_name, None))

    T = min(len(pred_poses), len(gt_poses), len(visible))
    pred_poses = pred_poses[:T]
    gt_poses = gt_poses[:T]
    visible = visible[:T]

    # Mesh
    candidates_mesh = [
        os.path.join(args.model_path, canonical, "textured.obj"),
        os.path.join(args.model_path, "models", canonical, "textured_simple.obj"),
        os.path.join(args.model_path, canonical, "google_16k", "textured.obj"),
    ]
    mesh_path = next((c for c in candidates_mesh if os.path.exists(c)), None)
    if mesh_path is None:
        raise SystemExit(f"Cannot find mesh in {candidates_mesh}")
    mesh = trimesh.load(mesh_path)
    mesh_pts = np.asarray(mesh.vertices, dtype=np.float64)

    report = compute_recovery_metrics(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        visible=visible,
        mesh_pts=mesh_pts,
        seq_name=args.video_name,
        method=args.method,
        min_occlusion_len=args.min_occlusion_len,
        horizons=tuple(args.horizons),
        tau_m=args.tau_m,
    )

    print(f"\n=== {args.method or 'method'} on {args.video_name}/{obj_name} ===")
    print(f"  occlusion events (>= {args.min_occlusion_len} frames): {report.n_events}")
    for N in report.horizons:
        rr = report.recovery_rate[N]
        adds = report.post_occ_adds_mean[N] * 100  # cm
        add = report.post_occ_add_mean[N] * 100    # cm
        print(f"  N={N:3d}: recovery_rate={rr:.3f}  "
              f"post_occ_ADDS={adds:.2f} cm  post_occ_ADD={add:.2f} cm")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    _cli()
