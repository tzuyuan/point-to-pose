#!/usr/bin/env python3
"""
Visualize a compact factor-graph formulation from logged meta_data.npz.

Given a sequence and frame number, this script:
1) Selects camera frames that are far apart in camera-center space.
2) Draws camera poses around the object map.
3) Samples a small set of landmark observations (<10 by default).
4) Draws:
   - landmark factors (camera-to-landmark edges)
   - between factors (camera-to-camera edges)
with different colors.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root is importable when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from point2pose.utils.transform import inverse_SE3


@dataclass(frozen=True)
class CameraNode:
    row_idx: int
    frame_id: int
    T_c2m: np.ndarray  # camera -> map
    center: np.ndarray  # (3,)


def _ragged_slice(npz, key: str, row_idx: int) -> np.ndarray:
    """Return flattened ragged payload for one row."""
    data = npz[f"{key}_data"]
    offsets = npz[f"{key}_offsets"]
    lengths = npz[f"{key}_lengths"]
    off = int(offsets[row_idx])
    L = int(lengths[row_idx])
    if L <= 0:
        return np.asarray([], dtype=data.dtype)
    return np.asarray(data[off : off + L])


def _ragged_slice_2d(
    npz, key: str, row_idx: int, d: int, *, dtype=None
) -> np.ndarray:
    """Return ragged payload reshaped as (-1, d)."""
    flat = _ragged_slice(npz, key, row_idx)
    if dtype is not None:
        flat = flat.astype(dtype, copy=False)
    if flat.size == 0:
        out_dtype = dtype if dtype is not None else flat.dtype
        return np.zeros((0, d), dtype=out_dtype)
    if flat.size % d != 0:
        raise ValueError(
            f"Ragged field {key} row {row_idx}: flat.size={flat.size} not divisible by d={d}"
        )
    return flat.reshape(-1, d)


def _parse_pose_fields(pose_fields: str) -> Tuple[str, ...]:
    fields = [x.strip() for x in pose_fields.split(",") if x.strip()]
    if not fields:
        raise ValueError("pose_fields cannot be empty")
    return tuple(fields)


def _choose_pose(npz, row_idx: int, pose_fields: Sequence[str]) -> Optional[np.ndarray]:
    for key in pose_fields:
        if key not in npz.files:
            continue
        p = np.asarray(npz[key][row_idx], dtype=float)
        if p.shape == (4, 4) and np.all(np.isfinite(p)):
            return p
    return None


def _resolve_meta_data_path(
    sequence: Optional[str], meta_data: Optional[str], results_dir: Optional[str]
) -> str:
    if meta_data:
        path = os.path.abspath(meta_data)
        if not os.path.exists(path):
            raise FileNotFoundError(f"meta_data path not found: {path}")
        return path

    if not sequence:
        raise ValueError("Provide either --meta-data or --sequence")

    candidates: List[str] = []
    if results_dir:
        candidates.append(
            os.path.abspath(
                os.path.join(results_dir, sequence, "meta_data", "meta_data.npz")
            )
        )
    else:
        pattern = os.path.join(
            "results", "*", sequence, "meta_data", "meta_data.npz"
        )
        candidates.extend(sorted(glob.glob(pattern)))

    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        if results_dir:
            raise FileNotFoundError(
                f"Could not find meta_data.npz for sequence='{sequence}' under results_dir='{results_dir}'"
            )
        raise FileNotFoundError(
            f"Could not find meta_data.npz for sequence='{sequence}'. "
            "Try passing --results-dir or --meta-data explicitly."
        )

    if len(existing) > 1:
        print("Multiple meta_data.npz candidates found; using the first one:")
        for c in existing:
            print(f"  - {c}")
    return os.path.abspath(existing[0])


def _nearest_row_for_frame(frame_ids: np.ndarray, frame_id: int) -> int:
    idx = int(np.argmin(np.abs(frame_ids.astype(int) - int(frame_id))))
    return idx


def _extract_map_points(npz, row_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    xyz = _ragged_slice_2d(npz, "obj_key_points", row_idx, 3, dtype=float)
    ids = np.arange(xyz.shape[0], dtype=int)

    if "obj_valid_data" in npz.files:
        valid = _ragged_slice(npz, "obj_valid", row_idx).astype(bool, copy=False).reshape(
            -1
        )
        if valid.size == xyz.shape[0]:
            xyz = xyz[valid]
            ids = ids[valid]

    finite = np.isfinite(xyz).all(axis=1)
    return ids[finite], xyz[finite]


def _make_camera_nodes(
    npz,
    frame_ids: np.ndarray,
    candidate_rows: Iterable[int],
    pose_fields: Sequence[str],
) -> List[CameraNode]:
    nodes: List[CameraNode] = []
    for row_idx in candidate_rows:
        p = _choose_pose(npz, int(row_idx), pose_fields)
        if p is None:
            continue
        T_c2m = inverse_SE3(p)
        c = np.asarray(T_c2m[:3, 3], dtype=float).reshape(3)
        if not np.all(np.isfinite(c)):
            continue
        nodes.append(
            CameraNode(
                row_idx=int(row_idx),
                frame_id=int(frame_ids[int(row_idx)]),
                T_c2m=T_c2m,
                center=c,
            )
        )
    return nodes


def _select_far_apart_nodes(
    nodes: Sequence[CameraNode], seed_row_idx: int, n_select: int
) -> List[CameraNode]:
    if len(nodes) <= n_select:
        return list(nodes)

    by_row = {n.row_idx: n for n in nodes}
    if seed_row_idx in by_row:
        selected = [by_row[seed_row_idx]]
    else:
        selected = [min(nodes, key=lambda n: abs(n.row_idx - seed_row_idx))]

    remaining = [n for n in nodes if n.row_idx != selected[0].row_idx]
    while remaining and len(selected) < n_select:
        best = None
        best_score = -np.inf
        for cand in remaining:
            d_min = min(np.linalg.norm(cand.center - s.center) for s in selected)
            if d_min > best_score:
                best_score = d_min
                best = cand
        if best is None:
            break
        selected.append(best)
        remaining = [n for n in remaining if n.row_idx != best.row_idx]

    selected.sort(key=lambda n: n.frame_id)
    return selected


def _unique_preserve_order(values: np.ndarray) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in values.tolist():
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _extract_observed_landmark_ids(npz, row_idx: int) -> List[int]:
    if "reg_key_points_idx_data" not in npz.files:
        return []
    ids = _ragged_slice(npz, "reg_key_points_idx", row_idx)
    if ids.size == 0:
        return []
    ids = np.asarray(ids, dtype=int).reshape(-1)
    return _unique_preserve_order(ids)


def _sample_landmarks_for_nodes(
    nodes: Sequence[CameraNode],
    obs_ids_by_row: Dict[int, List[int]],
    map_ids_set: set,
    max_landmarks: int,
) -> List[int]:
    if max_landmarks <= 0:
        return []

    support: Dict[int, int] = {}
    for n in nodes:
        valid_obs = [lid for lid in obs_ids_by_row.get(n.row_idx, []) if lid in map_ids_set]
        for lid in set(valid_obs):
            support[lid] = support.get(lid, 0) + 1

    if not support:
        return []

    ranked = sorted(support.keys(), key=lambda lid: (-support[lid], lid))
    return ranked[:max_landmarks]


def _make_axes_traces(
    nodes: Sequence[CameraNode], axis_len: float
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    x_x: List[float] = []
    y_x: List[float] = []
    z_x: List[float] = []
    x_y: List[float] = []
    y_y: List[float] = []
    z_y: List[float] = []
    x_z: List[float] = []
    y_z: List[float] = []
    z_z: List[float] = []

    for n in nodes:
        c = n.center
        R = n.T_c2m[:3, :3]
        x_tip = c + axis_len * R[:, 0]
        y_tip = c + axis_len * R[:, 1]
        z_tip = c + axis_len * R[:, 2]
        x_x.extend([c[0], x_tip[0], None])
        y_x.extend([c[1], x_tip[1], None])
        z_x.extend([c[2], x_tip[2], None])
        x_y.extend([c[0], y_tip[0], None])
        y_y.extend([c[1], y_tip[1], None])
        z_y.extend([c[2], y_tip[2], None])
        x_z.extend([c[0], z_tip[0], None])
        y_z.extend([c[1], z_tip[1], None])
        z_z.extend([c[2], z_tip[2], None])

    return x_x, y_x, z_x, x_y, y_y, z_y, x_z, y_z, z_z


def _make_camera_illustration_traces(
    nodes: Sequence[CameraNode],
    scale: float,
    *,
    object_center: Optional[np.ndarray] = None,
    square_faces_object: bool = True,
) -> Tuple[List[float], List[float], List[float]]:
    """Build line segments for a stylized frustum-like camera glyph."""
    x: List[float] = []
    y: List[float] = []
    z: List[float] = []

    for n in nodes:
        c = n.center
        R = n.T_c2m[:3, :3]

        # Camera viewing axis for glyph orientation.
        if square_faces_object and object_center is not None:
            v = np.asarray(object_center, dtype=float).reshape(3) - c
            vn = float(np.linalg.norm(v))
            if vn > 1e-9:
                z_axis = v / vn
            else:
                z_axis = R[:, 2]
        else:
            z_axis = R[:, 2]

        z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-9)

        # Preserve roll using camera x-axis, then orthogonalize.
        x_axis = R[:, 0] - float(np.dot(R[:, 0], z_axis)) * z_axis
        xn = float(np.linalg.norm(x_axis))
        if xn <= 1e-9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(tmp, z_axis))) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            x_axis = tmp - float(np.dot(tmp, z_axis)) * z_axis
            xn = float(np.linalg.norm(x_axis))
        x_axis = x_axis / max(xn, 1e-9)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)

        # Put the square face toward the object, with the tip away from it.
        square_c = c + 0.25 * scale * z_axis
        tip = c - 0.95 * scale * z_axis
        xr = 0.45 * scale * x_axis
        yr = 0.30 * scale * y_axis

        b0 = square_c + xr + yr
        b1 = square_c - xr + yr
        b2 = square_c - xr - yr
        b3 = square_c + xr - yr

        # pyramid sides from rear tip to front square corners
        x.extend([tip[0], b0[0], None, tip[0], b1[0], None, tip[0], b2[0], None, tip[0], b3[0], None])
        y.extend([tip[1], b0[1], None, tip[1], b1[1], None, tip[1], b2[1], None, tip[1], b3[1], None])
        z.extend([tip[2], b0[2], None, tip[2], b1[2], None, tip[2], b2[2], None, tip[2], b3[2], None])

        # front square face
        x.extend([b0[0], b1[0], b2[0], b3[0], b0[0], None])
        y.extend([b0[1], b1[1], b2[1], b3[1], b0[1], None])
        z.extend([b0[2], b1[2], b2[2], b3[2], b0[2], None])

    return x, y, z


def _plot_segmented_lines_matplotlib(ax, x, y, z, **kwargs) -> None:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for xv, yv, zv in zip(x, y, z):
        if xv is None or yv is None or zv is None:
            if xs:
                ax.plot(xs, ys, zs, **kwargs)
            xs, ys, zs = [], [], []
            continue
        xs.append(float(xv))
        ys.append(float(yv))
        zs.append(float(zv))
    if xs:
        ax.plot(xs, ys, zs, **kwargs)


def build_figure(
    meta_path: str,
    frame_number: int,
    num_cameras: int,
    max_landmarks: int,
    pose_fields: Sequence[str],
    backend: str = "auto",
    camera_distance_scale: float = 0.45,
    map_point_size: float = 5.0,
    map_point_alpha: float = 0.9,
    view_padding_scale: float = 0.05,
):
    meta = np.load(meta_path, allow_pickle=True)
    required = ("frame_id", "obj_key_points_data")
    for k in required:
        if k not in meta.files:
            raise KeyError(f"{meta_path} missing required key: {k}")

    frame_ids = np.asarray(meta["frame_id"], dtype=int)
    if "is_key_frame" in meta.files:
        is_key_frame = np.asarray(meta["is_key_frame"], dtype=bool)
    else:
        is_key_frame = np.zeros((len(frame_ids),), dtype=bool)

    target_row = _nearest_row_for_frame(frame_ids, int(frame_number))
    target_fid = int(frame_ids[target_row])

    candidate_rows = np.where(is_key_frame)[0].astype(int).tolist()
    if len(candidate_rows) < num_cameras:
        candidate_rows = list(range(len(frame_ids)))

    nodes_all = _make_camera_nodes(
        npz=meta,
        frame_ids=frame_ids,
        candidate_rows=candidate_rows,
        pose_fields=pose_fields,
    )
    if len(nodes_all) == 0:
        raise RuntimeError(
            f"No valid camera poses found with pose fields={pose_fields}."
        )

    selected_nodes = _select_far_apart_nodes(
        nodes=nodes_all, seed_row_idx=target_row, n_select=num_cameras
    )
    if len(selected_nodes) < 2:
        raise RuntimeError("Need at least 2 valid camera nodes to draw factors.")

    map_ids, map_xyz = _extract_map_points(meta, target_row)
    if map_xyz.shape[0] == 0:
        raise RuntimeError("No finite map keypoints found at target frame.")
    map_id_to_xyz = {int(i): map_xyz[k] for k, i in enumerate(map_ids.tolist())}
    map_ids_set = set(map_id_to_xyz.keys())

    if camera_distance_scale <= 0.0:
        raise ValueError("camera_distance_scale must be > 0")
    if map_point_size <= 0.0:
        raise ValueError("map_point_size must be > 0")
    if not (0.0 <= map_point_alpha <= 1.0):
        raise ValueError("map_point_alpha must be in [0, 1]")
    if view_padding_scale < 0.0:
        raise ValueError("view_padding_scale must be >= 0")

    # Visualization-only camera scaling around the map center.
    map_center = np.mean(map_xyz, axis=0)
    plot_nodes: List[CameraNode] = []
    for n in selected_nodes:
        center_plot = map_center + float(camera_distance_scale) * (n.center - map_center)
        plot_nodes.append(
            CameraNode(
                row_idx=n.row_idx,
                frame_id=n.frame_id,
                T_c2m=n.T_c2m,
                center=center_plot,
            )
        )
    center_by_row = {n.row_idx: n.center for n in plot_nodes}

    obs_ids_by_row: Dict[int, List[int]] = {}
    for n in plot_nodes:
        obs_ids_by_row[n.row_idx] = _extract_observed_landmark_ids(meta, n.row_idx)

    sampled_landmark_ids = _sample_landmarks_for_nodes(
        nodes=plot_nodes,
        obs_ids_by_row=obs_ids_by_row,
        map_ids_set=map_ids_set,
        max_landmarks=max_landmarks,
    )
    sampled_landmark_xyz = np.asarray(
        [map_id_to_xyz[i] for i in sampled_landmark_ids], dtype=float
    ) if sampled_landmark_ids else np.zeros((0, 3), dtype=float)

    # Landmark factor edges (camera -> landmark)
    x_lm: List[float] = []
    y_lm: List[float] = []
    z_lm: List[float] = []
    sampled_set = set(sampled_landmark_ids)
    for n in plot_nodes:
        obs = [lid for lid in obs_ids_by_row.get(n.row_idx, []) if lid in sampled_set]
        for lid in obs:
            p = map_id_to_xyz[lid]
            c = center_by_row[n.row_idx]
            x_lm.extend([c[0], p[0], None])
            y_lm.extend([c[1], p[1], None])
            z_lm.extend([c[2], p[2], None])

    # Between factors (camera_i <-> camera_{i+1}, ordered by frame_id)
    x_bt: List[float] = []
    y_bt: List[float] = []
    z_bt: List[float] = []
    for i in range(len(plot_nodes) - 1):
        a = plot_nodes[i].center
        b = plot_nodes[i + 1].center
        x_bt.extend([a[0], b[0], None])
        y_bt.extend([a[1], b[1], None])
        z_bt.extend([a[2], b[2], None])

    # Axis scaling
    all_pts = [map_xyz]
    all_pts.append(np.asarray([n.center for n in plot_nodes], dtype=float))
    if sampled_landmark_xyz.size > 0:
        all_pts.append(sampled_landmark_xyz)
    pts = np.concatenate(all_pts, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(spans))
    pad = float(view_padding_scale) * max_span
    axis_len = 0.08 * max_span
    cam_glyph_size = 0.11 * max_span

    center = 0.5 * (mins + maxs)
    half = 0.5 * max_span + pad
    xr = [float(center[0] - half), float(center[0] + half)]
    yr = [float(center[1] - half), float(center[1] + half)]
    zr = [float(center[2] - half), float(center[2] + half)]

    (
        x_ax,
        y_ax,
        z_ax,
        x_ay,
        y_ay,
        z_ay,
        x_az,
        y_az,
        z_az,
    ) = _make_axes_traces(plot_nodes, axis_len=axis_len)
    x_cam_glyph, y_cam_glyph, z_cam_glyph = _make_camera_illustration_traces(
        plot_nodes,
        scale=cam_glyph_size,
        object_center=map_center,
        square_faces_object=True,
    )

    cam_centers = np.asarray([n.center for n in plot_nodes], dtype=float)
    summary = {
        "target_row": int(target_row),
        "target_frame_resolved": int(target_fid),
        "selected_rows": [int(n.row_idx) for n in plot_nodes],
        "selected_frames": [int(n.frame_id) for n in plot_nodes],
        "num_map_points": int(map_xyz.shape[0]),
        "num_sampled_landmarks": int(len(sampled_landmark_ids)),
        "num_landmark_factor_edges": int(len(x_lm) // 3),
        "num_between_factor_edges": int(len(x_bt) // 3),
        "pose_fields_used": list(pose_fields),
        "camera_distance_scale": float(camera_distance_scale),
        "view_padding_scale": float(view_padding_scale),
    }

    title = "Factor Graph Illustration"

    # Try Plotly first unless matplotlib is explicitly requested.
    if backend not in ("auto", "plotly", "matplotlib"):
        raise ValueError(f"Unsupported backend: {backend}")

    use_plotly = backend in ("auto", "plotly")
    if use_plotly:
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            if backend == "plotly":
                raise ImportError(
                    "Plotly backend requested but plotly is not installed. "
                    "Install with `pip install plotly`, or run with `--backend matplotlib`."
                ) from exc
            use_plotly = False

    if use_plotly:
        map_color = f"rgba(70,130,180,{float(map_point_alpha):.3f})"
        landmark_color = "rgba(31,119,180,0.95)"
        camera_color = "rgba(220,20,60,0.95)"
        landmark_factor_color = "rgba(34,139,34,0.85)"
        between_factor_color = "rgba(255,140,0,0.95)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=map_xyz[:, 0],
                y=map_xyz[:, 1],
                z=map_xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=float(map_point_size),
                    color=map_color,
                    line=dict(color="rgba(20,20,20,0.35)", width=0.6),
                ),
                name="Map keypoints",
                hoverinfo="skip",
            )
        )

        if sampled_landmark_xyz.size > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=sampled_landmark_xyz[:, 0],
                    y=sampled_landmark_xyz[:, 1],
                    z=sampled_landmark_xyz[:, 2],
                    mode="markers+text",
                    marker=dict(size=8, color=landmark_color, symbol="diamond"),
                    text=[str(i) for i in sampled_landmark_ids],
                    textposition="middle right",
                    name="Landmark nodes",
                    hovertemplate="landmark id=%{text}<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=x_cam_glyph,
                y=y_cam_glyph,
                z=z_cam_glyph,
                mode="lines",
                line=dict(color=camera_color, width=8),
                name="Camera nodes",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=cam_centers[:, 0],
                y=cam_centers[:, 1],
                z=cam_centers[:, 2],
                mode="markers",
                marker=dict(size=4, color=camera_color, symbol="circle"),
                name="Camera centers",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=x_bt,
                y=y_bt,
                z=z_bt,
                mode="lines",
                line=dict(color=between_factor_color, width=6),
                name="Between factors",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=x_lm,
                y=y_lm,
                z=z_lm,
                mode="lines",
                line=dict(color=landmark_factor_color, width=4),
                name="Landmark factors",
                hoverinfo="skip",
            )
        )

        # Camera axes (x=red, y=green, z=blue)
        fig.add_trace(
            go.Scatter3d(
                x=x_ax,
                y=y_ax,
                z=z_ax,
                mode="lines",
                line=dict(color="rgba(220,20,60,0.7)", width=4),
                name="Cam +X",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=x_ay,
                y=y_ay,
                z=z_ay,
                mode="lines",
                line=dict(color="rgba(34,139,34,0.7)", width=4),
                name="Cam +Y",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=x_az,
                y=y_az,
                z=z_az,
                mode="lines",
                line=dict(color="rgba(30,144,255,0.75)", width=4),
                name="Cam +Z",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=title,
            width=1200,
            height=850,
            template="plotly_white",
            scene=dict(
                xaxis=dict(
                    range=xr,
                    visible=False,
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    showbackground=False,
                    title="",
                ),
                yaxis=dict(
                    range=yr,
                    visible=False,
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    showbackground=False,
                    title="",
                ),
                zaxis=dict(
                    range=zr,
                    visible=False,
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    showbackground=False,
                    title="",
                ),
                aspectmode="cube",
                bgcolor="white",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.65)"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode=False,
        )
        return fig, summary, "plotly"

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Remove the default 3D pane tint/grid so the background is clean.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
        with np.errstate(all="ignore"):
            try:
                axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
                axis._axinfo["grid"]["linewidth"] = 0.0
            except Exception:
                pass
    ax.grid(False)

    ax.scatter(
        map_xyz[:, 0],
        map_xyz[:, 1],
        map_xyz[:, 2],
        s=max(10.0, float(map_point_size) * 8.0),
        c=[(70 / 255.0, 130 / 255.0, 180 / 255.0, float(map_point_alpha))],
        edgecolors=[(0.1, 0.1, 0.1, 0.3)],
        linewidths=0.4,
        label="Map keypoints",
    )

    if sampled_landmark_xyz.size > 0:
        ax.scatter(
            sampled_landmark_xyz[:, 0],
            sampled_landmark_xyz[:, 1],
            sampled_landmark_xyz[:, 2],
            s=70,
            c=[(31 / 255.0, 119 / 255.0, 180 / 255.0, 0.95)],
            marker="D",
            label="Landmark nodes",
        )
        for p, lid in zip(sampled_landmark_xyz, sampled_landmark_ids):
            ax.text(
                p[0],
                p[1],
                p[2],
                str(lid),
                fontsize=8,
                color=(0.1, 0.1, 0.1, 0.9),
            )

    ax.scatter(
        cam_centers[:, 0],
        cam_centers[:, 1],
        cam_centers[:, 2],
        s=55,
        c=[(220 / 255.0, 20 / 255.0, 60 / 255.0, 0.95)],
        marker="o",
        label="Camera centers",
    )

    _plot_segmented_lines_matplotlib(
        ax,
        x_bt,
        y_bt,
        z_bt,
        color=(255 / 255.0, 140 / 255.0, 0.0, 0.95),
        linewidth=2.8,
        label="Between factors",
    )
    _plot_segmented_lines_matplotlib(
        ax,
        x_lm,
        y_lm,
        z_lm,
        color=(34 / 255.0, 139 / 255.0, 34 / 255.0, 0.85),
        linewidth=1.8,
        label="Landmark factors",
    )
    _plot_segmented_lines_matplotlib(
        ax, x_ax, y_ax, z_ax, color=(220 / 255.0, 20 / 255.0, 60 / 255.0, 0.7), linewidth=1.5
    )
    _plot_segmented_lines_matplotlib(
        ax, x_ay, y_ay, z_ay, color=(34 / 255.0, 139 / 255.0, 34 / 255.0, 0.7), linewidth=1.5
    )
    _plot_segmented_lines_matplotlib(
        ax, x_az, y_az, z_az, color=(30 / 255.0, 144 / 255.0, 255 / 255.0, 0.75), linewidth=1.5
    )
    _plot_segmented_lines_matplotlib(
        ax,
        x_cam_glyph,
        y_cam_glyph,
        z_cam_glyph,
        color=(220 / 255.0, 20 / 255.0, 60 / 255.0, 0.95),
        linewidth=2.4,
        label="Camera nodes",
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim(xr[0], xr[1])
    ax.set_ylim(yr[0], yr[1])
    ax.set_zlim(zr[0], zr[1])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc="upper left")
    fig.tight_layout()

    return fig, summary, "matplotlib"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize factor-graph style camera/landmark factors from meta_data.npz."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence/video name. Used with --results-dir if --meta-data is not provided.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory containing <sequence>/meta_data/meta_data.npz.",
    )
    parser.add_argument(
        "--meta-data",
        type=str,
        default=None,
        help="Direct path to meta_data.npz (overrides --sequence/--results-dir).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Target frame_id used to seed camera selection.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=4,
        help="Number of camera nodes to select (default: 4).",
    )
    parser.add_argument(
        "--max-landmarks",
        type=int,
        default=8,
        help="Maximum sampled landmarks to visualize (<10 recommended).",
    )
    parser.add_argument(
        "--pose-fields",
        type=str,
        default="obj_pose,pose_local,pose_frontend",
        help="Comma-separated pose fields in priority order.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Plotly writes HTML; Matplotlib writes PNG.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "plotly", "matplotlib"],
        help="Rendering backend: auto (prefer plotly), plotly, or matplotlib.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=0.45,
        help="Visualization-only camera distance scaling around map center (<1 closer, >1 farther).",
    )
    parser.add_argument(
        "--map-point-size",
        type=float,
        default=5.0,
        help="Map keypoint marker size.",
    )
    parser.add_argument(
        "--map-point-alpha",
        type=float,
        default=0.9,
        help="Map keypoint marker opacity in [0,1].",
    )
    parser.add_argument(
        "--view-padding-scale",
        type=float,
        default=0.05,
        help="Padding ratio for axis range around all plotted points (smaller => object appears larger).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    pose_fields = _parse_pose_fields(args.pose_fields)

    if args.max_landmarks >= 10:
        print(
            f"Warning: max_landmarks={args.max_landmarks} (recommend <10 for clarity)."
        )
    if args.num_cameras < 2:
        raise ValueError("--num-cameras must be >= 2")

    meta_path = _resolve_meta_data_path(
        sequence=args.sequence, meta_data=args.meta_data, results_dir=args.results_dir
    )
    print(f"Using meta_data: {meta_path}")

    fig, summary, backend_used = build_figure(
        meta_path=meta_path,
        frame_number=int(args.frame),
        num_cameras=int(args.num_cameras),
        max_landmarks=int(args.max_landmarks),
        pose_fields=pose_fields,
        backend=args.backend,
        camera_distance_scale=float(args.camera_distance_scale),
        map_point_size=float(args.map_point_size),
        map_point_alpha=float(args.map_point_alpha),
        view_padding_scale=float(args.view_padding_scale),
    )
    print(f"Rendered with backend: {backend_used}")

    if args.output:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if backend_used == "plotly":
            fig.write_html(out_path, include_plotlyjs="cdn")
            print(f"Saved interactive HTML: {out_path}")
        else:
            root, ext = os.path.splitext(out_path)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
                out_path = f"{root}.png" if ext else f"{out_path}.png"
            fig.savefig(out_path, dpi=180)
            print(f"Saved static figure: {out_path}")

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if not args.no_show:
        if backend_used == "plotly":
            fig.show()
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
