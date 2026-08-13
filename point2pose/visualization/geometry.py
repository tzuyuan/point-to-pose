"""Open3D geometry builders shared by the 3D demo views.

All helpers are pure: they build numpy wireframes/colors or copy them into
persistent Open3D geometry objects. Nothing here reads pipeline state.
"""

import numpy as np
import open3d as o3d

GOLDEN_RATIO_CONJUGATE = 0.6180339887498949

# Shared render background; also used to hide degenerate placeholder geometry.
BACKGROUND_COLOR = np.array([0.08, 0.09, 0.11])

EMPTY_POINTS = np.zeros((0, 3), dtype=np.float64)
EMPTY_LINES = np.zeros((0, 2), dtype=np.int32)
EMPTY_COLORS = np.zeros((0, 3), dtype=np.float64)

# Per-object base colors (RGB in [0, 1]). Same order as the BGR palette in the
# 2D realsense demo so each object keeps its color identity across windows.
_OBJECT_PALETTE = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.5],
    ]
)

_BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def object_color(obj_id: int) -> np.ndarray:
    return _OBJECT_PALETTE[obj_id % len(_OBJECT_PALETTE)].copy()


def hsv_to_rgb(h, s, v) -> np.ndarray:
    """Vectorized HSV -> RGB. `h` must be a 1D array; `s`/`v` broadcastable."""
    h, s, v = np.broadcast_arrays(np.asarray(h, dtype=np.float64) % 1.0, s, v)
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    lut = np.stack(
        [
            np.stack([v, t, p], axis=-1),
            np.stack([q, v, p], axis=-1),
            np.stack([p, v, t], axis=-1),
            np.stack([p, q, v], axis=-1),
            np.stack([t, p, v], axis=-1),
            np.stack([v, p, q], axis=-1),
        ],
        axis=0,
    )
    return np.take_along_axis(lut, i[None, ..., None], axis=0)[0]


def frame_id_colors(frame_ids: np.ndarray) -> np.ndarray:
    """Stable, distinctive color per frame id (golden-ratio hue walk, like the
    `frame_id` point coloring mode of the 2D demo)."""
    fid = np.asarray(frame_ids, dtype=np.float64)
    fid = np.where(fid < 0, 0.0, fid)
    return hsv_to_rgb((fid * GOLDEN_RATIO_CONJUGATE) % 1.0, 0.85, 1.0)


def track_id_colors(track_ids: np.ndarray, visible: np.ndarray = None, dim: float = 0.15) -> np.ndarray:
    """Stable color per global track id (golden-ratio hue walk), so the same
    physical point keeps its color across frames. Points flagged not-visible
    are darkened by ``dim`` rather than recolored — "known but currently
    untracked" vs "locked on" at a glance. Returns float RGB in [0, 1]."""
    tid = np.asarray(track_ids, dtype=np.float64).reshape(-1)
    val = np.full(len(tid), 0.95)
    if visible is not None:
        val = np.where(np.asarray(visible, dtype=bool).reshape(-1), 0.95, 0.95 * dim)
    return hsv_to_rgb((tid * GOLDEN_RATIO_CONJUGATE) % 1.0, 0.85, val)


def inlier_colors(status: np.ndarray) -> np.ndarray:
    """Registration-status colors: 1 (inlier) -> green, 0 (outlier) -> red,
    -1 (not used this frame) -> dim gray. Returns float RGB in [0, 1]."""
    status = np.asarray(status, dtype=np.int64).reshape(-1)
    colors = np.tile(np.array([0.42, 0.44, 0.50]), (len(status), 1))
    colors[status == 1] = (0.15, 0.9, 0.35)
    colors[status == 0] = (0.95, 0.22, 0.18)
    return colors


def uncertainty_colors(u: np.ndarray, u_min: float = 0.0, u_max: float = 1.0) -> np.ndarray:
    """Green (certain) -> red (uncertain)."""
    un = np.clip(
        (np.asarray(u, dtype=np.float64) - u_min) / max(u_max - u_min, 1e-9), 0.0, 1.0
    )
    return hsv_to_rgb((1.0 - un) * 0.333, 0.9, 1.0)


def frustum_wireframe(T_cam_in_target: np.ndarray, scale: float = 0.06, aspect: float = 4.0 / 3.0):
    """Camera frustum wireframe posed by ``T_cam_in_target`` (camera -> target).

    OpenCV convention (+z forward, +y down); a small "up" spike marks the image
    top so roll is visible. Returns ``(points (6,3), lines (10,2))``.
    """
    w = 0.5 * aspect * scale
    h = 0.5 * scale
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [-w, -h, scale],
            [w, -h, scale],
            [w, h, scale],
            [-w, h, scale],
            [0.0, -1.6 * h, scale],
        ]
    )
    lines = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [2, 5]],
        dtype=np.int32,
    )
    T = np.asarray(T_cam_in_target, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3], lines


def box_wireframe(extent, T: np.ndarray = None):
    """Wireframe of a box centered at the origin with the given extent,
    optionally posed by ``T``. Returns ``(corners (8,3), edges (12,2))``."""
    half = 0.5 * np.asarray(extent, dtype=np.float64).reshape(3)
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners = signs * half
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        corners = corners @ T[:3, :3].T + T[:3, 3]
    return corners, _BOX_EDGES.copy()


def axes_lines(T: np.ndarray, scale: float = 0.05):
    """RGB xyz axis lines at pose ``T``. Returns ``(points, lines, colors)``."""
    T = np.asarray(T, dtype=np.float64)
    o = T[:3, 3]
    pts = np.vstack([o, o + scale * T[:3, 0], o + scale * T[:3, 1], o + scale * T[:3, 2]])
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.array([[0.9, 0.15, 0.15], [0.15, 0.9, 0.15], [0.25, 0.4, 1.0]])
    return pts, lines, colors


def polyline(points: np.ndarray, color_old, color_new):
    """Connect consecutive points, colors blending old -> new along the line."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return pts, EMPTY_LINES.copy(), EMPTY_COLORS.copy()
    lines = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    tt = np.linspace(0.0, 1.0, n - 1)[:, None]
    colors = np.asarray(color_old, dtype=np.float64) * (1.0 - tt) + np.asarray(
        color_new, dtype=np.float64
    ) * tt
    return pts, lines, colors


class LineAccumulator:
    """Merge many wireframes into one LineSet payload (index-offset bookkeeping)."""

    def __init__(self):
        self._pts = []
        self._lines = []
        self._colors = []
        self._offset = 0

    def add(self, points, lines, colors):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        lines = np.asarray(lines, dtype=np.int32).reshape(-1, 2)
        if len(lines) == 0:
            return
        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim == 1:
            colors = np.tile(colors, (len(lines), 1))
        self._pts.append(points)
        self._lines.append(lines + self._offset)
        self._colors.append(colors.reshape(-1, 3))
        self._offset += len(points)

    def build(self):
        if not self._lines:
            return EMPTY_POINTS.copy(), EMPTY_LINES.copy(), EMPTY_COLORS.copy()
        return (
            np.concatenate(self._pts, axis=0),
            np.concatenate(self._lines, axis=0),
            np.concatenate(self._colors, axis=0),
        )


def merged_frustums(poses, scale: float, color):
    """One wireframe payload holding a frustum for every pose in ``poses``."""
    acc = LineAccumulator()
    for T in poses:
        pts, lines = frustum_wireframe(T, scale)
        acc.add(pts, lines, color)
    return acc.build()


def trail_segments(history, color_old, color_new):
    """Fading per-point trails from a time-ordered list of (K_t, 3) arrays.

    Rows are stable track indices; NaN rows (invisible/invalid points) break
    the trail. Colors fade from ``color_old`` (oldest) to ``color_new``.
    """
    acc = LineAccumulator()
    steps = len(history) - 1
    for t in range(steps):
        a = np.asarray(history[t])
        b = np.asarray(history[t + 1])
        n = min(len(a), len(b))
        if n == 0:
            continue
        good = np.isfinite(a[:n]).all(axis=1) & np.isfinite(b[:n]).all(axis=1)
        m = int(good.sum())
        if m == 0:
            continue
        blend = (t + 1) / steps
        color = np.asarray(color_old, dtype=np.float64) * (1.0 - blend) + np.asarray(
            color_new, dtype=np.float64
        ) * blend
        pts = np.concatenate([a[:n][good], b[:n][good]], axis=0)
        lines = np.stack([np.arange(m), np.arange(m) + m], axis=1)
        acc.add(pts, lines, color)
    return acc.build()


def set_lineset(ls: o3d.geometry.LineSet, points, lines, colors):
    """Fill a persistent LineSet. Empty payloads are padded with an invisible
    zero-length line: truly empty geometry makes the legacy Open3D renderer
    print a binding warning on every frame."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lines = np.asarray(lines, dtype=np.int32).reshape(-1, 2)
    if len(lines) == 0:
        points = np.zeros((1, 3))
        lines = np.zeros((1, 2), dtype=np.int32)
        colors = BACKGROUND_COLOR
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))


def set_pointcloud(pcd: o3d.geometry.PointCloud, points, colors):
    """Fill a persistent PointCloud; empty payloads are padded with a single
    background-colored point (see set_lineset)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        points = np.zeros((1, 3))
        colors = BACKGROUND_COLOR
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(points), 1))
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))


def set_degenerate_mesh(mesh: o3d.geometry.TriangleMesh):
    """Replace a mesh with an invisible zero-area triangle (empty meshes
    trigger the same per-frame renderer warning as empty linesets)."""
    mesh.vertices = o3d.utility.Vector3dVector(np.zeros((3, 3)))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(BACKGROUND_COLOR, (3, 1))
    )
    mesh.compute_vertex_normals()
