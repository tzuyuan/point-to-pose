"""
Rerun visualization helpers for live tracking demos.

Mirrors the ../kv_tracker/kv_tracker/rerun_tools.py pattern (camera frustums via
Transform3D + Pinhole, trajectories via LineStrips3D) but scoped to point2pose's
object-frame convention: ``obj.pose`` is ``T_obj2cam`` (object -> camera), so the
camera's pose in the (object-frame == world) is ``T_cam2obj = inverse_SE3(obj.pose)``.
"""

import colorsys
from collections import deque

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from point2pose.utils.transform import inverse_SE3, transform_pts

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def colors_for_track_ids(track_ids, visible=None, dim_factor=0.15, sat=0.85, val=0.95):
    """Deterministic, well-spread-out RGB color per track id (golden-ratio hue stepping, like
    the frame-id colors in realsense_tracking.py), so the same physical point keeps the same
    color across frames. If ``visible`` is given, invisible points get darkened by
    ``dim_factor`` instead of a different hue, so you can tell "known but currently untracked"
    from "actively locked on" at a glance.

    Args:
        track_ids: (N,) int array of global track ids.
        visible: optional (N,) bool array; False entries are darkened.
        dim_factor: value multiplier applied to invisible points' color (0..1, smaller = darker).
    Returns:
        (N,3) uint8 RGB array.
    """
    track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    n = track_ids.shape[0]
    colors = np.zeros((n, 3), dtype=np.uint8)
    if n == 0:
        return colors

    hues = (track_ids.astype(np.float64) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    vis = np.ones(n, dtype=bool) if visible is None else np.asarray(visible, dtype=bool).reshape(-1)
    vals = np.where(vis, val, val * dim_factor)

    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(hues[i], sat, vals[i])
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def rr_init(app_name="point2pose live tracking"):
    rr.init(app_name, spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )
    rr.log("fps_plot", rr.SeriesLines(colors=[0, 255, 0], widths=[2.5], names="FPS"), static=True)

    # Explicit layout: two side-by-side 3D views, since Rerun's auto-layout otherwise merges
    # the "world/*" and "camframe/*" entity trees into a single spatial view.
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="World (object fixed)"),
            rrb.Spatial3DView(origin="camframe", name="Camera frame (camera fixed)"),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def rr_log_camera(path, T_wc, K, w, h, image=None, color=(255, 255, 0), scale=0.08):
    """Log a camera frustum at world pose T_wc (camera -> world), CV convention."""
    rr.log(path, rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))
    rr.log(
        path,
        rr.Pinhole(
            image_from_camera=K, width=w, height=h,
            image_plane_distance=scale, color=color,
        ),
    )
    if image is not None:
        rr.log(f"{path}/image", rr.Image(image))


def rr_log_live_camera(path, T_obj2cam, K, w, h, image=None, color=(255, 255, 0), scale=0.08):
    """Convenience: log a camera frustum from an object->camera pose (point2pose convention)."""
    T_cam2obj = inverse_SE3(np.asarray(T_obj2cam, dtype=np.float64))
    rr_log_camera(path, T_cam2obj, K, w, h, image=image, color=color, scale=scale)


def rr_log_trajectory(path, positions_wc, color=(0, 127, 255)):
    """positions_wc: list/array of (3,) camera centers in world (object) frame."""
    if len(positions_wc) < 2:
        return
    rr.log(path, rr.LineStrips3D([np.asarray(positions_wc, dtype=np.float32)], colors=[color]))


def rr_log_points3d(path, pts3d, colors=None, radii=0.003):
    rr.log(path, rr.Points3D(np.asarray(pts3d, dtype=np.float32), colors=colors, radii=radii))


def rr_log_bbox(path, extent, center, color=(255, 255, 0)):
    rr.log(
        path,
        rr.Boxes3D(
            half_sizes=[0.5 * np.asarray(extent, dtype=np.float32)],
            centers=[np.asarray(center, dtype=np.float32)],
            colors=[color],
        ),
    )


def rr_log_fps(idx, fps, extra_text=""):
    rr.set_time("frame", sequence=idx)
    rr.log("fps_plot", rr.Scalars(fps))
    rr.log("fps_text", rr.TextDocument(f"# FPS: {fps:.1f}\n{extra_text}", media_type=rr.MediaType.MARKDOWN))


class PointTraceBuffer:
    """Rolling per-point (keyed by global track id) history of camera-frame positions, for the
    camera-fixed / object-moving panel. Only visible points get a new sample each frame; a
    point's trace is dropped once it hasn't been seen for ``max_gap`` frames so stale trails
    don't linger after a point goes invisible for good."""

    def __init__(self, max_len=30, max_gap=5):
        self.max_len = max_len
        self.max_gap = max_gap
        self._traces = {}  # track_id -> deque[(3,) float]
        self._last_seen = {}  # track_id -> frame idx

    def update(self, frame_idx, track_ids, pts_cam):
        seen = set()
        for tid, p in zip(track_ids, pts_cam):
            tid = int(tid)
            seen.add(tid)
            self._traces.setdefault(tid, deque(maxlen=self.max_len)).append(np.asarray(p, dtype=np.float32))
            self._last_seen[tid] = frame_idx

        stale = [tid for tid, last in self._last_seen.items() if frame_idx - last > self.max_gap]
        for tid in stale:
            self._traces.pop(tid, None)
            self._last_seen.pop(tid, None)

    def strips(self, min_len=2):
        return [np.stack(list(pts)) for pts in self._traces.values() if len(pts) >= min_len]


def rr_log_object_centric(path, T_obj2cam, K, w, h, obj_pts_obj, extent=None, T_bbox2cam=None,
                          cam_color=(255, 255, 0), obj_color=(120, 120, 255)):
    """Camera-fixed / object-moving panel: log a static camera frustum (no image -- this panel
    is 3D-only) at the origin, and the object's map points transformed into the (fixed) camera
    frame via T_obj2cam. ``obj_pts_obj`` must already be in the same frame T_obj2cam maps (the
    "map" frame -- see point2pose's obj.key_points / obj.pose convention), NOT necessarily the
    same frame the bbox is defined in; pass ``T_bbox2cam`` separately for that (point2pose's
    bbox is defined in a canonical local frame centered at its own origin, placed into the map
    frame via obj.init_pose -- see draw_posed_3d_box usage in realsense_tracking.py)."""
    rr_log_camera(f"{path}/fixed_cam", np.eye(4), K, w, h, image=None, color=cam_color)

    if obj_pts_obj is not None and len(obj_pts_obj):
        pts_cam = transform_pts(T_obj2cam, np.asarray(obj_pts_obj, dtype=np.float64))
        rr_log_points3d(f"{path}/map_points_cam", pts_cam, colors=obj_color, radii=0.002)
    if extent is not None and T_bbox2cam is not None:
        # bbox is defined in its own local frame at the origin; T_bbox2cam places it directly.
        center_cam = T_bbox2cam[:3, 3]
        rr_log_bbox(f"{path}/bbox_cam", extent, center_cam, color=(255, 200, 0))


def rr_log_point_traces(path, strips, color=(0, 255, 0)):
    if not strips:
        return
    rr.log(path, rr.LineStrips3D(strips, colors=[color] * len(strips)))
