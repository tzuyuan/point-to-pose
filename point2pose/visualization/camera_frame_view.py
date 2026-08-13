"""Camera-frame ("trails") window.

The sensor stays fixed at the origin (OpenCV convention: +z forward, +y down).
For every tracked object the view shows:
  * the currently tracked 3D points on the object,
  * fading trails of where each point was over the last N frames,
  * the object's bounding box and xyz axes at its current pose.
"""

from collections import deque

import numpy as np
import open3d as o3d

from point2pose.visualization.geometry import (
    LineAccumulator,
    axes_lines,
    box_wireframe,
    frame_id_colors,
    frustum_wireframe,
    object_color,
    set_lineset,
    set_pointcloud,
    trail_segments,
    uncertainty_colors,
)
from point2pose.visualization.view_base import Base3DView

_TRAIL_FADE = np.array([0.28, 0.29, 0.33])  # oldest trail color (near background)
_SENSOR_COLOR = np.array([0.55, 0.58, 0.65])
_BBOX_DIM = 0.75  # bbox drawn slightly dimmer than the object color


class CameraFrameView(Base3DView):
    def __init__(self, opts, visible: bool = True):
        super().__init__(
            title="Point2Pose 3D | Camera Frame (point trails)",
            width=opts.window_width,
            height=opts.window_height,
            left=opts.window_left,
            top=opts.window_top,
            visible=visible,
        )
        self._opts = opts
        self._points = o3d.geometry.PointCloud()
        self._trails = o3d.geometry.LineSet()
        self._bboxes = o3d.geometry.LineSet()
        self._obj_axes = o3d.geometry.LineSet()
        self._sensor = o3d.geometry.LineSet()
        sensor_pts, sensor_lines = frustum_wireframe(
            np.eye(4), float(opts.frustum_scale)
        )
        set_lineset(self._sensor, sensor_pts, sensor_lines, _SENSOR_COLOR)
        self._axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5 * float(opts.frustum_scale)
        )
        # obj_id -> deque of (K_t, 3) NaN-padded camera-frame point arrays
        self._history = {}
        self._latest_points = None

    def _geometries(self):
        return [
            self._axes,
            self._sensor,
            self._points,
            self._trails,
            self._bboxes,
            self._obj_axes,
        ]

    def update(self, scene):
        """Refresh from a SceneSnapshot (all objects)."""
        all_pts, all_colors = [], []
        trails = LineAccumulator()
        bboxes = LineAccumulator()
        axes = LineAccumulator()

        for snap in scene.objects:
            hist = self._history.setdefault(
                snap.obj_id, deque(maxlen=int(self._opts.trail_length))
            )
            hist.append(snap.track_points_cam)

            good = np.isfinite(snap.track_points_cam).all(axis=1)
            if good.any():
                all_pts.append(snap.track_points_cam[good])
                all_colors.append(self._point_colors(snap, good))

            base = object_color(snap.obj_id)
            if bool(self._opts.get("show_trails", True)):
                trails.add(*trail_segments(list(hist), _TRAIL_FADE, base))

            if bool(self._opts.show_bbox) and snap.bbox_extent is not None:
                corners, edges = box_wireframe(snap.bbox_extent, snap.T_cam_obj)
                bboxes.add(corners, edges, base * _BBOX_DIM)
            axes.add(*axes_lines(snap.T_cam_obj, 0.75 * float(self._opts.frustum_scale)))

        if all_pts:
            self._latest_points = np.concatenate(all_pts, axis=0)
            set_pointcloud(
                self._points, self._latest_points, np.concatenate(all_colors, axis=0)
            )
        else:
            set_pointcloud(self._points, np.zeros((0, 3)), np.zeros((0, 3)))
        set_lineset(self._trails, *trails.build())
        set_lineset(self._bboxes, *bboxes.build())
        set_lineset(self._obj_axes, *axes.build())

        self.show()

    def _point_colors(self, snap, good):
        mode = str(self._opts.point_color_mode)
        if mode == "frame_id":
            return frame_id_colors(snap.track_point_frames[good])
        if mode == "uncertainty":
            return uncertainty_colors(snap.track_point_uncertainties[good])
        return object_color(snap.obj_id)

    def _set_initial_view(self, view_control):
        # Slightly above/behind the sensor, looking at the tracked points.
        lookat = np.array([0.0, 0.0, 0.4])
        if self._latest_points is not None and len(self._latest_points) > 0:
            lookat = np.median(self._latest_points, axis=0)
        front = np.array([0.0, -0.25, -1.0])
        view_control.set_lookat(lookat)
        view_control.set_front(front / np.linalg.norm(front))
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_zoom(0.5)

    def reset(self):
        """Drop trail history and per-frame geometry."""
        self._history = {}
        self._latest_points = None
        set_pointcloud(self._points, [], [])
        for geom in (self._trails, self._bboxes, self._obj_axes):
            set_lineset(geom, [], [], [])
