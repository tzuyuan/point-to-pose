"""Object-frame ("map") window.

The tracked object stays fixed at the origin. Around it, the view shows:
  * the keypoint map (colored by first-seen frame id, uncertainty, or object),
  * the full camera trajectory (fading old -> bright new),
  * the current camera frustum (green; red when the object is lost),
  * a small frustum at every keyframe camera,
  * the object bounding box, and (optionally) the live SDF reconstruction.
"""

import numpy as np
import open3d as o3d

from point2pose.visualization.geometry import (
    box_wireframe,
    frame_id_colors,
    frustum_wireframe,
    merged_frustums,
    object_color,
    polyline,
    set_degenerate_mesh,
    set_lineset,
    set_pointcloud,
    uncertainty_colors,
)
from point2pose.visualization.view_base import Base3DView

_TRAJ_OLD = np.array([0.35, 0.37, 0.45])
_TRAJ_NEW = np.array([1.0, 0.8, 0.1])
_CAM_COLOR = np.array([0.1, 1.0, 0.35])
_LOST_COLOR = np.array([1.0, 0.2, 0.2])
_KF_COLOR = np.array([0.85, 0.5, 0.15])
_BBOX_COLOR = np.array([0.6, 0.65, 0.75])


class ObjectFrameView(Base3DView):
    def __init__(self, opts, visible: bool = True):
        super().__init__(
            title=f"Point2Pose 3D | Object Frame (obj {int(opts.obj_id)})",
            width=opts.window_width,
            height=opts.window_height,
            left=opts.window_left,
            top=opts.window_top,
            visible=visible,
        )
        self._opts = opts
        self._map_pcd = o3d.geometry.PointCloud()
        self._traj = o3d.geometry.LineSet()
        self._cam = o3d.geometry.LineSet()
        self._kfs = o3d.geometry.LineSet()
        self._bbox = o3d.geometry.LineSet()
        self._mesh = o3d.geometry.TriangleMesh()
        set_degenerate_mesh(self._mesh)  # empty meshes spam renderer warnings
        self._axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(opts.frustum_scale)
        )
        self._cam_centers = []
        self._latest_snap = None

    def _geometries(self):
        return [
            self._axes,
            self._map_pcd,
            self._mesh,
            self._traj,
            self._kfs,
            self._cam,
            self._bbox,
        ]

    def update(self, snap, mesh=None):
        """Refresh from an ObjectSnapshot; ``mesh`` (already in the object
        frame) is only passed in when the reconstruction changed."""
        self._latest_snap = snap
        scale = float(self._opts.frustum_scale)

        set_pointcloud(self._map_pcd, snap.map_points, self._map_colors(snap))

        center = snap.T_obj_cam[:3, 3]
        if (
            not self._cam_centers
            or np.linalg.norm(center - self._cam_centers[-1]) > 1e-9
        ):
            self._cam_centers.append(center.copy())
            if len(self._cam_centers) > int(self._opts.max_trajectory):
                self._cam_centers.pop(0)
        set_lineset(
            self._traj, *polyline(np.asarray(self._cam_centers), _TRAJ_OLD, _TRAJ_NEW)
        )

        cam_pts, cam_lines = frustum_wireframe(snap.T_obj_cam, scale)
        set_lineset(self._cam, cam_pts, cam_lines, _LOST_COLOR if snap.lost else _CAM_COLOR)

        if bool(self._opts.show_keyframes):
            set_lineset(
                self._kfs,
                *merged_frustums(snap.keyframe_T_obj_cam, 0.6 * scale, _KF_COLOR),
            )
        else:
            set_lineset(self._kfs, [], [], [])

        if bool(self._opts.show_bbox) and snap.bbox_extent is not None:
            corners, edges = box_wireframe(snap.bbox_extent)
            set_lineset(self._bbox, corners, edges, _BBOX_COLOR)
        else:
            set_lineset(self._bbox, [], [], [])

        if not bool(self._opts.show_mesh):
            set_degenerate_mesh(self._mesh)
        elif mesh is not None:
            self._mesh.vertices = mesh.vertices
            self._mesh.triangles = mesh.triangles
            self._mesh.vertex_colors = mesh.vertex_colors
            self._mesh.vertex_normals = mesh.vertex_normals

        self.show()

    def _map_colors(self, snap):
        mode = str(self._opts.point_color_mode)
        if mode == "uncertainty":
            return uncertainty_colors(snap.map_point_uncertainties)
        if mode == "object":
            return object_color(snap.obj_id)
        return frame_id_colors(snap.map_point_frames)

    def _set_initial_view(self, view_control):
        # Start from (roughly) the physical camera's side of the object.
        front = np.array([0.4, -0.4, -0.8])
        up = np.array([0.0, -1.0, 0.0])
        if self._latest_snap is not None:
            cam_center = self._latest_snap.T_obj_cam[:3, 3]
            norm = np.linalg.norm(cam_center)
            if norm > 1e-6:
                front = cam_center / norm
            up = -self._latest_snap.T_obj_cam[:3, 1]  # image "up" of the camera
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(0.7)

    def reset(self):
        """Drop accumulated state (camera trajectory, mesh, map)."""
        self._cam_centers = []
        self._latest_snap = None
        set_pointcloud(self._map_pcd, [], [])
        for geom in (self._traj, self._cam, self._kfs, self._bbox):
            set_lineset(geom, [], [], [])
        set_degenerate_mesh(self._mesh)
