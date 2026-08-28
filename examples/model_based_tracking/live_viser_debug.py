"""
Optional viser debug view for live model tracking.

Shows the object mesh (fixed at the origin, object frame) and the CURRENT estimated camera
frustum each frame. Optionally overlays the PnP inlier map points on the mesh.

Debug workflow for "position is right but rotation is wrong":
  1. Run with --viser. Watch the frustum + magenta inlier points on the mesh.
  2. Tick "pause" in the viser GUI. The tracking loop stops grabbing frames, so the last
     frame's inliers stay on screen.
  3. Drag the "inlier" slider. The selected inlier is highlighted (yellow) on the mesh, and
     its SOURCE map view (the rendered image + that view's keypoint) is shown -- so you can
     see exactly which camera / keypoint that correspondence came from.

Requires the ModelMap (for per-inlier -> seed-view/keypoint traceback).
"""

import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from point2pose.utils.transform import inverse_SE3


class LiveViserDebug:
    def __init__(self, mesh_path, render_K, img_hw, model_map=None, mesh_scale=1.0, port=8081):
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.model_map = model_map

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh_scale != 1.0:
            mesh.apply_scale(mesh_scale)
        mesh.apply_translation(-mesh.vertices.mean(0))  # match map centering
        self.server.scene.add_mesh_trimesh("/mesh", mesh)
        self.server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002)

        self.H, self.W = int(img_hw[0]), int(img_hw[1])
        self.fov = float(2 * np.arctan2(self.H / 2, render_K[1, 1]))
        self.aspect = self.W / self.H

        # Per-map-point -> seed view index, and cumulative offsets to find the keypoint within
        # that view. Built once from the ModelMap's per-view keypoint list.
        self._view_of_row = None
        self._row_offset_in_view = None
        if model_map is not None:
            rows_view, rows_off = [], []
            for vi, kps in enumerate(model_map.view_kps):
                rows_view.extend([vi] * len(kps))
                rows_off.extend(range(len(kps)))
            self._view_of_row = np.asarray(rows_view, dtype=int)
            self._row_offset_in_view = np.asarray(rows_off, dtype=int)

        # GUI
        self.gui_pause = self.server.gui.add_checkbox("pause", initial_value=False)
        # Fixed max (a slider needs min<max; a runtime max=0 makes viser emit NaN). The actual
        # inlier count is usually far below this; the selection is clamped to it in code.
        self.gui_sel = self.server.gui.add_slider("inlier", min=0, max=500, step=1,
                                                  initial_value=0)
        self.gui_info = self.server.gui.add_markdown("")

        # cached state of the last frame (for the slider to inspect while paused)
        self._last_inlier_pts = None       # (K,3) object frame
        self._last_inlier_rows = None      # (K,) map rows
        self.gui_sel.on_update(lambda _: self._show_selected())
        print(f"[viser-debug] serving at http://localhost:{port}")

    @property
    def paused(self):
        return bool(self.gui_pause.value)

    # ------------------------------------------------------------------ #
    def update(self, T_obj2cam, rgb=None, lost=False, inlier_pts_obj=None, inlier_rows=None):
        """Called every tracking frame (skip when paused). Draws frustum + inliers."""
        # inlier point cloud (magenta) on the mesh
        if inlier_pts_obj is not None and len(inlier_pts_obj):
            self.server.scene.add_point_cloud(
                "/inliers",
                points=np.asarray(inlier_pts_obj, dtype=np.float32),
                colors=np.tile(np.array([[255, 0, 255]], np.uint8), (len(inlier_pts_obj), 1)),
                point_size=0.004,
            )
            self._last_inlier_pts = np.asarray(inlier_pts_obj, dtype=np.float32)
            self._last_inlier_rows = (np.asarray(inlier_rows, dtype=int)
                                      if inlier_rows is not None else None)

        T_cam2obj = inverse_SE3(np.asarray(T_obj2cam, dtype=np.float64))
        R, t = T_cam2obj[:3, :3], T_cam2obj[:3, 3]
        self.server.scene.add_camera_frustum(
            "/live_cam", fov=self.fov, aspect=self.aspect, scale=0.06,
            color=(255, 60, 60) if lost else (60, 200, 60),
            wxyz=vtf.SO3.from_matrix(R).wxyz, position=t.astype(np.float32), image=rgb,
        )

    # ------------------------------------------------------------------ #
    def _show_selected(self):
        """Highlight the selected inlier and show its source map view + keypoint."""
        if self._last_inlier_pts is None or not len(self._last_inlier_pts):
            return
        val = self.gui_sel.value
        if val is None or not np.isfinite(val):
            return
        k = int(np.clip(int(val), 0, len(self._last_inlier_pts) - 1))
        pt = self._last_inlier_pts[k]
        # highlight the selected 3D inlier (yellow, larger)
        self.server.scene.add_point_cloud(
            "/inlier_sel", points=pt[None].astype(np.float32),
            colors=np.array([[255, 255, 0]], np.uint8), point_size=0.012,
        )

        info = [f"inlier {k}/{len(self._last_inlier_pts)-1}",
                f"obj-frame xyz = {np.round(pt, 4).tolist()}"]

        # trace back to the source map view + keypoint, if we have the ModelMap
        if (self.model_map is not None and self._last_inlier_rows is not None
                and self._view_of_row is not None):
            row = int(self._last_inlier_rows[k])
            if 0 <= row < len(self._view_of_row):
                vi = int(self._view_of_row[row])
                off = int(self._row_offset_in_view[row])
                info.append(f"source map view = {vi}, keypoint #{off}")
                # draw that view's render with the source keypoint marked, in the GUI
                self._show_source_view_image(vi, off)
        self.gui_info.content = "  \n".join(info)

    def _show_source_view_image(self, vi, off):
        """Show the source map view as a camera frustum AT that view's actual pose, with the
        source keypoint marked -- so its position/orientation in space is meaningful (the raw
        image floating near the origin looked mirrored because it had no pose)."""
        import cv2

        if vi >= len(self.model_map.view_rgbs):
            return
        img = cv2.cvtColor(self.model_map.view_rgbs[vi].copy(), cv2.COLOR_RGB2BGR)
        kps = self.model_map.view_kps[vi]
        for j, (x, y) in enumerate(np.asarray(kps)):
            c = (0, 255, 255) if j == off else (0, 120, 0)
            r = 4 if j == off else 2
            cv2.circle(img, (int(round(x)), int(round(y))), r, c, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # place a frustum at the source view's camera pose (object frame)
        T_obj2cam = self.model_map.view_poses_obj2cam[vi]
        T_cam2obj = inverse_SE3(np.asarray(T_obj2cam, dtype=np.float64))
        R, t = T_cam2obj[:3, :3], T_cam2obj[:3, 3]
        self.server.scene.add_camera_frustum(
            "/source_view", fov=self.fov, aspect=self.aspect, scale=0.08,
            color=(255, 200, 0),
            wxyz=vtf.SO3.from_matrix(R).wxyz, position=t.astype(np.float32), image=img,
        )
