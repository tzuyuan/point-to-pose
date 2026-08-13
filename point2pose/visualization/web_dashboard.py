"""viser-based web UI — the style of viewer used by Gaussian-splatting /
Nerfstudio-class demos.

A viser server hosts a single browser scene with two stages side by side:

  * ``/map``  — object frame: the keypoint map, SDF mesh, bounding box, the
    full camera trajectory, the current camera frustum textured with the live
    RGB frame, and keyframe frustums with RGB thumbnails.
  * ``/live`` — camera frame (oriented like the camera was at start): the
    sensor frustum with the live image, currently tracked points, and fading
    point trails on the object.

The right-hand panel carries the live RGB feed, display toggles and sliders,
color modes, view shortcuts, pose/health stats, and a registration-residual
plot. Rendering happens client-side in the browser, so orbiting stays smooth
regardless of the pipeline frame rate.

This module imports viser at construction time only — environments without
viser can still use the ``combined`` / ``windows`` UI modes.
"""

import time
import webbrowser
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.visualization.geometry import (
    box_wireframe,
    frame_id_colors,
    object_color,
    trail_segments,
    uncertainty_colors,
)

_TRAJ_OLD = np.array([90, 95, 115])
_TRAJ_NEW = np.array([255, 204, 26])
_CAM_COLOR = (26, 255, 90)
_LOST_COLOR = (255, 60, 60)
_KF_COLOR = (255, 140, 50)
_BBOX_COLOR = (150, 165, 190)
_TRAIL_FADE = np.array([70, 74, 84])
_ACCENT = (20, 200, 120)


def _wxyz_position(T):
    q = scipy_R.from_matrix(np.asarray(T)[:3, :3]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]]), np.asarray(T)[:3, 3]


def _to_rgb(img_bgr, width):
    """BGR -> RGB, resized to the given width (keeps aspect)."""
    import cv2

    h, w = img_bgr.shape[:2]
    height = max(1, int(round(h * width / w)))
    return cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)[..., ::-1]


def _u8(colors01):
    return (np.asarray(colors01) * 255.0).clip(0, 255).astype(np.uint8)


class WebDashboard:
    """Owns the viser server and mirrors pipeline state into the browser."""

    def __init__(self, cfg):
        import viser  # deferred: optional dependency

        self.cfg = cfg
        web = cfg.web
        self._server = viser.ViserServer(
            host=str(web.host), port=int(web.port), label="Point2Pose"
        )
        try:
            self._server.gui.configure_theme(
                dark_mode=True, control_layout="collapsible", brand_color=_ACCENT
            )
        except Exception:
            pass

        self._build_gui()

        # runtime state
        self._nodes = {}  # dynamic scene nodes, re-added per frame
        self._kf_handles = {}  # kf_idx -> frustum handle (pose updated in place)
        self._grid = None
        self._history = {}  # obj_id -> deque of NaN-padded (K,3) arrays
        self._cam_centers = []
        self._last_mesh = None  # cached trimesh so toggling mesh back on works
        self._stage_ready = False
        self._live_T = np.eye(4)
        self._orbit = None  # (position, look_at) applied to connecting clients
        self._fps = None
        self._last_t = None
        self._residual_log = deque(maxlen=400)
        self._frame_log = deque(maxlen=400)
        self._plot_counter = 0

        @self._server.on_client_connect
        def _(client):  # new browser tabs start on the orbit view
            if self._orbit is not None:
                client.camera.position, client.camera.look_at = self._orbit

        url = f"http://localhost:{int(web.port)}"
        print(f"[viz3d] web UI ready at {url}")
        if bool(web.open_browser):
            try:
                webbrowser.open(url)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GUI panel
    # ------------------------------------------------------------------
    def _build_gui(self):
        gui = self._server.gui
        self._md_status = gui.add_markdown(
            "**Waiting for tracking** — click prompt points in the tracker "
            "window, then press `s`."
        )
        try:
            self._img_panel = gui.add_image(
                np.zeros((45, 80, 3), dtype=np.uint8), label="RGB feed"
            )
        except Exception:
            self._img_panel = None

        with gui.add_folder("Display"):
            self._cb_map = gui.add_checkbox("Keypoint map", True)
            self._cb_mesh = gui.add_checkbox(
                "SDF mesh", bool(self.cfg.object_view.show_mesh)
            )
            self._cb_kfs = gui.add_checkbox(
                "Keyframes", bool(self.cfg.object_view.show_keyframes)
            )
            self._cb_traj = gui.add_checkbox("Camera trajectory", True)
            self._cb_bbox = gui.add_checkbox(
                "Bounding box", bool(self.cfg.object_view.show_bbox)
            )
            self._cb_live = gui.add_checkbox("Live stage", True)
            self._cb_trails = gui.add_checkbox(
                "Point trails", bool(self.cfg.camera_view.get("show_trails", True))
            )
            self._cb_grid = gui.add_checkbox("Ground grid", True)
            self._sl_psize = gui.add_slider(
                "Point size", min=0.001, max=0.02, step=0.001, initial_value=0.004
            )
            self._sl_trail = gui.add_slider(
                "Trail length", min=5, max=100, step=5,
                initial_value=int(self.cfg.camera_view.trail_length),
            )
            self._dd_map_color = gui.add_dropdown(
                "Map colors", ("frame_id", "uncertainty", "object"),
                initial_value=str(self.cfg.object_view.point_color_mode),
            )
            self._dd_pt_color = gui.add_dropdown(
                "Point colors", ("object", "frame_id", "uncertainty"),
                initial_value=str(self.cfg.camera_view.point_color_mode),
            )

        with gui.add_folder("Views"):
            gui.add_button("Orbit object").on_click(lambda _: self._goto_orbit())
            gui.add_button("Current camera").on_click(lambda _: self._goto_camera())
            gui.add_button("Live stage").on_click(lambda _: self._goto_live())

        with gui.add_folder("Stats"):
            self._md_stats = gui.add_markdown("*no data yet*")

        self._plot = None
        if bool(self.cfg.web.residual_plot):
            try:
                import plotly.graph_objects as go

                with gui.add_folder("Health"):
                    self._plot = gui.add_plotly(go.Figure(), aspect=2.2)
            except Exception:
                self._plot = None

    # ------------------------------------------------------------------
    # per-frame update
    # ------------------------------------------------------------------
    def update(self, scene, obj_snap, mesh, overlay_bgr, fov, aspect, new_kf_thumbs):
        """Mirror the current pipeline state into the browser scene.

        ``mesh`` is an (object-frame) Open3D mesh only when it changed;
        ``new_kf_thumbs`` is a list of (kf_idx, rgb_thumbnail) for keyframes
        created since the last call.
        """
        now = time.time()
        if self._last_t is not None:
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._fps = inst if self._fps is None else 0.9 * self._fps + 0.1 * inst
        self._last_t = now

        rgb = None
        if overlay_bgr is not None:
            rgb = _to_rgb(overlay_bgr, int(self.cfg.web.image_width))

        with self._server.atomic():
            if obj_snap is not None:
                self._ensure_stages(obj_snap)
                self._update_map_stage(obj_snap, mesh, rgb, fov, aspect, new_kf_thumbs)
            self._update_live_stage(scene, rgb, fov, aspect)
        self._update_panel(scene, obj_snap, rgb)

    # ------------------------------------------------------------------
    # scene: static setup
    # ------------------------------------------------------------------
    def _ensure_stages(self, snap):
        if self._stage_ready:
            return
        self._stage_ready = True
        scene = self._server.scene

        # Viewer up = the physical camera's image "up" at start.
        up = -snap.T_obj_cam[:3, 1]
        try:
            scene.set_up_direction(up)
        except Exception:
            pass

        # Ground grid below the object, perpendicular to the viewer up.
        extent = 0.25 if snap.bbox_extent is None else float(np.linalg.norm(snap.bbox_extent))
        try:
            rot = scipy_R.align_vectors([up], [[0.0, 0.0, 1.0]])[0]
            q = rot.as_quat()
            self._grid = scene.add_grid(
                "/map/grid",
                width=6 * extent,
                height=6 * extent,
                plane="xy",
                cell_size=0.5 * extent,
                cell_color=(62, 66, 76),
                cell_thickness=0.6,
                section_size=2 * extent,
                section_color=(88, 94, 108),
                wxyz=np.array([q[3], q[0], q[1], q[2]]),
                position=-0.6 * extent * up,
            )
        except Exception:
            self._grid = None
            self._cb_grid.visible = False

        scene.add_frame(
            "/map/axes", show_axes=True, axes_length=0.05, axes_radius=0.0015
        )
        try:
            scene.add_label("/map/label", "object frame · map", position=0.9 * extent * up)
        except Exception:
            pass

        # Live stage: frozen initial-camera axes, offset to the camera's right.
        R0 = snap.T_obj_cam[:3, :3]
        live_pos = R0[:, 0] * float(self.cfg.web.live_stage_offset)
        self._live_T = np.eye(4)
        self._live_T[:3, :3] = R0
        self._live_T[:3, 3] = live_pos
        wxyz, pos = _wxyz_position(self._live_T)
        scene.add_frame("/live", show_axes=False, wxyz=wxyz, position=pos)
        try:
            scene.add_label(
                "/live/label", "camera frame · trails", position=(0.0, -0.5 * extent, 0.0)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # scene: object-frame stage
    # ------------------------------------------------------------------
    def _update_map_stage(self, snap, mesh, rgb, fov, aspect, new_kf_thumbs):
        scene = self._server.scene
        fscale = float(self.cfg.object_view.frustum_scale)

        self._set_node(
            "/map/points",
            self._cb_map.value and len(snap.map_points) > 0,
            lambda: scene.add_point_cloud(
                "/map/points",
                points=snap.map_points.astype(np.float32),
                colors=self._map_colors(snap),
                point_size=float(self._sl_psize.value),
                point_shape="circle",
            ),
        )

        # SDF mesh: cache the latest one so toggling it back on re-adds it.
        if mesh is not None:
            self._last_mesh = self._as_trimesh(mesh)
        want_mesh = self._cb_mesh.value and self._last_mesh is not None
        if mesh is not None or want_mesh != ("/map/mesh" in self._nodes):
            self._set_node(
                "/map/mesh",
                want_mesh,
                lambda: scene.add_mesh_trimesh("/map/mesh", self._last_mesh),
            )

        # camera trajectory
        center = snap.T_obj_cam[:3, 3]
        if not self._cam_centers or np.linalg.norm(center - self._cam_centers[-1]) > 1e-9:
            self._cam_centers.append(center.copy())
            if len(self._cam_centers) > int(self.cfg.object_view.max_trajectory):
                self._cam_centers.pop(0)
        n = len(self._cam_centers)
        self._set_node(
            "/map/trajectory",
            self._cb_traj.value and n >= 2,
            lambda: scene.add_line_segments(
                "/map/trajectory",
                points=self._traj_segments(),
                colors=self._traj_colors(),
                line_width=3.0,
            ),
        )

        # current camera frustum with the live image
        wxyz, pos = _wxyz_position(snap.T_obj_cam)
        self._set_node(
            "/map/camera",
            True,
            lambda: scene.add_camera_frustum(
                "/map/camera",
                fov=fov,
                aspect=aspect,
                scale=1.3 * fscale,
                color=_LOST_COLOR if snap.lost else _CAM_COLOR,
                image=rgb,
                jpeg_quality=75,
                wxyz=wxyz,
                position=pos,
            ),
        )

        # keyframe frustums: created once (with thumbnail), poses kept fresh
        for kf_idx, thumb in new_kf_thumbs:
            self._add_kf_frustum(kf_idx, thumb, fov, aspect, fscale)
        for kf_idx in range(len(snap.keyframe_T_obj_cam)):
            if kf_idx not in self._kf_handles:
                self._add_kf_frustum(kf_idx, None, fov, aspect, fscale)
        for kf_idx, handle in self._kf_handles.items():
            if kf_idx >= len(snap.keyframe_T_obj_cam):
                continue
            try:
                handle.visible = bool(self._cb_kfs.value)
                handle.wxyz, handle.position = _wxyz_position(
                    snap.keyframe_T_obj_cam[kf_idx]
                )
            except Exception:
                pass

        self._set_node(
            "/map/bbox",
            self._cb_bbox.value and snap.bbox_extent is not None,
            lambda: scene.add_line_segments(
                "/map/bbox",
                points=self._box_segments(snap.bbox_extent, None),
                colors=np.array(_BBOX_COLOR, dtype=np.uint8),
                line_width=2.0,
            ),
        )

        if self._grid is not None:
            try:
                self._grid.visible = bool(self._cb_grid.value)
            except Exception:
                pass

    def _add_kf_frustum(self, kf_idx, thumb, fov, aspect, fscale):
        self._kf_handles[kf_idx] = self._server.scene.add_camera_frustum(
            f"/map/keyframes/kf_{kf_idx}",
            fov=fov,
            aspect=aspect,
            scale=0.55 * fscale,
            color=_KF_COLOR,
            image=thumb,
            jpeg_quality=70,
        )

    # ------------------------------------------------------------------
    # scene: camera-frame ("live") stage
    # ------------------------------------------------------------------
    def _update_live_stage(self, scene_snap, rgb, fov, aspect):
        scene = self._server.scene
        show = self._cb_live.value and self._stage_ready
        fscale = float(self.cfg.camera_view.frustum_scale)

        self._set_node(
            "/live/sensor",
            show,
            lambda: scene.add_camera_frustum(
                "/live/sensor",
                fov=fov,
                aspect=aspect,
                scale=fscale,
                color=(150, 160, 180),
                image=rgb,
                jpeg_quality=70,
            ),
        )

        # trail history is recorded even while hidden, so toggling is seamless
        maxlen = int(self._sl_trail.value)
        all_pts, all_cols, segs, seg_cols = [], [], [], []
        for snap in scene_snap.objects:
            hist = self._history.get(snap.obj_id)
            if hist is None or hist.maxlen != maxlen:
                hist = deque(list(hist or []), maxlen=maxlen)
                self._history[snap.obj_id] = hist
            hist.append(snap.track_points_cam)

            good = np.isfinite(snap.track_points_cam).all(axis=1)
            if good.any():
                all_pts.append(snap.track_points_cam[good])
                all_cols.append(self._live_point_colors(snap, good))

            base = _u8(object_color(snap.obj_id))
            pts, lines, cols = trail_segments(
                list(hist), _TRAIL_FADE / 255.0, base / 255.0
            )
            if len(lines):
                segs.append(pts[lines])
                seg_cols.append(np.repeat(_u8(cols)[:, None, :], 2, axis=1))

            if snap.bbox_extent is not None:
                self._set_node(
                    f"/live/bbox_{snap.obj_id}",
                    show and self._cb_bbox.value,
                    lambda s=snap: scene.add_line_segments(
                        f"/live/bbox_{s.obj_id}",
                        points=self._box_segments(s.bbox_extent, s.T_cam_obj),
                        colors=_u8(object_color(s.obj_id) * 0.85),
                        line_width=2.0,
                    ),
                )
            wxyz, pos = _wxyz_position(snap.T_cam_obj)
            self._set_node(
                f"/live/obj_axes_{snap.obj_id}",
                show,
                lambda w=wxyz, p=pos, s=snap: scene.add_frame(
                    f"/live/obj_axes_{s.obj_id}",
                    show_axes=True,
                    axes_length=0.6 * fscale,
                    axes_radius=0.0015,
                    wxyz=w,
                    position=p,
                ),
            )

        self._set_node(
            "/live/points",
            show and bool(all_pts),
            lambda: scene.add_point_cloud(
                "/live/points",
                points=np.concatenate(all_pts).astype(np.float32),
                colors=np.concatenate(all_cols),
                point_size=float(self._sl_psize.value),
                point_shape="circle",
            ),
        )
        self._set_node(
            "/live/trails",
            show and self._cb_trails.value and bool(segs),
            lambda: scene.add_line_segments(
                "/live/trails",
                points=np.concatenate(segs).astype(np.float32),
                colors=np.concatenate(seg_cols),
                line_width=2.0,
            ),
        )

    # ------------------------------------------------------------------
    # panel updates + view shortcuts
    # ------------------------------------------------------------------
    def _update_panel(self, scene_snap, obj_snap, rgb):
        if rgb is not None and self._img_panel is not None:
            try:
                self._img_panel.image = rgb
            except Exception:
                pass

        fps = f" · {self._fps:.1f} fps" if self._fps else ""
        self._md_status.content = f"**Tracking** · frame {scene_snap.frame_id}{fps}"

        rows = [
            "| obj | t [m] | rpy [deg] | pts | map | kf | res mm | inl |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for snap in scene_snap.objects:
            t = snap.T_cam_obj[:3, 3]
            rpy = scipy_R.from_matrix(snap.T_cam_obj[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            n_vis = int(np.isfinite(snap.track_points_cam).all(axis=1).sum())
            name = f"**{snap.obj_id}** ⚠" if snap.lost else f"**{snap.obj_id}**"
            rows.append(
                f"| {name} "
                f"| {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f} "
                f"| {rpy[0]:+.0f} {rpy[1]:+.0f} {rpy[2]:+.0f} "
                f"| {n_vis}/{len(snap.track_points_cam)} "
                f"| {len(snap.map_points)} "
                f"| {len(snap.keyframe_T_obj_cam)} "
                f"| {snap.mean_residual * 1000.0:.1f} "
                f"| {snap.num_inliers if snap.num_inliers >= 0 else '-'} |"
            )
        self._md_stats.content = "\n".join(rows)

        if obj_snap is not None:
            # camera shortcuts follow the current pose
            center = obj_snap.T_obj_cam[:3, 3]
            dist = max(float(np.linalg.norm(center)), 0.3)
            direction = center / max(np.linalg.norm(center), 1e-6)
            self._orbit = (1.4 * dist * direction, np.zeros(3))
            self._residual_log.append(obj_snap.mean_residual * 1000.0)
            self._frame_log.append(scene_snap.frame_id)
            self._plot_counter += 1
            if self._plot is not None and self._plot_counter % 10 == 0:
                self._update_plot()

    def _update_plot(self):
        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Scatter(
                    x=list(self._frame_log), y=list(self._residual_log),
                    mode="lines", line=dict(color="rgb(20,200,120)", width=2),
                )
            )
            fig.update_layout(
                title=dict(text="registration residual [mm]", font=dict(size=12)),
                margin=dict(l=30, r=10, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.05)",
                font=dict(size=9, color="#aaa"),
                showlegend=False,
            )
            self._plot.figure = fig
        except Exception:
            self._plot = None

    def _goto_orbit(self):
        if self._orbit is not None:
            self._apply_camera(*self._orbit)

    def _goto_camera(self):
        if self._orbit is None or not self._cam_centers:
            return
        pos = self._cam_centers[-1]
        self._apply_camera(pos, np.zeros(3))

    def _goto_live(self):
        if not self._stage_ready:
            return
        anchor = self._live_T[:3, 3] + self._live_T[:3, :3] @ np.array([0, 0, 0.4])
        pos = self._live_T[:3, 3] + self._live_T[:3, :3] @ np.array([0, -0.25, -0.6])
        self._apply_camera(pos, anchor)

    def _apply_camera(self, position, look_at):
        for client in self._server.get_clients().values():
            client.camera.position = np.asarray(position)
            client.camera.look_at = np.asarray(look_at)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _set_node(self, name, enabled, add_fn):
        """Re-add a dynamic node (same path replaces it) or remove it."""
        if enabled:
            self._nodes[name] = add_fn()
        else:
            handle = self._nodes.pop(name, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

    def _map_colors(self, snap):
        mode = str(self._dd_map_color.value)
        if mode == "uncertainty":
            return _u8(uncertainty_colors(snap.map_point_uncertainties))
        if mode == "object":
            return _u8(object_color(snap.obj_id))
        return _u8(frame_id_colors(snap.map_point_frames))

    def _live_point_colors(self, snap, good):
        mode = str(self._dd_pt_color.value)
        if mode == "frame_id":
            return _u8(frame_id_colors(snap.track_point_frames[good]))
        if mode == "uncertainty":
            return _u8(uncertainty_colors(snap.track_point_uncertainties[good]))
        return np.tile(_u8(object_color(snap.obj_id)), (int(good.sum()), 1))

    def _traj_segments(self):
        pts = np.asarray(self._cam_centers, dtype=np.float32)
        return np.stack([pts[:-1], pts[1:]], axis=1)

    def _traj_colors(self):
        n = len(self._cam_centers) - 1
        tt = np.linspace(0.0, 1.0, n)[:, None]
        cols = (_TRAJ_OLD * (1 - tt) + _TRAJ_NEW * tt).astype(np.uint8)
        return np.repeat(cols[:, None, :], 2, axis=1)

    @staticmethod
    def _box_segments(extent, T):
        corners, edges = box_wireframe(extent, T)
        return corners[edges].astype(np.float32)

    @staticmethod
    def _as_trimesh(o3d_mesh):
        import trimesh

        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        kwargs = {}
        colors = np.asarray(o3d_mesh.vertex_colors)
        if len(colors) == len(vertices) and len(colors) > 0:
            kwargs["vertex_colors"] = _u8(colors)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False, **kwargs)

    # ------------------------------------------------------------------
    def reset(self):
        """Clear per-run scene content; the server and panel stay up."""
        for handle in list(self._nodes.values()) + list(self._kf_handles.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._nodes = {}
        self._kf_handles = {}
        self._history = {}
        self._cam_centers = []
        self._last_mesh = None
        self._fps = None
        self._last_t = None
        self._residual_log.clear()
        self._frame_log.clear()
        self._md_status.content = (
            "**Waiting for tracking** — click prompt points in the tracker "
            "window, then press `s`."
        )

    def close(self):
        # Drain queued messages first: stopping mid-flush makes viser's
        # background loop print a harmless-but-noisy asyncio traceback.
        try:
            self._server.flush()
            time.sleep(0.1)
        except Exception:
            pass
        try:
            self._server.stop()
        except Exception:
            pass
        # viser registers its own atexit stop; after an explicit stop that
        # callback hits a closed event loop and prints a spurious traceback.
        try:
            import atexit

            atexit.unregister(self._server._websock_server.stop)
        except Exception:
            pass
