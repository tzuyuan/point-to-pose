"""Facade tying the 3D demo visualization to a running ModularPipeline.

Plug-in usage (no pipeline or runner modifications required):

    from point2pose.visualization import Pose3DVisualizer

    viz = Pose3DVisualizer(cfg.get("visualization_3d"))  # None -> defaults
    ...
    pipeline.step(frame)
    canvas = viz.update(pipeline, frame, overlay_2d=display_bgr)
    if canvas is not None:          # combined mode only
        cv2.imshow(window, canvas)
    ...
    viz.close()

Four UI modes (``ui_mode``):
  * ``rerun`` (default) — logs into the Rerun viewer (the kv_tracker-style
    native app): object-fixed + camera-fixed 3D views, image-textured
    frustums, keyframe thumbnails, health plots, event log, and a timeline
    scrubber that replays the whole session; optional ``.rrd`` recording.
    ``update`` returns None. Falls back to ``web``, then ``combined``.
  * ``web`` — a viser server renders the scene in the browser.
  * ``combined`` — one SLAM-style cv2 dashboard canvas; ``update`` returns
    the canvas and mouse events go to ``handle_mouse``.
  * ``windows`` — two separate interactive Open3D windows.

The visualizer only *reads* pipeline state (via SceneSnapshot); it works with
any runner built on ModularPipeline (RealSense demo, dataset playback, ...).
"""

import numpy as np
from omegaconf import OmegaConf

from point2pose.visualization.snapshot import SceneSnapshot

DEFAULT_CONFIG = {
    "enabled": True,
    "ui_mode": "rerun",  # rerun | web | combined | windows
    # Save PNGs every frame (combined: dashboard canvas; windows: one PNG per
    # view; rerun/web: not applicable — use .rrd recording / the browser).
    "save_images": False,
    "output_image_dir": "./debug/viz3d",
    # Rerun UI (kv_tracker-style viewer).
    "rerun": {
        "app_name": "point2pose",
        "spawn": True,  # launch the native Rerun viewer
        "save_rrd": None,  # path to also record the session (.rrd), or null
        "collapse_panels": True,
        "image_width": 480,  # streamed RGB overlay width
        "keyframe_images": True,  # RGB thumbnails on keyframe frustums
        "keyframe_image_width": 160,
        "map_color_mode": "track_id",  # track_id | frame_id | uncertainty | object
        # track_id | inlier | frame_id | uncertainty | object
        # (the pts:<mode> button on the control strip cycles this at runtime).
        # The camera-frame panel always shows the FULL keypoint map; visible
        # points are lit, the rest dimmed. track_id = stable color per point.
        "point_color_mode": "track_id",
        "point_radius": 0.003,  # meters
        "trail_length": 8,  # frames of trace history per point
        "trail_max_gap": 5,  # drop a trace unseen for this many frames
        # residual plot: values clipped here, y-axis pinned to (-1, 1.05*cap)
        "residual_cap_mm": 20.0,
        # initial states of the show/hide buttons on the control strip
        "show": {
            "map": True,
            "mesh": True,
            "kfs": True,
            "traj": True,
            "bbox": False,
            "traces": True,
            "2d": True,
            "mask": False,
            "reproj": True,
        },
    },
    # Web (viser) UI.
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "open_browser": True,  # open the viewer in the default browser
        "image_width": 360,  # streamed RGB width (frustum texture + panel)
        "keyframe_images": True,  # RGB thumbnails on keyframe frustums
        "keyframe_image_width": 160,
        "live_stage_offset": 0.8,  # meters, camera-frame stage offset
        "residual_plot": True,  # registration-residual plot in the panel
    },
    # SDF-mesh continuity filter (same as the mesh visualizer script's
    # filter_disconnected_components): keep the largest connected components,
    # dropping marching-cubes debris before display. Applies to all UI modes.
    "mesh_filter": {
        "enabled": True,
        "keep_components": 1,
        "min_component_triangles": 0,
    },
    # Combined-dashboard layout; panel sizes override the window sizes below.
    "combined": {
        "main_panel_width": 880,  # object-frame panel
        "side_panel_width": 440,  # camera-frame + RGB panels (4:3)
        "record_fps": 15,  # fps stamped into recorded mp4 files
    },
    # Object-frame view: object fixed at the origin; cameras + map around it.
    "object_view": {
        "obj_id": 0,  # the object whose frame anchors this view
        "window_width": 800,
        "window_height": 600,
        "window_left": 60,
        "window_top": 80,
        "point_color_mode": "frame_id",  # frame_id | uncertainty | object
        "show_keyframes": True,
        "show_mesh": True,  # live SDF reconstruction
        "show_bbox": True,
        "max_trajectory": 4000,  # camera-trajectory vertices kept
        "frustum_scale": 0.06,  # meters
    },
    # Camera-frame view: sensor fixed at the origin; points leave trails.
    "camera_view": {
        "window_width": 800,
        "window_height": 600,
        "window_left": 880,
        "window_top": 80,
        "point_color_mode": "object",  # object | frame_id | uncertainty
        "trail_length": 30,  # frames of trail history
        "show_trails": True,
        "show_bbox": True,
        "frustum_scale": 0.08,  # meters
    },
}


class Pose3DVisualizer:
    """Owns the 3D UI (web, combined dashboard, or separate windows) and
    feeds it from pipeline state each frame."""

    def __init__(self, cfg=None):
        base = OmegaConf.create(DEFAULT_CONFIG)
        self.cfg = OmegaConf.merge(base, cfg) if cfg is not None else base
        self.enabled = bool(self.cfg.enabled)
        self.mode = str(self.cfg.ui_mode)

        self._rerun = None
        self._web = None
        self._dashboard = None
        self._object_view = None
        self._camera_view = None
        self._mesh_versions = {}
        self._kf_thumb_counts = {}
        self._capture_dir = None
        self._mesh_filter = None
        self._mesh_filter_failed = False

        if not self.enabled:
            return

        if self.mode == "rerun":
            try:
                from point2pose.visualization.rerun_dashboard import RerunDashboard

                self._rerun = RerunDashboard(self.cfg)
            except Exception as exc:
                print(f"[viz3d] rerun UI unavailable ({exc}); trying the web UI")
                self.mode = "web"

        if self.mode == "web":
            try:
                from point2pose.visualization.web_dashboard import WebDashboard

                self._web = WebDashboard(self.cfg)
            except Exception as exc:
                print(
                    f"[viz3d] web UI unavailable ({exc}); "
                    "falling back to the combined dashboard"
                )
                self.mode = "combined"

        if self.mode == "combined":
            from point2pose.visualization.dashboard import Dashboard

            self._dashboard = Dashboard(self.cfg)
            self._object_view = self._dashboard.object_view
            self._camera_view = self._dashboard.camera_view
        elif self.mode == "windows":
            from point2pose.visualization.camera_frame_view import CameraFrameView
            from point2pose.visualization.object_frame_view import ObjectFrameView

            self._object_view = ObjectFrameView(self.cfg.object_view)
            self._camera_view = CameraFrameView(self.cfg.camera_view)

        if bool(self.cfg.save_images) and self.mode != "web":
            from pathlib import Path

            self._capture_dir = Path(str(self.cfg.output_image_dir))
            self._capture_dir.mkdir(parents=True, exist_ok=True)

    @property
    def combined(self) -> bool:
        return self.mode == "combined"

    def update(self, pipeline, frame, overlay_2d=None):
        """Refresh the UI from the pipeline's current state.

        ``overlay_2d`` (BGR) is the runner's 2D tracking overlay: shown as a
        dashboard panel (combined) or streamed to the browser (web). Returns
        the dashboard canvas in combined mode, else None.
        """
        if not self.enabled:
            return None
        scene = SceneSnapshot.from_pipeline(pipeline, frame)

        target = int(self.cfg.object_view.obj_id)
        obj_snap = next((s for s in scene.objects if s.obj_id == target), None)
        mesh = self._updated_mesh(pipeline, obj_snap) if obj_snap is not None else None

        if self._rerun is not None:
            if overlay_2d is None and getattr(frame, "rgb", None) is not None:
                overlay_2d = np.asarray(frame.rgb)[..., ::-1]  # RGB -> BGR
            small = self._resized(overlay_2d, int(self.cfg.rerun.image_width))
            K, w, h = self._intrinsics(frame)
            mask_classes = None
            if small is not None:
                mask_classes = self._mask_class_image(
                    frame, (small.shape[1], small.shape[0])
                )
            thumbs = (
                self._new_keyframe_thumbs(pipeline, obj_snap)
                if obj_snap is not None
                else []
            )
            self._rerun.update(
                scene, obj_snap, mesh, small, K, w, h, thumbs, mask_classes
            )
            # the cv2 window shows the overlay plus the show/hide buttons
            if overlay_2d is not None:
                return self._rerun.attach_controls(np.ascontiguousarray(overlay_2d))
            return None

        if self._web is not None:
            if overlay_2d is None and getattr(frame, "rgb", None) is not None:
                overlay_2d = np.asarray(frame.rgb)[..., ::-1]  # RGB -> BGR
            fov, aspect = self._fov_aspect(frame)
            thumbs = (
                self._new_keyframe_thumbs(pipeline, obj_snap)
                if obj_snap is not None
                else []
            )
            self._web.update(scene, obj_snap, mesh, overlay_2d, fov, aspect, thumbs)
            return None

        if self._dashboard is not None:
            canvas = self._dashboard.update(scene, obj_snap, mesh, overlay_2d)
            if self._capture_dir is not None:
                import cv2

                cv2.imwrite(
                    str(self._capture_dir / f"dashboard_{scene.frame_id:06d}.png"),
                    canvas,
                )
            return canvas

        if obj_snap is not None:
            self._object_view.update(obj_snap, mesh)
        self._camera_view.update(scene)
        if self._capture_dir is not None:
            self._object_view.capture(
                self._capture_dir / f"object_frame_{scene.frame_id:06d}.png"
            )
            self._camera_view.capture(
                self._capture_dir / f"camera_frame_{scene.frame_id:06d}.png"
            )
        return None

    def handle_mouse(self, event, x, y, flags=0):
        """Forward mouse events from the host cv2 window: dashboard
        navigation (combined mode) or show/hide buttons (rerun mode)."""
        if not self.enabled:
            return
        if self._rerun is not None:
            self._rerun.handle_mouse(event, x, y, flags)
        elif self._dashboard is not None:
            self._dashboard.handle_mouse(event, x, y, flags)

    def set_display_window(self, name):
        """Name of the host cv2 window showing the dashboard; enables
        immediate redraws while dragging (combined mode)."""
        if self._dashboard is not None:
            self._dashboard.set_live_window(name)

    @staticmethod
    def _fov_aspect(frame):
        """Vertical FoV (rad) and aspect from the frame's intrinsics."""
        rgb = getattr(frame, "rgb", None)
        intr = getattr(frame, "intrinsics", None)
        if rgb is None or intr is None:
            return np.deg2rad(55.0), 4.0 / 3.0
        h, w = rgb.shape[:2]
        fy = float(np.asarray(intr)[1, 1])
        return 2.0 * np.arctan(h / (2.0 * fy)), w / h

    @staticmethod
    def _intrinsics(frame):
        """(K, width, height) from the frame, with sane defaults."""
        rgb = getattr(frame, "rgb", None)
        h, w = (480, 640) if rgb is None else rgb.shape[:2]
        intr = getattr(frame, "intrinsics", None)
        if intr is None:
            f = h / (2.0 * np.tan(np.deg2rad(55.0) / 2.0))
            intr = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        return np.asarray(intr, dtype=np.float64), w, h

    @staticmethod
    def _resized(img, width):
        """Downscale a BGR image to the given width (no-op if smaller)."""
        if img is None or img.shape[1] <= width:
            return img
        import cv2

        height = max(1, int(round(img.shape[0] * width / img.shape[1])))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _mask_class_image(frame, size_wh):
        """Collapse per-object segmentation logits ([N,1,H,W], torch or
        numpy) into one uint8 class image (0 = background, i+1 = object i),
        resized to ``size_wh``. Returns None if the frame has no mask."""
        mask = getattr(frame, "mask", None)
        if mask is None:
            return None
        try:
            if hasattr(mask, "detach"):
                mask = mask.detach()
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            mask = np.asarray(mask)
            if mask.ndim != 4 or mask.shape[0] == 0:
                return None
            classes = np.zeros(mask.shape[2:], dtype=np.uint8)
            for i in range(mask.shape[0]):
                classes[mask[i, 0] > 0] = i + 1
            import cv2

            return cv2.resize(classes, size_wh, interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None

    def _mesh_filter_fn(self):
        """Lazy-load ``filter_disconnected_components`` from the mesh
        visualizer script (scripts/debug_visualization) so the live SDF gets
        the exact same continuity filtering as the offline tool."""
        if self._mesh_filter is not None or self._mesh_filter_failed:
            return self._mesh_filter
        try:
            import importlib.util
            from pathlib import Path

            script = (
                Path(__file__).resolve().parents[2]
                / "scripts/debug_visualization/visualize_textured_mesh.py"
            )
            spec = importlib.util.spec_from_file_location("p2p_mesh_visualizer", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._mesh_filter = module.filter_disconnected_components
        except Exception as exc:
            print(f"[viz3d] mesh continuity filter unavailable ({exc})")
            self._mesh_filter_failed = True
        return self._mesh_filter

    def _new_keyframe_thumbs(self, pipeline, snap):
        """RGB thumbnails for keyframes created since the last call (shown on
        the keyframe frustums in the rerun/web UIs)."""
        node = self.cfg.rerun if self._rerun is not None else self.cfg.web
        if not bool(node.keyframe_images):
            return []
        keyframes = getattr(getattr(pipeline, "kf_manager", None), "keyframes", {})
        kf_list = keyframes.get(snap.obj_id, [])
        start = self._kf_thumb_counts.get(snap.obj_id, 0)
        self._kf_thumb_counts[snap.obj_id] = len(kf_list)

        out = []
        for i in range(start, len(kf_list)):
            rgb = getattr(getattr(kf_list[i], "frame", None), "rgb", None)
            thumb = None
            if rgb is not None:
                import cv2

                rgb = np.asarray(rgb)
                width = int(node.keyframe_image_width)
                height = max(1, int(round(rgb.shape[0] * width / rgb.shape[1])))
                thumb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            out.append((i, thumb))
        return out

    def _updated_mesh(self, pipeline, snap):
        """Extract the SDF mesh (in the object frame) when it changed;
        returns None while the cached version is still current."""
        if not bool(self.cfg.object_view.show_mesh) and self._web is None:
            # cv2/Open3D modes poll this toggle here; the web UI has its own
            # mesh checkbox and caches the last mesh itself.
            self._mesh_versions.clear()
            return None
        if self._mesh_versions.get(snap.obj_id) == snap.mesh_version:
            return None
        self._mesh_versions[snap.obj_id] = snap.mesh_version

        builder = getattr(pipeline, "sdf_builder", None)
        obj = pipeline.objects[snap.obj_id]
        if builder is None or getattr(obj, "sdf_volume", None) is None:
            return None
        try:
            mesh = builder._build_colored_o3d_mesh(obj)
        except Exception as exc:  # mesh display is best-effort
            print(f"[viz3d] SDF mesh extraction failed for obj {snap.obj_id}: {exc}")
            return None
        if mesh is None:
            return None

        # continuity filtering (same as the offline mesh visualizer)
        fcfg = self.cfg.mesh_filter
        if bool(fcfg.enabled) and len(mesh.triangles) > 0:
            filter_fn = self._mesh_filter_fn()
            if filter_fn is not None:
                try:
                    mesh, stats = filter_fn(
                        mesh,
                        keep_components=int(fcfg.keep_components),
                        min_component_triangles=int(fcfg.min_component_triangles),
                    )
                    if stats.get("removed_triangles", 0) > 0:
                        print(
                            f"[viz3d] mesh filter obj {snap.obj_id}: kept "
                            f"{stats['kept_components']}/{stats['num_components']} "
                            f"components, removed {stats['removed_triangles']} tris"
                        )
                except Exception as exc:
                    print(f"[viz3d] mesh continuity filter failed: {exc}")

        mesh.transform(snap.T_obj_world)  # first-camera frame -> object frame
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    def reset(self):
        """Clear accumulated state (trajectory, trails, mesh cache); the UI
        stays up and repopulates once tracking restarts."""
        self._mesh_versions = {}
        self._kf_thumb_counts = {}
        if self._rerun is not None:
            self._rerun.reset()
        elif self._web is not None:
            self._web.reset()
        elif self._dashboard is not None:
            self._dashboard.reset()
        elif self._object_view is not None:
            self._object_view.reset()
            self._camera_view.reset()

    def close(self):
        if self._rerun is not None:
            self._rerun.close()
        elif self._web is not None:
            self._web.close()
        elif self._dashboard is not None:
            self._dashboard.close()
        elif self._object_view is not None:
            self._object_view.close()
            self._camera_view.close()
