"""Single-window SLAM-style dashboard.

Composites both offscreen-rendered 3D views, the RGB tracking overlay, a
status bar (pose readout, tracking health, FPS), and clickable controls into
one BGR canvas. The canvas is returned to the caller for display — in the
RealSense demo it is shown by the *base* demo's own cv2 window, so the whole
UI lives in a single window.

Layout (gaps omitted):

    +--------------------------------+----------------------+
    |                                |  CAMERA FRAME        |
    |   OBJECT FRAME                 |  (point trails)      |
    |   (map + camera trajectory)    +----------------------+
    |                                |  RGB                 |
    |                                |  (tracking overlay)  |
    +--------------------------------+----------------------+
    |  [buttons]                                  fps  REC  |
    |  (base demo text zone)   per-object pose/health stats |
    +--------------------------------+----------------------+

Mouse on the 3D panels: left-drag = rotate, shift+left-drag or middle-drag =
pan, right-drag (vertical) or wheel = zoom. Buttons toggle display options
and start/stop mp4 recording of the dashboard.
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from point2pose.visualization.camera_frame_view import CameraFrameView
from point2pose.visualization.geometry import object_color
from point2pose.visualization.object_frame_view import ObjectFrameView

_GAP = 8
_BAR_H = 96
_LABEL_H = 22
# Leave the bottom-left of the status bar empty: the base demo draws its own
# "Frame: N" / "Objects: K" text there after we return the canvas.
_BASE_TEXT_ZONE_W = 270

_BG = (24, 21, 18)  # BGR
_PANEL_BG = (38, 34, 30)
_BORDER = (80, 72, 62)
_LABEL_BG = (52, 46, 40)
_TEXT = (232, 228, 222)
_DIM = (160, 152, 142)
_BTN_BG = (58, 52, 45)
_BTN_HOVER = (84, 76, 64)
_ACCENT = (120, 210, 140)
_LOST = (70, 70, 235)
_REC = (60, 60, 235)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_COLOR_MODES = ["frame_id", "uncertainty", "object"]


class Dashboard:
    """Owns the two offscreen 3D views and turns them + the RGB overlay into
    a single interactive canvas."""

    def __init__(self, cfg):
        self.cfg = cfg

        side_w = int(cfg.combined.side_panel_width)
        side_h = (side_w * 3) // 4  # match the 4:3 RealSense stream
        main_w = int(cfg.combined.main_panel_width)
        main_h = 2 * side_h + _GAP

        # The panels dictate the offscreen render sizes.
        cfg.object_view.window_width = main_w
        cfg.object_view.window_height = main_h
        cfg.camera_view.window_width = side_w
        cfg.camera_view.window_height = side_h
        self.object_view = ObjectFrameView(cfg.object_view, visible=False)
        self.camera_view = CameraFrameView(cfg.camera_view, visible=False)

        self._obj_rect = (_GAP, _GAP, main_w, main_h)
        self._cam_rect = (_GAP + main_w + _GAP, _GAP, side_w, side_h)
        self._rgb_rect = (_GAP + main_w + _GAP, _GAP + side_h + _GAP, side_w, side_h)
        self.width = _GAP + main_w + _GAP + side_w + _GAP
        self.height = _GAP + main_h + _GAP + _BAR_H + _GAP
        self._bar_rect = (_GAP, _GAP + main_h + _GAP, self.width - 2 * _GAP, _BAR_H)

        self._buttons = [
            (lambda: f"map:{self.cfg.object_view.point_color_mode}",
             lambda: self._cycle_mode(self.cfg.object_view)),
            (lambda: f"pts:{self.cfg.camera_view.point_color_mode}",
             lambda: self._cycle_mode(self.cfg.camera_view)),
            (lambda: self._toggle_label("mesh", self.cfg.object_view.show_mesh),
             lambda: self._toggle(self.cfg.object_view, "show_mesh")),
            (lambda: self._toggle_label("kfs", self.cfg.object_view.show_keyframes),
             lambda: self._toggle(self.cfg.object_view, "show_keyframes")),
            (lambda: self._toggle_label("trails", self.cfg.camera_view.get("show_trails", True)),
             lambda: self._toggle(self.cfg.camera_view, "show_trails")),
            (lambda: self._toggle_label("bbox", self.cfg.object_view.show_bbox),
             self._toggle_boxes),
            (lambda: "reset view", self._reset_views),
            (lambda: "rec:on" if self._recording else "rec:off", self._toggle_record),
        ]
        self._button_hits = []  # (x1, y1, x2, y2, action), filled while drawing

        # runtime state
        self._scene = None
        self._obj_snap = None
        self._overlay = None
        self._fps = None
        self._last_update_t = None
        self._drag = None  # {"view", "mode", "last"}
        self._hover = (-1, -1)
        self._recording = False
        self._writer = None
        self._writer_path = None
        self._live_window = None  # cv2 window name for live drag redraws
        self._last_live_draw = 0.0

    # ------------------------------------------------------------------
    # per-frame update
    # ------------------------------------------------------------------
    def update(self, scene, obj_snap, mesh, overlay_2d):
        """Push new pipeline state into the views and return the composited
        dashboard canvas (BGR uint8)."""
        now = time.time()
        if self._last_update_t is not None:
            dt = max(now - self._last_update_t, 1e-6)
            inst = 1.0 / dt
            self._fps = inst if self._fps is None else 0.9 * self._fps + 0.1 * inst
        self._last_update_t = now

        self._scene = scene
        self._obj_snap = obj_snap
        self._overlay = overlay_2d

        if obj_snap is not None:
            self.object_view.update(obj_snap, mesh)
        self.camera_view.update(scene)

        canvas = self._compose()
        if self._recording:
            self._write_video_frame(canvas)
        return canvas

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------
    def _compose(self):
        canvas = np.full((self.height, self.width, 3), _BG, dtype=np.uint8)
        self._blit_view(canvas, self._obj_rect, self.object_view,
                        "OBJECT FRAME  |  keypoint map + camera trajectory")
        self._blit_view(canvas, self._cam_rect, self.camera_view,
                        "CAMERA FRAME  |  point trails")
        self._blit_rgb(canvas)
        self._draw_bar(canvas)
        return canvas

    def _blit_view(self, canvas, rect, view, label):
        x, y, w, h = rect
        img = view.snapshot_image()
        if img is None:
            self._placeholder(canvas, rect, "waiting for tracking ...")
        else:
            if img.shape[0] != h or img.shape[1] != w:  # e.g. HiDPI scaling
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            canvas[y:y + h, x:x + w] = img
        self._frame_panel(canvas, rect, label)

    def _blit_rgb(self, canvas):
        x, y, w, h = self._rgb_rect
        if self._overlay is None:
            self._placeholder(canvas, self._rgb_rect, "no RGB overlay")
        else:
            canvas[y:y + h, x:x + w] = cv2.resize(
                self._overlay, (w, h), interpolation=cv2.INTER_AREA
            )
        self._frame_panel(canvas, self._rgb_rect, "RGB  |  tracking overlay")

    def _placeholder(self, canvas, rect, text):
        x, y, w, h = rect
        canvas[y:y + h, x:x + w] = _PANEL_BG
        cv2.putText(canvas, text, (x + w // 2 - 90, y + h // 2), _FONT, 0.5, _DIM, 1,
                    cv2.LINE_AA)

    def _frame_panel(self, canvas, rect, label):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), _BORDER, 1)
        strip = canvas[y:y + _LABEL_H, x:x + w]
        strip[:] = (0.35 * strip + 0.65 * np.asarray(_LABEL_BG)).astype(np.uint8)
        cv2.putText(canvas, label, (x + 8, y + 15), _FONT, 0.42, _TEXT, 1, cv2.LINE_AA)

    def _draw_bar(self, canvas):
        x, y, w, h = self._bar_rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _PANEL_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _BORDER, 1)

        # buttons (top row)
        self._button_hits = []
        bx = x + 8
        by = y + 6
        bh = 26
        for label_fn, action in self._buttons:
            label = label_fn()
            (tw, _), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
            bw = tw + 16
            hover = bx <= self._hover[0] <= bx + bw and by <= self._hover[1] <= by + bh
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh),
                          _BTN_HOVER if hover else _BTN_BG, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), _BORDER, 1)
            color = _REC if label == "rec:on" else _TEXT
            cv2.putText(canvas, label, (bx + 8, by + 18), _FONT, 0.45, color, 1,
                        cv2.LINE_AA)
            self._button_hits.append((bx, by, bx + bw, by + bh, action))
            bx += bw + 6

        # fps + record indicator (top right)
        info = f"{self._fps:5.1f} fps" if self._fps else ""
        if self._recording:
            cv2.circle(canvas, (x + w - 130, by + 13), 6, _REC, -1)
        cv2.putText(canvas, info, (x + w - 110, by + 18), _FONT, 0.5, _ACCENT, 1,
                    cv2.LINE_AA)

        # per-object stats (bottom rows, right of the base demo's text zone)
        if self._scene is None:
            return
        sx = x + _BASE_TEXT_ZONE_W
        sy = y + 52
        for snap in self._scene.objects[:2]:
            chip = (object_color(snap.obj_id)[::-1] * 255).astype(np.uint8).tolist()
            cv2.rectangle(canvas, (sx, sy - 10), (sx + 10, sy), chip, -1)
            cv2.putText(canvas, self._stats_line(snap), (sx + 16, sy), _FONT, 0.42,
                        _TEXT, 1, cv2.LINE_AA)
            if snap.lost:
                cv2.putText(canvas, "LOST", (x + w - 60, sy), _FONT, 0.5, _LOST, 2,
                            cv2.LINE_AA)
            sy += 22

    @staticmethod
    def _stats_line(snap):
        t = snap.T_cam_obj[:3, 3]
        rpy = scipy_R.from_matrix(snap.T_cam_obj[:3, :3]).as_euler("xyz", degrees=True)
        n_vis = int(np.isfinite(snap.track_points_cam).all(axis=1).sum())
        inl = str(snap.num_inliers) if snap.num_inliers >= 0 else "-"
        return (
            f"obj{snap.obj_id}"
            f"  t[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]m"
            f"  rpy[{rpy[0]:+6.1f} {rpy[1]:+6.1f} {rpy[2]:+6.1f}]deg"
            f"  pts {n_vis}/{len(snap.track_points_cam)}"
            f"  map {len(snap.map_points)}"
            f"  kf {len(snap.keyframe_T_obj_cam)}"
            f"  res {snap.mean_residual * 1000.0:.1f}mm"
            f"  inl {inl}"
        )

    # ------------------------------------------------------------------
    # mouse interaction (routed from the host cv2 window)
    # ------------------------------------------------------------------
    def handle_mouse(self, event, x, y, flags):
        """Route cv2 mouse events: buttons on click, 3D navigation on drag."""
        self._hover = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, action in self._button_hits:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    action()
                    self._redraw_live()
                    return
            view = self._view_at(x, y)
            if view is not None:
                shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
                self._drag = {"view": view, "mode": "pan" if shift else "rotate",
                              "last": (x, y)}
        elif event == cv2.EVENT_MBUTTONDOWN:
            view = self._view_at(x, y)
            if view is not None:
                self._drag = {"view": view, "mode": "pan", "last": (x, y)}
        elif event == cv2.EVENT_RBUTTONDOWN:
            view = self._view_at(x, y)
            if view is not None:
                self._drag = {"view": view, "mode": "zoom", "last": (x, y)}
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            dx = x - self._drag["last"][0]
            dy = y - self._drag["last"][1]
            self._drag["last"] = (x, y)
            view = self._drag["view"]
            if self._drag["mode"] == "rotate":
                view.rotate(dx, dy)
            elif self._drag["mode"] == "pan":
                view.pan(dx, dy)
            else:  # zoom: drag up = in, down = out
                view.zoom(dy * 0.05)
            self._redraw_live()
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._drag = None
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", -1):
            view = self._view_at(x, y)
            if view is not None:
                delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
                view.zoom(-1.0 if delta > 0 else 1.0)
                self._redraw_live()

    def _view_at(self, x, y):
        for rect, view in ((self._obj_rect, self.object_view),
                           (self._cam_rect, self.camera_view)):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return view if view.created else None
        return None

    def set_live_window(self, name):
        """cv2 window to redraw immediately during drags (between pipeline
        frames), so navigation stays responsive at low tracking FPS."""
        self._live_window = name

    def _redraw_live(self):
        if self._live_window is None or self._scene is None:
            return
        now = time.time()
        if now - self._last_live_draw < 0.03:
            return
        self._last_live_draw = now
        cv2.imshow(self._live_window, self._compose())

    # ------------------------------------------------------------------
    # button actions
    # ------------------------------------------------------------------
    @staticmethod
    def _toggle_label(name, value):
        return f"{name}:{'on' if bool(value) else 'off'}"

    @staticmethod
    def _toggle(node, key):
        node[key] = not bool(node.get(key, True))

    @staticmethod
    def _cycle_mode(node):
        cur = str(node.point_color_mode)
        idx = _COLOR_MODES.index(cur) if cur in _COLOR_MODES else 0
        node.point_color_mode = _COLOR_MODES[(idx + 1) % len(_COLOR_MODES)]

    def _toggle_boxes(self):
        new = not bool(self.cfg.object_view.show_bbox)
        self.cfg.object_view.show_bbox = new
        self.cfg.camera_view.show_bbox = new

    def _reset_views(self):
        self.object_view.reset_view()
        self.camera_view.reset_view()

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def _toggle_record(self):
        self._recording = not self._recording
        if not self._recording:
            self._release_writer()

    def _write_video_frame(self, canvas):
        if self._writer is None:
            out_dir = Path(str(self.cfg.output_image_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._writer_path = out_dir / f"dashboard_{stamp}.mp4"
            self._writer = cv2.VideoWriter(
                str(self._writer_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.cfg.combined.record_fps),
                (self.width, self.height),
            )
            print(f"[viz3d] recording dashboard to {self._writer_path}")
        self._writer.write(canvas)

    def _release_writer(self):
        if self._writer is not None:
            self._writer.release()
            print(f"[viz3d] saved recording: {self._writer_path}")
            self._writer = None

    # ------------------------------------------------------------------
    def reset(self):
        """Clear accumulated state; keeps windows and recording running."""
        self.object_view.reset()
        self.camera_view.reset()
        self._scene = None
        self._obj_snap = None
        self._overlay = None
        self._fps = None
        self._last_update_t = None
        self._drag = None

    def close(self):
        self._release_writer()
        self.object_view.close()
        self.camera_view.close()
