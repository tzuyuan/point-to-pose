"""Lazy Open3D window wrapper shared by the two demo views.

The window is created on the first ``show()`` call (i.e. once tracking starts
and there is something to display). With ``visible=False`` the window stays
hidden and acts as an offscreen renderer: ``snapshot_image()`` returns the
rendering as a BGR image, and ``rotate/pan/zoom`` drive the view control the
way real mouse input would — this is what the combined dashboard uses.
"""

import numpy as np
import open3d as o3d

from point2pose.visualization.geometry import BACKGROUND_COLOR


class Base3DView:
    def __init__(
        self,
        title: str,
        width: int,
        height: int,
        left: int,
        top: int,
        background=BACKGROUND_COLOR,
        point_size: float = 4.0,
        visible: bool = True,
    ):
        self._title = title
        self._width = int(width)
        self._height = int(height)
        self._left = int(left)
        self._top = int(top)
        self._background = np.asarray(background, dtype=np.float64)
        self._point_size = float(point_size)
        self._visible = bool(visible)
        self._vis = None

    @property
    def created(self) -> bool:
        return self._vis is not None

    # -- subclass hooks -------------------------------------------------
    def _geometries(self):
        """Persistent geometry objects registered with the window."""
        return []

    def _set_initial_view(self, view_control):
        """Set the initial viewpoint; called once, right after creation."""

    # -- window lifecycle ----------------------------------------------
    def _ensure_window(self):
        if self._vis is not None:
            return False
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            window_name=self._title,
            width=self._width,
            height=self._height,
            left=self._left,
            top=self._top,
            visible=self._visible,
        )
        opt = self._vis.get_render_option()
        if opt is not None:
            opt.background_color = self._background
            opt.point_size = self._point_size
            opt.mesh_show_back_face = True
            try:
                opt.line_width = 2.0
            except Exception:
                pass  # not supported on all Open3D builds
        for geom in self._geometries():
            self._vis.add_geometry(geom, reset_bounding_box=True)
        self._set_initial_view(self._vis.get_view_control())
        return True

    def show(self):
        """Create the window on first call, then push geometry updates and
        pump the event loop. Returns True on the creating call."""
        first = self._ensure_window()
        if not first:
            for geom in self._geometries():
                self._vis.update_geometry(geom)
        self._vis.poll_events()
        self._vis.update_renderer()
        return first

    # -- offscreen rendering -------------------------------------------
    def snapshot_image(self):
        """Render the current state and return it as a BGR uint8 image
        (None until the window exists). Does not push geometry updates."""
        if self._vis is None:
            return None
        buf = self._vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(buf) * 255.0).clip(0, 255).astype(np.uint8)
        return img[..., ::-1]  # RGB -> BGR

    def capture(self, path):
        """Save a PNG screenshot of the current rendering."""
        img = self.snapshot_image()
        if img is None:
            return
        import cv2

        cv2.imwrite(str(path), img)

    # -- interaction (used by the dashboard's mouse routing) ------------
    def rotate(self, dx: float, dy: float):
        if self._vis is not None:
            self._vis.get_view_control().rotate(dx, dy)

    def pan(self, dx: float, dy: float):
        if self._vis is not None:
            self._vis.get_view_control().translate(dx, dy)

    def zoom(self, amount: float):
        """Negative zooms in, positive zooms out (Open3D scale convention)."""
        if self._vis is not None:
            self._vis.get_view_control().scale(amount)

    def reset_view(self):
        if self._vis is not None:
            self._set_initial_view(self._vis.get_view_control())

    def close(self):
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
