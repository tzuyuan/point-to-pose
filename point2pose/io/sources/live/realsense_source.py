import numpy as np
import pyrealsense2 as rs

from point2pose.data_types.frame import Frame
from point2pose.io.sources.base import BaseLiveSource


class RealSenseSource(BaseLiveSource):
    def __init__(
        self,
        serial: str | None = None,
        width=640,
        height=480,
        fps=30,
        enable_depth=True,
    ):
        self.serial, self.w, self.h, self.fps, self.enable_depth = (
            serial,
            width,
            height,
            fps,
            enable_depth,
        )
        self._pipe, self._cfg, self._align = None, None, None

    def open(self):
        self._pipe, self._cfg = rs.pipeline(), rs.config()
        if self.serial:
            self._cfg.enable_device(self.serial)
        self._cfg.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps
        )
        if self.enable_depth:
            self._cfg.enable_stream(
                rs.stream.depth, self.w, self.h, rs.format.z16, self.fps
            )
            self._align = rs.align(rs.stream.color)
        self._pipe.start(self._cfg)

    def close(self):
        if self._pipe:
            self._pipe.stop()
            self._pipe = None

    def __iter__(self):
        while True:
            frames = self._pipe.wait_for_frames()
            if self._align:
                frames = self._align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = (
                np.asanyarray(frames.get_depth_frame().get_data())
                if self.enable_depth
                else None
            )
            ts = frames.get_timestamp() / 1000.0
            yield Frame(rgb=color, depth=depth, timestamp=ts, meta={})
