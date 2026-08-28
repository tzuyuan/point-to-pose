"""
Live point-tracker sandbox: RealSense RGB stream + click-to-add query points.

Purpose: test whether a point tracker (TAPIR / CoTracker3-online) can accept NEW query
points added mid-stream, while frames keep flowing -- both trackers already implement
``add_query_points`` for this (TAPIR expands its causal state, CoTracker appends to its
online query buffer and re-commits on the next window step). This script just exercises
that interface interactively so you can eyeball tracking quality and add/remove points live.

Controls:
    Left click   : add a query point at the clicked pixel (added to the RUNNING tracker,
                   no restart -- this is exactly the "add query mid-stream" behavior)
    'c'          : clear all points and reset the tracker (restarts the model)
    'q' / Esc    : quit

Run (point2pose env, from repo root):
    python examples/model_based_tracking/live_tracker_test.py --tracker tapir
    python examples/model_based_tracking/live_tracker_test.py --tracker cotracker3_online

Options:
    --rs-serial <serial>   RealSense device serial (default: first available device)
    --width/--height       Capture resolution (default 640x480)
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
if str(REPO / "tapnet") not in sys.path:
    sys.path.insert(0, str(REPO / "tapnet"))

import cv2
import numpy as np
import pyrealsense2 as rs

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import TRACKER
from point2pose.data_types.frame import Frame

WINDOW_NAME = "live_tracker_test"

TRACKER_CONFIGS = {
    "tapir": {
        "type": "tapir",
        "params": {
            "img_width": 640,
            "img_height": 480,
            "resize_width": 256,
            "resize_height": 256,
            "device": "cuda",
            "checkpoint_path": str(REPO / "checkpoints/tapir/causal_bootstapir_checkpoint.pt"),
        },
    },
    "cotracker3_online": {
        "type": "cotracker3_online",
        "params": {
            "img_width": 640,
            "img_height": 480,
            "resize_width": 640,
            "resize_height": 480,
            "device": "cuda",
            "window_len": 16,
            "v2": False,
            "checkpoint_path": str(REPO / "checkpoints/cotracker/scaled_online.pth"),
        },
    },
}


def build_tracker(tracker_name, width, height, device):
    cfg = TRACKER_CONFIGS[tracker_name]
    cfg["params"]["img_width"] = width
    cfg["params"]["img_height"] = height
    cfg["params"]["device"] = device
    return build_from_cfg(cfg, TRACKER)


class LiveTrackerTest:
    def __init__(self, tracker_name, width, height, device, rs_serial=None):
        self.tracker_name = tracker_name
        self.width = width
        self.height = height
        self.device = device
        self.tracker = build_tracker(tracker_name, width, height, device)

        self._init_realsense(rs_serial, width, height)

        self.frame_id = 0
        self.num_points = 0
        self.initialized = False
        self.last_fps = 0.0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_cb)

        print(f"[live_tracker_test] tracker = {tracker_name}")
        print("Left click: add query point | 'c': clear/reset | 'q'/Esc: quit")

    def _init_realsense(self, rs_serial, width, height):
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        if rs_serial is not None:
            config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.rs_pipeline.start(config)

    def _mouse_cb(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame = Frame(id=self.frame_id, rgb=self._last_rgb, timestamp=time.time())
        new_points = np.array([[x, y]], dtype=np.float32)
        indices = self.tracker.add_query_points(frame, new_points)
        self.initialized = self.tracker.initialize(frame) if not self.initialized else True
        self.num_points += len(indices)
        print(f"[live_tracker_test] added point ({x},{y}) -> index {indices}, total {self.num_points}")

    def reset(self):
        self.tracker = build_tracker(self.tracker_name, self.width, self.height, self.device)
        self.num_points = 0
        self.initialized = False
        print("[live_tracker_test] tracker reset, points cleared")

    def _grab_rgb(self):
        frames = self.rs_pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        bgr = np.asanyarray(color.get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def run(self):
        try:
            while True:
                rgb = self._grab_rgb()
                if rgb is None:
                    continue
                self._last_rgb = rgb

                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if self.num_points > 0:
                    t0 = time.time()
                    frame = Frame(id=self.frame_id, rgb=rgb, timestamp=time.time())
                    tracks, uncertainties, visibles = self.tracker.track_once(frame)
                    dt = time.time() - t0
                    self.last_fps = 1.0 / dt if dt > 0 else 0.0

                    for (x, y), vis in zip(tracks, visibles):
                        color = (0, 255, 0) if vis else (0, 0, 255)
                        cv2.circle(display, (int(x), int(y)), 5, color, -1)

                cv2.putText(
                    display,
                    f"tracker={self.tracker_name} points={self.num_points} fps={self.last_fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "left click: add point | 'c': clear | 'q': quit",
                    (10, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(WINDOW_NAME, display)
                self.frame_id += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    self.reset()
        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=list(TRACKER_CONFIGS.keys()), default="tapir")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rs-serial", default=None)
    args = ap.parse_args()

    test = LiveTrackerTest(
        args.tracker, args.width, args.height, args.device, rs_serial=args.rs_serial
    )
    test.run()


if __name__ == "__main__":
    main()
