"""Record raw RGB-D from a RealSense camera in the YCBMultiTrack layout.

Saves:
    <out>/rgb/000000.png       color (BGR png)
    <out>/depth/000000.png     depth aligned to color, uint16 millimeters
    <out>/cam_K.txt            3x3 color intrinsics

The output folder can be fed directly to the offline pipeline / poster
visualization scripts.

Usage:
    python record_rgbd.py --out ~/data/poster_recordings/take01 [--serial N]

Keys (focus the preview window):
    r / space   start-stop recording
    q / esc     quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output sequence directory")
    ap.add_argument("--serial", default=None, help="camera serial (default: first)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(str(args.serial))
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    np.savetxt(out / "cam_K.txt", K)
    print("depth_scale:", depth_scale, "\nK:\n", K)

    # auto-exposure warmup
    for _ in range(30):
        pipe.wait_for_frames()

    recording = False
    idx = 0
    t0 = time.time()
    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())

            if recording:
                cv2.imwrite(str(out / "rgb" / f"{idx:06d}.png"), color)
                depth_mm = (depth.astype(np.float32) * depth_scale * 1000.0)
                cv2.imwrite(str(out / "depth" / f"{idx:06d}.png"),
                            depth_mm.astype(np.uint16))
                idx += 1

            view = color.copy()
            status = f"REC {idx}" if recording else "idle (r to record)"
            col = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(view, f"{status}  {idx / max(time.time()-t0,1e-6):.0f} fps",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            if recording:
                cv2.circle(view, (args.width - 30, 26), 10, (0, 0, 255), -1)
            cv2.imshow("record_rgbd", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k in (ord("r"), ord(" ")):
                recording = not recording
                if recording:
                    t0 = time.time()
                    idx = idx  # continue numbering across takes in same folder
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print(f"saved {idx} frames to {out}")


if __name__ == "__main__":
    main()
