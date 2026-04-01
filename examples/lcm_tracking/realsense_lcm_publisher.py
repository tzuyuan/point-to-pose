import argparse
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

from point2pose.io.lcm import RgbdLcmPublisher
from point2pose.io.lcm.data_models import CameraInfoPacket, RGBDFramePacket
from point2pose.io.lcm.messages import rgbd_t


class RealSenseLcmPublisherApp:
    def __init__(self, config_path: str, preview: bool = False):
        self.cfg = OmegaConf.load(config_path)
        self._rs_cfg = self.cfg.realsense.params
        self._lcm_cfg = self.cfg.get("lcm", {})
        self._preview = bool(preview)

        self._rgbd_channel = str(self._lcm_cfg.get("rgbd_channel", "d455_1"))
        self._camera_info_channel = str(
            self._lcm_cfg.get("camera_info_channel", f"{self._rgbd_channel}_info")
        )
        self._camera_info_pub_hz = max(
            0.1, float(self._lcm_cfg.get("camera_info_pub_hz", 1.0))
        )
        self._verbose = bool(self._lcm_cfg.get("verbose", False))

        self._drop_stale_frames = bool(self._rs_cfg.get("drop_stale_frames", True))
        self._max_frame_drain = max(0, int(self._rs_cfg.get("max_frame_drain", 8)))
        self._frames_queue_size = int(self._rs_cfg.get("frames_queue_size", 1))
        self._timing_debug_every = int(self._rs_cfg.get("timing_debug_every", 30))

        self.publisher = RgbdLcmPublisher(
            rgbd_channel=self._rgbd_channel,
            camera_info_channel=self._camera_info_channel,
            verbose=self._verbose,
        )
        self.rs_pipeline = None
        self._rs_align = None
        self.camera_intrinsics = None
        self.depth_factor = 1000.0
        self._frame_height = 0
        self._frame_width = 0
        self._window_name = "RealSense LCM Publisher"

        self._init_realsense()

    def _init_realsense(self):
        rs_serial = self._rs_cfg.get("rs_serial", None)
        width = int(self._rs_cfg.get("width", 640))
        height = int(self._rs_cfg.get("height", 480))
        color_fps = int(self._rs_cfg.get("color_fps", 30))
        depth_fps = int(self._rs_cfg.get("depth_fps", 30))

        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        if rs_serial not in (None, ""):
            config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, depth_fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, color_fps)

        self._rs_align = rs.align(rs.stream.color)
        profile = self.rs_pipeline.start(config)
        self._configure_realsense_sensors(profile)

        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self._frame_height = int(intrinsics.height)
        self._frame_width = int(intrinsics.width)
        self.camera_intrinsics = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        print("RealSense LCM publisher initialized:")
        print(f"  RGBD channel: {self._rgbd_channel}")
        print(f"  Camera info channel: {self._camera_info_channel}")
        print(f"  Resolution: {self._frame_width}x{self._frame_height}")
        print(f"  FPS: color={color_fps}, depth={depth_fps}")
        print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
        if self._preview:
            print("  Preview: enabled (press 'q' to quit)")

    def _configure_realsense_sensors(self, profile):
        try:
            sensors = list(profile.get_device().query_sensors())
        except Exception as exc:
            print(f"Warning: failed to query RealSense sensors for configuration ({exc})")
            return

        color_sensor = None
        depth_sensor = None
        for sensor in sensors:
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                name = ""
            if "RGB" in name and color_sensor is None:
                color_sensor = sensor
            elif ("Stereo" in name or "Depth" in name) and depth_sensor is None:
                depth_sensor = sensor

        if color_sensor is not None:
            self._set_sensor_option(
                color_sensor,
                getattr(rs.option, "frames_queue_size", None),
                self._frames_queue_size,
            )
            self._set_sensor_option(
                color_sensor,
                rs.option.enable_auto_exposure,
                self._rs_cfg.get("color_auto_exposure", None),
            )
            self._set_sensor_option(
                color_sensor,
                rs.option.auto_exposure_priority,
                self._rs_cfg.get("color_auto_exposure_priority", None),
            )
            self._set_sensor_option(
                color_sensor,
                rs.option.exposure,
                self._rs_cfg.get("color_exposure_us", None),
            )
            self._set_sensor_option(
                color_sensor, rs.option.gain, self._rs_cfg.get("color_gain", None)
            )
            self._set_sensor_option(
                color_sensor,
                rs.option.sharpness,
                self._rs_cfg.get("color_sharpness", None),
            )

        if depth_sensor is not None:
            self._set_sensor_option(
                depth_sensor,
                getattr(rs.option, "frames_queue_size", None),
                self._frames_queue_size,
            )
            emitter_enabled = self._rs_cfg.get("depth_emitter_enabled", None)
            if emitter_enabled is not None:
                self._set_sensor_option(
                    depth_sensor,
                    rs.option.emitter_enabled,
                    1 if bool(emitter_enabled) else 0,
                )

    def _set_sensor_option(self, sensor, option, value):
        if value is None or option is None:
            return
        try:
            if sensor.supports(option):
                sensor.set_option(option, float(value))
        except Exception as exc:
            print(f"Warning: failed to set RealSense option {option}={value} ({exc})")

    def _wait_for_latest_frames(self):
        t0 = time.perf_counter()
        frames = self.rs_pipeline.wait_for_frames()
        wait_s = time.perf_counter() - t0

        dropped_frames = 0
        if self._drop_stale_frames:
            latest_frames = frames
            for _ in range(self._max_frame_drain):
                polled = self.rs_pipeline.poll_for_frames()
                if not polled:
                    break
                latest_frames = polled
                dropped_frames += 1
            frames = latest_frames

        return frames, {"wait_s": wait_s, "dropped_frames": dropped_frames}

    def _build_camera_info_packet(self, timestamp: float) -> CameraInfoPacket:
        return CameraInfoPacket(
            timestamp=float(timestamp),
            height=int(self._frame_height),
            width=int(self._frame_width),
            intrinsics=self.camera_intrinsics.copy(),
            world_to_camera=np.eye(4, dtype=np.float64),
            depth_factor=float(self.depth_factor),
            fixed=True,
            attached_body="realsense",
        )

    def _build_rgbd_packet(
        self, timestamp: float, frame_rgb: np.ndarray, frame_depth: np.ndarray
    ) -> RGBDFramePacket:
        return RGBDFramePacket(
            timestamp=float(timestamp),
            height=int(frame_rgb.shape[0]),
            width=int(frame_rgb.shape[1]),
            num_rgb_channels=int(frame_rgb.shape[2]),
            rgb_channel_type=rgbd_t.CHANNEL_TYPE_UINT8,
            depth_channel_type=rgbd_t.CHANNEL_TYPE_UINT16,
            rgb_image=frame_rgb,
            depth_image=frame_depth,
        )

    def run(self):
        self.publisher.start()
        last_camera_info_publish_s = 0.0
        frame_count = 0

        if self._preview:
            cv2.startWindowThread()
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                frames, wait_stats = self._wait_for_latest_frames()

                t_align = time.perf_counter()
                aligned_frames = self._rs_align.process(frames)
                align_s = time.perf_counter() - t_align

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                t_convert = time.perf_counter()
                frame_bgr = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_depth = np.asanyarray(depth_frame.get_data())
                convert_s = time.perf_counter() - t_convert

                timestamp = float(color_frame.get_timestamp()) / 1000.0
                camera_info_period_s = 1.0 / self._camera_info_pub_hz
                if (timestamp - last_camera_info_publish_s) >= camera_info_period_s:
                    self.publisher.publish_camera_info(
                        self._build_camera_info_packet(timestamp)
                    )
                    last_camera_info_publish_s = timestamp

                self.publisher.publish_rgbd(
                    self._build_rgbd_packet(timestamp, frame_rgb, frame_depth)
                )

                if self._timing_debug_every > 0 and ((frame_count + 1) % self._timing_debug_every) == 0:
                    loop_total_s = wait_stats["wait_s"] + align_s + convert_s
                    approx_fps = 1.0 / max(loop_total_s, 1e-6)
                    print(
                        "[RealSenseLcmPublisher] "
                        f"frame={frame_count + 1} "
                        f"publish_rgbd={self._rgbd_channel} "
                        f"publish_info={self._camera_info_channel} "
                        f"wait={1000.0 * wait_stats['wait_s']:.1f}ms "
                        f"align={1000.0 * align_s:.1f}ms "
                        f"convert={1000.0 * convert_s:.1f}ms "
                        f"dropped={wait_stats['dropped_frames']} "
                        f"loop_total={1000.0 * loop_total_s:.1f}ms "
                        f"(~{approx_fps:.1f} FPS)"
                    )

                frame_count += 1

                if self._preview:
                    display_frame = frame_bgr.copy()
                    cv2.putText(
                        display_frame,
                        f"RGBD: {self._rgbd_channel}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Info: {self._camera_info_channel}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        "Press 'q' to quit",
                        (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(self._window_name, display_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.publisher.stop()
            if self.rs_pipeline is not None:
                self.rs_pipeline.stop()
            if self._preview:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/pipeline_test.yaml",
        help="Path to the Point2Pose config with realsense and lcm sections.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the live color stream locally while publishing over LCM.",
    )
    args = parser.parse_args()

    app = RealSenseLcmPublisherApp(config_path=args.config, preview=args.preview)
    app.run()


if __name__ == "__main__":
    main()
