# realsense_pipeline_tracker_mp.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # fine on X11; helps on Wayland

import sys
from pathlib import Path
import time
import multiprocessing as mp
from queue import Empty, Full

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

# Your pipeline imports (Torch/LightGlue stay in the main process)
sys.path.append(str(Path(__file__).resolve().parents[2]))
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.data_types.frame import Frame
from point2pose.utils.visualization import (
    draw_xyz_axis,
    draw_posed_3d_box,
    draw_points_on_image,
    get_n_uncertainty_colors,
)


# --------------------------- GUI PROCESS ---------------------------


def _gui_process(frame_q: mp.Queue, event_q: mp.Queue, window_name: str):
    """
    Separate process that ONLY shows frames and collects mouse/keyboard events.
    IMPORTANT: Do NOT import Torch/LightGlue in this process.
    """
    import cv2
    import numpy as np

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Mouse callback lives in this process; send events back to main
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            event_q.put(("mouse", "left", int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            event_q.put(("mouse", "right", int(x), int(y)))

    cv2.setMouseCallback(window_name, on_mouse)

    last_frame = None
    try:
        while True:
            # Drain the frame queue quickly to show the latest frame
            try:
                # get without blocking to avoid lag; show the most recent
                while True:
                    last_frame = frame_q.get_nowait()
            except Empty:
                pass

            if last_frame is not None:
                if isinstance(last_frame, bytes):
                    # if you later switch to JPEG bytes, decode here
                    img = cv2.imdecode(
                        np.frombuffer(last_frame, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                else:
                    img = last_frame
                cv2.imshow(window_name, img)
                last_frame = None

            # Keyboard handling in GUI process, then forward as events
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                if k in (ord("q"), ord("s"), ord("r")):
                    event_q.put(("key", chr(k)))
                # You can extend more keys here if needed

            # Check for sentinel to exit
            # Use a small non-blocking poll for a special control message if you want.
            # (We rely on parent terminating this process normally.)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


# --------------------------- MAIN PROCESS CLASS ---------------------------


class RealSensePipelineTracker:
    """
    RealSense tracker that integrates SAM2 segmentation with the point2pose pipeline.
    GUI is in a separate process. This class owns camera+pipeline and does all drawing.
    """

    def __init__(
        self,
        frame_q: mp.Queue,
        event_q: mp.Queue,
        window_name: str = "RealSense Pipeline Tracker",
        config_path: str = "configs/pipeline/pipeline_test2.yaml",
    ):
        self.frame_q = frame_q
        self.event_q = event_q
        self.window_name = window_name

        # Load configuration
        self.cfg = OmegaConf.load(config_path)
        rs_serial = self.cfg.realsense.params.rs_serial

        self._visualize_points = self.cfg.visualization.params.visualize_points
        self._points_vis_method = self.cfg.visualization.params.points_vis_method
        self._save_images = self.cfg.visualization.params.save_images
        self._output_image_dir = Path(self.cfg.visualization.params.output_image_dir)

        if self._save_images:
            self._output_image_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Image saving enabled. Images will be saved to: {self._output_image_dir}"
            )

        # Heavy pipeline stays in main process
        self.pipeline = PipelineSingleProcess(self.cfg)

        # RealSense
        self._init_realsense(rs_serial)

        # Interaction state
        self.click_points = []  # [[x,y], ...]
        self.click_labels = []  # [1 for +, 0 for -]
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # Colors cache for frame-id coloring
        self._frame_color_lookup = {}
        self._frame_color_used_hsv = set()

        print("Instructions:")
        print("- Left click: Add positive point")
        print("- Right click: Add negative point")
        print("- Press 's': Start tracking")
        print("- Press 'q': Quit")
        print("- Press 'r': Reset points")

    # ---------------- Camera ----------------

    def _init_realsense(self, rs_serial):
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        align_to = rs.stream.color
        self._rs_align = rs.align(align_to)

        profile = self.rs_pipeline.start(config)

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_intrinsics = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float64,
        )
        self.depth_factor = 1000.0  # RealSense depth in mm

        print("Camera initialized:")
        print(f"  Resolution: {intr.width}x{intr.height}")
        print(f"  Focal length: fx={intr.fx:.2f}, fy={intr.fy:.2f}")
        print(f"  Principal point: cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")

    # ---------------- Helpers ----------------

    def _generate_next_frame_color(self):
        golden = 0.6180339887498949
        base_index = len(self._frame_color_lookup)
        saturation_cycle = (255, 230, 200, 180)
        value_cycle = (255, 235, 215)
        attempt = 0
        while True:
            idx = base_index + attempt
            hue = int(round(((idx * golden) % 1.0) * 179)) % 180
            saturation = saturation_cycle[idx % len(saturation_cycle)]
            value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]
            hsv_tuple = (hue, saturation, value)
            if hsv_tuple not in self._frame_color_used_hsv:
                self._frame_color_used_hsv.add(hsv_tuple)
                hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            attempt += 1

    def _colors_for_frame_ids(self, frame_ids: np.ndarray) -> np.ndarray:
        if frame_ids.size == 0:
            return np.empty((0, 3), dtype=np.uint8)
        colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
        unique_ids = np.unique(frame_ids.astype(np.int64))
        for fid in unique_ids:
            fid_int = int(fid)
            if fid_int not in self._frame_color_lookup:
                self._frame_color_lookup[fid_int] = self._generate_next_frame_color()
            colors[frame_ids == fid] = self._frame_color_lookup[fid_int]
        return colors

    def _poll_events(self):
        """
        Poll GUI events without blocking.
        Returns a tuple (quit, start_pressed, reset_pressed) to act on,
        and updates click_points/labels inline.
        """
        quit_flag = False
        start_flag = False
        reset_flag = False
        while True:
            try:
                ev = self.event_q.get_nowait()
            except Empty:
                break
            etype = ev[0]
            if etype == "mouse":
                _, btn, x, y = ev
                if btn == "left":
                    self.click_points.append([x, y])
                    self.click_labels.append(1)
                    print(f"Added positive point: ({x}, {y})")
                elif btn == "right":
                    self.click_points.append([x, y])
                    self.click_labels.append(0)
                    print(f"Added negative point: ({x}, {y})")
            elif etype == "key":
                _, k = ev
                if k == "q":
                    quit_flag = True
                elif k == "s":
                    start_flag = True
                elif k == "r":
                    reset_flag = True
        return quit_flag, start_flag, reset_flag

    # ---------------- Pipeline control ----------------

    def reset_points(self):
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None
        self.pipeline = PipelineSingleProcess(self.cfg)
        print("Points reset. Click to add new points.")

    def start_tracking(self):
        if len(self.click_points) == 0:
            print("No points collected! Please click on objects first.")
            return False

        frames = self.rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Failed to get frames for initialization")
            return False

        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        frame = Frame(
            id=0,
            rgb=frame_rgb,
            depth=frame_depth,
            intrinsics=self.camera_intrinsics,
            depth_factor=self.depth_factor,
            timestamp=time.time(),
        )

        self.pipeline.add_user_points(self.click_points, self.click_labels)
        self.current_poses = self.pipeline.step(frame)

        self.tracking_started = True
        self.frame_count = 1

        print(f"Started tracking with {len(self.click_points)} points")
        print(f"Number of objects: {len(self.current_poses)}")
        return True

    def create_frame_from_realsense(self, frame_id):
        frames = self.rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        return Frame(
            id=frame_id,
            rgb=frame_rgb,
            depth=frame_depth,
            intrinsics=self.camera_intrinsics,
            depth_factor=self.depth_factor,
            timestamp=time.time(),
        )

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        display_frame = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        h, w = display_frame.shape[:2]
        if hasattr(frame, "mask") and frame.mask is not None:
            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            mask_overlay[..., 1] = 255
            for i in range(len(frame.mask)):
                obj_mask = frame.mask[i, 0] > 0.0
                obj_mask = obj_mask.cpu().numpy()
                if np.any(obj_mask):
                    hue = (i + 3) / (len(frame.mask) + 3) * 255
                    mask_overlay[obj_mask, 0] = hue
                    mask_overlay[obj_mask, 2] = 255
            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
            display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)

        if self._visualize_points:
            if self._points_vis_method == "uncertainty":
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue
                    uncertainty_color = get_n_uncertainty_colors(
                        self.pipeline.track_table.uncertainty[
                            self.pipeline.track_table.obj2track_map[i]
                        ]
                    )
                    draw_points_on_image(
                        display_frame,
                        self.pipeline.track_table.track_2d[
                            self.pipeline.track_table.obj2track_map[i]
                        ],
                        uncertainty_color,
                    )
            elif self._points_vis_method == "visible":
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue
                    track_2d_points = self.pipeline.track_table.track_2d[
                        self.pipeline.track_table.obj2track_map[i]
                    ]
                    N = len(track_2d_points)
                    visible_color = np.full((N, 3), (0, 0, 255), dtype=np.uint8)
                    visible_color[
                        self.pipeline.track_table.visible[
                            self.pipeline.track_table.obj2track_map[i]
                        ]
                    ] = (0, 255, 0)
                    draw_points_on_image(display_frame, track_2d_points, visible_color)

            elif self._points_vis_method == "visible_uncertainty":
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue
                    track_idx = self.pipeline.track_table.obj2track_map[i]
                    track_2d_points = self.pipeline.track_table.track_2d[track_idx]
                    visible_mask = self.pipeline.track_table.visible[track_idx]
                    if np.any(visible_mask):
                        uncertainty_color = get_n_uncertainty_colors(
                            self.pipeline.track_table.uncertainty[track_idx]
                        )
                        draw_points_on_image(
                            display_frame,
                            track_2d_points[visible_mask],
                            uncertainty_color[visible_mask],
                        )

            elif self._points_vis_method == "frame_id":
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue
                    track_idx = self.pipeline.track_table.obj2track_map[i]
                    track_2d_points = self.pipeline.track_table.track_2d[track_idx]
                    visible_mask = self.pipeline.track_table.visible[track_idx]
                    if not np.any(visible_mask):
                        continue
                    if obj.key_point_frames.shape[0] == 0:
                        continue
                    num_tracks_for_obj = len(track_idx)
                    kp_frames_for_obj = obj.key_point_frames[
                        :num_tracks_for_obj
                    ].astype(np.int32)
                    frame_ids = kp_frames_for_obj[visible_mask]
                    if frame.id is not None:
                        frame_ids = frame_ids.copy()
                        frame_ids[frame_ids == -1] = int(frame.id)
                    colors_bgr = self._colors_for_frame_ids(frame_ids)
                    draw_points_on_image(
                        display_frame, track_2d_points[visible_mask], colors_bgr
                    )

        for i, obj in enumerate(objects):
            if obj.pose is not None:
                pose = obj.pose @ obj.init_pose
                half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
                bbox_min_max_local = np.vstack([-half, +half])
                display_frame = draw_posed_3d_box(
                    self.camera_intrinsics, display_frame, pose, bbox_min_max_local
                )
                display_frame = draw_xyz_axis(
                    image=display_frame, ob_in_cam=pose, K=self.camera_intrinsics
                )

        if self._save_images and frame_id is not None:
            out = self._output_image_dir / f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(out), display_frame)

        return display_frame

    # ---------------- Main loop ----------------

    def run_tracking(self):
        try:
            while True:
                quit_flag, start_flag, reset_flag = self._poll_events()
                if quit_flag:
                    break
                if reset_flag:
                    self.reset_points()

                if not self.tracking_started:
                    # Show live color with current point markers and instructions
                    frames = self.rs_pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    display_frame = np.asanyarray(color_frame.get_data()).copy()
                    # Overlay current clicks
                    for i, (pt, label) in enumerate(
                        zip(self.click_points, self.click_labels)
                    ):
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)
                        cv2.circle(
                            display_frame, (int(pt[0]), int(pt[1])), 5, color, -1
                        )
                        cv2.putText(
                            display_frame,
                            f"{i+1}",
                            (int(pt[0]) + 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

                    cv2.putText(
                        display_frame,
                        "Left click: +, Right click: -, Press 's' to start",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Points: {len(self.click_points)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    if start_flag:
                        if self.start_tracking():
                            print("Pipeline tracking started!")
                        else:
                            print("Failed to start pipeline tracking!")

                else:
                    frame = self.create_frame_from_realsense(self.frame_count)
                    if frame is None:
                        continue

                    self.pipeline.step(frame)

                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects, self.frame_count
                    )
                    h, _ = display_frame.shape[:2]
                    cv2.putText(
                        display_frame,
                        f"Frame: {self.frame_count}",
                        (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Objects: {len(self.current_poses) if self.current_poses is not None else 0}",
                        (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    self.frame_count += 1

                # Send frame to GUI (drop if queue is full to stay real-time)
                try:
                    self.frame_q.put_nowait(display_frame)
                except Full:
                    pass  # drop frame

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.rs_pipeline.stop()


# --------------------------- ENTRY POINT ---------------------------


def main():
    mp.set_start_method("spawn", force=True)  # avoid inheriting Torch/CUDA/Qt state

    window_name = "RealSense Pipeline Tracker"
    frame_q = mp.Queue(maxsize=2)  # small to keep latency low
    event_q = mp.Queue(maxsize=64)

    # Start GUI process FIRST (it won't import Torch/LightGlue)
    gui = mp.Process(
        target=_gui_process, args=(frame_q, event_q, window_name), daemon=True
    )
    gui.start()

    # Now start the main tracker (Torch/LightGlue is inside PipelineSingleProcess)
    tracker = RealSensePipelineTracker(frame_q, event_q, window_name=window_name)
    try:
        tracker.run_tracking()
    finally:
        # tell GUI to close (send sentinel by killing queue or just terminate process)
        if gui.is_alive():
            gui.terminate()
            gui.join(timeout=2.0)


if __name__ == "__main__":
    main()
