import os
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Helps OpenCV HighGUI come up more reliably on Linux/Wayland setups.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3, transform_pts

# from point2pose.data_types.object import Object
from point2pose.utils.visualization import (
    _extract_bbox_info,
    draw_xyz_axis,
    draw_oriented_3d_box,
    draw_posed_3d_box,
    draw_points_on_image,
    get_n_uncertainty_colors,
    _resolve_pose_and_bbox,
)


class RealSensePipelineTracker:
    """
    RealSense tracker that integrates SAM2 segmentation with the point2pose pipeline.
    This tracker performs object tracking and pose estimation using the complete pipeline.
    """

    def __init__(
        self,
        config_path="configs/pipeline/pipeline_test.yaml",
        # rs_serial=242422304947
        # rs_serial=941322070969,
    ):
        """
        Args:
            config_path (str): Path to the pipeline configuration file.
            rs_serial (int): Serial number of the RealSense camera.
        """
        # Load configuration using OmegaConf
        self.cfg = OmegaConf.load(config_path)

        self._rs_cfg = self.cfg.realsense.params
        rs_serial = self._rs_cfg.rs_serial

        self._visualize_points = self.cfg.visualization.params.visualize_points
        self._points_vis_method = self.cfg.visualization.params.points_vis_method
        self._save_images = bool(self.cfg.visualization.params.get("save_images", False))
        self._output_image_dir = Path(
            self.cfg.visualization.params.get(
            "output_image_dir", "/home/justin/code/point-to-pose/debug/output_images"
            )
        )
        self._save_final_sdf_on_exit = bool(
            self.cfg.visualization.params.get("save_final_sdf_on_exit", False)
        )
        self._final_sdf_output_dir = Path(
            self.cfg.visualization.params.get(
                "final_sdf_output_dir",
                "/home/justin/code/point-to-pose/debug/final_sdf",
            )
        )
        self._window_name = "RealSense Pipeline Tracker"
        bbox_mode = self.cfg.pipeline.params.get("bbox_estimation_mode", None)
        if bbox_mode is None:
            bbox_mode = (
                "first_frame_dense"
                if self.cfg.pipeline.params.get("estimate_init_pose", False)
                else "continuous"
            )
        self._bbox_estimation_mode = str(bbox_mode).lower()
        self._drop_stale_frames = bool(self._rs_cfg.get("drop_stale_frames", True))
        self._max_frame_drain = int(self._rs_cfg.get("max_frame_drain", 8))
        self._frames_queue_size = self._rs_cfg.get("frames_queue_size", 1)
        self._timing_debug_every = int(self._rs_cfg.get("timing_debug_every", 30))
        self._last_capture_stats = {
            "wait_s": 0.0,
            "align_s": 0.0,
            "convert_s": 0.0,
            "dropped_frames": 0,
        }
        self._last_visualization_timings = {
            "prepare": 0.0,
            "masks": 0.0,
            "points": 0.0,
            "pose": 0.0,
            "save": 0.0,
            "total": 0.0,
        }

        # Create output directory if saving images is enabled
        if self._save_images:
            self._output_image_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Image saving enabled. Images will be saved to: {self._output_image_dir}"
            )
        if self._save_final_sdf_on_exit:
            self._final_sdf_output_dir.mkdir(parents=True, exist_ok=True)
            print(
                "Final SDF export enabled. Outputs will be saved to: "
                f"{self._final_sdf_output_dir}"
            )

        # Mouse click storage, grouped by object.
        self.click_point_groups = []
        self.click_label_groups = []
        self.current_click_points = []
        self.current_click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # Cache for consistent, distinctive frame-based point colors
        self._frame_color_lookup = {}
        self._frame_color_used_hsv = set()

        # Pipeline is initialized lazily when tracking actually starts so the
        # OpenCV window can appear immediately and remain responsive.
        self.pipeline = None

        # Create window and set mouse callback
        cv2.startWindowThread()
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self.mouse_callback)
        self._show_status_screen("Initializing RealSense camera...")

        # Initialize RealSense camera
        self._init_realsense(rs_serial)

        self._show_status_screen(
            "Camera ready. Click prompts for object 1, press 's' for next object, 'r' to run."
        )

        print("Instructions:")
        print("- Left click: Add positive point")
        print("- Right click: Add negative point")
        print("- Press 's': Finalize current object and move to the next one")
        print("- Press 'r': Start tracking with the collected object prompts")
        print("- Press 'c': Clear prompts and reset")
        if self._bbox_estimation_mode == "manual":
            print("- Press 'b': Estimate bounding box from the configured bbox source")
        print("- Press 'q': Quit")

    def _init_realsense(self, rs_serial):
        """Initialize RealSense camera"""
        width = int(self._rs_cfg.get("width", 640))
        height = int(self._rs_cfg.get("height", 480))
        color_fps = int(self._rs_cfg.get("color_fps", 30))
        depth_fps = int(self._rs_cfg.get("depth_fps", 30))

        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, depth_fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, color_fps)

        align_to = rs.stream.color
        self._rs_align = rs.align(align_to)

        # Start streaming
        profile = self.rs_pipeline.start(config)
        self._configure_realsense_sensors(profile)

        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Store camera parameters
        self.camera_intrinsics = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.depth_factor = 1000.0  # RealSense depth is in millimeters

        print("Camera initialized:")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  FPS: color={color_fps}, depth={depth_fps}")
        print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    def _configure_realsense_sensors(self, profile):
        """Apply optional camera controls from config."""
        try:
            device = profile.get_device()
            sensors = list(device.query_sensors())
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
            auto_exposure = self._rs_cfg.get("color_auto_exposure", None)
            auto_exposure_priority = self._rs_cfg.get(
                "color_auto_exposure_priority", None
            )
            exposure_us = self._rs_cfg.get("color_exposure_us", None)
            gain = self._rs_cfg.get("color_gain", None)
            sharpness = self._rs_cfg.get("color_sharpness", None)

            self._set_sensor_option(
                color_sensor, rs.option.enable_auto_exposure, auto_exposure
            )
            self._set_sensor_option(
                color_sensor,
                rs.option.auto_exposure_priority,
                auto_exposure_priority,
            )
            if exposure_us is not None:
                self._set_sensor_option(color_sensor, rs.option.exposure, exposure_us)
            if gain is not None:
                self._set_sensor_option(color_sensor, rs.option.gain, gain)
            if sharpness is not None:
                self._set_sensor_option(color_sensor, rs.option.sharpness, sharpness)

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
            if not sensor.supports(option):
                return
            sensor.set_option(option, float(value))
        except Exception as exc:
            try:
                opt_name = str(option)
            except Exception:
                opt_name = "unknown"
            print(f"Warning: failed to set RealSense option {opt_name}={value} ({exc})")

    def _wait_for_latest_frames(self):
        """Fetch frames and optionally drain buffered stale frames."""
        t0 = time.perf_counter()
        frames = self.rs_pipeline.wait_for_frames()
        wait_s = time.perf_counter() - t0

        dropped_frames = 0
        if self._drop_stale_frames:
            latest_frames = frames
            for _ in range(max(self._max_frame_drain, 0)):
                polled = self.rs_pipeline.poll_for_frames()
                if not polled:
                    break
                latest_frames = polled
                dropped_frames += 1
            frames = latest_frames

        return frames, {"wait_s": wait_s, "dropped_frames": dropped_frames}

    def mouse_callback(self, event, x, y, _flags, _param):
        """Handle mouse clicks for point collection"""
        if self.tracking_started:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point
            self.current_click_points.append([x, y])
            self.current_click_labels.append(1)
            print(
                f"Added positive point for object {self._active_prompt_object_index()}: ({x}, {y})"
            )

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point
            self.current_click_points.append([x, y])
            self.current_click_labels.append(0)
            print(
                f"Added negative point for object {self._active_prompt_object_index()}: ({x}, {y})"
            )

    def reset_points(self):
        """Reset collected points and restart tracking"""
        self.click_point_groups = []
        self.click_label_groups = []
        self.current_click_points = []
        self.current_click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # Recreate the pipeline lazily on the next tracking start.
        self.pipeline = None

        self._show_status_screen(
            "Prompts reset. Add clicks for object 1, press 's' for next object, 'r' to run."
        )
        print("Prompts reset. Add clicks for object 1.")

    def _count_positive_labels(self, labels) -> int:
        return int(np.sum(np.asarray(labels, dtype=np.int32) == 1))

    def _active_prompt_object_index(self) -> int:
        return len(self.click_point_groups) + 1

    def _prompt_object_color(self, obj_idx: int):
        palette = [
            (0, 255, 0),
            (255, 200, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (0, 128, 255),
        ]
        return palette[obj_idx % len(palette)]

    def _finalize_current_object(self) -> bool:
        if len(self.current_click_points) == 0:
            print("No points collected for the current object yet.")
            return False

        if self._count_positive_labels(self.current_click_labels) == 0:
            print(
                "The current object needs at least one positive click before it can be finalized."
            )
            return False

        self.click_point_groups.append([list(pt) for pt in self.current_click_points])
        self.click_label_groups.append([int(label) for label in self.current_click_labels])
        obj_idx = len(self.click_point_groups)
        num_pos = self._count_positive_labels(self.current_click_labels)
        num_neg = len(self.current_click_labels) - num_pos
        self.current_click_points = []
        self.current_click_labels = []
        print(
            f"Finalized object {obj_idx} with {num_pos} positive and {num_neg} negative point(s)."
        )
        print(f"Collecting prompts for object {self._active_prompt_object_index()} next.")
        return True

    def _prompt_groups_ready_for_tracking(self) -> bool:
        return len(self.click_point_groups) > 0 or len(self.current_click_points) > 0

    def _prepare_prompt_groups_for_tracking(self):
        if len(self.current_click_points) > 0:
            if not self._finalize_current_object():
                return None, None

        if len(self.click_point_groups) == 0:
            print("No object prompts collected yet. Click points before pressing 'r'.")
            return None, None

        prompt_points = [
            [list(pt) for pt in point_group] for point_group in self.click_point_groups
        ]
        prompt_labels = [
            [int(label) for label in label_group] for label_group in self.click_label_groups
        ]
        return prompt_points, prompt_labels

    def _draw_prompt_group(self, image, points, labels, obj_idx, is_current=False):
        color = self._prompt_object_color(obj_idx)
        label_prefix = f"O{obj_idx + 1}"
        for point_idx, (point, label) in enumerate(zip(points, labels), start=1):
            px = int(point[0])
            py = int(point[1])
            if int(label) == 1:
                cv2.circle(image, (px, py), 6 if is_current else 5, color, -1)
                cv2.circle(image, (px, py), 8 if is_current else 7, (255, 255, 255), 1)
            else:
                cv2.drawMarker(
                    image,
                    (px, py),
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12 if is_current else 10,
                    thickness=2,
                )

            cv2.putText(
                image,
                f"{label_prefix}:{point_idx}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color if int(label) == 1 else (0, 0, 255),
                1,
            )

    def _draw_prompt_collection_overlay(self, display_frame):
        for obj_idx, (points, labels) in enumerate(
            zip(self.click_point_groups, self.click_label_groups)
        ):
            self._draw_prompt_group(display_frame, points, labels, obj_idx=obj_idx)

        if len(self.current_click_points) > 0:
            self._draw_prompt_group(
                display_frame,
                self.current_click_points,
                self.current_click_labels,
                obj_idx=len(self.click_point_groups),
                is_current=True,
            )

        status_lines = [
            "L:+  R:-  s:next object  r:run  c:clear  q:quit",
            f"Finalized objects: {len(self.click_point_groups)} | Active object: {self._active_prompt_object_index()}",
            (
                f"Current points: {len(self.current_click_points)}"
                f" | Ready to run: {'yes' if self._prompt_groups_ready_for_tracking() else 'no'}"
            ),
        ]

        for line_idx, line in enumerate(status_lines):
            cv2.putText(
                display_frame,
                line,
                (10, 30 + 30 * line_idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _show_status_screen(self, message: str):
        """Display a simple status frame so the UI becomes visible early."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lines = [
            message,
            "L:+  R:-  s:next object  r:run  c:clear  q:quit",
        ]
        if self._bbox_estimation_mode == "manual":
            lines.append("Press 'b' during tracking to estimate the bounding box")
        y = 180
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y += 40
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)

    def _initialize_pipeline(self):
        """Construct the heavy tracking pipeline on demand."""
        if self.pipeline is not None:
            return

        self._show_status_screen("Initializing tracking pipeline...")
        print("Initializing tracking pipeline...")

        if self.cfg.pipeline.type == "single_process":
            from point2pose.pipeline.pipeline_single_process import (
                PipelineSingleProcess,
            )

            self.pipeline = PipelineSingleProcess(self.cfg)
        elif self.cfg.pipeline.type == "modular":
            from point2pose.pipeline.modular_pipeline import ModularPipeline

            self.pipeline = ModularPipeline(self.cfg)
        else:
            raise ValueError(f"Invalid pipeline type: {self.cfg.pipeline.type}")

        print("Tracking pipeline initialized.")

    def _generate_next_frame_color(self):
        """Generate a new distinctive HSV-based color and convert it to BGR."""
        golden_ratio_conjugate = 0.6180339887498949
        base_index = len(self._frame_color_lookup)
        saturation_cycle = (255, 230, 200, 180)
        value_cycle = (255, 235, 215)

        attempt = 0
        while True:
            idx = base_index + attempt
            hue = int(round(((idx * golden_ratio_conjugate) % 1.0) * 179)) % 180
            saturation = saturation_cycle[idx % len(saturation_cycle)]
            value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]

            hsv_tuple = (hue, saturation, value)
            if hsv_tuple not in self._frame_color_used_hsv:
                self._frame_color_used_hsv.add(hsv_tuple)
                hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

            attempt += 1

    def _colors_for_frame_ids(self, frame_ids: np.ndarray) -> np.ndarray:
        """Return consistent BGR colors for the provided frame ids."""
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

    def start_tracking(self):
        """Initialize pipeline tracking with collected points"""
        prompt_points, prompt_labels = self._prepare_prompt_groups_for_tracking()
        if prompt_points is None or prompt_labels is None:
            return False

        self._initialize_pipeline()

        # Get current frame for initialization
        frames, _ = self._wait_for_latest_frames()
        color_frame = frames.get_color_frame()

        # Align the depth frame to color frame
        aligned_frames = self._rs_align.process(frames)
        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not aligned_depth_frame:
            print("Failed to get frames for initialization")
            return False

        # Convert frames to numpy arrays
        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_depth = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)

        # Create Frame object for pipeline
        frame = Frame(
            id=0,
            rgb=frame_rgb,
            depth=frame_depth,
            intrinsics=self.camera_intrinsics,
            depth_factor=self.depth_factor,
            timestamp=time.time(),
        )

        # Add user points to pipeline
        self.pipeline.add_user_points(prompt_points, prompt_labels)

        # Initialize pipeline with first frame
        self.current_poses = self.pipeline.step(frame)

        self.tracking_started = True
        self.frame_count = 1

        total_points = int(sum(len(group) for group in prompt_points))
        print(
            f"Started tracking with {total_points} prompt point(s) across "
            f"{len(prompt_points)} object(s)."
        )
        print(f"Number of objects: {len(self.current_poses)}")

        return True

    def _save_sdf_volume_npz(self, obj, save_path: Path) -> bool:
        sdf_meta = getattr(obj, "sdf", None)
        tsdf = None
        color = None
        vol_bnds = None
        vol_origin = None
        voxel_size = None
        num_integrated = None

        if isinstance(sdf_meta, dict):
            tsdf = sdf_meta.get("tsdf", None)
            color = sdf_meta.get("color", None)
            vol_bnds = sdf_meta.get("vol_bnds", None)
            vol_origin = sdf_meta.get("vol_origin", None)
            voxel_size = sdf_meta.get("voxel_size", None)
            num_integrated = sdf_meta.get("num_integrated", None)

        sdf_volume = getattr(obj, "sdf_volume", None)
        if tsdf is None and sdf_volume is not None and hasattr(sdf_volume, "get_volume"):
            try:
                tsdf, color = sdf_volume.get_volume()
            except Exception:
                tsdf, color = None, None

        if vol_bnds is None and sdf_volume is not None and hasattr(sdf_volume, "_vol_bnds"):
            vol_bnds = getattr(sdf_volume, "_vol_bnds", None)
        if vol_origin is None and sdf_volume is not None and hasattr(sdf_volume, "_vol_origin"):
            vol_origin = getattr(sdf_volume, "_vol_origin", None)
        if voxel_size is None and sdf_volume is not None and hasattr(sdf_volume, "_voxel_size"):
            voxel_size = getattr(sdf_volume, "_voxel_size", None)
        if num_integrated is None:
            num_integrated = getattr(obj, "sdf_num_integrated", None)

        if tsdf is None or vol_bnds is None or voxel_size is None:
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tsdf": np.asarray(tsdf, dtype=np.float32),
            "vol_bnds": np.asarray(vol_bnds, dtype=np.float32),
            "voxel_size": np.asarray([float(voxel_size)], dtype=np.float32),
            "obj_id": np.asarray([int(getattr(obj, "obj_id", -1))], dtype=np.int32),
        }
        if color is not None:
            payload["color"] = np.asarray(color)
        if vol_origin is not None:
            payload["vol_origin"] = np.asarray(vol_origin, dtype=np.float32)
        if num_integrated is not None:
            payload["num_integrated"] = np.asarray([int(num_integrated)], dtype=np.int32)

        np.savez_compressed(save_path, **payload)
        return True

    def _save_final_sdf_outputs(self):
        if not self._save_final_sdf_on_exit:
            return

        if self.pipeline is None or not getattr(self.pipeline, "objects", None):
            print("Final SDF export skipped: no tracked objects were available.")
            return

        try:
            from point2pose.utils.mesh_eval import export_final_meshes_from_pipeline
        except Exception as exc:
            print(f"Final SDF export skipped: failed to import mesh export helper ({exc}).")
            return

        mesh_dir = self._final_sdf_output_dir / "meshes"
        volume_dir = self._final_sdf_output_dir / "volumes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        volume_dir.mkdir(parents=True, exist_ok=True)

        try:
            saved_meshes = export_final_meshes_from_pipeline(
                self.pipeline,
                str(mesh_dir),
                prefix="realsense_sdf_obj",
            )
        except Exception as exc:
            saved_meshes = {}
            print(f"Final SDF mesh export failed: {exc}")

        saved_any = False
        for obj_idx, obj in enumerate(self.pipeline.objects):
            volume_path = volume_dir / f"realsense_sdf_obj_{obj_idx}.npz"
            if self._save_sdf_volume_npz(obj, volume_path):
                saved_any = True
                print(f"Saved SDF volume for object {obj_idx} to {volume_path}")

            mesh_path = saved_meshes.get(obj_idx, None)
            if mesh_path:
                saved_any = True
                print(f"Saved SDF mesh for object {obj_idx} to {mesh_path}")

        if not saved_any:
            print(
                "Final SDF export finished, but no SDF volumes or meshes were available to save."
            )

    def create_frame_from_realsense(self, frame_id):
        """Create Frame object from RealSense data"""
        t_wait = time.perf_counter()
        frames, wait_stats = self._wait_for_latest_frames()
        color_frame = frames.get_color_frame()

        # Align the depth frame to color frame
        t_align = time.perf_counter()
        aligned_frames = self._rs_align.process(frames)
        align_s = time.perf_counter() - t_align
        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not aligned_depth_frame:
            print("Failed to get frames for initialization")
            return False

        # Convert frames to numpy arrays
        t_convert = time.perf_counter()
        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_depth = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)
        convert_s = time.perf_counter() - t_convert

        self._last_capture_stats = {
            "wait_s": wait_stats["wait_s"],
            "align_s": align_s,
            "convert_s": convert_s,
            "dropped_frames": wait_stats["dropped_frames"],
        }

        # Create Frame object
        frame = Frame(
            id=frame_id,
            rgb=frame_rgb,
            depth=frame_depth,
            intrinsics=self.camera_intrinsics,
            depth_factor=self.depth_factor,
            timestamp=time.time(),
        )

        return frame

    def _bbox_from_sdf_volume(self, obj):
        """Extract a (2, 3) min/max box from the reconstructed SDF volume, if available."""
        sdf_meta = getattr(obj, "sdf", None)
        if isinstance(sdf_meta, dict):
            vol_bnds = sdf_meta.get("vol_bnds", None)
            if vol_bnds is not None:
                vol_bnds = np.asarray(vol_bnds, dtype=float)
                if vol_bnds.shape == (3, 2):
                    return vol_bnds.T.copy()

        sdf_volume = getattr(obj, "sdf_volume", None)
        vol_bnds = getattr(sdf_volume, "_vol_bnds", None)
        if vol_bnds is None:
            return None

        vol_bnds = np.asarray(vol_bnds, dtype=float)
        if vol_bnds.shape != (3, 2):
            return None

        return vol_bnds.T.copy()

    def _subsample_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        """Deterministically subsample a point set to keep bbox fitting lightweight."""
        if points.shape[0] <= max_points:
            return points
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        return points[idx]

    def _collect_object_frame_dense_points(
        self, obj, max_keyframes: int = 8, max_points_per_keyframe: int = 1200
    ) -> np.ndarray:
        """
        Gather dense object points from recent keyframes and convert them into object coordinates.
        """
        keyframes = list(getattr(obj, "keyframes", []) or [])
        if not keyframes:
            return np.empty((0, 3), dtype=float)

        pts_obj_all = []
        for kf in keyframes[-max_keyframes:]:
            dense_pts = np.asarray(getattr(kf, "dense_pts", np.empty((0, 3))), dtype=float)
            if dense_pts.ndim != 2 or dense_pts.shape[1] != 3 or dense_pts.shape[0] == 0:
                continue

            finite_mask = np.all(np.isfinite(dense_pts), axis=1)
            dense_pts = dense_pts[finite_mask]
            if dense_pts.shape[0] == 0:
                continue

            dense_pts = self._subsample_points(dense_pts, max_points_per_keyframe)
            pts_obj = transform_pts(inverse_SE3(np.asarray(kf.pose, dtype=float)), dense_pts)
            pts_obj_all.append(pts_obj)

        if not pts_obj_all:
            return np.empty((0, 3), dtype=float)

        pts_obj = np.concatenate(pts_obj_all, axis=0)
        finite_mask = np.all(np.isfinite(pts_obj), axis=1)
        pts_obj = pts_obj[finite_mask]
        if pts_obj.shape[0] == 0:
            return pts_obj

        # Robust trim to remove occasional mask leakage / bad depth.
        center = np.median(pts_obj, axis=0)
        dist = np.linalg.norm(pts_obj - center[None, :], axis=1)
        if dist.shape[0] >= 50:
            keep = dist <= np.percentile(dist, 97.5)
            if np.count_nonzero(keep) >= 20:
                pts_obj = pts_obj[keep]

        return pts_obj

    def _bbox_from_dense_keyframes(self, obj):
        """
        Fit a bbox from accumulated dense multi-view geometry in object coordinates.

        This is a much better proxy for object extent than TSDF volume bounds, which
        include reconstruction padding and are not intended to be visualized as the
        object bbox.
        """
        pts_obj = self._collect_object_frame_dense_points(obj)
        if pts_obj.shape[0] < 20:
            return None

        pts_obj = self._subsample_points(pts_obj, 6000)

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_obj.astype(np.float64))

            obb = None
            if hasattr(pcd, "get_minimal_oriented_bounding_box"):
                try:
                    obb = pcd.get_minimal_oriented_bounding_box(robust=True)
                except TypeError:
                    obb = pcd.get_minimal_oriented_bounding_box()
                except RuntimeError:
                    obb = None

            if obb is None:
                try:
                    obb = pcd.get_oriented_bounding_box(robust=True)
                except TypeError:
                    obb = pcd.get_oriented_bounding_box()

            center = np.asarray(obb.center, dtype=float).reshape(3)
            extent = np.asarray(obb.extent, dtype=float).reshape(3)
            rot = np.asarray(obb.R, dtype=float).reshape(3, 3)
            if np.all(np.isfinite(center)) and np.all(np.isfinite(extent)) and np.all(
                extent > 1e-4
            ):
                return {"center": center, "extent": extent, "rot": rot}
        except Exception:
            pass

        mn = pts_obj.min(axis=0)
        mx = pts_obj.max(axis=0)
        extent = np.maximum(mx - mn, 1e-3)
        pad = np.maximum(0.005, 0.05 * extent)
        return np.vstack([mn - pad, mx + pad])

    def _reject_bbox_outliers_from_key_points(self, points: np.ndarray) -> np.ndarray:
        """
        Robustly reject sparse keypoint outliers before fitting a fallback bbox.

        The thresholds are intentionally loose so we remove isolated bad depth points
        without cutting away legitimate object extremities.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 6:
            return pts

        bbox_points = pts

        # Pass 1: coordinate-wise MAD gate around the median.
        center = np.median(bbox_points, axis=0)
        abs_dev = np.abs(bbox_points - center[None, :])
        mad = np.median(abs_dev, axis=0)
        spread = np.ptp(bbox_points, axis=0)
        axis_scale = np.maximum(1.4826 * mad, 0.10 * spread + 0.003)
        axis_keep = np.all(abs_dev <= (3.5 * axis_scale)[None, :], axis=1)
        if np.count_nonzero(axis_keep) >= 4:
            bbox_points = bbox_points[axis_keep]

        # Pass 2: radial MAD gate to catch points that still sit far from the cloud.
        if bbox_points.shape[0] >= 6:
            center = np.median(bbox_points, axis=0)
            dist = np.linalg.norm(bbox_points - center[None, :], axis=1)
            dist_med = np.median(dist)
            dist_mad = np.median(np.abs(dist - dist_med))
            dist_scale = max(1.4826 * dist_mad, 0.02)
            radial_keep = dist <= (dist_med + 3.5 * dist_scale)
            if np.count_nonzero(radial_keep) >= 4:
                bbox_points = bbox_points[radial_keep]

        # Final soft percentile trim for larger sets.
        if bbox_points.shape[0] >= 12:
            lo = np.percentile(bbox_points, 2.5, axis=0)
            hi = np.percentile(bbox_points, 97.5, axis=0)
            soft_keep = np.all((bbox_points >= lo) & (bbox_points <= hi), axis=1)
            if np.count_nonzero(soft_keep) >= 4:
                bbox_points = bbox_points[soft_keep]

        return bbox_points

    def _bbox_from_key_points(self, obj):
        """
        Build a conservative fallback AABB from sparse keypoints in frame-0 coordinates.
        This is only for visualization when the pipeline has not produced an explicit bbox.
        """
        key_points = np.asarray(getattr(obj, "key_points", np.empty((0, 3))), dtype=float)
        if key_points.ndim != 2 or key_points.shape[1] != 3:
            return None

        finite_mask = np.all(np.isfinite(key_points), axis=1)
        key_points = key_points[finite_mask]
        if key_points.shape[0] < 3:
            return None

        bbox_points = self._reject_bbox_outliers_from_key_points(key_points)
        if bbox_points.shape[0] < 3:
            bbox_points = key_points

        mn = bbox_points.min(axis=0)
        mx = bbox_points.max(axis=0)
        extent = np.maximum(mx - mn, 1e-3)
        pad = np.maximum(0.01, 0.1 * extent)
        return np.vstack([mn - pad, mx + pad])

    def _resolve_visualization_box(self, obj):
        """
        Resolve the pose/bbox source used for visualization.

        Returns:
            pose_in_cam: pose to use with the returned bbox source
            bbox_source: Open3D bbox-like object or (2, 3) min/max array
            assume_pose_is_bbox_center: whether pose_in_cam already points at bbox center
        """
        pose_rel = np.asarray(getattr(obj, "pose", np.eye(4)), dtype=float)
        init_pose = np.asarray(getattr(obj, "init_pose", np.eye(4)), dtype=float)

        bbox_source = getattr(obj, "bbox", None)
        if bbox_source is None:
            bbox_source = getattr(obj, "init_bbox", None)
        if bbox_source is not None:
            if isinstance(bbox_source, dict):
                bbox_frame = str(bbox_source.get("frame", "bbox_center")).lower()
                if bbox_frame in {"object", "object_local", "mesh"}:
                    return pose_rel, bbox_source, False
            return pose_rel @ init_pose, bbox_source, True

        dense_bbox = self._bbox_from_dense_keyframes(obj)
        if dense_bbox is not None:
            return pose_rel, dense_bbox, False

        kp_bbox = self._bbox_from_key_points(obj)
        if kp_bbox is not None:
            return pose_rel, kp_bbox, False

        sdf_bbox = self._bbox_from_sdf_volume(obj)
        if sdf_bbox is not None:
            return pose_rel, sdf_bbox, False

        return pose_rel @ init_pose, None, True

    def _resolve_visualization_axis_pose(
        self, pose_in_cam, bbox_source, assume_pose_is_bbox_center
    ):
        """
        Draw the axis in the same frame as the visualized bbox.

        When the bbox is defined in object coordinates with its own center and optional
        orientation, we place the axis at that bbox-centered frame instead of the raw
        object origin.
        """
        pose_in_cam = np.asarray(pose_in_cam, dtype=float)
        if bbox_source is None or assume_pose_is_bbox_center:
            return pose_in_cam

        info = _extract_bbox_info(bbox_source)
        center = info["center"]
        rot = info["rot"]
        mn = info["mn"]
        mx = info["mx"]

        if center is None and (mn is not None) and (mx is not None):
            center = 0.5 * (np.asarray(mn, dtype=float) + np.asarray(mx, dtype=float))

        if center is None and rot is None:
            return pose_in_cam

        T_local = np.eye(4, dtype=float)
        if rot is not None:
            T_local[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
        if center is not None:
            T_local[:3, 3] = np.asarray(center, dtype=float).reshape(3)

        return pose_in_cam @ T_local

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        """Visualize tracking results on the frame"""
        viz_start = time.perf_counter()
        viz_timings = {
            "prepare": 0.0,
            "masks": 0.0,
            "points": 0.0,
            "pose": 0.0,
            "save": 0.0,
        }

        t0 = time.perf_counter()
        display_frame = frame.rgb.copy()
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        viz_timings["prepare"] = time.perf_counter() - t0

        height, width = display_frame.shape[:2]

        # Draw segmentation masks if available
        if hasattr(frame, "mask") and frame.mask is not None:
            t0 = time.perf_counter()
            mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            mask_overlay[..., 1] = 255  # Green base

            for i in range(len(frame.mask)):
                obj_mask = frame.mask[i, 0] > 0.0
                ## TODO: optimize this by removing the cpu().numpy()
                obj_mask = obj_mask.cpu().numpy()
                if np.any(obj_mask):
                    # Color each object differently
                    hue = (i + 3) / (len(frame.mask) + 3) * 255
                    mask_overlay[obj_mask, 0] = hue
                    mask_overlay[obj_mask, 2] = 255
            # masks: [K,1,H,W] -> [K,H,W] bool on CUDA
            # masks = frame.mask[:, 0] > 0

            # K, H, W = masks.shape

            # # For pixels with multiple objects, take the first object's index.
            # # (With 0/1 masks, argmax returns the first True along dim=0.)
            # idx_first = torch.argmax(masks.int(), dim=0)  # [H,W]
            # has_any = masks.any(dim=0)  # [H,W] bool

            # # Precompute hues per object on GPU
            # hues = (
            #     (
            #         (
            #             (torch.arange(K, device=masks.device, dtype=torch.float32) + 3)
            #             / (K + 3)
            #         )
            #         * 255.0
            #     )
            #     .round()
            #     .to(torch.uint8)
            # )  # [K]

            # # Build overlay on GPU: [H,W,3] uint8
            # overlay = torch.zeros((H, W, 3), dtype=torch.uint8, device=masks.device)

            # # Write hue to channel 0 and set channel 2 = 255 where masked
            # overlay[..., 0] = torch.where(has_any, hues[idx_first], overlay[..., 0])
            # overlay[..., 2] = torch.where(
            #     has_any,
            #     torch.tensor(255, dtype=torch.uint8, device=masks.device),
            #     overlay[..., 2],
            # )

            # mask_overlay = overlay.cpu().numpy()

            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
            display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)
            viz_timings["masks"] = time.perf_counter() - t0

        if self._visualize_points:
            t0 = time.perf_counter()
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

                    # Generate N by 3 array with (0,255,0) for each row
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
                    draw_points_on_image(
                        display_frame,
                        track_2d_points,
                        visible_color,
                    )
            elif self._points_vis_method == "visible_uncertainty":
                # Plot only visible points, colored by their uncertainty colors
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
                # Color each point based on the frame id it was first seen (object.key_point_frames)
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue

                    track_idx = self.pipeline.track_table.obj2track_map[i]
                    track_2d_points = self.pipeline.track_table.track_2d[track_idx]
                    visible_mask = self.pipeline.track_table.visible[track_idx]

                    # Only proceed if there are visible points
                    if not np.any(visible_mask):
                        continue
                    if obj.key_point_frames.shape[0] == 0:
                        continue
                    # Align per-object track order with object's key point order
                    # Assume key_point_frames order corresponds to obj2track_map order
                    num_tracks_for_obj = len(track_idx)
                    kp_frames_for_obj = obj.key_point_frames[
                        :num_tracks_for_obj
                    ].astype(np.int32)

                    # Frame ids for visible points; replace unknown -1 with current frame id if available
                    frame_ids = kp_frames_for_obj[visible_mask]
                    if frame_id is not None:
                        frame_ids = frame_ids.copy()
                        frame_ids[frame_ids == -1] = int(frame_id)

                    # Use cached, distinctive colors per frame id
                    colors_bgr = self._colors_for_frame_ids(frame_ids)

                    # Draw only visible points for this object, using aligned colors
                    draw_points_on_image(
                        display_frame,
                        track_2d_points[visible_mask],
                        colors_bgr,
                    )
            viz_timings["points"] = time.perf_counter() - t0

        # Draw pose information
        t0 = time.perf_counter()
        for i, obj in enumerate(objects):
            if obj.pose is not None:
                pose_in_cam, bbox_source, assume_pose_is_bbox_center = (
                    self._resolve_visualization_box(obj)
                )
                pose_draw = pose_in_cam
                axis_pose = self._resolve_visualization_axis_pose(
                    pose_in_cam=pose_in_cam,
                    bbox_source=bbox_source,
                    assume_pose_is_bbox_center=assume_pose_is_bbox_center,
                )

                if bbox_source is not None:
                    pose_draw, bbox_min_max_local, bbox_corners_local = (
                        _resolve_pose_and_bbox(
                            pose_in_cam=pose_in_cam,
                            bbox_source=bbox_source,
                            bbox_frame="mesh",
                            assume_pose_is_bbox_center=assume_pose_is_bbox_center,
                        )
                    )

                    if bbox_corners_local is not None:
                        display_frame = draw_oriented_3d_box(
                            self.camera_intrinsics,
                            display_frame,
                            pose_draw,
                            bbox_corners_local,
                        )
                    elif bbox_min_max_local is not None:
                        display_frame = draw_posed_3d_box(
                            self.camera_intrinsics,
                            display_frame,
                            pose_draw,
                            bbox_min_max_local,
                        )

                display_frame = draw_xyz_axis(
                    image=display_frame, ob_in_cam=axis_pose, K=self.camera_intrinsics
                )
        viz_timings["pose"] = time.perf_counter() - t0

        # Save image if flag is enabled and frame_id is provided
        if self._save_images and frame_id is not None:
            t0 = time.perf_counter()
            image_filename = self._output_image_dir / f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(image_filename), display_frame)
            viz_timings["save"] = time.perf_counter() - t0

        viz_timings["total"] = time.perf_counter() - viz_start
        self._last_visualization_timings = viz_timings

        return display_frame

    def run_tracking(self):
        """Main tracking loop"""
        try:
            while True:
                if not self.tracking_started:
                    # Get frames for point collection visualization
                    frames = self.rs_pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        continue

                    # Convert to display format
                    frame = np.asanyarray(color_frame.get_data())
                    display_frame = frame.copy()

                    self._draw_prompt_collection_overlay(display_frame)

                    # Save image if flag is enabled (for point collection phase)
                    if self._save_images:
                        image_filename = (
                            self._output_image_dir
                            / f"point_collection_{self.frame_count:06d}.png"
                        )
                        cv2.imwrite(str(image_filename), display_frame)

                else:
                    # Create frame for pipeline
                    frame = self.create_frame_from_realsense(self.frame_count)
                    if frame is None:
                        continue

                    # Run pipeline step
                    t_pipeline = time.perf_counter()
                    self.pipeline.step(frame)
                    pipeline_s = time.perf_counter() - t_pipeline

                    # Visualize results
                    t_viz = time.perf_counter()
                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects, self.frame_count
                    )
                    viz_s = time.perf_counter() - t_viz

                    # Show tracking info
                    height, _ = display_frame.shape[:2]
                    cv2.putText(
                        display_frame,
                        f"Frame: {self.frame_count}",
                        (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Objects: {len(self.current_poses) if self.current_poses is not None else 0}",
                        (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    self.frame_count += 1

                # Display the frame
                t_display = time.perf_counter()
                cv2.imshow(self._window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                display_s = time.perf_counter() - t_display

                if (
                    self.tracking_started
                    and self._timing_debug_every > 0
                    and (self.frame_count % self._timing_debug_every) == 0
                ):
                    capture_stats = self._last_capture_stats
                    viz_stats = self._last_visualization_timings
                    pipeline_stats = (
                        getattr(self.pipeline, "last_step_module_times", {})
                        if self.pipeline is not None
                        else {}
                    )
                    frontend_stats = (
                        getattr(self.pipeline, "last_frontend_timings", {})
                        if self.pipeline is not None
                        else {}
                    )
                    loop_total_s = (
                        capture_stats["wait_s"]
                        + capture_stats["align_s"]
                        + capture_stats["convert_s"]
                        + pipeline_s
                        + viz_s
                        + display_s
                    )
                    approx_fps = 1.0 / max(loop_total_s, 1e-6)
                    print(
                        "[RealSenseExample] "
                        f"Frame {self.frame_count}: "
                        f"capture={1000.0 * (capture_stats['wait_s'] + capture_stats['align_s'] + capture_stats['convert_s']):.1f}ms "
                        f"(wait={1000.0 * capture_stats['wait_s']:.1f}ms, "
                        f"align={1000.0 * capture_stats['align_s']:.1f}ms, "
                        f"convert={1000.0 * capture_stats['convert_s']:.1f}ms, "
                        f"dropped={capture_stats['dropped_frames']}), "
                        f"pipeline={1000.0 * pipeline_s:.1f}ms "
                        f"(frontend={1000.0 * pipeline_stats.get('frontend', 0.0):.1f}ms, "
                        f"track_table={1000.0 * pipeline_stats.get('track_table', 0.0):.1f}ms, "
                        f"track_compact={1000.0 * pipeline_stats.get('track_compact', 0.0):.1f}ms, "
                        f"recovery={1000.0 * pipeline_stats.get('recovery', 0.0):.1f}ms, "
                        f"local_opt={1000.0 * pipeline_stats.get('local_opt', 0.0):.1f}ms, "
                        f"keyframe={1000.0 * pipeline_stats.get('keyframe', 0.0):.1f}ms, "
                        f"global_opt={1000.0 * pipeline_stats.get('global_opt', 0.0):.1f}ms, "
                        f"logging={1000.0 * pipeline_stats.get('logging', 0.0):.1f}ms), "
                        f"frontend_detail="
                        f"(segmenter={1000.0 * frontend_stats.get('segmenter', 0.0):.1f}ms, "
                        f"tracker={1000.0 * frontend_stats.get('tracker', 0.0):.1f}ms, "
                        f"2d_to_3d={1000.0 * frontend_stats.get('2d_to_3d', 0.0):.1f}ms, "
                        f"extract_valid={1000.0 * frontend_stats.get('extract_valid', 0.0):.1f}ms, "
                        f"registration={1000.0 * frontend_stats.get('registration', 0.0):.1f}ms, "
                        f"dense_recovery={1000.0 * frontend_stats.get('dense_recovery', 0.0):.1f}ms), "
                        f"visualize={1000.0 * viz_s:.1f}ms "
                        f"(prepare={1000.0 * viz_stats.get('prepare', 0.0):.1f}ms, "
                        f"masks={1000.0 * viz_stats.get('masks', 0.0):.1f}ms, "
                        f"points={1000.0 * viz_stats.get('points', 0.0):.1f}ms, "
                        f"pose={1000.0 * viz_stats.get('pose', 0.0):.1f}ms, "
                        f"save={1000.0 * viz_stats.get('save', 0.0):.1f}ms), "
                        f"display={1000.0 * display_s:.1f}ms, "
                        f"loop_total={1000.0 * loop_total_s:.1f}ms "
                        f"(~{approx_fps:.1f} FPS)"
                    )

                if key == ord("q"):
                    break
                elif key == ord("s") and not self.tracking_started:
                    self._finalize_current_object()
                elif key == ord("r") and not self.tracking_started:
                    if self.start_tracking():
                        print("Pipeline tracking started!")
                    else:
                        print("Failed to start pipeline tracking!")
                elif (
                    key == ord("b")
                    and self.tracking_started
                    and self._bbox_estimation_mode == "manual"
                ):
                    if self.pipeline is None or not hasattr(
                        self.pipeline, "update_object_bboxes"
                    ):
                        print("Manual bbox estimation is not available in this pipeline.")
                    else:
                        updated = self.pipeline.update_object_bboxes(force=True)
                        print(f"Manual bbox estimation updated {updated} object(s).")
                elif key == ord("c"):
                    self.reset_points()

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            self._save_final_sdf_outputs()
            # Cleanup
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    """Main function to run the RealSense pipeline tracker"""
    try:
        tracker = RealSensePipelineTracker(
            config_path="configs/pipeline/pipeline_test.yaml",
        )
        tracker.run_tracking()
    except (RuntimeError, OSError, ImportError) as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. RealSense camera connected")
        print("2. Pipeline configuration file exists")
        print("3. Required dependencies installed")
        print("4. SAM2 and TAPIR checkpoints available")


if __name__ == "__main__":
    main()
