import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.data_types.frame import Frame

# from point2pose.data_types.object import Object
from point2pose.utils.visualization import (
    draw_xyz_axis,
    draw_posed_3d_box,
    draw_points_on_image,
    get_n_uncertainty_colors,
)


class RealSensePipelineTracker:
    """
    RealSense tracker that integrates SAM2 segmentation with the point2pose pipeline.
    This tracker performs object tracking and pose estimation using the complete pipeline.
    """

    def __init__(
        self,
        config_path="configs/pipeline/pipeline_test2.yaml",
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

        rs_serial = self.cfg.realsense.params.rs_serial

        self._visualize_points = self.cfg.visualization.params.visualize_points
        self._points_vis_method = self.cfg.visualization.params.points_vis_method
        self._save_images = self.cfg.visualization.params.save_images
        self._output_image_dir = Path(self.cfg.visualization.params.output_image_dir)

        # Create output directory if saving images is enabled
        if self._save_images:
            self._output_image_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Image saving enabled. Images will be saved to: {self._output_image_dir}"
            )

        # Initialize pipeline
        # self.pipeline = Pipeline(self.cfg)

        # Initialize RealSense camera
        self._init_realsense(rs_serial)

        # Per-object prompt-point storage. Each entry holds one object's clicked
        # points/labels. Start with a single (empty) object; press 'n' to begin the
        # next object.
        self.object_points = [[]]
        self.object_labels = [[]]
        self.current_obj = 0
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # SAM2 mask preview (recomputed only when the point set changes)
        self._preview_masks = None
        self._preview_dirty = False
        # Distinct BGR colors used to draw points/overlays per object
        self._obj_palette = [
            (0, 255, 0),
            (255, 128, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (0, 128, 255),
            (255, 255, 0),
            (128, 255, 0),
        ]

        # Cache for consistent, distinctive frame-based point colors
        self._frame_color_lookup = {}
        self._frame_color_used_hsv = set()

        # Create window and set mouse callback
        cv2.startWindowThread()
        cv2.namedWindow("RealSense Pipeline Tracker", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense Pipeline Tracker", self.mouse_callback)

        if self.cfg.pipeline.type != "modular":
            raise ValueError(
                f"Only 'modular' pipeline is supported, got: {self.cfg.pipeline.type}"
            )
        self.pipeline = ModularPipeline(self.cfg)

        print("Instructions:")
        print("- Left click:  Add positive point to the CURRENT object")
        print("- Right click: Add negative point to the CURRENT object")
        print("- Press 'n':   Start a NEW object (query the next object)")
        print("- Press 's':   Start tracking")
        print("- Press 'r':   Reset points")
        print("- Press 'q':   Quit")

    def _init_realsense(self, rs_serial):
        """Initialize RealSense camera"""
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        align_to = rs.stream.color
        self._rs_align = rs.align(align_to)

        # Start streaming
        profile = self.rs_pipeline.start(config)

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
        print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    def mouse_callback(self, event, x, y, _flags, _param):
        """Collect prompt points for the object currently being annotated."""
        if self.tracking_started:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point for the current object
            self.object_points[self.current_obj].append([x, y])
            self.object_labels[self.current_obj].append(1)
            self._preview_dirty = True
            print(f"[obj {self.current_obj}] +positive point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point for the current object
            self.object_points[self.current_obj].append([x, y])
            self.object_labels[self.current_obj].append(0)
            self._preview_dirty = True
            print(f"[obj {self.current_obj}] -negative point: ({x}, {y})")

    def next_object(self):
        """Finish the current object and start collecting points for a new one."""
        if len(self.object_points[self.current_obj]) == 0:
            print(
                f"[obj {self.current_obj}] no points yet; click before pressing 'n'."
            )
            return
        if not any(lbl == 1 for lbl in self.object_labels[self.current_obj]):
            print(
                f"[obj {self.current_obj}] add at least one positive (left-click) "
                "point first."
            )
            return
        self.object_points.append([])
        self.object_labels.append([])
        self.current_obj += 1
        self._preview_dirty = True
        print(
            f"Started object {self.current_obj}. Click its points, "
            "or press 's' to start tracking."
        )

    def _draw_mask_overlay(self, display_bgr, masks):
        """Overlay per-object SAM2 masks (logits [N,1,H,W], >0 = fg) onto a BGR image."""
        if masks is None or len(masks) == 0:
            return display_bgr

        height, width = display_bgr.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[..., 1] = 255  # green base (HSV)
        any_mask = False

        num_obj = len(masks)
        for i in range(num_obj):
            obj_mask = masks[i, 0] > 0.0
            if hasattr(obj_mask, "cpu"):
                obj_mask = obj_mask.cpu().numpy()
            obj_mask = np.asarray(obj_mask)
            if obj_mask.shape != (height, width):
                obj_mask = cv2.resize(
                    obj_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if np.any(obj_mask):
                any_mask = True
                hue = int((i + 3) / (num_obj + 3) * 255)
                overlay[obj_mask, 0] = hue
                overlay[obj_mask, 2] = 255

        if not any_mask:
            return display_bgr

        overlay = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(display_bgr, 1, overlay, 0.5, 0)

    def reset_points(self):
        """Reset collected points and restart tracking"""
        self.object_points = [[]]
        self.object_labels = [[]]
        self.current_obj = 0
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None
        self._preview_masks = None
        self._preview_dirty = False

        # Reset pipeline
        # self.pipeline = Pipeline(self.cfg)
        if self.cfg.pipeline.type != "modular":
            raise ValueError(
                f"Only 'modular' pipeline is supported, got: {self.cfg.pipeline.type}"
            )
        self.pipeline = ModularPipeline(self.cfg)

        print("Points reset. Click to add new points.")

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
        # Keep only objects that actually have points.
        groups = [
            (pts, lbls)
            for pts, lbls in zip(self.object_points, self.object_labels)
            if len(pts) > 0
        ]
        if len(groups) == 0:
            print("No points collected! Please click on objects first.")
            return False

        # Get current frame for initialization
        frames = self.rs_pipeline.wait_for_frames()
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

        # Add user points to pipeline, grouped per object
        objects_points = [pts for pts, _ in groups]
        objects_labels = [lbls for _, lbls in groups]
        self.pipeline.add_user_points(objects_points, objects_labels)

        # Initialize pipeline with first frame
        self.current_poses = self.pipeline.step(frame)

        self.tracking_started = True
        self.frame_count = 1

        total_points = sum(len(pts) for pts in objects_points)
        print(
            f"Started tracking with {len(objects_points)} object(s), "
            f"{total_points} points"
        )
        print(f"Number of objects: {len(self.current_poses)}")

        return True

    def create_frame_from_realsense(self, frame_id):
        """Create Frame object from RealSense data"""
        # Get current frame for initialization
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

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

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        """Visualize tracking results on the frame"""
        display_frame = frame.rgb.copy()
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        height, width = display_frame.shape[:2]

        # Draw segmentation masks if available
        if hasattr(frame, "mask") and frame.mask is not None:
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

        # Draw pose information
        for i, obj in enumerate(objects):
            if obj.pose is not None:
                pose = obj.pose @ obj.init_pose
                half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
                bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

                display_frame = draw_posed_3d_box(
                    self.camera_intrinsics,
                    display_frame,
                    pose,
                    bbox_min_max_local,
                )
                display_frame = draw_xyz_axis(
                    image=display_frame, ob_in_cam=pose, K=self.camera_intrinsics
                )

        # Save image if flag is enabled and frame_id is provided
        if self._save_images and frame_id is not None:
            image_filename = self._output_image_dir / f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(image_filename), display_frame)

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

                    # Convert to display format (BGR)
                    frame = np.asanyarray(color_frame.get_data())
                    display_frame = frame.copy()

                    # Recompute the SAM2 preview mask only when the point set changed.
                    if self._preview_dirty:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            self._preview_masks = self.pipeline.preview_user_masks(
                                rgb, self.object_points, self.object_labels
                            )
                        except Exception as exc:  # preview is best-effort
                            print(f"[preview] SAM2 preview failed: {exc}")
                            self._preview_masks = None
                        self._preview_dirty = False

                    # Overlay the SAM2 mask preview for the clicked points
                    display_frame = self._draw_mask_overlay(
                        display_frame, self._preview_masks
                    )

                    # Show collected points, colored per object
                    total_points = 0
                    for obj_idx, (pts, lbls) in enumerate(
                        zip(self.object_points, self.object_labels)
                    ):
                        obj_color = self._obj_palette[obj_idx % len(self._obj_palette)]
                        for point, label in zip(pts, lbls):
                            total_points += 1
                            px, py = int(point[0]), int(point[1])
                            if label == 1:
                                # positive: filled circle in the object's color
                                cv2.circle(display_frame, (px, py), 5, obj_color, -1)
                            else:
                                # negative: red cross
                                cv2.drawMarker(
                                    display_frame,
                                    (px, py),
                                    (0, 0, 255),
                                    cv2.MARKER_TILTED_CROSS,
                                    12,
                                    2,
                                )

                    # Show instructions + status
                    cv2.putText(
                        display_frame,
                        "L-click +, R-click -, 'n' next obj, 's' start, 'r' reset",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    num_objects = sum(1 for pts in self.object_points if pts)
                    cv2.putText(
                        display_frame,
                        f"Object {self.current_obj} | objects: {num_objects} | "
                        f"points: {total_points}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self._obj_palette[self.current_obj % len(self._obj_palette)],
                        2,
                    )

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
                    self.pipeline.step(frame)

                    # Visualize results
                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects, self.frame_count
                    )

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
                cv2.imshow("RealSense Pipeline Tracker", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and not self.tracking_started:
                    if self.start_tracking():
                        print("Pipeline tracking started!")
                    else:
                        print("Failed to start pipeline tracking!")
                elif key == ord("n") and not self.tracking_started:
                    self.next_object()
                elif key == ord("r"):
                    self.reset_points()

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            # Cleanup
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    """Main function to run the RealSense pipeline tracker"""
    try:
        tracker = RealSensePipelineTracker()
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
