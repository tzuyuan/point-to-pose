import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from omegaconf import OmegaConf

from point2pose.pipeline.pipeline import Pipeline
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

        # Initialize pipeline
        self.pipeline = Pipeline(self.cfg)

        # Initialize RealSense camera
        self._init_realsense(rs_serial)

        # Mouse click storage
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # Create window and set mouse callback
        cv2.namedWindow("RealSense Pipeline Tracker", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense Pipeline Tracker", self.mouse_callback)

        print("Instructions:")
        print("- Left click: Add positive point")
        print("- Right click: Add negative point")
        print("- Press 's': Start tracking")
        print("- Press 'q': Quit")
        print("- Press 'r': Reset points")

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
        """Handle mouse clicks for point collection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point
            self.click_points.append([x, y])
            self.click_labels.append(1)
            print(f"Added positive point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point
            self.click_points.append([x, y])
            self.click_labels.append(0)
            print(f"Added negative point: ({x}, {y})")

    def reset_points(self):
        """Reset collected points and restart tracking"""
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # Reset pipeline
        self.pipeline = Pipeline(self.cfg)

        print("Points reset. Click to add new points.")

    def start_tracking(self):
        """Initialize pipeline tracking with collected points"""
        if len(self.click_points) == 0:
            print("No points collected! Please click on objects first.")
            return False

        # Get current frame for initialization
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

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
        self.pipeline.add_user_points(self.click_points, self.click_labels)

        # Initialize pipeline with first frame
        self.current_poses = self.pipeline.step(frame)

        self.tracking_started = True
        self.frame_count = 1

        print(f"Started tracking with {len(self.click_points)} points")
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

    def visualize_tracking_results(self, frame, objects):
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

        # Draw pose information
        for i, obj in enumerate(objects):
            if obj.pose is not None:
                half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
                bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

                display_frame = draw_posed_3d_box(
                    self.camera_intrinsics,
                    display_frame,
                    obj.pose,
                    bbox_min_max_local,
                )
                display_frame = draw_xyz_axis(image=display_frame, ob_in_cam=obj.pose)

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

                    # Show collected points
                    for i, (point, label) in enumerate(
                        zip(self.click_points, self.click_labels)
                    ):
                        color = (
                            (0, 255, 0) if label == 1 else (0, 0, 255)
                        )  # Green for positive, red for negative
                        cv2.circle(
                            display_frame, (int(point[0]), int(point[1])), 5, color, -1
                        )
                        cv2.putText(
                            display_frame,
                            f"{i+1}",
                            (int(point[0]) + 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

                    # Show instructions
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

                else:
                    # Create frame for pipeline
                    frame = self.create_frame_from_realsense(self.frame_count)
                    if frame is None:
                        continue

                    # Run pipeline step
                    self.pipeline.step(frame)

                    # Visualize results
                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects
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
