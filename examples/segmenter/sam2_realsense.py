import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from types import SimpleNamespace

from point2pose.modules.segmenter.sam2_real_time_segmenter import Sam2RealTimeSegmenter

import time

# use bfloat16 for the entire script (if CUDA available)
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class SAM2RealSenseRealTimeSegmenter:
    """
    RealSense tracker for SAM2 segmentation.
    The tracker segments objects in real-time using a RealSense camera.
    """

    def __init__(
        self,
        sam2_checkpoint="checkpoints/sam2.1/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        rs_serial=242422304947,
    ):
        """
        Args:
            sam2_checkpoint (str): Path to the SAM2 checkpoint.
                            [Note] This path is relative to the current repository directory.
            model_cfg (str): Path to the SAM2 model configuration.
                             [Note] This path is relative to the segment-anything-2-real-time/ directory.
            rs_serial (int): Serial number of the RealSense camera.
        """
        # Build a minimal config object for the segmenter
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The segmenter expects config.get(...) and config.device
        self.segmenter = Sam2RealTimeSegmenter(
            SimpleNamespace(
                device=device,
                get=lambda key, default=None: {
                    "model_cfg": model_cfg,
                    "checkpoint": sam2_checkpoint,
                }.get(key, default),
            )
        )

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Mouse click storage
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0

        # Create window and set mouse callback
        cv2.namedWindow("RealSense SAM2 Tracker", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense SAM2 Tracker", self.mouse_callback)

        print("Instructions:")
        print("- Left click: Add positive point")
        print("- Right click: Add negative point")
        print("- Press 's': Start tracking")
        print("- Press 'q': Quit")
        print("- Press 'r': Reset points")

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
        """Reset collected points"""
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        print("Points reset. Click to add new points.")

    def start_tracking(self):
        """Initialize SAM2 tracking with collected points using the segmenter"""
        if len(self.click_points) == 0:
            print("No points collected! Please click on objects first.")
            return False

        # Convert to numpy arrays
        points = np.array(self.click_points, dtype=np.float32)
        labels = np.array(self.click_labels, dtype=np.int32)

        # Get current frame for initialization
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Failed to get color frame for initialization")
            return False

        frame = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add points into the segmenter and initialize
        self.segmenter.add_input_points(points.tolist(), labels.tolist())
        self.segmenter.initialize(frame_rgb)

        self.tracking_started = self.segmenter.tracking_started
        if self.tracking_started:
            print(f"Started tracking with {len(points)} points")
        else:
            print("No objects added. Add positive points before starting.")
        return self.tracking_started

    def run(self):
        """Main tracking loop"""
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame_rgb.shape[:2]

                # Display frame
                display_frame = frame.copy()

                if not self.tracking_started:
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
                    # Perform tracking via segmenter
                    out_obj_ids, out_mask_logits = self.segmenter.segment(frame_rgb)

                    self.frame_count += 1

                    # Create mask visualization
                    all_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    all_mask[..., 1] = 255  # Green base

                    for i, _ in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0).permute(
                            1, 2, 0
                        ).cpu().numpy().astype(np.uint8) * 255
                        # Color each object differently
                        hue = (i + 3) / (len(out_obj_ids) + 3) * 255
                        all_mask[out_mask[..., 0] == 255, 0] = hue
                        all_mask[out_mask[..., 0] == 255, 2] = 255

                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2BGR)
                    display_frame = cv2.addWeighted(display_frame, 1, all_mask, 0.5, 0)

                    # Show tracking info
                    cv2.putText(
                        display_frame,
                        f"Tracking frame: {self.frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Objects: {len(out_obj_ids)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Display the frame
                cv2.imshow("RealSense SAM2 Tracker", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and not self.tracking_started:
                    if self.start_tracking():
                        print("Tracking started!")
                    else:
                        print("Failed to start tracking!")
                elif key == ord("r"):
                    self.reset_points()

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if torch.cuda.is_available():
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


def main():
    """Main function to run the SAM2 RealSense segmentation tracker"""
    try:
        segmenter = SAM2RealSenseRealTimeSegmenter()
        segmenter.run()
    except (RuntimeError, OSError, ImportError) as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. RealSense camera connected")
        print("2. SAM2 checkpoint file at the specified path")
        print("3. Required dependencies installed (pyrealsense2, sam2, etc.)")


if __name__ == "__main__":
    main()
