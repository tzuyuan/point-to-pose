import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.sources.dataset.datareader import YcbineoatReader


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YCBInEOAT data (RGB, Mask, Depth)"
    )
    parser.add_argument(
        "--data_path",
        "-d",
        default="/home/justin/data/YCBInEOAT",
        help="Root directory containing video folders",
    )
    parser.add_argument(
        "--video_name",
        "-v",
        default="bleach0",
        help="Name of the video folder (e.g. bleach0)",
    )
    args = parser.parse_args()

    video_path = os.path.join(args.data_path, args.video_name)
    if not os.path.exists(video_path):
        print(f"Error: Video path not found: {video_path}")
        return

    print(f"Loading data from {video_path}...")
    try:
        reader = YcbineoatReader(video_path)
    except Exception as e:
        print(f"Failed to initialize reader: {e}")
        return

    print(f"Found {len(reader)} frames.")
    print("Controls: Press 'q' to quit, any other key to advance to next frame.")

    for i in range(len(reader)):
        # Read data
        rgb = reader.get_color(i)  # RGB
        depth = reader.get_depth(i)  # Meters
        mask = reader.get_mask(i)  # (H, W) uint8, 0 or 1 (or 255?)

        # Prepare visualizations

        # 1. RGB -> BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Mask Overlay
        # Ensure mask is boolean
        mask_bool = mask > 0

        mask_overlay = bgr.copy()
        # Add green overlay
        mask_overlay[mask_bool] = (
            mask_overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        # 3. Depth Visualization
        depth_vis = depth.copy()
        valid_mask = depth_vis > 0.001  # Filter out 0 or very small values

        if valid_mask.any():
            d_min, d_max = depth_vis[valid_mask].min(), depth_vis[valid_mask].max()
            # Normalize to 0-1
            if d_max > d_min:
                depth_norm = (depth_vis - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_vis)

            depth_norm = np.clip(depth_norm, 0, 1)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            # Black out invalid pixels in depth viz
            depth_color[~valid_mask] = 0
        else:
            depth_color = np.zeros_like(bgr)

        # Combine horizontally
        # Ensure same height/width (they should be from reader)
        combined = np.hstack([bgr, mask_overlay, depth_color])

        # Add text info
        cv2.putText(
            combined,
            f"Frame {i}/{len(reader)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "RGB",
            (10, combined.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            combined,
            "Mask Overlay",
            (10 + bgr.shape[1], combined.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            combined,
            "Depth (Jet)",
            (10 + 2 * bgr.shape[1], combined.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("YCBInEOAT Data Viewer", combined)

        key = cv2.waitKey(0)
        if key == ord("q"):
            print("Quitting...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
