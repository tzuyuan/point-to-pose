#!/usr/bin/env python3
"""
Interactive key points visualization for debug/key_points folder.
Navigate through key point clouds using 'n' for next frame, 'p' for previous frame, 'q' to quit.
Key points are colored based on the frame they were added.
"""

import os
import argparse
import glob
from typing import List, Optional

import numpy as np
import open3d as o3d


def find_available_frames(keypoints_folder: str, object_number: int) -> List[int]:
    """Find all available frame numbers for a given object's key points."""
    pattern = os.path.join(
        keypoints_folder, f"obj_{object_number}_keypoints_frame_*.ply"
    )
    files = glob.glob(pattern)

    frame_numbers = []
    for file in files:
        filename = os.path.basename(file)
        # Extract frame number from filename like "obj_0_keypoints_frame_128.ply"
        try:
            frame_part = filename.split("_frame_")[1].split(".")[0]
            frame_numbers.append(int(frame_part))
        except (IndexError, ValueError):
            continue

    return sorted(frame_numbers)


def load_keypoints_cloud(
    keypoints_folder: str, object_number: int, frame_number: int
) -> Optional[o3d.geometry.PointCloud]:
    """Load a key points cloud from the keypoints folder."""
    filename = f"obj_{object_number}_keypoints_frame_{frame_number}.ply"
    filepath = os.path.join(keypoints_folder, filename)

    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist")
        return None

    try:
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            print(f"Warning: Key points cloud {filename} is empty")
            return None
        return pcd
    except (IOError, ValueError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def get_keypoints_info(pcd: o3d.geometry.PointCloud) -> dict:
    """Extract information about the key points cloud."""
    info = {
        "num_points": len(pcd.points),
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
        "bounds": None,
        "color_info": None,
    }

    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        info["bounds"] = {
            "min": np.min(points, axis=0),
            "max": np.max(points, axis=0),
            "center": np.mean(points, axis=0),
        }

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        unique_colors = np.unique(colors, axis=0)
        info["color_info"] = {
            "num_unique_colors": len(unique_colors),
            "unique_colors": unique_colors,
        }

    return info


def interactive_visualization(
    keypoints_folder: str,
    object_number: int,
    available_frames: List[int],
    start_frame: int,
):
    """Interactive visualization with keyboard navigation."""
    current_frame_idx = available_frames.index(start_frame)

    print("\nNavigation controls:")
    print("  'n' or 'N': Next frame")
    print("  'p' or 'P': Previous frame")
    print("  'q' or 'Q': Quit")
    print("  'i' or 'I': Show key points information")
    print("  'c' or 'C': Show color information")

    while True:
        frame_number = available_frames[current_frame_idx]
        print(
            f"\nCurrent frame: {frame_number} (frame {current_frame_idx + 1} of {len(available_frames)})"
        )

        # Load key points cloud
        pcd = load_keypoints_cloud(keypoints_folder, object_number, frame_number)
        if pcd is None:
            print(f"Failed to load frame {frame_number}")
            # Move to next frame if current one fails
            if current_frame_idx < len(available_frames) - 1:
                current_frame_idx += 1
                continue
            else:
                print("No more frames available")
                return

        info = get_keypoints_info(pcd)
        print(
            f"Displaying: obj_{object_number}_keypoints_frame_{frame_number}.ply ({info['num_points']} points)"
        )

        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Object {object_number} Key Points - Frame {frame_number}"
        )
        vis.add_geometry(pcd)

        # Set up render options
        render_option = vis.get_render_option()
        render_option.point_size = 8.0  # Slightly smaller than register points
        render_option.show_coordinate_frame = True
        render_option.background_color = [1.0, 1.0, 1.0]  # White background

        # Run visualization (this will block until window is closed)
        vis.run()
        vis.destroy_window()

        # Get user input for next action
        while True:
            try:
                user_input = (
                    input(
                        "\nEnter command (n=next, p=previous, i=info, c=colors, q=quit): "
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nExiting visualization...")
                return

            if user_input in ["n", "next"]:
                if current_frame_idx < len(available_frames) - 1:
                    current_frame_idx += 1
                    break
                else:
                    print("Already at the last frame!")
                    continue
            elif user_input in ["p", "prev", "previous"]:
                if current_frame_idx > 0:
                    current_frame_idx -= 1
                    break
                else:
                    print("Already at the first frame!")
                    continue
            elif user_input in ["i", "info"]:
                print(f"\nKey Points Information for Frame {frame_number}:")
                print(f"  - Number of points: {info['num_points']}")
                print(f"  - Has colors: {info['has_colors']}")
                print(f"  - Has normals: {info['has_normals']}")
                if info["bounds"] is not None:
                    print(f"  - Bounds:")
                    print(
                        f"    - Min: [{info['bounds']['min'][0]:.3f}, {info['bounds']['min'][1]:.3f}, {info['bounds']['min'][2]:.3f}]"
                    )
                    print(
                        f"    - Max: [{info['bounds']['max'][0]:.3f}, {info['bounds']['max'][1]:.3f}, {info['bounds']['max'][2]:.3f}]"
                    )
                    print(
                        f"    - Center: [{info['bounds']['center'][0]:.3f}, {info['bounds']['center'][1]:.3f}, {info['bounds']['center'][2]:.3f}]"
                    )
                continue
            elif user_input in ["c", "colors"]:
                if info["color_info"] is not None:
                    print(f"\nColor Information for Frame {frame_number}:")
                    print(
                        f"  - Number of unique colors: {info['color_info']['num_unique_colors']}"
                    )
                    print("  - Unique colors (RGB):")
                    for i, color in enumerate(info["color_info"]["unique_colors"]):
                        print(
                            f"    {i+1}: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]"
                        )
                else:
                    print("No color information available")
                continue
            elif user_input in ["q", "quit"]:
                print("Exiting visualization...")
                return
            else:
                print("Invalid command. Please enter 'n', 'p', 'i', 'c', or 'q'.")
                continue


def visualize_all_keypoints(keypoints_folder: str, object_number: int):
    """Visualize all key points from all frames in a single view."""
    pattern = os.path.join(
        keypoints_folder, f"obj_{object_number}_keypoints_frame_*.ply"
    )
    files = glob.glob(pattern)

    if not files:
        print(f"No key points files found for object {object_number}")
        return

    print(f"Loading all key points from {len(files)} frames...")

    all_points = []
    all_colors = []
    frame_info = []

    for file in sorted(files):
        try:
            pcd = o3d.io.read_point_cloud(file)
            if len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                all_points.append(points)

                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    all_colors.append(colors)
                else:
                    # Default color if no colors
                    colors = np.tile([0.5, 0.5, 0.5], (len(points), 1))
                    all_colors.append(colors)

                # Extract frame number for info
                filename = os.path.basename(file)
                frame_num = int(filename.split("_frame_")[1].split(".")[0])
                frame_info.append((frame_num, len(points)))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not all_points:
        print("No valid key points found")
        return

    # Combine all points and colors
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    print(
        f"Combined key points: {len(combined_points)} points from {len(frame_info)} frames"
    )
    print("Frame breakdown:")
    for frame_num, num_points in frame_info:
        print(f"  Frame {frame_num}: {num_points} points")

    # Visualize combined point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Object {object_number} - All Key Points (Combined)")
    vis.add_geometry(combined_pcd)

    # Set up render options
    render_option = vis.get_render_option()
    render_option.point_size = 6.0
    render_option.show_coordinate_frame = True
    render_option.background_color = [1.0, 1.0, 1.0]  # White background

    print("\nPress any key in the visualization window to close...")
    vis.run()
    vis.destroy_window()


def main(args):
    """Main visualization function."""
    # Construct keypoints folder path
    if args.keypoints_folder:
        keypoints_folder = args.keypoints_folder
    else:
        # Use default debug/key_points folder
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        keypoints_folder = os.path.join(project_root, "debug", "key_points")

    if not os.path.exists(keypoints_folder):
        print(f"Error: Key points folder {keypoints_folder} does not exist")
        return

    # Find available frames for the specified object
    available_frames = find_available_frames(keypoints_folder, args.object_number)

    if not available_frames:
        print(
            f"No key points frames found for object {args.object_number} in {keypoints_folder}"
        )
        print("Available files:")
        all_files = glob.glob(os.path.join(keypoints_folder, "*.ply"))
        for file in all_files:
            print(f"  {os.path.basename(file)}")
        return

    print(
        f"Found {len(available_frames)} key points frames for object {args.object_number}: {available_frames}"
    )

    # Handle combined visualization
    if args.combined:
        visualize_all_keypoints(keypoints_folder, args.object_number)
        return

    # Determine starting frame
    if args.frame_number is not None:
        if args.frame_number not in available_frames:
            print(
                f"Frame {args.frame_number} not found. Available frames: {available_frames}"
            )
            print(f"Using first available frame: {available_frames[0]}")
            start_frame = available_frames[0]
        else:
            start_frame = args.frame_number
    else:
        start_frame = available_frames[0]

    # Load initial key points cloud
    print(f"Loading initial frame {start_frame}...")
    pcd = load_keypoints_cloud(keypoints_folder, args.object_number, start_frame)

    if pcd is None:
        print(
            f"Failed to load initial key points cloud for object {args.object_number}, frame {start_frame}"
        )
        return

    print(
        f"Successfully loaded: obj_{args.object_number}_keypoints_frame_{start_frame}.ply"
    )
    info = get_keypoints_info(pcd)
    print("Key points info:")
    print(f"  - Number of points: {info['num_points']}")
    print(f"  - Has colors: {info['has_colors']}")
    print(f"  - Has normals: {info['has_normals']}")

    # Start interactive visualization
    interactive_visualization(
        keypoints_folder, args.object_number, available_frames, start_frame
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive key points visualization for debug/key_points folder"
    )

    parser.add_argument(
        "--object_number",
        "-o",
        type=int,
        required=True,
        help="Object number to visualize (e.g., 0, 1, 2...)",
    )

    parser.add_argument(
        "--frame_number",
        "-f",
        type=int,
        default=None,
        help="Starting frame number (optional, defaults to first available frame)",
    )

    parser.add_argument(
        "--keypoints_folder",
        "-k",
        type=str,
        default=None,
        help="Path to keypoints folder (optional, defaults to project_root/debug/key_points)",
    )

    parser.add_argument(
        "--combined",
        "-c",
        action="store_true",
        help="Visualize all key points from all frames in a single view",
    )

    args = parser.parse_args()

    main(args)
