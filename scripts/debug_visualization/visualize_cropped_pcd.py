#!/usr/bin/env python3
"""
Visualize cropped point clouds from debug/cropped_pcd folder based on frame ID.

Usage:
    python visualize_cropped_pcd.py --frame_id 10
    python visualize_cropped_pcd.py --frame_id 10 --object_number 0
    python visualize_cropped_pcd.py --frame_id 10 --cropped_pcd_dir /path/to/cropped_pcd
"""

import os
import argparse
import glob
from typing import List, Optional

import numpy as np
import open3d as o3d


def find_available_frames(cropped_pcd_dir: str, object_number: Optional[int] = None) -> List[int]:
    """Find all available frame numbers for a given object (or all objects if None)."""
    if object_number is not None:
        pattern = os.path.join(cropped_pcd_dir, f"obj_{object_number}_frame_*.ply")
    else:
        pattern = os.path.join(cropped_pcd_dir, f"obj_*_frame_*.ply")
    
    files = glob.glob(pattern)
    
    frame_numbers = set()
    for file in files:
        filename = os.path.basename(file)
        # Extract frame number from filename like "obj_0_frame_11.ply"
        try:
            # Split by "_frame_" and take the part after it, then remove .ply
            frame_part = filename.split("_frame_")[1].split(".")[0]
            frame_numbers.add(int(frame_part))
        except (IndexError, ValueError):
            continue
    
    return sorted(frame_numbers)


def find_available_objects(cropped_pcd_dir: str) -> List[int]:
    """Find all available object numbers."""
    pattern = os.path.join(cropped_pcd_dir, f"obj_*_frame_*.ply")
    files = glob.glob(pattern)
    
    object_numbers = set()
    for file in files:
        filename = os.path.basename(file)
        # Extract object number from filename like "obj_0_frame_11.ply"
        try:
            # Split by "obj_" and take the number before "_frame_"
            obj_part = filename.split("obj_")[1].split("_frame_")[0]
            object_numbers.add(int(obj_part))
        except (IndexError, ValueError):
            continue
    
    return sorted(object_numbers)


def load_cropped_pcd(
    cropped_pcd_dir: str, object_number: int, frame_number: int
) -> Optional[o3d.geometry.PointCloud]:
    """Load a cropped point cloud from the cropped_pcd directory."""
    filename = f"obj_{object_number}_frame_{frame_number}.ply"
    filepath = os.path.join(cropped_pcd_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            print(f"Warning: Point cloud {filename} is empty")
            return None
        return pcd
    except (IOError, ValueError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def get_pcd_info(pcd: o3d.geometry.PointCloud) -> dict:
    """Extract information about the point cloud."""
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
            "extent": np.max(points, axis=0) - np.min(points, axis=0),
        }
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        info["color_info"] = {
            "min": np.min(colors, axis=0),
            "max": np.max(colors, axis=0),
            "mean": np.mean(colors, axis=0),
        }
    
    return info


def visualize_frame(
    cropped_pcd_dir: str,
    frame_number: int,
    object_numbers: Optional[List[int]] = None,
    show_info: bool = True,
):
    """Visualize cropped point clouds for a specific frame ID.
    
    Args:
        cropped_pcd_dir: Directory containing cropped PCD files
        frame_number: Frame ID to visualize
        object_numbers: List of object numbers to visualize. If None, visualize all available objects.
        show_info: Whether to print point cloud information
    """
    if object_numbers is None:
        object_numbers = find_available_objects(cropped_pcd_dir)
    
    if not object_numbers:
        print(f"No objects found in {cropped_pcd_dir}")
        return
    
    # Load point clouds for all specified objects at this frame
    pcds = []
    pcd_labels = []
    
    for obj_id in object_numbers:
        pcd = load_cropped_pcd(cropped_pcd_dir, obj_id, frame_number)
        if pcd is not None:
            pcds.append(pcd)
            pcd_labels.append(f"Object {obj_id}")
            
            if show_info:
                info = get_pcd_info(pcd)
                print(f"\nObject {obj_id} - Frame {frame_number}:")
                print(f"  - Number of points: {info['num_points']}")
                print(f"  - Has colors: {info['has_colors']}")
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
                    print(
                        f"    - Extent: [{info['bounds']['extent'][0]:.3f}, {info['bounds']['extent'][1]:.3f}, {info['bounds']['extent'][2]:.3f}]"
                    )
                if info["color_info"] is not None:
                    print(f"  - Colors (RGB):")
                    print(
                        f"    - Min: [{info['color_info']['min'][0]:.3f}, {info['color_info']['min'][1]:.3f}, {info['color_info']['min'][2]:.3f}]"
                    )
                    print(
                        f"    - Max: [{info['color_info']['max'][0]:.3f}, {info['color_info']['max'][1]:.3f}, {info['color_info']['max'][2]:.3f}]"
                    )
                    print(
                        f"    - Mean: [{info['color_info']['mean'][0]:.3f}, {info['color_info']['mean'][1]:.3f}, {info['color_info']['mean'][2]:.3f}]"
                    )
    
    if not pcds:
        print(f"No point clouds found for frame {frame_number} in objects {object_numbers}")
        print(f"\nAvailable frames for these objects:")
        available_frames = find_available_frames(cropped_pcd_dir, object_numbers[0] if len(object_numbers) > 0 else None)
        if available_frames:
            print(f"  {available_frames[:20]}{'...' if len(available_frames) > 20 else ''}")
            print(f"  Total: {len(available_frames)} frames")
        return
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    window_name = f"Cropped PCDs - Frame {frame_number}"
    if len(object_numbers) == 1:
        window_name += f" (Object {object_numbers[0]})"
    vis.create_window(window_name=window_name)
    
    # Add all point clouds to visualization
    for pcd in pcds:
        vis.add_geometry(pcd)
    
    # Set up render options
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.show_coordinate_frame = True
    render_option.background_color = [0.1, 0.1, 0.1]  # Dark background
    
    # Print visualization instructions
    print(f"\n{'='*60}")
    print(f"Visualizing frame {frame_number}")
    print(f"Loaded {len(pcds)} object(s): {object_numbers}")
    print(f"{'='*60}")
    print("\nVisualization controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Shift + Mouse drag: Pan view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - 'Q' or close window: Quit")
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def main(args):
    """Main visualization function."""
    # Determine cropped PCD directory
    if args.cropped_pcd_dir:
        cropped_pcd_dir = args.cropped_pcd_dir
    else:
        # Use default debug/cropped_pcd folder
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cropped_pcd_dir = os.path.join(project_root, "debug", "cropped_pcd")
    
    if not os.path.exists(cropped_pcd_dir):
        print(f"Error: Cropped PCD directory {cropped_pcd_dir} does not exist")
        print("\nPlease specify the directory with --cropped_pcd_dir or ensure debug/cropped_pcd exists")
        return
    
    # Determine object numbers to visualize
    if args.object_number is not None:
        object_numbers = [args.object_number]
    else:
        # Find all available objects
        object_numbers = find_available_objects(cropped_pcd_dir)
        if not object_numbers:
            print(f"No cropped PCD files found in {cropped_pcd_dir}")
            return
        print(f"Found objects: {object_numbers}")
        if not args.all_objects:
            # Default to first object if not specified
            object_numbers = [object_numbers[0]]
            print(f"Visualizing object {object_numbers[0]} (use --all_objects to visualize all)")
    
    # Check if frame exists
    available_frames = find_available_frames(cropped_pcd_dir, object_numbers[0])
    if args.frame_id not in available_frames:
        print(f"Frame {args.frame_id} not found in {cropped_pcd_dir}")
        print(f"\nAvailable frames: {available_frames[:50]}{'...' if len(available_frames) > 50 else ''}")
        print(f"Total: {len(available_frames)} frames")
        if available_frames:
            print(f"\nClosest frame: {min(available_frames, key=lambda x: abs(x - args.frame_id))}")
        return
    
    # Visualize the specified frame
    visualize_frame(
        cropped_pcd_dir,
        args.frame_id,
        object_numbers if args.all_objects else None,
        show_info=not args.no_info,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize cropped point clouds based on frame ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize frame 10 (defaults to first available object)
  python visualize_cropped_pcd.py --frame_id 10
  
  # Visualize specific object and frame
  python visualize_cropped_pcd.py --frame_id 10 --object_number 0
  
  # Visualize all objects for a frame
  python visualize_cropped_pcd.py --frame_id 10 --all_objects
  
  # Specify custom directory
  python visualize_cropped_pcd.py --frame_id 10 --cropped_pcd_dir /path/to/cropped_pcd
        """
    )
    
    parser.add_argument(
        "--frame_id",
        "-f",
        type=int,
        required=True,
        help="Frame ID to visualize (required)",
    )
    
    parser.add_argument(
        "--object_number",
        "-o",
        type=int,
        default=None,
        help="Object number to visualize (default: first available object)",
    )
    
    parser.add_argument(
        "--all_objects",
        "-a",
        action="store_true",
        help="Visualize all objects for the specified frame",
    )
    
    parser.add_argument(
        "--cropped_pcd_dir",
        "-d",
        type=str,
        default=None,
        help="Path to cropped PCD directory (default: project_root/debug/cropped_pcd)",
    )
    
    parser.add_argument(
        "--no_info",
        action="store_true",
        help="Don't print point cloud information",
    )
    
    args = parser.parse_args()
    
    main(args)

