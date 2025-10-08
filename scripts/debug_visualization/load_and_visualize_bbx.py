import os
import sys

import argparse

import open3d as o3d


def main(args):

    # Determine file paths based on whether object number is provided
    if args.object_number is not None:
        # Construct paths using object number
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        pcd_path = os.path.join(
            project_root,
            f"debug/pipeline/initial_bbx/initial_pcd_{args.object_number}.ply",
        )
        obb_ls_path = os.path.join(
            project_root,
            f"debug/pipeline/initial_bbx/initial_bbx_{args.object_number}.ply",
        )
    else:
        # Use provided paths
        pcd_path = args.pcd_path
        obb_ls_path = args.obb_ls_path

    pcd = o3d.io.read_point_cloud(pcd_path)
    obb_ls = o3d.io.read_line_set(obb_ls_path)

    # Debug: Check if colors are loaded
    print(f"Point cloud has colors: {pcd.has_colors()}")
    print(f"Number of points: {len(pcd.points)}")
    if pcd.has_colors():
        print(f"Number of colors: {len(pcd.colors)}")
        print(f"First few colors: {pcd.colors[:3] if len(pcd.colors) > 0 else 'None'}")

    # Set line width if specified
    if args.line_width is not None:
        obb_ls.paint_uniform_color([1, 0, 0])  # Set color to red for visibility

    # Create visualization with line width control
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(obb_ls)

    # Set line width in the renderer
    render_option = vis.get_render_option()
    if args.line_width is not None:
        render_option.line_width = args.line_width
    else:
        render_option.line_width = 7.0

    # Enhance color visualization
    render_option.point_size = 3.0  # Make points more visible
    render_option.show_coordinate_frame = True  # Show coordinate frame for reference

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    ## Parse args
    parser = argparse.ArgumentParser()

    # Add object number argument (optional)
    parser.add_argument(
        "--object_number",
        "-o",
        type=int,
        default=None,
        help="Object number to visualize (will use initial_pcd_{i}.ply and initial_bbx_{i}.ply)",
    )

    # Add line width argument (optional)
    parser.add_argument(
        "--line_width",
        "-w",
        type=float,
        default=None,
        help="Line width for the bounding box visualization (default: 1.0)",
    )

    # get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # set default to /home/justin/code/point-to-pose/debug/pipeline/initial_bbx/initial_pcd_0.ply
    parser.add_argument(
        "--pcd_path",
        "-p",
        type=str,
        default=os.path.join(
            project_root, "debug/pipeline/initial_bbx/initial_pcd_0.ply"
        ),
        help="Path to point cloud file (required if object_number not provided)",
    )
    # set default to /home/justin/code/point-to-pose/debug/pipeline/initial_bbx/initial_bbx_0.ply
    parser.add_argument(
        "--obb_ls_path",
        "-b",
        type=str,
        default=os.path.join(
            project_root, "debug/pipeline/initial_bbx/initial_bbx_0.ply"
        ),
        help="Path to oriented bounding box line set file (required if object_number not provided)",
    )

    args = parser.parse_args()

    main(args)
