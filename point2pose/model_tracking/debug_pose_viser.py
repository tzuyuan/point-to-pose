"""
Viser 3D debug viewer for the pose-jump filter (reconstruct.py's
filter_frames_by_pose_jump): plots every exported frame's camera frustum in 3D,
colored by keep/reject, with a slider to step through frames and see the
corresponding RGB + a highlighted (yellow) frustum for the selected index.

Colors:
    green  = kept (passed the pose-jump filter)
    red    = rejected (dropped -- flanks a pose jump)
    yellow = currently selected frame (slider/arrow keys)

Usage (in the ``point2pose`` conda env):
    python -m point2pose.model_tracking.debug_pose_viser \
        --results_path debug/kv_tracker_adapted \
        [--pose_jump_max_trans 0.05] [--pose_jump_max_rot_deg 15] [--stride 1]
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import viser

from point2pose.model_tracking.reconstruct import (
    filter_frames_by_pose_jump,
    load_all_frames,
    rotmat_to_wxyz,
)

GREEN = (60, 200, 60)
RED = (220, 60, 60)
HIGHLIGHT = (255, 0, 255)  # magenta -- doesn't collide with green/red at a glance
BASE_SCALE = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True, type=str)
    ap.add_argument("--pose_jump_max_trans", default=0.05, type=float)
    ap.add_argument("--pose_jump_max_rot_deg", default=15.0, type=float)
    ap.add_argument("--stride", default=1, type=int,
                     help="Only plot every Nth frame's frustum (all frames are still "
                     "selectable via the slider; use this if 769 frustums lags viser).")
    ap.add_argument("--port", default=8080, type=int)
    args = ap.parse_args()

    results_path = Path(args.results_path)
    _points, _colors, images, _masks, poses, K, frame_idx = load_all_frames(results_path)
    n = len(images)
    height, width = images.shape[1:3]

    keep, deltas = filter_frames_by_pose_jump(
        results_path, poses, frame_idx,
        max_trans=args.pose_jump_max_trans, max_rot_deg=args.pose_jump_max_rot_deg,
    )
    print(f"{keep.sum()}/{n} frames kept "
          f"(max_trans={args.pose_jump_max_trans}, max_rot_deg={args.pose_jump_max_rot_deg})")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    fx = K[0, 0]
    fov = 2 * math.atan(width / (2 * fx))
    aspect = width / height

    plot_idx = np.arange(0, n, max(1, args.stride))
    handles = {}
    for i in plot_idx:
        c2w = poses[i]
        color = GREEN if keep[i] else RED
        h = server.scene.add_camera_frustum(
            f"cameras/frustum_{i:05d}",
            fov=fov, aspect=aspect, scale=BASE_SCALE, color=color,
            wxyz=rotmat_to_wxyz(c2w[:3, :3]), position=c2w[:3, 3],
            image=images[i],
        )
        handles[int(i)] = h

    # A standalone marker (not a per-frame frustum) at the selected pose, so the
    # highlight is unmistakable regardless of whether per-object property mutation
    # (e.g. .scale) actually round-trips to the client -- only .position/.wxyz/
    # .color, which we already rely on elsewhere, are used here.
    marker = server.scene.add_icosphere(
        "cameras/selected_marker", radius=BASE_SCALE * 2.5, color=HIGHLIGHT,
        position=poses[int(plot_idx[0])][:3, 3],
    )

    gui_slider = server.gui.add_slider(
        "Frame index", min=0, max=n - 1, step=1, initial_value=int(plot_idx[0])
    )
    gui_info = server.gui.add_text("Status", initial_value="")
    gui_rgb = server.gui.add_image(images[int(plot_idx[0])], label="RGB (selected frame)")

    selected_idx = [int(plot_idx[0])]

    def restore_style(i):
        if i in handles:
            handles[i].color = GREEN if keep[i] else RED

    def highlight(i):
        restore_style(selected_idx[0])
        selected_idx[0] = i
        if i in handles:
            handles[i].color = HIGHLIGHT
        marker.position = poses[i][:3, 3]
        status = "KEEP" if keep[i] else "REJECT"
        dt_prev, dr_prev = deltas[i - 1] if i > 0 else (float("nan"), float("nan"))
        dt_next, dr_next = deltas[i] if i < n - 1 else (float("nan"), float("nan"))
        gui_info.value = (
            f"frame_idx={int(frame_idx[i])} [{status}]\n"
            f"delta to prev: dt={dt_prev:.4f}m dtheta={dr_prev:.2f}deg\n"
            f"delta to next: dt={dt_next:.4f}m dtheta={dr_next:.2f}deg"
        )
        gui_rgb.image = images[i]

    @gui_slider.on_update
    def _(_):
        highlight(gui_slider.value)

    highlight(int(plot_idx[0]))

    print(f"Viewer running at http://localhost:{args.port} -- Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
