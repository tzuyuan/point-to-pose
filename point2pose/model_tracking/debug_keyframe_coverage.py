"""
Debug viewer for the pose-jump filter used by ``reconstruct.py``'s
``filter_frames_by_pose_jump`` (see ``main``'s ``--pose_jump_max_trans`` /
``--pose_jump_max_rot_deg``).

Loads every exported frame from ReconstructionExporter's all_frames_rgb/depth/
mask/poses (reconstruct.py's load_all_frames), and steps through temporally-adjacent
frame pairs (i, i+1) side by side with the relative pose delta (translation in
meters, rotation in degrees) between them. A pair straddling a jump beyond the
configured thresholds is flagged REJECT -- both frames get dropped from training.

Controls:
    n / p       next / previous pair
    q / esc     quit

Usage (in the ``point2pose`` conda env):
    python -m point2pose.model_tracking.debug_keyframe_coverage \
        --results_path debug/export_test \
        [--pose_jump_max_trans 0.05] [--pose_jump_max_rot_deg 15]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from point2pose.model_tracking.reconstruct import (
    filter_frames_by_pose_jump,
    load_all_frames,
    relative_pose_delta,
)

WIN = "Pose jump filter debugger"


def draw_pair(img_i, img_j, dt, dtheta_deg, frame_id_i, frame_id_j, keep_i, keep_j,
              max_trans, max_rot_deg):
    disp = np.concatenate([img_i[..., ::-1], img_j[..., ::-1]], axis=1).copy()  # RGB->BGR, side by side

    is_jump = dt > max_trans or dtheta_deg > max_rot_deg
    status = "JUMP (both dropped)" if is_jump else "OK"
    color = (0, 0, 255) if is_jump else (0, 255, 0)

    cv2.putText(disp, f"frame {frame_id_i}  |  frame {frame_id_j}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(disp, f"dt={dt:.4f}m (max {max_trans})  dtheta={dtheta_deg:.2f}deg (max {max_rot_deg})  [{status}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    left_status = "KEEP" if keep_i else "REJECT"
    right_status = "KEEP" if keep_j else "REJECT"
    left_color = (0, 255, 0) if keep_i else (0, 0, 255)
    right_color = (0, 255, 0) if keep_j else (0, 0, 255)
    cv2.putText(disp, left_status, (10, disp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
    cv2.putText(disp, right_status, (img_i.shape[1] + 10, disp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
    cv2.putText(disp, "n/p: next/prev pair   q: quit",
                (10, disp.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True, type=str)
    ap.add_argument("--pose_jump_max_trans", default=0.05, type=float)
    ap.add_argument("--pose_jump_max_rot_deg", default=15.0, type=float)
    args = ap.parse_args()

    results_path = Path(args.results_path)
    _points, _colors, images, _masks, poses, _K, frame_idx = load_all_frames(results_path)
    n = len(images)
    if n < 2:
        raise RuntimeError(f"Need at least 2 frames, got {n}")

    keep, deltas = filter_frames_by_pose_jump(
        results_path, poses, frame_idx,
        max_trans=args.pose_jump_max_trans, max_rot_deg=args.pose_jump_max_rot_deg,
    )
    print(f"{keep.sum()}/{n} frames kept at "
          f"max_trans={args.pose_jump_max_trans}, max_rot_deg={args.pose_jump_max_rot_deg}")

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    i = 0

    while True:
        dt, dtheta_deg = relative_pose_delta(poses[i], poses[i + 1])
        disp = draw_pair(
            images[i], images[i + 1], dt, dtheta_deg,
            int(frame_idx[i]), int(frame_idx[i + 1]), bool(keep[i]), bool(keep[i + 1]),
            args.pose_jump_max_trans, args.pose_jump_max_rot_deg,
        )
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("n"):
            i = (i + 1) % (n - 1)
        elif key == ord("p"):
            i = (i - 1) % (n - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
