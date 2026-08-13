"""RealSense demo with live 3D visualization (plug-in; the original demo is
untouched — this subclasses it and only adds hooks).

Default is the Rerun viewer (the kv_tracker-style native app; entity
conventions match the ``model_based_tracking`` branch's reference viewer):

  * "Object frame · map" view — keypoint map (track-id colors, dimmed when
    untracked), SDF mesh, camera trajectory, the current camera frustum
    textured with the live RGB frame, keyframe frustums with thumbnails.
  * "Camera frame · trails" view — the sensor frustum, tracked points, and
    fading per-point traces on the object.
  * RGB / Events tabs + Health plots (residual, inliers, tracked pts, FPS).
  * Built-in timeline scrubber replays the whole session; set
    ``visualization_3d.rerun.save_rrd`` to also record a ``.rrd`` file.

The cv2 window is only used for prompt clicks and the 2D overlay. Other
``visualization_3d.ui_mode`` values: ``web`` (viser browser viewer),
``combined`` (single cv2 dashboard), ``windows`` (two Open3D windows).

Run from the repo root:

    python examples/realsense_tracking/realsense_tracking_3d.py \
        --config configs/pipeline/pipeline_test2.yaml \
        --viz-config configs/visualization/pose_3d_demo.yaml

Both arguments are optional; without --viz-config the visualizer uses the
``visualization_3d`` section of the pipeline config if present, else defaults.
Keyboard controls are unchanged (click points, 'n' next object, 's' start,
'r' reset, 'q' quit).
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.append(str(_HERE.parents[2]))  # repo root (same as the base demo)
sys.path.append(str(_HERE.parent))  # to import the base demo module directly

import cv2
from omegaconf import OmegaConf

from realsense_tracking import RealSensePipelineTracker
from point2pose.visualization import Pose3DVisualizer

# Must match the window name created by the base demo: during tracking we
# reuse that window to display the combined dashboard.
_WINDOW_NAME = "RealSense Pipeline Tracker"


class RealSenseTrackerWith3DViz(RealSensePipelineTracker):
    """The original demo plus the 3D UI — hooks only, no base edits."""

    def __init__(
        self,
        config_path="configs/pipeline/pipeline_test2.yaml",
        viz_config_path=None,
    ):
        super().__init__(config_path)
        viz_cfg = self.cfg.get("visualization_3d", None)
        if viz_config_path is not None:
            viz_cfg = OmegaConf.load(viz_config_path).get("visualization_3d", viz_cfg)
        self.viz3d = Pose3DVisualizer(viz_cfg)
        self.viz3d.set_display_window(_WINDOW_NAME)
        # Rebind the window's mouse callback: annotation clicks before
        # tracking (base behavior), dashboard navigation/buttons after.
        cv2.setMouseCallback(_WINDOW_NAME, self._routed_mouse_callback)

    def _routed_mouse_callback(self, event, x, y, flags, param):
        if not self.tracking_started:
            self.mouse_callback(event, x, y, flags, param)
        else:
            self.viz3d.handle_mouse(event, x, y, flags)

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        # Called by the base loop once per tracked frame. In combined mode we
        # return the dashboard canvas, which the base loop then displays in
        # its own window; in windows mode the 2D overlay passes through.
        display_frame = super().visualize_tracking_results(frame, objects, frame_id)
        try:
            canvas = self.viz3d.update(self.pipeline, frame, overlay_2d=display_frame)
            if canvas is not None:
                return canvas
        except Exception as exc:  # never let a viz hiccup kill the demo
            print(f"[viz3d] update failed: {exc}")
        return display_frame

    def reset_points(self):
        super().reset_points()
        self.viz3d.reset()

    def run_tracking(self):
        try:
            super().run_tracking()
        finally:
            self.viz3d.close()


def main():
    parser = argparse.ArgumentParser(
        description="RealSense point2pose demo with 3D visualization"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline/pipeline_test2.yaml",
        help="pipeline config (same as the base demo)",
    )
    parser.add_argument(
        "--viz-config",
        default=None,
        help="YAML with a visualization_3d section (see configs/visualization/)",
    )
    args = parser.parse_args()

    tracker = RealSenseTrackerWith3DViz(args.config, args.viz_config)
    tracker.run_tracking()


if __name__ == "__main__":
    main()
