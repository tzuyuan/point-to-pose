"""
Live SLAM tracking with frame/pose export for offline 2DGS/3DGS reconstruction.

Same flow as examples/realsense_tracking/realsense_tracking.py (click points, 's' to
start, live tracking with the ModularPipeline/SDF map -- no pre-existing mesh needed),
except every tracked frame is also written to --export-dir via ReconstructionExporter
for train_gaussians.py to consume afterward.

Run:
    python examples/realsense_tracking/reconstruction/capture.py \
        --config configs/pipeline/reconstruction_capture.yaml --export-dir debug/capture_test
"""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_rt_path = REPO / "examples/realsense_tracking/realsense_tracking.py"
_spec = importlib.util.spec_from_file_location("realsense_tracking", _rt_path)
_rt = importlib.util.module_from_spec(_spec)
sys.modules["realsense_tracking"] = _rt
_spec.loader.exec_module(_rt)

from point2pose.pipeline.components.reconstruction_exporter import ReconstructionExporter

DEFAULT_CONFIG = REPO / "configs/pipeline/reconstruction_capture.yaml"


class RealSenseCaptureTracker(_rt.RealSensePipelineTracker):
    """Exports each frame's pose right after ModularPipeline.step() updates it.
    visualize_tracking_results(frame, objects, frame_id) is called exactly once per
    tracked frame, right after step() in run_tracking()'s main loop -- the natural
    post-step hook without duplicating that loop here."""

    def __init__(self, config_path, export_dir):
        super().__init__(config_path)
        self.exporter = ReconstructionExporter(export_dir)

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        if objects:
            self.exporter.export_frame(frame, objects[0], frame_id)
        return super().visualize_tracking_results(frame, objects, frame_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--export-dir", required=True,
                    help="directory to write all_frames_{rgb,depth,mask}/ + poses to")
    args = ap.parse_args()
    tracker = RealSenseCaptureTracker(args.config, args.export_dir)
    tracker.run_tracking()


if __name__ == "__main__":
    main()
