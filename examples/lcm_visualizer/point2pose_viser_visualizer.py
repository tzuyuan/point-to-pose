import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.lcm import ViserLcmVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/pipeline_test.yaml",
        help="Path to the Point2Pose config with lcm and viser sections.",
    )
    parser.add_argument(
        "--filter_disconnected",
        action="store_true",
        help="Keep only the largest connected mesh component when loading meshes.",
    )
    args = parser.parse_args()

    viewer = ViserLcmVisualizer(
        config_path=args.config,
        filter_disconnected=args.filter_disconnected,
    )
    viewer.run()


if __name__ == "__main__":
    main()
