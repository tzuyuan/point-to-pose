import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.lcm import LcmTrackingRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/pipeline_test.yaml",
        help="Path to the Point2Pose config with the new lcm section.",
    )
    args = parser.parse_args()

    runner = LcmTrackingRunner(config_path=args.config)
    runner.run()


if __name__ == "__main__":
    main()
