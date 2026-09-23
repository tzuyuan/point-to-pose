"""
Point2Pose over LCM with an additional SAM2-only tracked obstacle.

Identical to point2pose_lcm_tracking.py, plus: press 'o' in the prompt window to
switch to OBSTACLE mode and click the obstacle. Its sphere approximation
([x y z r], world frame) is published on lcm.obstacle_pose_channel alongside the
unchanged object pose channels.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from point2pose.io.lcm.obstacle_tracking_runner import LcmObstacleTrackingRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/lcm_obstacle_tracking.yaml",
        help="Point2Pose config with the lcm and obstacle sections.",
    )
    args = parser.parse_args()

    runner = LcmObstacleTrackingRunner(config_path=args.config)
    runner.run()


if __name__ == "__main__":
    main()
