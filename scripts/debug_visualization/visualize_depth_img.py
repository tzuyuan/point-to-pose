import sys
import os

# Add project root to path for imports (same pattern as plot_registration_stats.py)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import cv2
from point2pose.io.sources.dataset.datareader import YCBInIsaacReader


def main():

    video_path = "/home/justin/data/Test/cracker_box"
    reader = YCBInIsaacReader(video_path)

    depth = reader.get_depth(0)

    print(depth)
    cv2.imshow("depth", depth)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
