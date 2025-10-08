# RealSense Pipeline Tracker

This example demonstrates how to use the point2pose pipeline with a RealSense camera for real-time object tracking and pose estimation.

## Features

- **Real-time object tracking** using SAM2 segmentation
- **Pose estimation** using the complete point2pose pipeline
- **Interactive point selection** via mouse clicks
- **Visual feedback** showing segmentation masks and pose information
- **Multiple object support** with individual pose tracking

## Requirements

- Intel RealSense camera (tested with D435i)
- Python dependencies: `pyrealsense2`, `opencv-python`, `numpy`, `torch`
- SAM2 and TAPIR model checkpoints (see main project README)
- Pipeline configuration file

## Usage

1. **Connect your RealSense camera** and note the serial number (default: 242422304947)

2. **Run the tracker**:
   ```bash
   python realsense_tracking.py
   ```

3. **Interactive controls**:
   - **Left click**: Add positive point (object to track)
   - **Right click**: Add negative point (background/exclusion)
   - **Press 's'**: Start tracking
   - **Press 'r'**: Reset points and restart
   - **Press 'q'**: Quit

## How it works

1. **Point Collection**: Click on objects you want to track (green circles for positive points, red for negative)

2. **Initialization**: Press 's' to start the pipeline:
   - SAM2 segmenter initializes with your points
   - TAPIR tracker sets up point tracking
   - Initial pose estimation (if enabled in config)

3. **Real-time Tracking**: The pipeline continuously:
   - Segments objects using SAM2
   - Tracks points using TAPIR
   - Estimates 3D poses using SVD registration
   - Updates point sampling based on criteria

4. **Visualization**: The display shows:
   - Segmentation masks overlaid on the camera feed
   - Object poses (translation and rotation in degrees)
   - Frame count and number of tracked objects

## Configuration

The tracker uses the pipeline configuration file (`configs/pipeline/pipeline_test.yaml`). Key settings:

- `estimate_init_pose`: Whether to estimate initial bounding boxes
- `debug_level`: Debug output level (0-2)
- `debug_dir`: Directory for debug visualizations
- Model checkpoints and parameters for SAM2, TAPIR, etc.

## Customization

You can modify the script to:

- Change camera serial number in the constructor
- Use different configuration files
- Add custom visualization overlays
- Integrate with other systems (ROS, etc.)

## Troubleshooting

- **Camera not found**: Check the serial number and USB connection
- **Poor tracking**: Ensure good lighting and clear object boundaries
- **High CPU usage**: Consider reducing resolution or debug level
- **Import errors**: Make sure all dependencies are installed
