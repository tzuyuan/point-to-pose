# Pose Logging Documentation

This document describes the pose logging functionality added to the point-to-pose pipeline.

## Configuration

Add the following parameters to your pipeline configuration:

```yaml
pipeline:
  params:
    save_pose: true                    # Enable pose logging
    pose_save_path: /path/to/poses     # Directory to save pose files
```

## Output Files

When pose logging is enabled, the following files will be created:

### 1. Object Pose Files (`obj_i_pose.txt`)

Each object gets its own pose file named `obj_i_pose.txt` where `i` is the object ID (0, 1, 2, etc.).

**Format:** TUM format
```
# timestamp tx ty tz qx qy qz qw
1234567890.123456 0.100000 0.200000 0.300000 0.000000 0.000000 0.000000 1.000000
1234567890.223456 0.105000 0.205000 0.305000 0.001000 0.002000 0.003000 0.999000
```

**Columns:**
- `timestamp`: Unix timestamp in seconds
- `tx ty tz`: Translation components (x, y, z in meters)
- `qx qy qz qw`: Quaternion rotation components (x, y, z, w)

### 2. Registration Statistics File (`registration_stats.txt`)

Contains detailed statistics from the registration process for each frame.

**Format:** Tab-separated values
```
timestamp	frame_id	obj_id	num_points	iter	thr	res_mean	res_median	res_max	num_inliers	total_points	mean_residual_inliers	mean_residual_outliers
1234567890.123456	1	0	25	2	0.150000	0.120000	0.110000	0.250000	22	25	0.110000	0.250000
1234567890.223456	2	0	28	1	0.140000	0.115000	0.105000	0.230000	25	28	0.105000	0.230000
```

**Columns:**
- `timestamp`: Unix timestamp in seconds when registration was performed
- `frame_id`: Frame number
- `obj_id`: Object ID
- `num_points`: Number of points used for registration
- `iter`: Number of iterations performed
- `thr`: Threshold used for inlier detection
- `res_mean`: Mean residual error
- `res_median`: Median residual error
- `res_max`: Maximum residual error
- `num_inliers`: Number of inlier points
- `total_points`: Total correspondences considered for residuals
- `mean_residual_inliers`: Mean residual over inlier set
- `mean_residual_outliers`: Mean residual over outlier set

## Usage Examples

### Python
```python
from point2pose.pipeline.pipeline import Pipeline

# Your existing configuration with pose logging enabled
pipeline = Pipeline(config)

# Poses will be automatically logged during pipeline execution
poses = pipeline.step(frame)
```

### Configuration Example
```yaml
pipeline:
  params:
    estimate_init_pose: true
    frame_to_map_reg: true
    debug_level: 2
    debug_dir: /path/to/debug
    save_pose: true
    pose_save_path: /path/to/debug/poses
```

## File Structure
```
pose_save_path/
├── obj_0_pose.txt           # Object 0 poses
├── obj_1_pose.txt           # Object 1 poses
├── obj_2_pose.txt           # Object 2 poses
└── registration_stats.txt   # Registration statistics
```

## Notes

- Pose files are created and opened during pipeline initialization
- Poses are logged in real-time as they are computed
- Files are automatically flushed after each pose update
- The pipeline will clean up file handles when destroyed
- If `save_pose` is `false`, no pose logging occurs and no files are created
