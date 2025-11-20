import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class FrontEndResult:
    """
    Results from the FrontEnd step containing pose estimates and tracking stats.
    """

    frame_id: int
    # Per-object results, keyed by obj_id
    obj_poses: Dict[int, np.ndarray] = field(default_factory=dict)  # 4x4 pose matrices
    rel_poses: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # 4x4 relative pose matrices

    # Registration stats per object
    # Key: obj_id, Value: Dict with stats (inliers, residuals, etc.)
    reg_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Mean residuals per object
    mean_residuals: Dict[int, float] = field(default_factory=dict)

    # Valid keypoints used for registration (for optimization later)
    # Key: obj_id
    valid_indices: Dict[int, np.ndarray] = field(default_factory=dict)
    valid_key_points: Dict[int, np.ndarray] = field(default_factory=dict)
    valid_curr_3d: Dict[int, np.ndarray] = field(default_factory=dict)

    # Extra debug/validation stats from extraction (e.g. masks)
    # Key: obj_id, Value: Dict of arrays
    valid_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Track table update data
    tracks: Optional[np.ndarray] = None
    uncertainties: Optional[np.ndarray] = None
    visibles: Optional[np.ndarray] = None
    track_3d: Optional[np.ndarray] = None
    track_valid: Optional[np.ndarray] = None
