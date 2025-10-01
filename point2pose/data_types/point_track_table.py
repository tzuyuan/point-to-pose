from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List


@dataclass
class PointTrackTable:
    """
    A table to store the track information of the points.
    N is number of points being tracked in the tracker.
    """

    # size == current number of tracker points
    obj_id: np.ndarray  # (N,) int, -1 if unassigned
    created_at: np.ndarray  # (N,) int frame id
    last_seen: np.ndarray  # (N,) int frame id
    visible: np.ndarray  # (N,) bool (from tracker)
    valid2d: np.ndarray  # (N,) bool (bounds etc.)
    valid3d: np.ndarray  # (N,) bool (depth projection valid)
    outlier_keep: np.ndarray  # (N,) bool (your composed outlier filter)
    active: np.ndarray  # (N,) bool (final “use in register” flag)

    # Optional: confidences/uncertainty per point
    uncertainty: np.ndarray  # (N,) float

    def __len__(self):
        return self.obj_id.shape[0]

    @classmethod
    def new(cls, n0=0):
        def arr(shape, dtype, fill):
            a = np.empty(shape, dtype=dtype)
            a.fill(fill)
            return a

        return cls(
            obj_id=arr(n0, np.int32, -1),
            created_at=arr(n0, np.int32, -1),
            last_seen=arr(n0, np.int32, -1),
            visible=arr(n0, np.bool_, False),
            valid2d=arr(n0, np.bool_, False),
            valid3d=arr(n0, np.bool_, False),
            outlier_keep=arr(n0, np.bool_, False),
            active=arr(n0, np.bool_, True),
            uncertainty=arr(n0, np.float32, np.nan),
        )

    def append(self, k: int, obj: int, frame_id: int):
        """Grow arrays by k and initialize newly added rows."""
        N = len(self)
        newN = N + k
        for name, arr in self.__dict__.items():
            self.__dict__[name] = np.resize(arr, newN)
        self.obj_id[N:newN] = obj
        self.created_at[N:newN] = frame_id
        self.last_seen[N:newN] = frame_id
        self.visible[N:newN] = False
        self.valid2d[N:newN] = False
        self.valid3d[N:newN] = False
        self.outlier_keep[N:newN] = True
        self.active[N:newN] = True
        self.uncertainty[N:newN] = np.nan
        return np.arange(N, newN, dtype=np.int32)  # tracker indices
