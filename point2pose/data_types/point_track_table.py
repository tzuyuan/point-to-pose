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
    track_2d: np.ndarray  # (N, 2) float  2d tracked points
    track_3d: np.ndarray  # (N, 3) float  3d tracked points
    visible: np.ndarray  # (N,) bool (from tracker)
    valid: np.ndarray  # (N,) bool (depth projection valid)
    uncertainty: np.ndarray  # (N,) float
    # outlier_keep: np.ndarray  # (N,) bool (your composed outlier filter)
    # active: np.ndarray  # (N,) bool (final “use in register” flag)

    # created_at: np.ndarray  # (N,) int frame id
    # last_seen: np.ndarray  # (N,) int frame id
    # track is the 2d tracked points in tracker
    # obj is the object id in the pipeline
    track2obj_map: Dict[int, int] = field(
        default_factory=dict
    )  # global index -> obj_id. (int -> int)
    obj2track_map: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # obj_id -> global indices. (int -> np.ndarray)

    def __len__(self):
        """
        Return the number of points being tracked in the tracker.
        """
        return self.track_3d.shape[0]

    @classmethod
    def new(cls, n0=0):
        def arr(shape, dtype, fill):
            a = np.empty(shape, dtype=dtype)
            a.fill(fill)
            return a

        return cls(
            track_2d=arr(n0, np.float32, np.nan),
            track_3d=arr(n0, np.float32, np.nan),
            visible=arr(n0, np.bool_, False),
            valid=arr(n0, np.bool_, False),
            uncertainty=arr(n0, np.float32, np.nan),
            # outlier_keep=arr(n0, np.bool_, False),
            # active=arr(n0, np.bool_, True),
            # created_at=arr(n0, np.int32, -1),
            # last_seen=arr(n0, np.int32, -1),
        )

    # def append(
    #     self,
    #     k: int,
    #     obj_id: int,
    #     frame_id: int,
    #     track_3d: np.ndarray,
    #     visible: np.ndarray,
    #     valid: np.ndarray,
    #     uncertainty: np.ndarray,
    # ):
    #     """Grow arrays by k and initialize newly added rows."""
    #     N = len(self)
    #     newN = N + k

    #     # Efficiently resize arrays using np.concatenate (much faster than np.resize)
    #     self.track_3d = np.concatenate([self.track_3d, track_3d])
    #     self.visible = np.concatenate([self.visible, visible])
    #     self.valid = np.concatenate([self.valid, valid])
    #     self.uncertainty = np.concatenate([self.uncertainty, uncertainty])

    #     # Create new arrays for the new points with appropriate defaults
    #     new_outlier_keep = np.full(k, True, dtype=np.bool_)
    #     new_active = np.full(k, True, dtype=np.bool_)
    #     new_created_at = np.full(k, frame_id, dtype=np.int32)
    #     new_last_seen = np.full(k, frame_id, dtype=np.int32)

    #     self.outlier_keep = np.concatenate([self.outlier_keep, new_outlier_keep])
    #     self.active = np.concatenate([self.active, new_active])
    #     self.created_at = np.concatenate([self.created_at, new_created_at])
    #     self.last_seen = np.concatenate([self.last_seen, new_last_seen])

    #     return np.arange(N, newN, dtype=np.int32)  # tracker indices

    def add_new_points_to_track_obj_maps(self, new_indices: np.ndarray, obj_id: int):
        # update track2obj_map and obj2track_map
        self.track2obj_map.update({int(idx): obj_id for idx in new_indices.tolist()})

        # update obj2track_map
        if obj_id in self.obj2track_map:
            self.obj2track_map[obj_id] = np.concatenate(
                (self.obj2track_map[obj_id], new_indices)
            )
        else:
            self.obj2track_map[obj_id] = new_indices

    def update_track_table(self, track_2d, track_3d, valid, uncertainties, visibles):
        self.track_2d = track_2d
        self.track_3d = track_3d
        self.valid = valid.reshape(-1)
        self.visible = visibles.reshape(-1)
        self.uncertainty = uncertainties.reshape(-1)

    def append_track_table(self, track_2d, track_3d, valid, uncertainties, visibles):
        self.track_2d = np.concatenate([self.track_2d, track_2d])
        self.track_3d = np.concatenate([self.track_3d, track_3d])
        self.valid = np.concatenate([self.valid, valid])
        self.visible = np.concatenate([self.visible, visibles])
        self.uncertainty = np.concatenate([self.uncertainty, uncertainties])

    def deactivate_tracks(self, track_ids: np.ndarray):
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        if track_ids.size == 0:
            return

        track_ids = track_ids[(track_ids >= 0) & (track_ids < len(self))]
        if track_ids.size == 0:
            return

        track_ids = np.unique(track_ids)
        self.visible[track_ids] = False
        self.valid[track_ids] = False
        if self.uncertainty.shape[0] == len(self):
            self.uncertainty[track_ids] = 1.0

        remove_set = set(int(t) for t in track_ids.tolist())
        for tid in remove_set:
            self.track2obj_map.pop(tid, None)

        for obj_id, obj_track_ids in list(self.obj2track_map.items()):
            obj_track_ids = np.asarray(obj_track_ids, dtype=np.int64).reshape(-1)
            if obj_track_ids.size == 0:
                continue
            keep = np.array([int(t) not in remove_set for t in obj_track_ids], dtype=bool)
            self.obj2track_map[obj_id] = obj_track_ids[keep]

    def get_visible_2d_and_uncertainties_for_obj(self, obj_id):
        """
        Returns:
            visible_2d: (M,2) float, pixels for this object
            visible_2d_idx: (M,) int, GLOBAL indices into the track_table arrays
            visible_uncertainties: (M,) float, uncertainties for the visible points
        """
        obj_idx = self.obj2track_map.get(obj_id, None)

        if obj_idx is None or len(obj_idx) == 0:
            return (
                np.empty((0, 2), dtype=self.track_2d.dtype),
                np.empty((0,), dtype=np.int64),
            )

        keep = self.visible[obj_idx]
        visible_2d_idx = obj_idx[keep]  # GLOBAL indices
        visible_2d = self.track_2d[visible_2d_idx]
        visible_uncertainties = self.uncertainty[visible_2d_idx]
        return visible_2d, visible_2d_idx, visible_uncertainties
