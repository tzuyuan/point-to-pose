"""Read-only capture of ModularPipeline state for visualization.

``SceneSnapshot.from_pipeline`` converts everything the 3D views need into
plain numpy arrays, so the views never hold references into live pipeline
state and the pipeline is never modified.

Frame conventions (matching the pipeline):
  * ``obj.init_pose``: object (bbox-centered) frame -> first camera frame.
  * ``obj.pose``:      first camera frame -> current camera frame.
  * ``obj.key_points`` and the SDF volume live in the first-camera ("world")
    frame; ``track_table.track_3d`` lives in the current camera frame.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from point2pose.utils.transform import inverse_SE3, transform_pts


@dataclass
class ObjectSnapshot:
    """Numpy-only state of one tracked object at one frame."""

    obj_id: int
    lost: bool
    T_cam_obj: np.ndarray  # (4,4) object frame -> current camera frame
    T_obj_cam: np.ndarray  # (4,4) current camera frame -> object frame
    T_obj_world: np.ndarray  # (4,4) first-camera ("world") frame -> object frame

    # Keypoint map, expressed in the object frame
    map_points: np.ndarray  # (M,3)
    map_point_frames: np.ndarray  # (M,) frame id each point was added at
    map_point_uncertainties: np.ndarray  # (M,)
    map_point_track_ids: np.ndarray  # (M,) global track ids
    map_point_visible: np.ndarray  # (M,) currently tracked & visible

    # Live tracker points in the current camera frame.
    # NaN rows mark points that are currently invisible or depth-invalid.
    track_points_cam: np.ndarray  # (K,3)
    track_points_2d: np.ndarray  # (K,2) tracked pixel positions (full-res)
    track_ids: np.ndarray  # (K,) global track ids (row-aligned)
    track_point_frames: np.ndarray  # (K,) first-seen frame ids
    track_point_uncertainties: np.ndarray  # (K,)
    # registration status: 1 = inlier, 0 = outlier, -1 = not used this frame
    track_point_inlier: np.ndarray  # (K,) int8

    # Cameras of this object's keyframes, expressed in the object frame
    keyframe_T_obj_cam: List[np.ndarray] = field(default_factory=list)

    bbox_extent: Optional[np.ndarray] = None  # (3,) or None
    mesh_version: int = 0  # obj.sdf_num_integrated; cache key for the SDF mesh

    # Registration health (for the dashboard status bar)
    mean_residual: float = 0.0  # meters
    num_inliers: int = -1  # -1 = unknown


@dataclass
class SceneSnapshot:
    frame_id: int
    timestamp: float
    objects: List[ObjectSnapshot] = field(default_factory=list)

    @classmethod
    def from_pipeline(cls, pipeline, frame) -> "SceneSnapshot":
        scene = cls(
            frame_id=int(frame.id),
            timestamp=float(getattr(frame, "timestamp", 0.0) or 0.0),
        )
        table = pipeline.track_table
        keyframes = getattr(getattr(pipeline, "kf_manager", None), "keyframes", {})

        for obj in pipeline.objects:
            pose = np.asarray(
                obj.pose if obj.pose is not None else np.eye(4), dtype=np.float64
            )
            init_pose = np.asarray(obj.init_pose, dtype=np.float64)
            T_cam_obj = pose @ init_pose
            T_obj_cam = inverse_SE3(T_cam_obj)
            T_obj_world = inverse_SE3(init_pose)

            snap = ObjectSnapshot(
                obj_id=int(obj.id),
                lost=bool(getattr(obj, "lost", False)),
                T_cam_obj=T_cam_obj,
                T_obj_cam=T_obj_cam,
                T_obj_world=T_obj_world,
                map_points=np.zeros((0, 3)),
                map_point_frames=np.zeros((0,), dtype=np.int64),
                map_point_uncertainties=np.zeros((0,)),
                map_point_track_ids=np.zeros((0,), dtype=np.int64),
                map_point_visible=np.zeros((0,), dtype=bool),
                track_points_cam=np.zeros((0, 3)),
                track_points_2d=np.zeros((0, 2)),
                track_ids=np.zeros((0,), dtype=np.int64),
                track_point_frames=np.zeros((0,), dtype=np.int64),
                track_point_uncertainties=np.zeros((0,)),
                track_point_inlier=np.zeros((0,), dtype=np.int8),
                mesh_version=int(getattr(obj, "sdf_num_integrated", 0)),
                mean_residual=float(getattr(obj, "mean_residual", 0.0) or 0.0),
            )
            inliers = getattr(obj, "inliers", None)
            if inliers is not None and np.size(inliers) > 0:
                snap.num_inliers = int(np.sum(np.asarray(inliers).astype(bool)))

            cls._fill_map(snap, obj, table, T_obj_world, scene.frame_id)
            cls._fill_tracks(snap, obj, table, scene.frame_id)

            for kf in keyframes.get(obj.id, []):
                if getattr(kf, "pose", None) is None:
                    continue
                kf_pose = np.asarray(kf.pose, dtype=np.float64)
                snap.keyframe_T_obj_cam.append(T_obj_world @ inverse_SE3(kf_pose))

            bbox = getattr(obj, "bbox", None)
            if bbox is not None:
                try:
                    snap.bbox_extent = np.asarray(bbox.extent, dtype=np.float64)
                except Exception:
                    snap.bbox_extent = None

            scene.objects.append(snap)
        return scene

    @staticmethod
    def _fill_map(snap, obj, table, T_obj_world, frame_id):
        key_points = np.asarray(obj.key_points, dtype=np.float64).reshape(-1, 3)
        n = len(key_points)
        if n == 0:
            return

        valid = np.ones(n, dtype=bool)
        obj_valid = getattr(obj, "valid", None)
        if obj_valid is not None and len(obj_valid) == n:
            valid = np.asarray(obj_valid).astype(bool)
        if not valid.any():
            return

        snap.map_points = transform_pts(T_obj_world, key_points[valid])

        frames = np.full(n, frame_id, dtype=np.int64)
        if getattr(obj, "key_point_frames", np.zeros(0)).shape[0] == n:
            frames = np.asarray(obj.key_point_frames, dtype=np.int64).copy()
            frames[frames < 0] = frame_id  # -1 = added this frame
        snap.map_point_frames = frames[valid]

        uncert = np.zeros(n)
        if getattr(obj, "uncertainties", np.zeros(0)).shape[0] == n:
            uncert = np.asarray(obj.uncertainties, dtype=np.float64)
        snap.map_point_uncertainties = uncert[valid]

        # global track id per map point + whether it is tracked right now
        ids = np.full(n, -1, dtype=np.int64)
        kp_ids = np.asarray(getattr(obj, "kp_track_indices", np.zeros(0)), dtype=np.int64)
        if kp_ids.shape[0] == n:
            ids = kp_ids
        snap.map_point_track_ids = ids[valid]
        vis = np.zeros(len(snap.map_point_track_ids), dtype=bool)
        table_vis = np.asarray(table.visible).reshape(-1)
        in_range = (snap.map_point_track_ids >= 0) & (
            snap.map_point_track_ids < table_vis.shape[0]
        )
        vis[in_range] = table_vis[snap.map_point_track_ids[in_range]]
        snap.map_point_visible = vis

    @staticmethod
    def _fill_tracks(snap, obj, table, frame_id):
        idx = table.obj2track_map.get(obj.id)
        if idx is None or len(idx) == 0 or len(table.track_3d) == 0:
            return
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[idx < len(table.track_3d)]
        if len(idx) == 0:
            return

        pts = np.asarray(table.track_3d[idx], dtype=np.float64).reshape(-1, 3).copy()
        ok = np.asarray(table.valid[idx]).astype(bool) & np.asarray(
            table.visible[idx]
        ).astype(bool)
        pts[~ok] = np.nan
        snap.track_points_cam = pts
        snap.track_ids = idx
        if len(table.track_2d) >= len(table.track_3d):
            snap.track_points_2d = np.asarray(
                table.track_2d[idx], dtype=np.float64
            ).reshape(-1, 2)
        snap.track_point_uncertainties = np.asarray(
            table.uncertainty[idx], dtype=np.float64
        )

        # Registration inlier status per track. obj.inliers[j] belongs to
        # global track id obj.curr_frame_indices[j] (front_end convention).
        status = np.full(len(idx), -1, dtype=np.int8)
        reg_ids = getattr(obj, "curr_frame_indices", None)
        reg_inliers = getattr(obj, "inliers", None)
        if reg_ids is not None and reg_inliers is not None:
            reg_ids = np.asarray(reg_ids, dtype=np.int64).reshape(-1)
            reg_inliers = np.asarray(reg_inliers).astype(bool).reshape(-1)
            if len(reg_ids) == len(reg_inliers) and len(reg_ids) > 0:
                order = np.argsort(idx)
                sorted_ids = idx[order]
                pos = np.searchsorted(sorted_ids, reg_ids)
                pos = np.clip(pos, 0, max(len(sorted_ids) - 1, 0))
                ok = sorted_ids[pos] == reg_ids
                status[order[pos[ok]]] = reg_inliers[ok].astype(np.int8)
        snap.track_point_inlier = status

        # First-seen frame per track: global track id -> object row -> frame id
        frames = np.full(len(idx), frame_id, dtype=np.int64)
        lut = np.asarray(getattr(obj, "track_idx_2_obj_idx", np.zeros(0, dtype=np.int32)))
        kp_frames = np.asarray(getattr(obj, "key_point_frames", np.zeros(0, dtype=int)))
        if lut.size > 0 and kp_frames.size > 0:
            safe = np.minimum(idx, lut.size - 1)
            rows = np.where(idx < lut.size, lut[safe], -1)
            good = (rows >= 0) & (rows < kp_frames.size)
            seen = kp_frames[rows[good]].astype(np.int64)
            seen[seen < 0] = frame_id
            frames[good] = seen
        snap.track_point_frames = frames
