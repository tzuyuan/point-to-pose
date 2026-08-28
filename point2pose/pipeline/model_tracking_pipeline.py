"""
Model-based tracking pipeline.

Given a textured 3D model, this pipeline pre-builds a fixed point map from rendered views
(``MapBuilder``) and then tracks the object's 6-DoF pose in a live RGB-D stream using TAPIR +
PnP frame-to-map. It deliberately omits everything the full ``ModularPipeline`` layers on top:
no SDF reconstruction, no local/global pose-graph optimization, no online keyframe/map growth,
no recovery manager.

Usage:
    pipe = ModelTrackingPipeline(cfg)
    pipe.build_map(intrinsics, img_size)      # render + seed TAPIR (once, at startup)
    for frame in stream:
        poses = pipe.step(frame)              # (num_obj, 4, 4)  T_obj2cam per object
"""

import numpy as np

from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.model_tracking.map_builder import MapBuilder
from point2pose.pipeline.components.model_tracking_front_end import ModelTrackingFrontEnd


class ModelTrackingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg.pipeline.params
        self.num_obj = int(p.get("max_num_obj", 1))

        mt = cfg.model_tracking.params
        self.map_path = mt.get("map_path", None)
        self.rebuild_map = bool(mt.get("rebuild_map", False))

        # Map builder owns the SuperPoint sampler + the (to-be-seeded) TAPIR tracker.
        self.map_builder = MapBuilder(cfg)

        # State, populated by build_map().
        self.tracker = None
        self.frontend = None
        self.track_table = PointTrackTable.new(n0=0)
        self.objects = []
        self.model_map = None
        self._map_ready = False
        self.last_fe_result = None

    # ------------------------------------------------------------------ #
    def build_map(self, intrinsics, img_size):
        """
        Build (or load) the model map and seed the TAPIR tracker. Must be called once before
        ``step``, after the live camera intrinsics are known so the renders match the camera.
        """
        model_map, tracker, track_table, obj = self.map_builder.build_and_maybe_cache(
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            img_size=img_size,
            map_path=self.map_path,
            rebuild=self.rebuild_map,
        )
        self.model_map = model_map
        self.tracker = tracker
        self.track_table = track_table
        self.objects = [obj]

        # The front-end consumes the already-seeded tracker.
        self.frontend = ModelTrackingFrontEnd(self.cfg, tracker)
        # TAPIR was initialized from render frames; initialize() only checks query points exist.
        self.tracker.initialize(_DummyFrame(img_size))
        self._map_ready = True
        print(
            f"[ModelTrackingPipeline] Map ready: {obj.key_points.shape[0]} points, "
            f"{len(self.objects)} object(s)."
        )

    def add_user_points(self, points, labels):
        """Only meaningful when segmentation is enabled; otherwise a no-op."""
        if self.frontend is not None:
            self.frontend.add_user_points(points, labels)

    def initialize_segmenter(self, frame):
        if self.frontend is not None:
            return self.frontend.initialize_segmenter(frame)
        return False

    @property
    def segmenter_ready(self):
        return bool(self.frontend is not None and self.frontend.segmenter_ready)

    # ------------------------------------------------------------------ #
    def step(self, frame):
        """Run one tracking step. Returns (num_obj, 4, 4) array of T_obj2cam."""
        if not self._map_ready:
            # Lazily build the map from this frame's intrinsics if not built yet.
            if frame.intrinsics is None:
                raise RuntimeError(
                    "[ModelTrackingPipeline] Map not built and frame has no intrinsics. "
                    "Call build_map(intrinsics, img_size) first."
                )
            self.build_map(frame.intrinsics, frame.rgb.shape[:2])

        fe_result = self.frontend.step(frame, self.track_table, self.objects)
        # Expose the last front-end result so callers can visualize exactly the points that
        # passed the (visible + low-uncertainty + in-mask) filter and were fed to PnP.
        self.last_fe_result = fe_result

        # Update live 2D/visibility in the track table (3D map stays fixed).
        if fe_result.tracks is not None:
            self.track_table.track_2d = fe_result.tracks
            self.track_table.visible = np.asarray(fe_result.visibles, dtype=bool).reshape(-1)
            self.track_table.uncertainty = np.asarray(
                fe_result.uncertainties, dtype=np.float32
            ).reshape(-1)

        out_pose = np.stack([obj.pose for obj in self.objects], axis=0)
        return out_pose


class _DummyFrame:
    """Minimal stand-in so TapirTracker.initialize can read image dimensions."""

    def __init__(self, img_size):
        H, W = int(img_size[0]), int(img_size[1])
        self.rgb = np.zeros((H, W, 3), dtype=np.uint8)
        self.id = 0
