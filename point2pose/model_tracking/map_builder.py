"""
Offline map builder for model-based tracking.

Given a textured 3D model placed at the origin (object frame == mesh frame), this module:

  1. Renders textured RGB + depth from many viewpoints around the object (pyrender).
  2. Runs SuperPoint on each render to pick surface feature points.
  3. Backprojects those keypoints with the render's depth to 3D coordinates in the
     *object frame* (the map points).
  4. Seeds those points as TAPIR query points, each stamped with its own timestamp `t`
     (one `t` per rendered view) -- reusing point2pose's existing multi-view mechanism.

The result is a ``ModelMap`` (3D map points in object frame, per-view render RGB, keypoints,
timestamps, poses) plus a TAPIR tracker whose query features are already initialized from the
renders and ready to track into a live video stream.

Coordinate conventions
-----------------------
- Camera poses are expressed in the OpenCV convention: ``T_obj2cam`` maps object-frame points
  to the camera frame where +Z points *into* the scene (toward the object), +X right, +Y down.
  ``T_cam2obj = inverse_SE3(T_obj2cam)`` is the camera pose in the object frame.
- Rendering uses Open3D's ``OffscreenRenderer``. Its ``setup_camera(intrinsic, extrinsic)``
  takes the extrinsic as ``T_obj2cam`` in the CV convention (+Z forward, +Y down), which is
  exactly the pose convention we use here -- no OpenGL basis flip is needed.
- Backprojection (``convert_pixel_to_world``) uses the CV convention, so the render depth
  (metric, +Z in view space) can be lifted directly and transformed to the object frame with
  ``T_cam2obj``.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from point2pose.core.build import build_from_cfg
from point2pose.core.module_registry import SAMPLER, TRACKER
from point2pose.data_types.frame import Frame
from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.data_types.sampler_context import SamplerContext
from point2pose.modules.object.object import Object
from point2pose.utils.camera import convert_pixel_to_world
from point2pose.utils.transform import inverse_SE3, transform_pts


@dataclass
class _AABB:
    """Minimal axis-aligned bounding box with .extent/.center (numpy), matching the
    attributes the pose-box visualization reads (avoids depending on open3d's API)."""

    extent: np.ndarray
    center: np.ndarray


@dataclass
class ModelMap:
    """Container for the pre-built model map."""

    mesh_path: str
    intrinsics: np.ndarray  # (3,3) the LIVE camera K (reference; used at runtime for PnP)
    img_size: tuple  # (H, W)
    render_intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))  # (3,3) K used to RENDER the map

    # Per-view render RGB (uint8, HxWx3), one per rendered view. Needed to rebuild the
    # TAPIR query features on load.
    view_rgbs: List[np.ndarray] = field(default_factory=list)
    # Per-view 2D keypoints in that render (list of (M_v, 2) float, (x, y) pixels).
    view_kps: List[np.ndarray] = field(default_factory=list)
    # Per-view timestamp assigned to the TAPIR queries for that view.
    view_timestamps: List[int] = field(default_factory=list)
    # Per-view camera pose T_obj2cam (object -> camera, CV convention), (4,4).
    view_poses_obj2cam: List[np.ndarray] = field(default_factory=list)

    # Flattened map: 3D map points in object frame, aligned to the concatenated per-view
    # keypoints in the order they were added to the tracker.
    map_pts_obj: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    map_valid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))

    # Object axis-aligned bounding box (object frame), for pose-box visualization.
    bbox_extent: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ModelMap] Saved map with {len(self.map_pts_obj)} points to {path}")

    @staticmethod
    def load(path: str) -> "ModelMap":
        with open(path, "rb") as f:
            m = pickle.load(f)
        print(f"[ModelMap] Loaded map with {len(m.map_pts_obj)} points from {path}")
        return m


class MapBuilder:
    """
    Builds a ``ModelMap`` from a textured mesh and seeds a TAPIR tracker with the map points.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: the full OmegaConf config. Uses ``cfg.model_tracking`` for build params and
                 ``cfg.sampler`` / ``cfg.tracker`` to construct the SuperPoint sampler and TAPIR
                 tracker (same module configs the live pipeline uses).
        """
        self.cfg = cfg
        mt = cfg.model_tracking.params

        self.mesh_path = mt.get("mesh_path", None)
        # Mapper backend: "nvdiffrast" (textured mesh) or "gsplat" (pretrained gaussian
        # splat checkpoint, e.g. from kv_tracker's reconstruct.py). gsplat_* params below
        # are only used when renderer == "gsplat".
        self.renderer_backend = mt.get("renderer", "nvdiffrast")
        assert self.renderer_backend in ("nvdiffrast", "gsplat"), (
            f"unknown model_tracking.params.renderer {self.renderer_backend!r}, "
            "expected 'nvdiffrast' or 'gsplat'"
        )
        self.gsplat_path = mt.get("gsplat_path", None)
        self.gsplat_model = mt.get("gsplat_model", "2dgs")
        self.gsplat_sh_degree = mt.get("gsplat_sh_degree", None)
        self.gsplat_depth_mode = mt.get("gsplat_depth_mode", "median")
        self.num_views = int(mt.get("num_views", 20))
        # In-plane (camera roll) augmentation: for each viewpoint, also render this many
        # views rolled about the optical axis by evenly-spaced angles in [0, 360). 1 = no aug.
        self.num_rotations = max(1, int(mt.get("num_rotations", 1)))
        # Hard cap on total map points (TAPIR tracks all of them every frame; keep it modest
        # for memory/speed). Points beyond the cap are dropped by evenly subsampling views.
        self.max_map_points = int(mt.get("max_map_points", 1500))
        self.render_radius = float(mt.get("render_radius", 0.4))
        # The map is rendered with a STANDARD centered pinhole (cx=cy=res/2, this vertical FOV),
        # NOT the live camera K. The map only needs 3D surface feature points, and those are the
        # same regardless of the (well-behaved) virtual camera used to see them; using the live
        # K's off-center principal point would only skew feature coverage in the render.
        self.render_fov_deg = float(mt.get("render_fov_deg", 55.0))
        # Environment-map (IBL) shading for the render, so the 3D shape reads and shading
        # rotates with the camera the way a real fixed studio light would (see
        # ``NvdiffrastRenderer`` docstring for why object frame doubles as world frame here).
        # envmap_path=None renders unlit (diffuse albedo only) instead of a fake fixed light.
        self.envmap_path = mt.get("envmap_path", None)
        self.envmap_intensity = float(mt.get("envmap_intensity", 1.0))
        self.envmap_cube_res = int(mt.get("envmap_cube_res", 128))
        # Phong-style flat term added to the diffuse irradiance everywhere (fraction of its
        # scene mean), so faces facing away from every bright HDRI direction don't go fully
        # black -- a cheap stand-in for the indirect bounce light a real room always has.
        # Adding rather than clamping preserves the lit/shadow shading shape. 0 disables it.
        self.envmap_ambient_term = float(mt.get("envmap_ambient_term", 0.25))
        self.render_specular = bool(mt.get("render_specular", False))
        self.render_roughness = float(mt.get("render_roughness", 0.5))
        self.min_depth = float(mt.get("min_depth", 0.05))
        self.max_depth = float(mt.get("max_depth", 2.0))
        self.mesh_scale = float(mt.get("mesh_scale", 1.0))
        self.device = cfg.pipeline.params.get("device", "cuda") if "pipeline" in cfg else "cuda"
        # Directory to dump per-view renders (RGB/depth) during the render phase. If unset,
        # a temp dir alongside the map cache (or the system temp) is used.
        self.render_dir = mt.get("render_dir", None)
        # Debug: save per-view "render + SuperPoint keypoints" overlays under
        # <pipeline.debug_dir>/map_debug/ so you can eyeball feature coverage per viewpoint.
        pdir = cfg.pipeline.params.get("debug_dir", None) if "pipeline" in cfg else None
        self.debug_save = bool(mt.get("debug_save_views", True)) and pdir is not None
        self.debug_dir = os.path.join(pdir, "map_debug") if pdir else None

        # Sampler (SuperPoint) and tracker (TAPIR): built from the same registries/configs
        # used by the live pipeline so features/scale match.
        self.sampler = build_from_cfg(cfg.sampler, SAMPLER)
        self.tracker = build_from_cfg(cfg.tracker, TRACKER)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, intrinsics: np.ndarray, img_size):
        """
        Build the map and seed ``self.tracker``.

        Args:
            intrinsics: (3,3) LIVE camera intrinsics. Stored on the map for reference/PnP; it is
                NOT used for rendering -- the map is rendered with a standard centered pinhole
                (see ``render_fov_deg``), which only affects how we *view* the model, not the
                object-frame 3D feature points that get stored.
            img_size: (H, W) render/live image size (square recommended; == TAPIR img_*).

        Returns:
            (model_map, tracker, track_table, obj)
              - model_map: ModelMap
              - tracker: TAPIR tracker seeded with all map query points (across views/timestamps)
              - track_table: PointTrackTable with obj2track_map / track_3d filled for obj 0
              - obj: Object with key_points (map 3D in object frame) and kp_track_indices
        """
        H, W = int(img_size[0]), int(img_size[1])
        live_K = np.asarray(intrinsics, dtype=np.float64)
        # Standard centered pinhole used ONLY for rendering the map.
        K = self._standard_render_K(H, W, self.render_fov_deg)

        poses_obj2cam = self._sample_viewpoints()

        if self.debug_save:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[MapBuilder] saving per-view keypoint overlays to {self.debug_dir}")

        model_map = ModelMap(
            mesh_path=self.mesh_path, intrinsics=live_K, render_intrinsics=K, img_size=(H, W)
        )
        track_table = PointTrackTable.new(n0=0)
        obj = Object(id=0)
        obj.init_pose = np.eye(4)  # object frame == mesh/map frame
        obj.pose = np.eye(4)

        # ---- Phase 1: render every view to disk, then free the renderer (releases GPU) ----
        # This keeps the nvdiffrast context and the growing TAPIR state from coexisting on
        # the GPU, which otherwise OOMs for many views / many points.
        renders = self._render_all_views(poses_obj2cam, K, H, W)

        # ---- Phase 2: SuperPoint + backprojection + TAPIR seeding from the saved renders ----
        # Even per-view budget so the cap is spread across all viewpoints, not consumed by the
        # first few. SuperPoint already returns points ordered by descending score, so keeping
        # the first `per_view_budget` keeps the strongest features.
        n_views = max(1, len(renders))
        per_view_budget = max(1, self.max_map_points // n_views)
        print(f"[MapBuilder] per-view budget = {per_view_budget} "
              f"(max_map_points={self.max_map_points} / {n_views} views). "
              f"Increase max_map_points or reduce num_views/num_rotations for more per view.")

        all_map_pts = []
        all_valid = []
        total_raw_kps = 0
        for v, (rgb, depth, T_obj2cam) in enumerate(renders):
            render_mask = depth > 0  # object mask = wherever geometry was rendered
            kps = self._extract_superpoint(rgb, render_mask, v)  # (M,2) (x,y)
            total_raw_kps += kps.shape[0]
            if kps.shape[0] == 0:
                continue

            # Backproject with render depth (CV convention). Depth is metric meters.
            pts_cam, valid = convert_pixel_to_world(
                pixel=kps,
                depth_image=depth,
                cam_intrinsics=K,
                depth_factor=1.0,  # nvdiffrast depth is already in meters
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            valid = np.asarray(valid, dtype=bool).reshape(-1)

            # Keep only points with valid 3D, then trim to the per-view budget. Use an even
            # subsample (not first-N) so the trim is agnostic to sampler ordering -- for the
            # FPS sampler the points are spatially ordered, not score-ordered, and an even
            # subsample preserves the spatial spread.
            keep = np.where(valid)[0]
            if keep.size > per_view_budget:
                sel = np.linspace(0, keep.size - 1, per_view_budget).round().astype(int)
                keep = keep[np.unique(sel)]
            kps = kps[keep]
            pts_cam = pts_cam[keep]
            valid = valid[keep]
            if kps.shape[0] == 0:
                continue

            # Transform camera-frame points to object frame.
            T_cam2obj = inverse_SE3(T_obj2cam)
            pts_obj = transform_pts(T_cam2obj, np.nan_to_num(pts_cam, nan=0.0))

            # Seed the tracker with these keypoints at a unique timestamp (= view index).
            render_frame = Frame(id=v, rgb=rgb, depth=depth, intrinsics=K, depth_factor=1.0)
            new_indices = self.tracker.add_query_points(render_frame, kps)
            # Commit this view's queries immediately. TAPIR already seeds incrementally in
            # add_query_points, so this is just a cheap extra step for it; CoTracker's online
            # predictor instead buffers queries and only commits them on the next track_once
            # -- without calling it here, every add_query_points() call but the last would be
            # silently dropped once live tracking starts.
            self.tracker.track_once(render_frame)

            # Register the new points into the track table + object map.
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id=0)
            obj.add_key_points(
                new_key_points=pts_obj,
                new_uncertainties=0.1 * np.ones(pts_obj.shape[0], dtype=float),
                new_valid=valid,
                new_indices=new_indices,
                frame_id=v,
            )

            model_map.view_rgbs.append(rgb)
            model_map.view_kps.append(kps)
            model_map.view_timestamps.append(v)
            model_map.view_poses_obj2cam.append(T_obj2cam)
            all_map_pts.append(pts_obj)
            all_valid.append(valid)

            if self.debug_save:
                self._save_view_overlay(v, rgb, kps, valid)

            print(
                f"[MapBuilder] seed view {v}/{len(renders)}: "
                f"{kps.shape[0]} kps, {int(valid.sum())} valid 3D"
            )

        if len(all_map_pts) == 0:
            raise RuntimeError(
                "[MapBuilder] No valid map points produced. Check mesh path / render / depth range."
            )

        model_map.map_pts_obj = np.concatenate(all_map_pts, axis=0)
        model_map.map_valid = np.concatenate(all_valid, axis=0)

        # Sanity: the flattened map must align 1:1 with obj.key_points ordering.
        assert model_map.map_pts_obj.shape[0] == obj.key_points.shape[0], (
            f"map/obj point count mismatch: {model_map.map_pts_obj.shape[0]} "
            f"vs {obj.key_points.shape[0]}"
        )

        # Fill the track_table 3D/valid so downstream code can read map 3D directly.
        n = obj.key_points.shape[0]
        track_table.track_3d = obj.key_points.copy()
        track_table.track_2d = np.full((n, 2), np.nan, dtype=np.float32)
        track_table.valid = model_map.map_valid.copy()
        track_table.visible = np.zeros(n, dtype=bool)
        track_table.uncertainty = 0.1 * np.ones(n, dtype=np.float32)

        # Object bounding box (object frame), for pose-box visualization. Use the mesh extent
        # (AABB) or, for gsplat, the fitted OBB -- Open3D's OrientedBoundingBox exposes the
        # center only via get_center() (no .center attribute), unlike our plain _AABB.
        self._set_object_bbox(obj)
        model_map.bbox_extent = np.asarray(obj.bbox.extent, dtype=float)
        center = getattr(obj.bbox, "center", None)
        if center is None:
            center = obj.bbox.get_center()
        model_map.bbox_center = np.asarray(center, dtype=float)

        n_used_views = len(model_map.view_rgbs)
        print(
            f"[MapBuilder] Built map: {n} points across {n_used_views} views "
            f"(SuperPoint raw kps total={total_raw_kps}, "
            f"avg {total_raw_kps / max(1, n_used_views):.1f}/view raw -> "
            f"{n / max(1, n_used_views):.1f}/view kept after budget {per_view_budget})."
        )
        return model_map, self.tracker, track_table, obj

    def _save_view_overlay(self, view_id, rgb, kps, valid):
        """Save a render + SuperPoint keypoint overlay PNG for one view."""
        import cv2

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        valid = np.asarray(valid, dtype=bool)
        for i, (x, y) in enumerate(np.asarray(kps)):
            # green = kept with valid 3D, red = no valid depth (dropped from the map)
            color = (0, 255, 0) if (i < valid.size and valid[i]) else (0, 0, 255)
            cv2.circle(img, (int(round(x)), int(round(y))), 2, color, -1)
        cv2.putText(img, f"view {view_id}: {int(valid.sum())}/{len(kps)} pts",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(self.debug_dir, f"view_{view_id:04d}.png"), img)

    def _set_object_bbox(self, obj):
        """Attach a bounding box (object frame) to obj: an oriented box (OBB, fit to the
        actual point distribution) for the gsplat backend, or an axis-aligned box (AABB,
        matching the centered mesh frame) for nvdiffrast."""
        if self.renderer_backend == "gsplat":
            obj.bbox = self._gaussian_obb()
        else:
            from point2pose.model_tracking.nvdiffrast_renderer import NvdiffrastRenderer

            # Reuse the renderer's centered vertices so the box matches the map coord frame.
            v = NvdiffrastRenderer._load_mesh(self.mesh_path, self.mesh_scale)[0]
            v = v - v.mean(0)  # same centering the renderer applies
            lo, hi = v.min(0), v.max(0)
            obj.bbox = _AABB(extent=(hi - lo), center=0.5 * (lo + hi))
        obj.init_bbox = obj.bbox

    def _gaussian_obb(self, opacity_thresh: float = 0.1):
        """Oriented bounding box (Open3D, same type `pipeline.py`'s point-cloud init uses)
        fit to the gaussian means, restricted to gaussians with activated opacity above
        ``opacity_thresh``. Stray low-opacity gaussians (background noise the optimizer
        never pruned) would otherwise inflate an axis-aligned/oriented fit far past the
        object's actual extent."""
        import open3d as o3d
        import torch

        splats = torch.load(self.gsplat_path, map_location="cpu")
        means = splats["means"].numpy()
        opacities = torch.sigmoid(splats["opacities"]).numpy()
        means = means[opacities > opacity_thresh]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        return pcd.get_oriented_bounding_box()

    def build_and_maybe_cache(self, intrinsics, img_size, map_path: Optional[str] = None,
                              rebuild: bool = False):
        """
        Build the map, or load a cached ModelMap and re-seed the tracker from its stored renders.

        Note: TAPIR query features are held in the tracker object (not trivially picklable), so
        even when loading a cached ModelMap we must re-run ``add_query_points`` on the stored
        per-view renders to rebuild the tracker state. This is fast relative to rendering.
        """
        if map_path and os.path.exists(map_path) and not rebuild:
            model_map = ModelMap.load(map_path)
            track_table, obj = self._reseed_tracker_from_map(model_map)
            return model_map, self.tracker, track_table, obj

        model_map, tracker, track_table, obj = self.build(intrinsics, img_size)
        if map_path:
            model_map.save(map_path)
        return model_map, tracker, track_table, obj

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _reseed_tracker_from_map(self, model_map: ModelMap):
        """Rebuild TAPIR query state and track_table/obj from a loaded ModelMap."""
        track_table = PointTrackTable.new(n0=0)
        obj = Object(id=0)
        obj.init_pose = np.eye(4)
        obj.pose = np.eye(4)

        cursor = 0
        for rgb, kps, t, T_obj2cam in zip(
            model_map.view_rgbs,
            model_map.view_kps,
            model_map.view_timestamps,
            model_map.view_poses_obj2cam,
        ):
            m = kps.shape[0]
            pts_obj = model_map.map_pts_obj[cursor: cursor + m]
            valid = model_map.map_valid[cursor: cursor + m]
            cursor += m

            render_frame = Frame(id=int(t), rgb=rgb, intrinsics=model_map.intrinsics,
                                 depth_factor=1.0)
            new_indices = self.tracker.add_query_points(render_frame, kps)
            # See build(): commit each view's queries now, or CoTracker's online predictor
            # drops all but the last add_query_points() batch once live tracking starts.
            self.tracker.track_once(render_frame)
            track_table.add_new_points_to_track_obj_maps(new_indices, obj_id=0)
            obj.add_key_points(
                new_key_points=pts_obj,
                new_uncertainties=0.1 * np.ones(m, dtype=float),
                new_valid=valid,
                new_indices=new_indices,
                frame_id=int(t),
            )

        n = obj.key_points.shape[0]
        track_table.track_3d = obj.key_points.copy()
        track_table.track_2d = np.full((n, 2), np.nan, dtype=np.float32)
        track_table.valid = model_map.map_valid.copy()
        track_table.visible = np.zeros(n, dtype=bool)
        track_table.uncertainty = 0.1 * np.ones(n, dtype=np.float32)
        self._set_object_bbox(obj)
        return track_table, obj

    def _extract_superpoint(self, rgb, render_mask, view_id):
        """Run the SuperPoint sampler on a render, restricted to the object mask."""
        import torch

        H, W = rgb.shape[:2]
        # SamplerContext.frame needs a Frame with rgb/depth/mask. The sampler filters by
        # frame.mask[obj_id, 0] and by depth. We provide a full-object mask and a dummy depth
        # of 1.0 (in-range) so the sampler's depth filter passes; real 3D comes from the render
        # depth via convert_pixel_to_world afterwards.
        mask = torch.from_numpy(render_mask.astype(np.uint8))[None, None].to(
            self.sampler.device if hasattr(self.sampler, "device") else "cuda"
        )
        dummy_depth = np.where(render_mask, 1.0, 0.0).astype(np.float32)
        frame = Frame(id=view_id, rgb=rgb, depth=dummy_depth, mask=mask,
                      intrinsics=np.eye(3), depth_factor=1.0)
        ctx = SamplerContext(frame=frame, track_table=None,
                             min_depth=self.min_depth, max_depth=self.max_depth)
        pts = self.sampler.sample(ctx, obj_id=0)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def _sample_viewpoints(self):
        """
        Camera positions on the vertices of a subdivided icosahedron (geodesic sphere) at
        radius ``render_radius``, each looking at the origin -- uniform full-sphere coverage.
        Returns a list of T_obj2cam (CV convention). ``num_views`` selects the icosphere
        subdivision level that yields at least that many vertices.
        """
        dirs = self._icosphere_dirs_for_count(self.num_views)

        # In-plane roll angles about the optical axis (evenly spaced in [0, 360)).
        roll_angles = np.linspace(0.0, 2 * np.pi, self.num_rotations, endpoint=False)

        poses = []
        for d in dirs:
            cam_pos = self.render_radius * d
            T_cam2obj = self._look_at(cam_pos, target=np.zeros(3))
            T_obj2cam = inverse_SE3(T_cam2obj)
            for theta in roll_angles:
                # Roll the camera about its own +Z (optical) axis: rotate in the camera frame,
                # i.e. pre-multiply the object->camera transform by Rz(theta).
                Rz = np.eye(4)
                c, s = np.cos(theta), np.sin(theta)
                Rz[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                poses.append(Rz @ T_obj2cam)
        print(
            f"[MapBuilder] {len(dirs)} icosahedron viewpoints x {self.num_rotations} "
            f"in-plane rotations = {len(poses)} views"
        )
        return poses

    @staticmethod
    def _standard_render_K(H, W, fov_deg):
        """Centered pinhole intrinsics for rendering the map (cx=W/2, cy=H/2)."""
        f = (H / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], dtype=np.float64)

    @staticmethod
    def _icosphere_dirs_for_count(min_count):
        """Unit vectors = vertices of a geodesic icosphere, subdivided until >= min_count."""
        import trimesh

        level = 0
        while True:
            ico = trimesh.creation.icosphere(subdivisions=level, radius=1.0)
            v = np.asarray(ico.vertices, dtype=np.float64)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if v.shape[0] >= min_count or level >= 4:
                return v
            level += 1

    @staticmethod
    def _look_at(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
        """
        Build T_cam2obj (camera pose in object frame), CV convention:
        camera +Z points from cam_pos toward target, +X right, +Y down.
        """
        forward = target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        down = np.cross(forward, right)  # CV: +Y is down
        R = np.stack([right, down, forward], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        return T

    # ---- renderer helpers (nvdiffrast textured mesh, or gsplat gaussian checkpoint) ----
    def _make_renderer(self, K, H, W):
        if self.renderer_backend == "gsplat":
            from point2pose.model_tracking.gsplat_renderer import GsplatRenderer

            assert self.gsplat_path, (
                "model_tracking.params.renderer == 'gsplat' requires gsplat_path "
                "(path to a gaussian checkpoint, e.g. kv_tracker's gaussians.pt)"
            )
            return GsplatRenderer(
                self.gsplat_path, K, H, W,
                device=self.device, model=self.gsplat_model,
                sh_degree=self.gsplat_sh_degree, depth_mode=self.gsplat_depth_mode,
            )

        from point2pose.model_tracking.nvdiffrast_renderer import NvdiffrastRenderer

        return NvdiffrastRenderer(
            self.mesh_path, K, H, W,
            device=self.device, mesh_scale=self.mesh_scale,
            envmap_path=self.envmap_path, envmap_intensity=self.envmap_intensity,
            envmap_cube_res=self.envmap_cube_res,
            envmap_ambient_term=self.envmap_ambient_term,
            specular=self.render_specular, roughness=self.render_roughness,
        )

    def _render_view(self, renderer, T_obj2cam):
        """Render one view. Returns (rgb uint8 HxWx3, depth float32 HxW metric Z)."""
        return renderer.render(T_obj2cam)

    @staticmethod
    def _delete_renderer(renderer):
        renderer.delete()

    def _render_all_views(self, poses_obj2cam, K, H, W):
        """
        Render every view to disk, then free the renderer and its GPU context before any
        TAPIR seeding runs. Renders are dumped to ``render_dir`` (RGB .png + depth .npy) and
        reloaded, so peak GPU memory is just the renderer -- not renderer + full TAPIR state.

        Returns a list of (rgb uint8, depth float32, T_obj2cam) tuples.
        """
        import cv2
        import gc
        import tempfile

        render_dir = self.render_dir or os.path.join(
            tempfile.gettempdir(), "model_tracking_renders"
        )
        os.makedirs(render_dir, exist_ok=True)

        renderer = self._make_renderer(K, H, W)
        paths = []
        try:
            for v, T in enumerate(poses_obj2cam):
                rgb, depth = self._render_view(renderer, T)
                rgb_path = os.path.join(render_dir, f"view_{v:04d}_rgb.png")
                depth_path = os.path.join(render_dir, f"view_{v:04d}_depth.npy")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(depth_path, depth)
                paths.append((rgb_path, depth_path, T))
            print(f"[MapBuilder] rendered {len(paths)} views to {render_dir}")
        finally:
            self._delete_renderer(renderer)
            # Release the nvdiffrast/CUDA context memory before TAPIR grows its state.
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        renders = []
        for rgb_path, depth_path, T in paths:
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = np.load(depth_path)
            renders.append((np.ascontiguousarray(rgb), depth.astype(np.float32), T))
        return renders
