"""Rerun-based UI — the viewer style used by kv_tracker and many recent
tracking/SLAM demos.

Follows the entity conventions of the ``model_based_tracking`` branch's
reference viewer (``point2pose/utils/rerun_viz.py`` by the project's
collaborator): ``world/obj_<i>`` for the object-fixed stage, ``camframe`` for
the camera-fixed stage, per-track-id point colors with dimming for currently
untracked points, and per-point trace buffers keyed by global track id.

On top of that reference this adds: the SDF mesh (logged at the frame it
changed, so scrubbing shows the reconstruction grow), keyframe frustums with
RGB thumbnails, the live RGB overlay as both frustum texture and 2D panel,
health time-series (residual / inliers / tracked points / FPS), keyframe and
lost-object event logs, a curated blueprint layout, and optional ``.rrd``
session recording for offline replay.

Everything rides rerun's ``frame`` timeline: the viewer's built-in scrubber
replays the whole session, and the entity sidebar gives per-element
visibility toggles for free.
"""

import time
from collections import deque

import numpy as np

from point2pose.visualization.geometry import (
    frame_id_colors,
    inlier_colors,
    object_color,
    track_id_colors,
    uncertainty_colors,
)

_POINT_COLOR_MODES = ["track_id", "inlier", "frame_id", "uncertainty", "object"]
_UNLIT_DIM = 0.18  # brightness of map points that are not currently visible

_TRAJ_COLOR = (60, 130, 255)
_CAM_COLOR = (26, 255, 90)
_LOST_COLOR = (255, 60, 60)
_KF_COLOR = (255, 140, 50)

# Display-side recovery evidence: the pipeline's lost flag only clears once a
# clean f2m registration passes the pose-jump guard, which can lag (or stall
# while inlier support stays weak). Strong current evidence shows green.
_RECOVERED_MIN_INLIERS = 5
_RECOVERED_MAX_RESIDUAL = 0.02  # meters
_BBOX_COLOR = (255, 200, 0)


def _u8(colors01):
    return (np.asarray(colors01) * 255.0).clip(0, 255).astype(np.uint8)


class _TraceBuffer:
    """Rolling per-point history keyed by global track id (mirrors the
    reference viewer's PointTraceBuffer): only visible points get samples,
    and a trace is dropped once its point hasn't been seen for ``max_gap``
    frames so stale trails don't linger."""

    def __init__(self, max_len=30, max_gap=5):
        self.max_len = max_len
        self.max_gap = max_gap
        self._traces = {}  # track_id -> deque[(3,) float32]
        self._last_seen = {}  # track_id -> frame idx

    def update(self, frame_idx, track_ids, pts_cam):
        for tid, p in zip(track_ids, pts_cam):
            tid = int(tid)
            trace = self._traces.setdefault(tid, deque(maxlen=self.max_len))
            if trace.maxlen != self.max_len:
                trace = deque(trace, maxlen=self.max_len)
                self._traces[tid] = trace
            trace.append(np.asarray(p, dtype=np.float32))
            self._last_seen[tid] = frame_idx
        stale = [t for t, last in self._last_seen.items() if frame_idx - last > self.max_gap]
        for tid in stale:
            self._traces.pop(tid, None)
            self._last_seen.pop(tid, None)

    def strips_with_ids(self, min_len=2):
        ids, strips = [], []
        for tid, pts in self._traces.items():
            if len(pts) >= min_len:
                ids.append(tid)
                strips.append(np.stack(list(pts)))
        return np.asarray(ids, dtype=np.int64), strips


class RerunDashboard:
    """Owns the rerun recording stream and mirrors pipeline state into it."""

    def __init__(self, cfg):
        import rerun as rr

        self._rr = rr
        self.cfg = cfg
        rcfg = cfg.rerun

        rr.init(str(rcfg.app_name), spawn=bool(rcfg.spawn))
        if rcfg.save_rrd:
            rr.save(str(rcfg.save_rrd))
            print(f"[viz3d] recording session to {rcfg.save_rrd}")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("camframe", rr.ViewCoordinates.RDF, static=True)  # OpenCV
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )
        # class colors/labels for the SAM2 segmentation overlay
        rr.log(
            "/",
            rr.AnnotationContext(
                [
                    rr.AnnotationInfo(
                        id=i + 1, label=f"obj{i}", color=_u8(object_color(i))
                    )
                    for i in range(8)
                ]
            ),
            static=True,
        )

        # metric series styling (legend names, colors, widths)
        for path, color, name in (
            ("metrics/residual_mm", (240, 90, 90), "residual [mm]"),
            ("metrics/inliers", (90, 200, 250), "inliers"),
            ("metrics/tracked", (250, 200, 90), "tracked pts"),
            ("metrics/fps", (90, 240, 130), "fps"),
        ):
            rr.log(path, rr.SeriesLines(colors=[color], widths=[2.0], names=[name]), static=True)

        self._anchor = int(cfg.object_view.obj_id)

        # show/hide toggles, driven by the control strip on the cv2 window
        # (blueprint-level: toggling affects the whole timeline, not the log).
        # Initial states come from cfg.rerun.show.
        self._show = {key: bool(value) for key, value in dict(rcfg.show).items()}
        self._button_rects = []  # filled by attach_controls
        self._obj_ids = set()

        # runtime state
        self._traces = {}  # obj_id -> _TraceBuffer
        self._traj = {}  # obj_id -> list of camera-center segments
        self._traj_break_pending = {}  # obj_id -> break trajectory on recovery
        self._kf_counts = {}  # obj_id -> keyframes seen so far
        self._lost_state = {}  # obj_id -> bool
        self._fps = None
        self._last_t = None

        self._send_blueprint()

    # ------------------------------------------------------------------
    # blueprint (layout + show/hide state)
    # ------------------------------------------------------------------
    def _blueprint_contents(self):
        """Entity-query contents for each view, honoring the toggles."""
        world = ["+ $origin/**"]
        cam = ["+ $origin/**"]
        img = ["+ $origin/**"]
        for oid in sorted(self._obj_ids) or [self._anchor]:
            if not self._show["map"]:
                world.append(f"- /world/obj_{oid}/map_points")
            if not self._show["mesh"]:
                world.append(f"- /world/obj_{oid}/mesh")
            if not self._show["kfs"]:
                world.append(f"- /world/obj_{oid}/keyframes/**")
            if not self._show["traj"]:
                world.append(f"- /world/obj_{oid}/trajectory")
            if not self._show["bbox"]:
                world.append(f"- /world/obj_{oid}/bbox")
                cam.append(f"- /camframe/obj_{oid}/bbox")
            if not self._show["traces"]:
                cam.append(f"- /camframe/obj_{oid}/traces")
        image_root = f"/world/obj_{self._anchor}/camera/image"
        for key, leaf in (("2d", "tracks"), ("mask", "mask"), ("reproj", "reproj")):
            if not self._show[key]:
                img.append(f"- {image_root}/{leaf}")
                world.append(f"- {image_root}/{leaf}")  # also on the frustum
        return world, cam, img

    def _send_blueprint(self):
        import rerun.blueprint as rrb

        world, cam, img = self._blueprint_contents()
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world", name="Object frame · map", contents=world
                ),
                rrb.Vertical(
                    rrb.Spatial3DView(
                        origin="camframe", name="Camera frame · trails", contents=cam
                    ),
                    rrb.Horizontal(
                        rrb.Tabs(
                            rrb.Spatial2DView(
                                origin=f"world/obj_{self._anchor}/camera/image",
                                name="RGB",
                                contents=img,
                            ),
                            rrb.TextLogView(origin="events", name="Events"),
                        ),
                        # residual gets its own pinned y-axis so one spike
                        # (or the larger count series) can't drown it out
                        rrb.Vertical(
                            rrb.TimeSeriesView(
                                origin="metrics",
                                contents=["+ /metrics/residual_mm"],
                                name="Residual [mm]",
                                axis_y=rrb.ScalarAxis(
                                    range=(
                                        -1.0,
                                        1.05 * float(self.cfg.rerun.residual_cap_mm),
                                    ),
                                    zoom_lock=False,
                                ),
                            ),
                            rrb.TimeSeriesView(
                                origin="metrics",
                                contents=[
                                    "+ /metrics/inliers",
                                    "+ /metrics/tracked",
                                    "+ /metrics/fps",
                                ],
                                name="Tracking",
                            ),
                        ),
                    ),
                    row_shares=[3, 2],
                ),
                column_shares=[5, 3],
            ),
            collapse_panels=bool(self.cfg.rerun.collapse_panels),
        )
        self._rr.send_blueprint(blueprint)

    # ------------------------------------------------------------------
    def update(
        self, scene, obj_snap, mesh, overlay_bgr, K, w, h, new_kf_thumbs,
        mask_classes=None,
    ):
        """Log the current pipeline state onto the ``frame`` timeline.

        ``mesh`` (object-frame Open3D mesh) is only passed when it changed;
        ``new_kf_thumbs`` is a list of (kf_idx, rgb_thumb) for keyframes of
        the anchored object created since the last call. ``mask_classes`` is
        an (h', w') uint8 class image (0 = background, i+1 = object i) at the
        same resolution as ``overlay_bgr``.
        """
        rr = self._rr
        now = time.time()
        if self._last_t is not None:
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._fps = inst if self._fps is None else 0.9 * self._fps + 0.1 * inst
        self._last_t = now

        rr.set_time("frame", sequence=int(scene.frame_id))

        rgb = None
        if overlay_bgr is not None:
            rgb = np.ascontiguousarray(overlay_bgr[..., ::-1])

        # a new object id may need existing show/hide rules applied to it
        new_ids = {s.obj_id for s in scene.objects} - self._obj_ids
        if new_ids:
            self._obj_ids |= new_ids
            if not all(self._show.values()):
                self._send_blueprint()

        anchor = self._anchor
        for snap in scene.objects:
            self._log_world(
                snap,
                rgb if snap.obj_id == anchor else None,
                K, w, h,
                mask_classes if snap.obj_id == anchor else None,
            )
            self._log_camframe(snap, scene.frame_id)
            self._log_events(snap, scene.frame_id)
        if obj_snap is not None:
            if mesh is not None:
                self._log_mesh(obj_snap.obj_id, mesh)
            for kf_idx, thumb in new_kf_thumbs:
                self._log_keyframe_thumb(obj_snap.obj_id, kf_idx, thumb)
            self._log_metrics(obj_snap)

        # camera-fixed sensor frustum (shared by all objects)
        rr.log("camframe/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
        rr.log(
            "camframe/camera",
            rr.Pinhole(
                image_from_camera=K, width=w, height=h,
                image_plane_distance=float(self.cfg.camera_view.frustum_scale),
                color=(150, 160, 180),
            ),
        )

    # ------------------------------------------------------------------
    # world (object-fixed) stage
    # ------------------------------------------------------------------
    def _log_world(self, snap, rgb, K, w, h, mask_classes=None):
        rr = self._rr
        ns = f"world/obj_{snap.obj_id}"
        rcfg = self.cfg.rerun

        # keypoint map
        if len(snap.map_points):
            rr.log(
                f"{ns}/map_points",
                rr.Points3D(
                    snap.map_points.astype(np.float32),
                    colors=self._map_colors(snap),
                    radii=float(rcfg.point_radius),
                ),
            )

        # live camera frustum (+ RGB texture for the anchored object).
        # The pinhole is logged at the *logged image's* scale so that 2D
        # annotations land exactly on the image in both the 2D view and on
        # the frustum plane in 3D.
        scale = rgb.shape[1] / float(w) if rgb is not None else 1.0
        rr.log(
            f"{ns}/camera",
            rr.Transform3D(
                translation=snap.T_obj_cam[:3, 3], mat3x3=snap.T_obj_cam[:3, :3]
            ),
        )
        rr.log(
            f"{ns}/camera",
            rr.Pinhole(
                image_from_camera=self._scaled_K(K, scale),
                width=int(round(w * scale)),
                height=int(round(h * scale)),
                image_plane_distance=1.3 * float(self.cfg.object_view.frustum_scale),
                color=_CAM_COLOR if self._tracking_ok(snap) else _LOST_COLOR,
            ),
        )
        if rgb is not None:
            img = rr.Image(rgb)
            try:
                img = img.compress(jpeg_quality=75)
            except Exception:
                pass
            rr.log(f"{ns}/camera/image", img)
            self._log_image_annotations(snap, K, scale, mask_classes, ns)

        # trajectory through camera centers (full history each frame, so the
        # timeline scrubber shows the trajectory as of that frame). The
        # history pauses while the object is flagged lost and resumes in a
        # NEW segment on recovery, so a re-acquired teleport is not drawn as
        # travelled path; all segments share one color.
        segments = self._traj_for(snap)
        strips = [np.asarray(s, dtype=np.float32) for s in segments if len(s) >= 2]
        if strips:
            rr.log(
                f"{ns}/trajectory",
                rr.LineStrips3D(strips, colors=[_TRAJ_COLOR] * len(strips)),
            )

        # bbox fixed at the origin of the object frame
        if snap.bbox_extent is not None:
            rr.log(
                f"{ns}/bbox",
                rr.Boxes3D(
                    half_sizes=[0.5 * snap.bbox_extent.astype(np.float32)],
                    centers=[[0.0, 0.0, 0.0]],
                    colors=[_BBOX_COLOR],
                ),
            )

        # keyframe frustum poses (kept fresh: global opt updates them)
        kf_scale = 1.0
        if bool(rcfg.keyframe_images):
            kf_scale = int(rcfg.keyframe_image_width) / float(w)
        for kf_idx, T in enumerate(snap.keyframe_T_obj_cam):
            path = f"{ns}/keyframes/kf_{kf_idx}"
            rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            if kf_idx >= self._kf_counts.get(snap.obj_id, 0):
                rr.log(
                    path,
                    rr.Pinhole(
                        image_from_camera=self._scaled_K(K, kf_scale),
                        width=int(round(w * kf_scale)),
                        height=int(round(h * kf_scale)),
                        image_plane_distance=0.55 * float(self.cfg.object_view.frustum_scale),
                        color=_KF_COLOR,
                    ),
                )

    @staticmethod
    def _tracking_ok(snap):
        """Green/red state for the camera frustum: trust the pipeline's lost
        flag, but treat strong current registration evidence as recovered
        even while the (jump-guard-gated) flag lags behind."""
        if not snap.lost:
            return True
        return (
            snap.num_inliers >= _RECOVERED_MIN_INLIERS
            and 0.0 < snap.mean_residual < _RECOVERED_MAX_RESIDUAL
        )

    @staticmethod
    def _scaled_K(K, scale):
        sK = np.asarray(K, dtype=np.float64).copy()
        sK[:2, :] *= scale
        return sK

    def _log_image_annotations(self, snap, K, scale, mask_classes, ns):
        """2D overlays on the logged camera image: tracked points (track-id
        colors), the SAM2 segmentation mask, and reprojection whiskers from
        each map point's predicted pixel to its tracked observation."""
        rr = self._rr
        base = f"{ns}/camera/image"

        good = np.isfinite(snap.track_points_cam).all(axis=1)
        if good.any() and len(snap.track_points_2d) == len(good):
            rr.log(
                f"{base}/tracks",
                rr.Points2D(
                    (snap.track_points_2d[good] * scale).astype(np.float32),
                    colors=self._live_point_colors(snap, good),
                    radii=3.0 * scale,
                ),
            )

        if mask_classes is not None:
            rr.log(f"{base}/mask", rr.SegmentationImage(mask_classes))

        strips, colors = self._reprojection_whiskers(snap, K, good)
        if strips:
            rr.log(
                f"{base}/reproj",
                rr.LineStrips2D(
                    [(s * scale).astype(np.float32) for s in strips],
                    colors=colors,
                    radii=1.2 * scale,
                ),
            )

    def _reprojection_whiskers(self, snap, K, good):
        """Predicted (map point through current pose) vs observed (tracked
        2D) pixel pairs, colored green -> red by pixel error."""
        if len(snap.map_points) == 0 or len(snap.track_points_2d) != len(good):
            return [], None

        # map point track id -> row in the track arrays
        order = np.argsort(snap.track_ids)
        sorted_ids = snap.track_ids[order]
        pos = np.searchsorted(sorted_ids, snap.map_point_track_ids)
        pos = np.clip(pos, 0, max(len(sorted_ids) - 1, 0))
        matched = (
            (len(sorted_ids) > 0)
            & (sorted_ids[pos] == snap.map_point_track_ids)
            & (snap.map_point_track_ids >= 0)
        )
        rows = order[pos]
        keep = matched & good[rows]
        if not keep.any():
            return [], None

        P = snap.map_points[keep] @ snap.T_cam_obj[:3, :3].T + snap.T_cam_obj[:3, 3]
        z = P[:, 2]
        front = z > 1e-6
        if not front.any():
            return [], None
        P, z = P[front], z[front]
        uv_pred = (P @ np.asarray(K, dtype=np.float64).T)[:, :2] / z[:, None]
        uv_obs = snap.track_points_2d[rows[keep]][front]

        err = np.linalg.norm(uv_pred - uv_obs, axis=1)
        colors = _u8(uncertainty_colors(err, 0.0, 15.0))  # green<2px .. red>15px
        strips = [np.stack([p, o]) for p, o in zip(uv_pred, uv_obs)]
        return strips, colors

    def _log_mesh(self, obj_id, mesh):
        rr = self._rr
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        kwargs = {}
        colors = np.asarray(mesh.vertex_colors)
        if len(colors) == len(vertices) and len(colors) > 0:
            kwargs["vertex_colors"] = _u8(colors)
        normals = np.asarray(mesh.vertex_normals)
        if len(normals) == len(vertices) and len(normals) > 0:
            kwargs["vertex_normals"] = normals.astype(np.float32)
        rr.log(
            f"world/obj_{obj_id}/mesh",
            rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, **kwargs),
        )

    def _log_keyframe_thumb(self, obj_id, kf_idx, thumb):
        if thumb is None:
            return
        rr = self._rr
        img = rr.Image(np.ascontiguousarray(thumb))
        try:
            img = img.compress(jpeg_quality=70)
        except Exception:
            pass
        # logged once, at the keyframe's creation frame on the timeline
        rr.log(f"world/obj_{obj_id}/keyframes/kf_{kf_idx}/image", img)

    # ------------------------------------------------------------------
    # camframe (camera-fixed) stage
    # ------------------------------------------------------------------
    def _log_camframe(self, snap, frame_id):
        rr = self._rr
        ns = f"camframe/obj_{snap.obj_id}"
        rcfg = self.cfg.rerun

        good = np.isfinite(snap.track_points_cam).all(axis=1)
        pts = snap.track_points_cam[good]
        ids = snap.track_ids[good] if len(snap.track_ids) == len(good) else np.arange(good.sum())

        # the FULL keypoint map in the current camera frame; currently
        # visible points light up, the rest stay dimmed
        map_pts, map_colors = self._map_panel_points(snap, good)
        if map_pts is not None:
            rr.log(
                f"{ns}/points",
                rr.Points3D(
                    map_pts,
                    colors=map_colors,
                    radii=float(rcfg.point_radius),
                ),
            )

        # fading per-point traces, keyed by global track id
        buf = self._traces.setdefault(
            snap.obj_id,
            _TraceBuffer(int(rcfg.trail_length), int(rcfg.trail_max_gap)),
        )
        buf.max_len = int(rcfg.trail_length)
        buf.update(frame_id, ids, pts)
        strip_ids, strips = buf.strips_with_ids()
        if strips:
            rr.log(
                f"{ns}/traces",
                rr.LineStrips3D(
                    strips,
                    colors=self._trace_colors(snap, strip_ids),
                    radii=0.35 * float(rcfg.point_radius),
                ),
            )

        # object bbox + axes at its current pose in the camera frame
        if snap.bbox_extent is not None:
            rr.log(
                f"{ns}/bbox",
                rr.Transform3D(
                    translation=snap.T_cam_obj[:3, 3], mat3x3=snap.T_cam_obj[:3, :3]
                ),
            )
            rr.log(
                f"{ns}/bbox",
                rr.Boxes3D(
                    half_sizes=[0.5 * snap.bbox_extent.astype(np.float32)],
                    centers=[[0.0, 0.0, 0.0]],
                    colors=[_u8(object_color(snap.obj_id))],
                ),
            )

    # ------------------------------------------------------------------
    # metrics + events
    # ------------------------------------------------------------------
    def _log_metrics(self, snap):
        rr = self._rr
        # clipped at the cap so a lost-frame spike pegs at the plot ceiling
        # instead of stretching the whole residual history flat
        cap = float(self.cfg.rerun.residual_cap_mm)
        rr.log("metrics/residual_mm", rr.Scalars(min(snap.mean_residual * 1000.0, cap)))
        if snap.num_inliers >= 0:
            rr.log("metrics/inliers", rr.Scalars(float(snap.num_inliers)))
        n_vis = int(np.isfinite(snap.track_points_cam).all(axis=1).sum())
        rr.log("metrics/tracked", rr.Scalars(float(n_vis)))
        if self._fps is not None:
            rr.log("metrics/fps", rr.Scalars(self._fps))

    def _log_events(self, snap, frame_id):
        rr = self._rr
        n_kf = len(snap.keyframe_T_obj_cam)
        seen = self._kf_counts.get(snap.obj_id, 0)
        if n_kf > seen and seen > 0:  # skip the burst at initialization
            rr.log(
                "events",
                rr.TextLog(
                    f"obj {snap.obj_id}: keyframe {n_kf - 1} created (frame {frame_id})",
                    level="INFO",
                ),
            )
        self._kf_counts[snap.obj_id] = max(seen, n_kf)

        was_lost = self._lost_state.get(snap.obj_id, False)
        if snap.lost != was_lost:
            level = "ERROR" if snap.lost else "INFO"
            what = "LOST" if snap.lost else "re-acquired"
            rr.log(
                "events",
                rr.TextLog(f"obj {snap.obj_id}: {what} (frame {frame_id})", level=level),
            )
        self._lost_state[snap.obj_id] = snap.lost

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _traj_for(self, snap):
        """Camera-center history as a list of segments, following the same
        rule as the camera frustum color (_tracking_ok): frozen while lost,
        resuming in a new segment on recovery — where recovery is either the
        lost flag clearing or strong current registration evidence (so a
        stuck flag can't freeze the line forever). All other motion, however
        fast, stays connected."""
        segments = self._traj.setdefault(snap.obj_id, [[]])
        if not self._tracking_ok(snap):
            self._traj_break_pending[snap.obj_id] = True
            return segments

        center = snap.T_obj_cam[:3, 3]
        seg = segments[-1]
        if seg and self._traj_break_pending.pop(snap.obj_id, False):
            segments.append([center.copy()])
        elif not seg:
            seg.append(center.copy())
        elif np.linalg.norm(center - seg[-1]) > 1e-9:
            seg.append(center.copy())

        total = sum(len(s) for s in segments)
        while total > int(self.cfg.object_view.max_trajectory):
            segments[0].pop(0)
            if not segments[0]:
                segments.pop(0)
            total -= 1
        return segments

    def _map_colors(self, snap):
        mode = str(self.cfg.rerun.map_color_mode)
        if mode == "frame_id":
            return _u8(frame_id_colors(snap.map_point_frames))
        if mode == "uncertainty":
            return _u8(uncertainty_colors(snap.map_point_uncertainties))
        if mode == "object":
            return _u8(object_color(snap.obj_id))
        return _u8(track_id_colors(snap.map_point_track_ids, snap.map_point_visible))

    @staticmethod
    def _rows_for_ids(ids, query):
        """Row index in the track arrays for each queried global track id
        (-1 where the id is not present)."""
        if len(ids) == 0 or len(query) == 0:
            return np.full(len(query), -1, dtype=np.int64)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        pos = np.clip(np.searchsorted(sorted_ids, query), 0, len(sorted_ids) - 1)
        return np.where(sorted_ids[pos] == query, order[pos], -1)

    def _trace_colors(self, snap, strip_ids):
        """Per-trace colors following point_color_mode; a trace takes its
        point's *current* value (e.g. this frame's inlier/outlier verdict)."""
        mode = str(self.cfg.rerun.point_color_mode)
        if mode == "object":
            return np.tile(_u8(object_color(snap.obj_id)), (len(strip_ids), 1))
        if mode == "track_id":
            return _u8(track_id_colors(strip_ids))

        rows = self._rows_for_ids(snap.track_ids, strip_ids)
        found = rows >= 0
        if mode == "uncertainty":
            vals = np.zeros(len(strip_ids))
            vals[found] = snap.track_point_uncertainties[rows[found]]
            return _u8(uncertainty_colors(vals))
        if mode == "frame_id":
            fids = np.zeros(len(strip_ids), dtype=np.int64)
            fids[found] = snap.track_point_frames[rows[found]]
            return _u8(frame_id_colors(fids))
        # inlier (default): unknown ids stay -1 -> gray
        status = np.full(len(strip_ids), -1, dtype=np.int8)
        if len(snap.track_point_inlier) == len(snap.track_ids):
            status[found] = snap.track_point_inlier[rows[found]]
        return _u8(inlier_colors(status))

    def _map_panel_points(self, snap, good):
        """All map keypoints in the current camera frame, with colors lit for
        currently visible points and dimmed otherwise.

        Visible points with a fresh measurement are drawn at their *tracked*
        3D position (so trails connect exactly); the rest sit at the map
        position predicted through the current pose."""
        if len(snap.map_points) == 0:
            return None, None
        pts = (
            snap.map_points @ snap.T_cam_obj[:3, :3].T + snap.T_cam_obj[:3, 3]
        ).astype(np.float32)

        rows = self._rows_for_ids(snap.track_ids, snap.map_point_track_ids)
        observed = rows >= 0
        if len(good) == len(snap.track_ids):
            observed[observed] = good[rows[observed]]
            pts[observed] = snap.track_points_cam[rows[observed]].astype(np.float32)

        vis = snap.map_point_visible
        mode = str(self.cfg.rerun.point_color_mode)
        if mode == "inlier":
            status = np.full(len(pts), -1, dtype=np.int8)
            ok = rows >= 0
            if len(snap.track_point_inlier) == len(snap.track_ids):
                status[ok] = snap.track_point_inlier[rows[ok]]
            colors = inlier_colors(status)
        elif mode == "frame_id":
            colors = frame_id_colors(snap.map_point_frames)
        elif mode == "uncertainty":
            colors = uncertainty_colors(snap.map_point_uncertainties)
        elif mode == "object":
            colors = np.tile(object_color(snap.obj_id), (len(pts), 1))
        else:  # track_id (default): stable hue per point, built-in dimming
            return pts, _u8(track_id_colors(snap.map_point_track_ids, visible=vis))

        colors = colors.copy()
        colors[~vis] *= _UNLIT_DIM
        return pts, _u8(colors)

    def _live_point_colors(self, snap, good):
        mode = str(self.cfg.rerun.point_color_mode)
        if mode == "inlier" and len(snap.track_point_inlier) == len(good):
            return _u8(inlier_colors(snap.track_point_inlier[good]))
        if mode == "frame_id":
            return _u8(frame_id_colors(snap.track_point_frames[good]))
        if mode == "uncertainty":
            return _u8(uncertainty_colors(snap.track_point_uncertainties[good]))
        if mode == "object":
            return np.tile(_u8(object_color(snap.obj_id)), (int(good.sum()), 1))
        ids = snap.track_ids[good] if len(snap.track_ids) == len(good) else np.arange(good.sum())
        return _u8(track_id_colors(ids))

    # ------------------------------------------------------------------
    # control strip (drawn onto the demo's cv2 window; clicks re-send the
    # blueprint, so toggles apply across the whole timeline instantly)
    # ------------------------------------------------------------------
    _STRIP_H = 34

    def attach_controls(self, canvas_bgr):
        """Prepend a clickable show/hide button strip to a BGR canvas."""
        import cv2

        strip = np.full((self._STRIP_H, canvas_bgr.shape[1], 3), (30, 27, 24), np.uint8)
        self._button_rects = []
        x = 6
        for key in self._show:
            on = self._show[key]
            (tw, _), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bw = tw + 14
            y1, y2 = 5, self._STRIP_H - 5
            cv2.rectangle(strip, (x, y1), (x + bw, y2),
                          (46, 60, 44) if on else (38, 36, 34), -1)
            cv2.rectangle(strip, (x, y1), (x + bw, y2),
                          (110, 190, 120) if on else (70, 66, 60), 1)
            cv2.putText(strip, key, (x + 7, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (215, 235, 215) if on else (125, 120, 112), 1, cv2.LINE_AA)
            self._button_rects.append((x, y1, x + bw, y2, key))
            x += bw + 5

        # point-color-mode cycler (data-level: applies from the next frame on)
        label = f"pts:{self.cfg.rerun.point_color_mode}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bw = tw + 14
        y1, y2 = 5, self._STRIP_H - 5
        cv2.rectangle(strip, (x, y1), (x + bw, y2), (66, 50, 38), -1)
        cv2.rectangle(strip, (x, y1), (x + bw, y2), (190, 150, 100), 1)
        cv2.putText(strip, label, (x + 7, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 210, 180), 1, cv2.LINE_AA)
        self._button_rects.append((x, y1, x + bw, y2, "__ptmode__"))
        return np.vstack([strip, canvas_bgr])

    def handle_mouse(self, event, x, y, _flags=0):
        """Toggle a button under a left click (host window coordinates)."""
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, key in self._button_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "__ptmode__":
                    mode = str(self.cfg.rerun.point_color_mode)
                    idx = (
                        _POINT_COLOR_MODES.index(mode)
                        if mode in _POINT_COLOR_MODES
                        else 0
                    )
                    self.cfg.rerun.point_color_mode = _POINT_COLOR_MODES[
                        (idx + 1) % len(_POINT_COLOR_MODES)
                    ]
                else:
                    self._show[key] = not self._show[key]
                    self._send_blueprint()
                return

    # ------------------------------------------------------------------
    def reset(self):
        """Clear the live scene (a recursive Clear on both stages) and all
        internal buffers; the viewer and timeline history stay."""
        rr = self._rr
        for root in ("world", "camframe"):
            rr.log(root, rr.Clear(recursive=True))
        self._traces = {}
        self._traj = {}
        self._traj_break_pending = {}
        self._kf_counts = {}
        self._lost_state = {}
        self._fps = None
        self._last_t = None

    def close(self):
        try:
            self._rr.flush(blocking=True)
        except Exception:
            pass
