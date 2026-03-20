from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np

from point2pose.io.lcm.data_models import NamedVecListPayload
from point2pose.io.lcm.pose_export import object_name_from_index
from point2pose.io.lcm.runtime import NamedVecListLcmSubscriber


_LIVE_MESH_RE = re.compile(r"^(obj_(\d+))_frame_\d+_kf_\d+\.(ply|obj|glb)$")
_FINAL_REALSENSE_MESH_RE = re.compile(
    r"^realsense_sdf_obj_(\d+)(?:_textured)?\.(ply|obj|glb)$"
)
_FINAL_PRED_MESH_RE = re.compile(r"^pred_mesh_obj_(\d+)(?:_textured)?\.(ply|obj|glb)$")
_SUFFIX_PRIORITY = {".ply": 0, ".obj": 1, ".glb": 2}


@dataclass(slots=True)
class MeshGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    mesh_obj: object | None = None


@dataclass(slots=True)
class ObjectSceneState:
    name: str
    bbox_pose: np.ndarray | None = None
    bbox_extents: np.ndarray | None = None
    mesh_pose: np.ndarray | None = None
    mesh_geometry: MeshGeometry | None = None
    mesh_path: Path | None = None
    mesh_mtime_ns: int | None = None
    bbox_handle_extents: np.ndarray | None = None
    mesh_handle: object | None = None
    bbox_handle: object | None = None
    axis_handle: object | None = None
    label_handle: object | None = None


def _default_viser_server_factory(host: str, port: int):
    viser = importlib.import_module("viser")
    return viser.ViserServer(host=host, port=port)


def _filter_trimesh_disconnected_components(mesh):
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = []
    if not components:
        return mesh
    return max(components, key=lambda comp: len(getattr(comp, "vertices", [])))


def _make_mesh_loader(filter_disconnected: bool = False):
    def _load(mesh_path: Path) -> MeshGeometry:
        mesh_path = Path(mesh_path)
        try:
            trimesh = importlib.import_module("trimesh")
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            if hasattr(mesh, "geometry"):
                geometries = [geom for geom in mesh.geometry.values() if geom is not None]
                if not geometries:
                    raise ValueError(f"No mesh geometry found in {mesh_path}")
                mesh = trimesh.util.concatenate(tuple(geometries))

            if filter_disconnected:
                mesh = _filter_trimesh_disconnected_components(mesh)

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                raise ValueError(f"Mesh {mesh_path} is empty")
            return MeshGeometry(vertices=vertices, faces=faces, mesh_obj=mesh)
        except Exception:
            o3d = importlib.import_module("open3d")
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if filter_disconnected and hasattr(mesh, "cluster_connected_triangles"):
                triangle_clusters, cluster_n_triangles, _cluster_area = (
                    mesh.cluster_connected_triangles()
                )
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                if triangle_clusters.size > 0 and cluster_n_triangles.size > 0:
                    keep_cluster = int(np.argmax(cluster_n_triangles))
                    remove_mask = triangle_clusters != keep_cluster
                    mesh.remove_triangles_by_mask(remove_mask)
                    mesh.remove_unreferenced_vertices()

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                raise ValueError(f"Mesh {mesh_path} is empty")
            return MeshGeometry(vertices=vertices, faces=faces, mesh_obj=None)

    return _load


def _default_mesh_loader(mesh_path: Path) -> MeshGeometry:
    return _make_mesh_loader(filter_disconnected=False)(mesh_path)


def _build_box_line_segments(extents: np.ndarray):
    ex, ey, ez = np.asarray(extents, dtype=np.float32).reshape(3)
    hx, hy, hz = 0.5 * ex, 0.5 * ey, 0.5 * ez
    corners = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    edge_indices = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int64,
    )
    points = corners[edge_indices]
    colors = np.broadcast_to(
        np.asarray([0, 255, 0], dtype=np.uint8), points.shape
    ).copy()
    return points, colors


def _mesh_sort_key(path: Path):
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    return (_SUFFIX_PRIORITY.get(path.suffix.lower(), 99), -mtime_ns, path.name)


def discover_mesh_files(mesh_watch_dir: str | Path) -> dict[str, Path]:
    mesh_dir = Path(mesh_watch_dir)
    if not mesh_dir.exists() or not mesh_dir.is_dir():
        return {}

    live_meshes: dict[str, tuple[int, Path]] = {}
    fallback_meshes: dict[str, Path] = {}
    for path in mesh_dir.iterdir():
        if not path.is_file():
            continue

        match = _LIVE_MESH_RE.match(path.name)
        if match is not None:
            obj_name = match.group(1)
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                continue
            current = live_meshes.get(obj_name, None)
            if current is None or mtime_ns > current[0]:
                live_meshes[obj_name] = (mtime_ns, path)
            continue

        match = _FINAL_REALSENSE_MESH_RE.match(path.name)
        if match is None:
            match = _FINAL_PRED_MESH_RE.match(path.name)
        if match is None:
            continue

        obj_name = object_name_from_index(int(match.group(1)))
        current = fallback_meshes.get(obj_name, None)
        if current is None or _mesh_sort_key(path) < _mesh_sort_key(current):
            fallback_meshes[obj_name] = path

    result = {name: path for name, (_mtime, path) in live_meshes.items()}
    for name, path in fallback_meshes.items():
        result.setdefault(name, path)
    return result


def _remove_handle(handle):
    if handle is None:
        return
    remove = getattr(handle, "remove", None)
    if callable(remove):
        remove()


def _set_handle_visible(handle, visible: bool):
    if handle is not None and hasattr(handle, "visible"):
        handle.visible = bool(visible)


def _apply_pose_to_handle(handle, pose: np.ndarray | None):
    if handle is None or pose is None:
        return
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 7:
        return
    if hasattr(handle, "position"):
        handle.position = pose[:3].copy()
    if hasattr(handle, "wxyz"):
        handle.wxyz = pose[3:7].copy()


def _add_mesh_handle(scene, path: str, geometry: MeshGeometry):
    if geometry.mesh_obj is not None and hasattr(scene, "add_mesh_trimesh"):
        return scene.add_mesh_trimesh(path, geometry.mesh_obj)
    if hasattr(scene, "add_mesh_simple"):
        return scene.add_mesh_simple(path, geometry.vertices, geometry.faces)
    raise AttributeError("Scene object does not support mesh rendering.")


def _add_line_segments_handle(scene, path: str, points: np.ndarray, colors: np.ndarray):
    if hasattr(scene, "add_line_segments"):
        return scene.add_line_segments(
            path,
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors),
            line_width=2.0,
        )
    raise AttributeError("Scene object does not support line rendering.")


class ViserSceneController:
    def __init__(
        self,
        scene,
        mesh_loader: Callable[[Path], MeshGeometry] | None = None,
        show_bbox: bool = True,
        show_axes: bool = True,
        show_labels: bool = True,
        show_world_frame: bool = True,
    ):
        self._scene = scene
        self._mesh_loader = mesh_loader or _default_mesh_loader
        self._show_bbox = bool(show_bbox)
        self._show_axes = bool(show_axes)
        self._show_labels = bool(show_labels)
        self._objects: dict[str, ObjectSceneState] = {}
        self._world_frame_handle = None
        self._world_label_handle = None
        if bool(show_world_frame):
            self._world_frame_handle = self._scene.add_frame("/world")
            if hasattr(self._world_frame_handle, "position"):
                self._world_frame_handle.position = np.zeros(3, dtype=np.float32)
            if hasattr(self._world_frame_handle, "wxyz"):
                self._world_frame_handle.wxyz = np.asarray(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                )
            if self._show_labels:
                self._world_label_handle = self._scene.add_label("/world/label", "world")
                if hasattr(self._world_label_handle, "position"):
                    self._world_label_handle.position = np.asarray(
                        [0.0, 0.0, 0.05], dtype=np.float32
                    )
                if hasattr(self._world_label_handle, "wxyz"):
                    self._world_label_handle.wxyz = np.asarray(
                        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )

    def _get_state(self, name: str) -> ObjectSceneState:
        name = str(name)
        state = self._objects.get(name, None)
        if state is None:
            state = ObjectSceneState(name=name)
            self._objects[name] = state
        return state

    def _object_axis_pose(self, state: ObjectSceneState) -> np.ndarray | None:
        if state.bbox_pose is not None:
            return state.bbox_pose
        return state.mesh_pose

    def _preferred_pose(self, state: ObjectSceneState) -> np.ndarray | None:
        if state.mesh_pose is not None:
            return state.mesh_pose
        return state.bbox_pose

    def _sync_bbox_handle(self, state: ObjectSceneState):
        if not self._show_bbox or state.bbox_extents is None:
            _remove_handle(state.bbox_handle)
            state.bbox_handle = None
            state.bbox_handle_extents = None
            return

        if (
            state.bbox_handle is None
            or state.bbox_handle_extents is None
            or not np.allclose(state.bbox_handle_extents, state.bbox_extents)
        ):
            _remove_handle(state.bbox_handle)
            bbox_points, bbox_colors = _build_box_line_segments(state.bbox_extents)
            state.bbox_handle = _add_line_segments_handle(
                self._scene,
                f"/objects/{state.name}/bbox_lines",
                bbox_points,
                bbox_colors,
            )
            state.bbox_handle_extents = np.asarray(
                state.bbox_extents, dtype=np.float32
            ).copy()

        _set_handle_visible(state.bbox_handle, True)
        _apply_pose_to_handle(state.bbox_handle, state.bbox_pose)

    def _sync_axis_handle(self, state: ObjectSceneState):
        if not self._show_axes:
            _remove_handle(state.axis_handle)
            state.axis_handle = None
            return

        pose = self._object_axis_pose(state)
        if pose is None:
            return

        if state.axis_handle is None:
            state.axis_handle = self._scene.add_frame(f"/objects/{state.name}/axes")
        _set_handle_visible(state.axis_handle, True)
        _apply_pose_to_handle(state.axis_handle, pose)

    def _sync_label_handle(self, state: ObjectSceneState):
        if not self._show_labels:
            _remove_handle(state.label_handle)
            state.label_handle = None
            return

        pose = self._preferred_pose(state)
        if pose is None:
            return

        if state.label_handle is None:
            state.label_handle = self._scene.add_label(
                f"/objects/{state.name}/label", state.name
            )

        label_pose = np.asarray(pose, dtype=np.float32).copy()
        if state.bbox_extents is not None:
            label_pose[2] += 0.5 * float(np.max(state.bbox_extents)) + 0.02
        _set_handle_visible(state.label_handle, True)
        _apply_pose_to_handle(state.label_handle, label_pose)

    def _sync_mesh_handle(self, state: ObjectSceneState):
        if state.mesh_geometry is None:
            return

        if state.mesh_handle is None:
            state.mesh_handle = _add_mesh_handle(
                self._scene, f"/objects/{state.name}/mesh", state.mesh_geometry
            )
        _set_handle_visible(state.mesh_handle, True)
        _apply_pose_to_handle(state.mesh_handle, state.mesh_pose)

    def update_bbox_payload(self, payload: NamedVecListPayload | None):
        if payload is None:
            return

        vecs = np.asarray(payload.vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        for name, row in zip(payload.names, vecs):
            if row.shape[0] < 10:
                continue
            state = self._get_state(name)
            state.bbox_pose = np.asarray(row[:7], dtype=np.float32).copy()
            state.bbox_extents = np.asarray(row[7:10], dtype=np.float32).copy()
            self._sync_bbox_handle(state)
            self._sync_axis_handle(state)
            self._sync_label_handle(state)

    def update_mesh_pose_payload(self, payload: NamedVecListPayload | None):
        if payload is None:
            return

        vecs = np.asarray(payload.vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        for name, row in zip(payload.names, vecs):
            if row.shape[0] < 7:
                continue
            state = self._get_state(name)
            state.mesh_pose = np.asarray(row[:7], dtype=np.float32).copy()
            _apply_pose_to_handle(state.mesh_handle, state.mesh_pose)
            self._sync_axis_handle(state)
            self._sync_label_handle(state)

    def sync_mesh_files(self, mesh_paths: dict[str, Path]):
        for name, mesh_path in mesh_paths.items():
            state = self._get_state(name)
            try:
                mtime_ns = mesh_path.stat().st_mtime_ns
            except OSError:
                continue

            if state.mesh_path == mesh_path and state.mesh_mtime_ns == mtime_ns:
                continue

            try:
                geometry = self._mesh_loader(mesh_path)
            except Exception:
                continue

            _remove_handle(state.mesh_handle)
            state.mesh_handle = None
            state.mesh_geometry = geometry
            state.mesh_path = mesh_path
            state.mesh_mtime_ns = mtime_ns
            self._sync_mesh_handle(state)
            self._sync_axis_handle(state)
            self._sync_label_handle(state)

    def sync_mesh_directory(self, mesh_watch_dir: str | Path):
        self.sync_mesh_files(discover_mesh_files(mesh_watch_dir))


class ViserLcmVisualizer:
    def __init__(
        self,
        config_path: str = "configs/pipeline/pipeline_test.yaml",
        server_factory: Callable[..., object] | None = None,
        mesh_loader: Callable[[Path], MeshGeometry] | None = None,
        filter_disconnected: bool | None = None,
    ):
        omegaconf = importlib.import_module("omegaconf")
        self.cfg = omegaconf.OmegaConf.load(config_path)
        self._lcm_cfg = self.cfg.get("lcm", {})
        self._viser_cfg = self.cfg.get("viser", {})
        self._pipeline_cfg = self.cfg.get("pipeline", {}).get("params", {})
        self._visual_cfg = self.cfg.get("visualization", {}).get("params", {})

        self._bbox_pose_channel = str(
            self._lcm_cfg.get("obj_pose_bb2world_channel", "hw_obj_pose")
        )
        self._mesh_pose_channel = str(
            self._lcm_cfg.get("obj_pose_mesh2world_channel", "hw_obj_mesh_pose")
        )
        self._sub_poll_hz = float(self._lcm_cfg.get("sub_poll_hz", 500.0))
        self._verbose = bool(self._lcm_cfg.get("verbose", False))

        self._host = str(self._viser_cfg.get("host", "0.0.0.0"))
        self._port = int(self._viser_cfg.get("port", 8080))
        debug_dir = self._pipeline_cfg.get("debug_dir", None)
        default_mesh_dir = self._pipeline_cfg.get("sdf_mesh_save_dir", None)
        if default_mesh_dir is None:
            if debug_dir:
                default_mesh_dir = str(Path(str(debug_dir)) / "sdf_mesh")
            else:
                default_mesh_dir = "./results/sdf_mesh"
        self._mesh_watch_dir = Path(
            str(self._viser_cfg.get("mesh_watch_dir", default_mesh_dir))
        )
        final_sdf_output_dir = self._visual_cfg.get("final_sdf_output_dir", None)
        if final_sdf_output_dir is None:
            self._final_mesh_watch_dir = None
        else:
            self._final_mesh_watch_dir = Path(str(final_sdf_output_dir)) / "meshes"
        self._mesh_poll_hz = max(0.1, float(self._viser_cfg.get("mesh_poll_hz", 2.0)))
        if filter_disconnected is None:
            self._filter_disconnected = bool(
                self._viser_cfg.get("filter_disconnected", False)
            )
        else:
            self._filter_disconnected = bool(filter_disconnected)
        self._show_bbox = bool(self._viser_cfg.get("show_bbox", True))
        self._show_axes = bool(self._viser_cfg.get("show_axes", True))
        self._show_labels = bool(self._viser_cfg.get("show_labels", True))
        self._show_world_frame = bool(self._viser_cfg.get("show_world_frame", True))

        self._server_factory = server_factory or _default_viser_server_factory
        self._mesh_loader = mesh_loader or _make_mesh_loader(
            filter_disconnected=self._filter_disconnected
        )
        self._bbox_subscriber = NamedVecListLcmSubscriber(
            channel=self._bbox_pose_channel,
            sub_poll_hz=self._sub_poll_hz,
            verbose=self._verbose,
        )
        self._mesh_pose_subscriber = NamedVecListLcmSubscriber(
            channel=self._mesh_pose_channel,
            sub_poll_hz=self._sub_poll_hz,
            verbose=self._verbose,
        )

    def run(self):
        server = self._server_factory(host=self._host, port=self._port)
        scene = getattr(server, "scene", server)
        if hasattr(scene, "set_up_direction"):
            scene.set_up_direction("+z")
        initial_camera = getattr(server, "initial_camera", None)
        if initial_camera is not None:
            initial_camera.look_at = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            initial_camera.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        on_client_connect = getattr(server, "on_client_connect", None)
        if callable(on_client_connect):
            @on_client_connect
            def _set_orbit_target(client):
                camera = getattr(client, "camera", None)
                if camera is None:
                    return
                if hasattr(camera, "look_at"):
                    camera.look_at = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
                if hasattr(camera, "up_direction"):
                    camera.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        controller = ViserSceneController(
            scene=scene,
            mesh_loader=self._mesh_loader,
            show_bbox=self._show_bbox,
            show_axes=self._show_axes,
            show_labels=self._show_labels,
            show_world_frame=self._show_world_frame,
        )

        self._bbox_subscriber.start()
        self._mesh_pose_subscriber.start()

        display_host = "localhost" if self._host == "0.0.0.0" else self._host
        print(f"Viser viewer listening on http://{display_host}:{self._port}")
        print(f"Watching meshes in: {self._mesh_watch_dir}")
        if self._final_mesh_watch_dir is not None:
            print(f"Using final-mesh fallback dir: {self._final_mesh_watch_dir}")
        if self._filter_disconnected:
            print("Filtering disconnected mesh components to the largest connected component.")
        print(
            f"Subscribing to bbox poses on '{self._bbox_pose_channel}' and mesh poses on '{self._mesh_pose_channel}'."
        )

        poll_period_s = 1.0 / self._mesh_poll_hz
        next_mesh_poll = 0.0

        try:
            while True:
                controller.update_bbox_payload(self._bbox_subscriber.pop_latest())
                controller.update_mesh_pose_payload(
                    self._mesh_pose_subscriber.pop_latest()
                )

                now = time.monotonic()
                if now >= next_mesh_poll:
                    mesh_paths = discover_mesh_files(self._mesh_watch_dir)
                    if (
                        self._final_mesh_watch_dir is not None
                        and self._final_mesh_watch_dir != self._mesh_watch_dir
                    ):
                        fallback_paths = discover_mesh_files(self._final_mesh_watch_dir)
                        for name, mesh_path in fallback_paths.items():
                            mesh_paths.setdefault(name, mesh_path)
                    controller.sync_mesh_files(mesh_paths)
                    next_mesh_poll = now + poll_period_s

                time.sleep(0.01)
        finally:
            self._bbox_subscriber.stop()
            self._mesh_pose_subscriber.stop()
