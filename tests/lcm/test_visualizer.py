from pathlib import Path
import time

import numpy as np

from point2pose.io.lcm.data_models import NamedVecListPayload
from point2pose.io.lcm.viser_visualizer import (
    MeshGeometry,
    ViserSceneController,
    discover_mesh_files,
)


class _MockHandle:
    def __init__(self, path: str, payload=None):
        self.path = path
        self.payload = payload
        self.position = None
        self.wxyz = None
        self.visible = True
        self.removed = False

    def remove(self):
        self.removed = True


class _MockScene:
    def __init__(self):
        self.mesh_handles = {}
        self.line_handles = {}
        self.frame_handles = {}
        self.label_handles = {}

    def add_mesh_trimesh(self, path, mesh):
        handle = _MockHandle(path, payload=mesh)
        self.mesh_handles[path] = handle
        return handle

    def add_mesh_simple(self, path, vertices, faces):
        handle = _MockHandle(path, payload=(vertices, faces))
        self.mesh_handles[path] = handle
        return handle

    def add_line_segments(self, path, points, colors, line_width):
        handle = _MockHandle(path, payload=(points, colors, line_width))
        self.line_handles[path] = handle
        return handle

    def add_frame(self, path):
        handle = _MockHandle(path)
        self.frame_handles[path] = handle
        return handle

    def add_label(self, path, text):
        handle = _MockHandle(path, payload=text)
        self.label_handles[path] = handle
        return handle


def _make_mesh_geometry() -> MeshGeometry:
    return MeshGeometry(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        mesh_obj=object(),
    )


def test_discover_mesh_files_prefers_live_and_falls_back(tmp_path):
    fallback = tmp_path / "realsense_sdf_obj_1.ply"
    fallback.write_text("fallback")

    live_old = tmp_path / "obj_0_frame_1_kf_1.ply"
    live_old.write_text("old")
    live_new = tmp_path / "obj_0_frame_2_kf_1.ply"
    live_new.write_text("new")
    now_ns = time.time_ns()
    live_old.touch()
    live_new.touch()
    Path(live_old).touch()
    Path(live_new).touch()
    # Make sure the second live file wins even on coarse filesystems.
    import os

    os.utime(live_old, ns=(now_ns - 10_000_000, now_ns - 10_000_000))
    os.utime(live_new, ns=(now_ns, now_ns))

    mesh_files = discover_mesh_files(tmp_path)
    assert mesh_files["obj_0"] == live_new
    assert mesh_files["obj_1"] == fallback


def test_visualizer_controller_updates_handles_in_place_and_keeps_last_good_mesh(tmp_path):
    good_mesh = tmp_path / "obj_0_frame_1_kf_1.ply"
    good_mesh.write_text("good")
    bad_mesh = tmp_path / "obj_0_frame_2_kf_1.ply"

    def loader(path: Path):
        if path == bad_mesh:
            raise RuntimeError("reload failed")
        return _make_mesh_geometry()

    scene = _MockScene()
    controller = ViserSceneController(scene=scene, mesh_loader=loader)
    assert "/world" in scene.frame_handles
    assert "/world/label" in scene.label_handles

    bbox_payload = NamedVecListPayload(
        channel="bbox",
        timestamp=1.0,
        names=["obj_0"],
        vecs=np.asarray([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2]], dtype=np.float32),
    )
    mesh_payload = NamedVecListPayload(
        channel="mesh",
        timestamp=1.0,
        names=["obj_0"],
        vecs=np.asarray([[1.5, 2.5, 3.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    controller.update_bbox_payload(bbox_payload)
    controller.update_mesh_pose_payload(mesh_payload)
    bbox_handle_before = controller._objects["obj_0"].bbox_handle
    axis_handle_before = controller._objects["obj_0"].axis_handle
    assert bbox_handle_before.path == "/objects/obj_0/bbox_lines"

    controller.sync_mesh_directory(tmp_path)
    state = controller._objects["obj_0"]
    mesh_handle_before = state.mesh_handle
    assert mesh_handle_before is not None
    assert np.allclose(mesh_handle_before.position, np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert np.allclose(state.axis_handle.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert state.mesh_path == good_mesh

    mesh_payload_2 = NamedVecListPayload(
        channel="mesh",
        timestamp=2.0,
        names=["obj_0"],
        vecs=np.asarray([[2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    controller.update_mesh_pose_payload(mesh_payload_2)
    state = controller._objects["obj_0"]
    assert state.mesh_handle is mesh_handle_before
    assert state.axis_handle is axis_handle_before
    assert state.bbox_handle is bbox_handle_before
    assert np.allclose(state.mesh_handle.position, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(state.axis_handle.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    bad_mesh.write_text("bad")
    controller.sync_mesh_directory(tmp_path)
    state = controller._objects["obj_0"]
    assert state.mesh_handle is mesh_handle_before
    assert state.mesh_path == good_mesh
