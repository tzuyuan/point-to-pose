import numpy as np

from point2pose.io.lcm.pose_export import (
    build_bbox_pose_vector,
    build_mesh_pose_vector,
    object_name_from_index,
)


class _DummyObject:
    def __init__(self):
        self.pose = np.eye(4)
        self.init_pose = np.eye(4)
        self.bbox = {
            "center": np.array([0.1, 0.2, 0.3], dtype=float),
            "extent": np.array([0.2, 0.5, 0.3], dtype=float),
            "rot": np.eye(3, dtype=float),
            "frame": "object",
        }
        self.init_bbox = None
        self.key_points = np.empty((0, 3), dtype=float)
        self.keyframes = []
        self.sdf = None
        self.sdf_volume = None


def test_build_bbox_pose_vector_orders_quaternion_and_sizes():
    obj = _DummyObject()
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    vec = build_bbox_pose_vector(obj, camera_to_world=camera_to_world)
    assert vec.shape == (10,)
    assert np.allclose(vec[:3], np.array([1.1, 2.2, 3.3], dtype=np.float32))
    assert np.allclose(np.abs(vec[3:7]), np.full(4, 0.5, dtype=np.float32))
    assert vec[7] >= vec[8] >= vec[9]


def test_build_bbox_pose_vector_defaults_dict_bbox_to_object_frame():
    obj = _DummyObject()
    obj.bbox = {
        "center": np.array([0.1, 0.2, 0.3], dtype=float),
        "extent": np.array([0.2, 0.5, 0.3], dtype=float),
        "rot": np.eye(3, dtype=float),
    }
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    vec = build_bbox_pose_vector(obj, camera_to_world=camera_to_world)
    assert vec.shape == (10,)
    assert np.allclose(vec[:3], np.array([1.1, 2.2, 3.3], dtype=np.float32))
    assert vec[7] >= vec[8] >= vec[9]


def test_build_bbox_pose_vector_uses_epsilon_extent_when_bbox_missing():
    obj = _DummyObject()
    obj.bbox = None
    vec = build_bbox_pose_vector(obj, camera_to_world=None, min_extent=1e-3)
    assert vec.shape == (10,)
    assert np.allclose(vec[7:], np.array([1e-3, 1e-3, 1e-3], dtype=np.float32))


def test_build_mesh_pose_vector_and_name_are_consistent():
    obj = _DummyObject()
    obj.pose[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=float)
    obj.init_pose[:3, 3] = np.array([0.4, 0.5, -0.1], dtype=float)
    camera_to_world = np.eye(4, dtype=float)
    camera_to_world[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    vec = build_mesh_pose_vector(obj, camera_to_world=camera_to_world)
    assert vec.shape == (7,)
    assert np.allclose(vec[:3], np.array([1.5, 2.3, 3.2], dtype=np.float32))
    assert np.allclose(vec[3:], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert object_name_from_index(2) == "obj_2"
