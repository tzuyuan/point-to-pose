import numpy as np

from point2pose.io.lcm.messages import camera_info_t, rgbd_t, vec_list_t


def test_rgbd_t_round_trip():
    msg = rgbd_t()
    msg.timestamp = 12.5
    msg.height = 2
    msg.width = 3
    msg.num_rgb_channels = 3
    msg.rgb_channel_type = rgbd_t.CHANNEL_TYPE_UINT8
    msg.rgb_image = bytes(range(18))
    msg.rgb_size = len(msg.rgb_image)
    msg.depth_channel_type = rgbd_t.CHANNEL_TYPE_UINT16
    msg.depth_image = np.arange(6, dtype=np.uint16).tobytes()
    msg.depth_size = len(msg.depth_image)

    decoded = rgbd_t.decode(msg.encode())
    assert decoded.timestamp == msg.timestamp
    assert decoded.height == msg.height
    assert decoded.width == msg.width
    assert decoded.num_rgb_channels == msg.num_rgb_channels
    assert decoded.rgb_channel_type == msg.rgb_channel_type
    assert decoded.rgb_image == msg.rgb_image
    assert decoded.depth_channel_type == msg.depth_channel_type
    assert decoded.depth_image == msg.depth_image


def test_camera_info_t_round_trip():
    msg = camera_info_t()
    msg.height = 480
    msg.width = 640
    msg.timestamp = 1.5
    msg.fx = 500.0
    msg.fy = 501.0
    msg.cx = 320.0
    msg.cy = 240.0
    msg.extrinsic = tuple(float(i) for i in range(12))
    msg.depth_factor = 1000.0
    msg.fixed = True
    msg.attached_body = "camera_link"

    decoded = camera_info_t.decode(msg.encode())
    assert decoded.height == msg.height
    assert decoded.width == msg.width
    assert decoded.fx == msg.fx
    assert decoded.fy == msg.fy
    assert decoded.cx == msg.cx
    assert decoded.cy == msg.cy
    assert tuple(decoded.extrinsic) == tuple(msg.extrinsic)
    assert decoded.depth_factor == msg.depth_factor
    assert decoded.fixed is True
    assert decoded.attached_body == msg.attached_body


def test_vec_list_t_round_trip():
    msg = vec_list_t()
    msg.timestamp = 7.0
    msg.num_vecs = 2
    msg.vec_dim = 3
    msg.name_list = ["obj_0", "obj_1"]
    msg.vec_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    decoded = vec_list_t.decode(msg.encode())
    assert decoded.timestamp == msg.timestamp
    assert decoded.num_vecs == msg.num_vecs
    assert decoded.vec_dim == msg.vec_dim
    assert decoded.name_list == msg.name_list
    assert decoded.vec_list == [tuple(row) for row in msg.vec_list]
