import queue
import time

import numpy as np

from point2pose.io.lcm.data_models import (
    CameraInfoPacket,
    NamedVecListPayload,
    RGBDFramePacket,
)
from point2pose.io.lcm.image_conversion import (
    pack_image_to_bytes,
    unpack_image_from_bytes,
)
from point2pose.io.lcm.messages import camera_info_t, rgbd_t, vec_list_t
from point2pose.io.lcm.runtime import (
    NamedVecListLcmPublisher,
    NamedVecListLcmSubscriber,
    RgbdLcmPublisher,
    RgbdLcmSubscriber,
)


class _FakeLcm:
    def __init__(self):
        self.published = []
        self._subscriptions = {}
        self._incoming = queue.Queue()

    def publish(self, channel, payload):
        self.published.append((channel, payload))

    def subscribe(self, channel, callback):
        self._subscriptions[channel] = callback
        return None

    def emit(self, channel, payload):
        self._incoming.put((channel, payload))

    def handle_timeout(self, timeout_ms):
        try:
            channel, payload = self._incoming.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return 0
        callback = self._subscriptions.get(channel)
        if callback is not None:
            callback(channel, payload)
        return 0


def _make_rgbd_message(value: int) -> bytes:
    rgb = np.full((1, 1, 3), value, dtype=np.uint8)
    depth = np.full((1, 1), value, dtype=np.uint16)

    msg = rgbd_t()
    msg.timestamp = float(value)
    msg.height = 1
    msg.width = 1
    msg.num_rgb_channels = 3
    msg.rgb_channel_type = rgbd_t.CHANNEL_TYPE_UINT8
    msg.rgb_image = pack_image_to_bytes(rgb, msg.rgb_channel_type)
    msg.rgb_size = len(msg.rgb_image)
    msg.depth_channel_type = rgbd_t.CHANNEL_TYPE_UINT16
    msg.depth_image = pack_image_to_bytes(depth, msg.depth_channel_type)
    msg.depth_size = len(msg.depth_image)
    return msg.encode()


def _make_vec_list_message(timestamp: float, value: float, vec_dim: int) -> bytes:
    msg = vec_list_t()
    msg.timestamp = float(timestamp)
    msg.num_vecs = 1
    msg.vec_dim = int(vec_dim)
    msg.name_list = ["obj_0"]
    msg.vec_list = [[float(value) for _ in range(vec_dim)]]
    return msg.encode()


def test_publisher_replaces_stale_payload_with_latest():
    fake = _FakeLcm()
    publisher = NamedVecListLcmPublisher(
        channel="pose", pub_hz=500.0, lcm_factory=lambda: fake
    )
    publisher.start()
    try:
        publisher.submit(
            NamedVecListPayload(
                channel="pose",
                timestamp=1.0,
                names=["obj_0"],
                vecs=np.ones((1, 10), dtype=np.float32),
            )
        )
        publisher.submit(
            NamedVecListPayload(
                channel="pose",
                timestamp=2.0,
                names=["obj_0"],
                vecs=np.full((1, 10), 2.0, dtype=np.float32),
            )
        )
    finally:
        time.sleep(0.02)
        publisher.stop()

    assert len(fake.published) >= 1


def test_subscriber_drops_stale_frames_and_keeps_newest():
    fake = _FakeLcm()
    subscriber = RgbdLcmSubscriber(
        rgbd_channel="rgbd",
        drop_stale_frames=True,
        max_frame_drain=8,
        lcm_factory=lambda: fake,
    )
    subscriber.start()
    try:
        fake.emit("rgbd", _make_rgbd_message(1))
        fake.emit("rgbd", _make_rgbd_message(2))
        time.sleep(0.02)
        packet = subscriber.pop_latest_rgbd()
    finally:
        subscriber.stop()

    assert packet is not None
    assert np.isclose(packet.timestamp, 2.0)
    assert int(packet.rgb_image[0, 0, 0]) == 2
    assert subscriber.pop_latest_rgbd() is None


def test_subscriber_keeps_recent_backlog_when_not_dropping_stale_frames():
    fake = _FakeLcm()
    subscriber = RgbdLcmSubscriber(
        rgbd_channel="rgbd",
        drop_stale_frames=False,
        max_frame_drain=2,
        lcm_factory=lambda: fake,
    )
    subscriber.start()
    try:
        fake.emit("rgbd", _make_rgbd_message(1))
        fake.emit("rgbd", _make_rgbd_message(2))
        fake.emit("rgbd", _make_rgbd_message(3))
        time.sleep(0.02)
        first = subscriber.pop_oldest_rgbd()
        second = subscriber.pop_oldest_rgbd()
    finally:
        subscriber.stop()

    assert first is not None
    assert second is not None
    assert np.isclose(first.timestamp, 2.0)
    assert np.isclose(second.timestamp, 3.0)


def test_rgbd_publisher_encodes_rgbd_packets():
    fake = _FakeLcm()
    publisher = RgbdLcmPublisher(
        rgbd_channel="rgbd",
        camera_info_channel="rgbd_info",
        lcm_factory=lambda: fake,
    )

    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.arange(4, dtype=np.uint16).reshape(2, 2)
    publisher.publish_rgbd(
        RGBDFramePacket(
            timestamp=3.5,
            height=2,
            width=2,
            num_rgb_channels=3,
            rgb_channel_type=rgbd_t.CHANNEL_TYPE_UINT8,
            depth_channel_type=rgbd_t.CHANNEL_TYPE_UINT16,
            rgb_image=rgb,
            depth_image=depth,
        )
    )

    assert len(fake.published) == 1
    channel, payload = fake.published[0]
    assert channel == "rgbd"

    decoded = rgbd_t.decode(payload)
    assert np.isclose(decoded.timestamp, 3.5)
    assert decoded.height == 2
    assert decoded.width == 2
    assert decoded.num_rgb_channels == 3
    decoded_rgb = unpack_image_from_bytes(
        decoded.rgb_image,
        height=decoded.height,
        width=decoded.width,
        num_channels=decoded.num_rgb_channels,
        channel_type=decoded.rgb_channel_type,
    )
    decoded_depth = unpack_image_from_bytes(
        decoded.depth_image,
        height=decoded.height,
        width=decoded.width,
        num_channels=1,
        channel_type=decoded.depth_channel_type,
    )
    assert np.array_equal(decoded_rgb, rgb)
    assert np.array_equal(decoded_depth, depth)


def test_rgbd_publisher_encodes_camera_info_packets():
    fake = _FakeLcm()
    publisher = RgbdLcmPublisher(
        rgbd_channel="rgbd",
        camera_info_channel="rgbd_info",
        lcm_factory=lambda: fake,
    )

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    publisher.publish_camera_info(
        CameraInfoPacket(
            timestamp=4.0,
            height=480,
            width=640,
            intrinsics=np.array(
                [[500.0, 0.0, 320.0], [0.0, 501.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            world_to_camera=world_to_camera,
            depth_factor=1000.0,
            fixed=True,
            attached_body="camera_link",
        )
    )

    assert len(fake.published) == 1
    channel, payload = fake.published[0]
    assert channel == "rgbd_info"

    decoded = camera_info_t.decode(payload)
    assert decoded.height == 480
    assert decoded.width == 640
    assert np.isclose(decoded.timestamp, 4.0)
    assert np.isclose(decoded.fx, 500.0)
    assert np.isclose(decoded.fy, 501.0)
    assert np.isclose(decoded.cx, 320.0)
    assert np.isclose(decoded.cy, 240.0)
    assert np.allclose(
        np.asarray(decoded.extrinsic, dtype=np.float32).reshape(3, 4),
        world_to_camera[:3, :].astype(np.float32),
    )
    assert np.isclose(decoded.depth_factor, 1000.0)
    assert decoded.fixed is True
    assert decoded.attached_body == "camera_link"


def test_named_vec_list_subscriber_keeps_newest_payload():
    fake = _FakeLcm()
    subscriber = NamedVecListLcmSubscriber(
        channel="pose",
        sub_poll_hz=500.0,
        lcm_factory=lambda: fake,
    )
    subscriber.start()
    try:
        fake.emit("pose", _make_vec_list_message(timestamp=1.0, value=1.0, vec_dim=7))
        fake.emit("pose", _make_vec_list_message(timestamp=2.0, value=2.0, vec_dim=10))
        time.sleep(0.02)
        payload = subscriber.pop_latest()
    finally:
        subscriber.stop()

    assert payload is not None
    assert payload.names == ["obj_0"]
    assert payload.vecs.shape == (1, 10)
    assert np.isclose(payload.timestamp, 2.0)
    assert np.allclose(payload.vecs[0], np.full(10, 2.0, dtype=np.float32))
