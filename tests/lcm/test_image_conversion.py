import numpy as np
import pytest

from point2pose.io.lcm.image_conversion import pack_image_to_bytes, unpack_image_from_bytes
from point2pose.io.lcm.messages import rgbd_t


def test_rgb_round_trip():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    payload = pack_image_to_bytes(image, rgbd_t.CHANNEL_TYPE_UINT8)
    decoded = unpack_image_from_bytes(
        payload,
        height=2,
        width=3,
        num_channels=3,
        channel_type=rgbd_t.CHANNEL_TYPE_UINT8,
    )
    assert np.array_equal(decoded, image)


def test_depth_round_trip():
    image = np.arange(6, dtype=np.uint16).reshape(2, 3)
    payload = pack_image_to_bytes(image, rgbd_t.CHANNEL_TYPE_UINT16)
    decoded = unpack_image_from_bytes(
        payload,
        height=2,
        width=3,
        num_channels=1,
        channel_type=rgbd_t.CHANNEL_TYPE_UINT16,
    )
    assert np.array_equal(decoded, image)


def test_unpack_rejects_wrong_size():
    with pytest.raises(ValueError):
        unpack_image_from_bytes(
            b"\x00" * 4,
            height=2,
            width=2,
            num_channels=3,
            channel_type=rgbd_t.CHANNEL_TYPE_UINT8,
        )
