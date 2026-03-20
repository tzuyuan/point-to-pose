"""LCM type definitions.

This module intentionally mirrors the `vision.rgbd_t` wire format used by the
existing external publisher so this repo can decode messages without importing
Manipulator-Software.
"""

from io import BytesIO
import struct


class rgbd_t(object):
    __slots__ = [
        "timestamp",
        "height",
        "width",
        "num_rgb_channels",
        "rgb_channel_type",
        "rgb_size",
        "rgb_image",
        "depth_channel_type",
        "depth_size",
        "depth_image",
    ]

    __typenames__ = [
        "double",
        "int32_t",
        "int32_t",
        "int32_t",
        "int8_t",
        "int32_t",
        "byte",
        "int8_t",
        "int32_t",
        "byte",
    ]

    __dimensions__ = [
        None,
        None,
        None,
        None,
        None,
        None,
        ["rgb_size"],
        None,
        None,
        ["depth_size"],
    ]

    CHANNEL_TYPE_INT8 = 0
    CHANNEL_TYPE_UINT8 = 1
    CHANNEL_TYPE_INT16 = 2
    CHANNEL_TYPE_UINT16 = 3
    CHANNEL_TYPE_INT32 = 4
    CHANNEL_TYPE_UINT32 = 5
    CHANNEL_TYPE_FLOAT32 = 6
    CHANNEL_TYPE_FLOAT64 = 7

    def __init__(self):
        self.timestamp = 0.0
        self.height = 0
        self.width = 0
        self.num_rgb_channels = 0
        self.rgb_channel_type = 0
        self.rgb_size = 0
        self.rgb_image = b""
        self.depth_channel_type = 0
        self.depth_size = 0
        self.depth_image = b""

    def encode(self):
        buf = BytesIO()
        buf.write(rgbd_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(
            struct.pack(
                ">diiibi",
                self.timestamp,
                self.height,
                self.width,
                self.num_rgb_channels,
                self.rgb_channel_type,
                self.rgb_size,
            )
        )
        buf.write(bytearray(self.rgb_image[: self.rgb_size]))
        buf.write(struct.pack(">bi", self.depth_channel_type, self.depth_size))
        buf.write(bytearray(self.depth_image[: self.depth_size]))

    @staticmethod
    def decode(data: bytes):
        buf = data if hasattr(data, "read") else BytesIO(data)
        if buf.read(8) != rgbd_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return rgbd_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = rgbd_t()
        (
            self.timestamp,
            self.height,
            self.width,
            self.num_rgb_channels,
            self.rgb_channel_type,
            self.rgb_size,
        ) = struct.unpack(">diiibi", buf.read(25))
        self.rgb_image = buf.read(self.rgb_size)
        self.depth_channel_type, self.depth_size = struct.unpack(">bi", buf.read(5))
        self.depth_image = buf.read(self.depth_size)
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if rgbd_t in parents:
            return 0
        tmphash = 0xDD27AF7737FDB731 & 0xFFFFFFFFFFFFFFFF
        tmphash = (((tmphash << 1) & 0xFFFFFFFFFFFFFFFF) + (tmphash >> 63)) & 0xFFFFFFFFFFFFFFFF
        return tmphash

    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if rgbd_t._packed_fingerprint is None:
            rgbd_t._packed_fingerprint = struct.pack(
                ">Q", rgbd_t._get_hash_recursive([])
            )
        return rgbd_t._packed_fingerprint

    def get_hash(self):
        return struct.unpack(">Q", rgbd_t._get_packed_fingerprint())[0]
