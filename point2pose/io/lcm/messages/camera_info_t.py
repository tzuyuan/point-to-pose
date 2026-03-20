"""LCM type definitions for camera info."""

from io import BytesIO
import struct


class camera_info_t(object):
    __slots__ = [
        "height",
        "width",
        "timestamp",
        "fx",
        "fy",
        "cx",
        "cy",
        "extrinsic",
        "depth_factor",
        "fixed",
        "attached_body",
    ]

    __typenames__ = [
        "int32_t",
        "int32_t",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "boolean",
        "string",
    ]

    __dimensions__ = [None, None, None, None, None, None, None, [12], None, None, None]

    def __init__(self):
        self.height = 0
        self.width = 0
        self.timestamp = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.extrinsic = [0.0 for _ in range(12)]
        self.depth_factor = 0.0
        self.fixed = False
        self.attached_body = ""

    def encode(self):
        buf = BytesIO()
        buf.write(camera_info_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(
            struct.pack(
                ">iifffff",
                self.height,
                self.width,
                self.timestamp,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            )
        )
        buf.write(struct.pack(">12f", *self.extrinsic[:12]))
        buf.write(struct.pack(">fb", self.depth_factor, self.fixed))
        attached_body = self.attached_body.encode("utf-8")
        buf.write(struct.pack(">I", len(attached_body) + 1))
        buf.write(attached_body)
        buf.write(b"\0")

    @staticmethod
    def decode(data: bytes):
        buf = data if hasattr(data, "read") else BytesIO(data)
        if buf.read(8) != camera_info_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return camera_info_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = camera_info_t()
        (
            self.height,
            self.width,
            self.timestamp,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        ) = struct.unpack(">iifffff", buf.read(28))
        self.extrinsic = struct.unpack(">12f", buf.read(48))
        self.depth_factor = struct.unpack(">f", buf.read(4))[0]
        self.fixed = bool(struct.unpack("b", buf.read(1))[0])
        attached_body_len = struct.unpack(">I", buf.read(4))[0]
        self.attached_body = buf.read(attached_body_len)[:-1].decode(
            "utf-8", "replace"
        )
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if camera_info_t in parents:
            return 0
        tmphash = 0x3C6C827F905C70FF & 0xFFFFFFFFFFFFFFFF
        tmphash = (((tmphash << 1) & 0xFFFFFFFFFFFFFFFF) + (tmphash >> 63)) & 0xFFFFFFFFFFFFFFFF
        return tmphash

    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if camera_info_t._packed_fingerprint is None:
            camera_info_t._packed_fingerprint = struct.pack(
                ">Q", camera_info_t._get_hash_recursive([])
            )
        return camera_info_t._packed_fingerprint

    def get_hash(self):
        return struct.unpack(">Q", camera_info_t._get_packed_fingerprint())[0]
