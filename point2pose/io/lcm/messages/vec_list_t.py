"""LCM type definitions for named vector lists."""

from io import BytesIO
import struct


class vec_list_t(object):
    __slots__ = ["timestamp", "num_vecs", "vec_dim", "name_list", "vec_list"]

    __typenames__ = ["double", "int32_t", "int32_t", "string", "float"]

    __dimensions__ = [None, None, None, ["num_vecs"], ["num_vecs", "vec_dim"]]

    def __init__(self):
        self.timestamp = 0.0
        self.num_vecs = 0
        self.vec_dim = 0
        self.name_list = []
        self.vec_list = []

    def encode(self):
        buf = BytesIO()
        buf.write(vec_list_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">dii", self.timestamp, self.num_vecs, self.vec_dim))
        for idx in range(self.num_vecs):
            encoded = self.name_list[idx].encode("utf-8")
            buf.write(struct.pack(">I", len(encoded) + 1))
            buf.write(encoded)
            buf.write(b"\0")
        for idx in range(self.num_vecs):
            buf.write(struct.pack(f">{self.vec_dim}f", *self.vec_list[idx][: self.vec_dim]))

    @staticmethod
    def decode(data: bytes):
        buf = data if hasattr(data, "read") else BytesIO(data)
        if buf.read(8) != vec_list_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return vec_list_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = vec_list_t()
        self.timestamp, self.num_vecs, self.vec_dim = struct.unpack(">dii", buf.read(16))
        self.name_list = []
        for _ in range(self.num_vecs):
            name_len = struct.unpack(">I", buf.read(4))[0]
            self.name_list.append(buf.read(name_len)[:-1].decode("utf-8", "replace"))
        self.vec_list = []
        for _ in range(self.num_vecs):
            self.vec_list.append(struct.unpack(f">{self.vec_dim}f", buf.read(self.vec_dim * 4)))
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if vec_list_t in parents:
            return 0
        tmphash = 0x8D0D22C73945E9E0 & 0xFFFFFFFFFFFFFFFF
        tmphash = (((tmphash << 1) & 0xFFFFFFFFFFFFFFFF) + (tmphash >> 63)) & 0xFFFFFFFFFFFFFFFF
        return tmphash

    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if vec_list_t._packed_fingerprint is None:
            vec_list_t._packed_fingerprint = struct.pack(
                ">Q", vec_list_t._get_hash_recursive([])
            )
        return vec_list_t._packed_fingerprint

    def get_hash(self):
        return struct.unpack(">Q", vec_list_t._get_packed_fingerprint())[0]
