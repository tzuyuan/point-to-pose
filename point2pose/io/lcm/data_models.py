from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RGBDFramePacket:
    timestamp: float
    height: int
    width: int
    num_rgb_channels: int
    rgb_channel_type: int
    depth_channel_type: int
    rgb_image: np.ndarray
    depth_image: np.ndarray

    def copy(self) -> "RGBDFramePacket":
        return RGBDFramePacket(
            timestamp=float(self.timestamp),
            height=int(self.height),
            width=int(self.width),
            num_rgb_channels=int(self.num_rgb_channels),
            rgb_channel_type=int(self.rgb_channel_type),
            depth_channel_type=int(self.depth_channel_type),
            rgb_image=self.rgb_image.copy(),
            depth_image=self.depth_image.copy(),
        )


@dataclass(slots=True)
class CameraInfoPacket:
    timestamp: float
    height: int
    width: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    depth_factor: float
    fixed: bool
    attached_body: str

    def copy(self) -> "CameraInfoPacket":
        return CameraInfoPacket(
            timestamp=float(self.timestamp),
            height=int(self.height),
            width=int(self.width),
            intrinsics=self.intrinsics.copy(),
            world_to_camera=self.world_to_camera.copy(),
            depth_factor=float(self.depth_factor),
            fixed=bool(self.fixed),
            attached_body=str(self.attached_body),
        )


@dataclass(slots=True)
class NamedVecListPayload:
    channel: str
    timestamp: float
    names: list[str]
    vecs: np.ndarray

    def copy(self) -> "NamedVecListPayload":
        return NamedVecListPayload(
            channel=str(self.channel),
            timestamp=float(self.timestamp),
            names=[str(name) for name in self.names],
            vecs=np.asarray(self.vecs, dtype=np.float32).copy(),
        )
