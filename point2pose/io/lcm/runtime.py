from __future__ import annotations

from collections import deque
import time
import importlib
import queue
import threading
from typing import Callable

import numpy as np

from point2pose.io.lcm.data_models import (
    CameraInfoPacket,
    NamedVecListPayload,
    RGBDFramePacket,
)
from point2pose.io.lcm.image_conversion import pack_image_to_bytes, unpack_image_from_bytes
from point2pose.io.lcm.messages import camera_info_t, rgbd_t, vec_list_t


def _default_lcm_factory():
    return importlib.import_module("lcm").LCM()


class RgbdLcmSubscriber:
    def __init__(
        self,
        rgbd_channel: str,
        camera_info_channel: str | None = None,
        sub_poll_hz: float = 500.0,
        drop_stale_frames: bool = True,
        max_frame_drain: int = 8,
        lcm_factory: Callable[[], object] | None = None,
        verbose: bool = False,
        sub_queue_capacity: int = 0,
    ):
        self.rgbd_channel = str(rgbd_channel)
        # LCM keeps its own per-subscription queue (default 30 messages) *before*
        # our handler runs. If the handler thread is starved, that queue fills
        # with old frames and they are delivered oldest-first. >0 caps it.
        self.sub_queue_capacity = int(sub_queue_capacity)
        self.camera_info_channel = (
            str(camera_info_channel)
            if camera_info_channel
            else f"{self.rgbd_channel}_info"
        )
        self.sub_poll_hz = max(1.0, float(sub_poll_hz))
        self.drop_stale_frames = bool(drop_stale_frames)
        self.max_frame_drain = max(1, int(max_frame_drain))
        self.verbose = bool(verbose)
        self._lcm_factory = lcm_factory or _default_lcm_factory

        self._lcm = None
        self._lock = threading.Lock()
        self._rgbd_packets = deque(maxlen=self.max_frame_drain)
        self._latest_camera_info: CameraInfoPacket | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="RgbdLcmSubscriber")
        # Communication stats (guarded by self._lock).
        self._rx_count = 0
        self._rx_dropped = 0  # frames discarded before anyone consumed them
        self._rx_last_latency = float("nan")  # recv_time - msg.timestamp, seconds
        self._rx_last_decode_s = float("nan")
        self._rx_last_time = float("nan")
        self._rx_last_gap = float("nan")  # inter-arrival time, seconds
        # Handler-thread diagnostics: how long handle_timeout() waited before a
        # message arrived (~0 repeatedly => LCM queue backlog being drained), and
        # the max latency seen since the last stats snapshot.
        self._rx_backlog_hits = 0
        self._rx_max_latency_since = 0.0
        self._rx_count_since = 0

    def start(self):
        if self._thread.is_alive():
            return
        self._lcm = self._lcm_factory()
        sub = self._lcm.subscribe(self.rgbd_channel, self._handle_rgbd)
        if self.sub_queue_capacity > 0 and hasattr(sub, "set_queue_capacity"):
            sub.set_queue_capacity(self.sub_queue_capacity)
        self._lcm.subscribe(self.camera_info_channel, self._handle_camera_info)
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        timeout_ms = max(1, int(round(1000.0 / self.sub_poll_hz)))
        while not self._stop_event.is_set():
            t0 = time.time()
            handled = self._lcm.handle_timeout(timeout_ms)
            if handled and (time.time() - t0) < 0.5e-3:
                # Message was already waiting inside LCM: we are behind.
                with self._lock:
                    self._rx_backlog_hits += 1

    def _handle_rgbd(self, _channel: str, data: bytes):
        t_rx = time.time()
        msg = rgbd_t.decode(data)
        rgb_image = unpack_image_from_bytes(
            msg.rgb_image,
            height=msg.height,
            width=msg.width,
            num_channels=msg.num_rgb_channels,
            channel_type=msg.rgb_channel_type,
        )
        depth_image = unpack_image_from_bytes(
            msg.depth_image,
            height=msg.height,
            width=msg.width,
            num_channels=1,
            channel_type=msg.depth_channel_type,
        )
        packet = RGBDFramePacket(
            timestamp=float(msg.timestamp),
            height=int(msg.height),
            width=int(msg.width),
            num_rgb_channels=int(msg.num_rgb_channels),
            rgb_channel_type=int(msg.rgb_channel_type),
            depth_channel_type=int(msg.depth_channel_type),
            rgb_image=np.array(rgb_image, copy=True),
            depth_image=np.array(depth_image, copy=True),
            recv_time=t_rx,
        )
        t_done = time.time()
        with self._lock:
            if self.drop_stale_frames:
                self._rx_dropped += len(self._rgbd_packets)
                self._rgbd_packets.clear()
            elif len(self._rgbd_packets) == self._rgbd_packets.maxlen:
                self._rx_dropped += 1  # deque(maxlen) evicts the oldest
            self._rgbd_packets.append(packet)
            self._rx_count += 1
            self._rx_count_since += 1
            self._rx_last_latency = t_rx - packet.timestamp
            self._rx_max_latency_since = max(
                self._rx_max_latency_since, self._rx_last_latency
            )
            self._rx_last_decode_s = t_done - t_rx
            if self._rx_last_time == self._rx_last_time:  # not nan
                self._rx_last_gap = t_rx - self._rx_last_time
            self._rx_last_time = t_rx

    def _handle_camera_info(self, _channel: str, data: bytes):
        msg = camera_info_t.decode(data)
        intrinsics = np.array(
            [
                [msg.fx, 0.0, msg.cx],
                [0.0, msg.fy, msg.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = np.asarray(msg.extrinsic, dtype=np.float64).reshape(
            3, 4
        )
        packet = CameraInfoPacket(
            timestamp=float(msg.timestamp),
            height=int(msg.height),
            width=int(msg.width),
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            depth_factor=float(msg.depth_factor),
            fixed=bool(msg.fixed),
            attached_body=str(msg.attached_body),
        )
        with self._lock:
            self._latest_camera_info = packet

    def peek_latest_rgbd(self) -> RGBDFramePacket | None:
        with self._lock:
            if not self._rgbd_packets:
                return None
            return self._rgbd_packets[-1].copy()

    def pop_latest_rgbd(self) -> RGBDFramePacket | None:
        with self._lock:
            if not self._rgbd_packets:
                return None
            packet = self._rgbd_packets[-1].copy()
            self._rx_dropped += len(self._rgbd_packets) - 1
            self._rgbd_packets.clear()
            return packet

    def comm_stats(self, reset_window: bool = False) -> dict:
        """Snapshot of receive-side communication counters (seconds).

        `rx_since`/`rx_max_latency_since`/`rx_backlog_hits` accumulate since the
        last call with reset_window=True.
        """
        with self._lock:
            out = {
                "rx_count": self._rx_count,
                "rx_dropped": self._rx_dropped,
                "rx_latency_s": self._rx_last_latency,
                "rx_decode_s": self._rx_last_decode_s,
                "rx_gap_s": self._rx_last_gap,
                "queued": len(self._rgbd_packets),
                "rx_since": self._rx_count_since,
                "rx_max_latency_since_s": self._rx_max_latency_since,
                "rx_backlog_hits": self._rx_backlog_hits,
            }
            if reset_window:
                self._rx_count_since = 0
                self._rx_max_latency_since = 0.0
                self._rx_backlog_hits = 0
            return out

    def pop_oldest_rgbd(self) -> RGBDFramePacket | None:
        with self._lock:
            if not self._rgbd_packets:
                return None
            packet = self._rgbd_packets.popleft().copy()
            return packet

    def get_latest_camera_info(self) -> CameraInfoPacket | None:
        with self._lock:
            return (
                None
                if self._latest_camera_info is None
                else self._latest_camera_info.copy()
            )

    def has_camera_info(self) -> bool:
        with self._lock:
            return self._latest_camera_info is not None

    def has_rgbd(self) -> bool:
        with self._lock:
            return bool(self._rgbd_packets)


class RgbdLcmPublisher:
    def __init__(
        self,
        rgbd_channel: str,
        camera_info_channel: str | None = None,
        lcm_factory: Callable[[], object] | None = None,
        verbose: bool = False,
    ):
        self.rgbd_channel = str(rgbd_channel)
        self.camera_info_channel = (
            str(camera_info_channel)
            if camera_info_channel
            else f"{self.rgbd_channel}_info"
        )
        self.verbose = bool(verbose)
        self._lcm_factory = lcm_factory or _default_lcm_factory
        self._lcm = None

    def start(self):
        if self._lcm is None:
            self._lcm = self._lcm_factory()

    def stop(self):
        self._lcm = None

    def _ensure_lcm(self):
        if self._lcm is None:
            self.start()
        return self._lcm

    def publish_rgbd(self, packet: RGBDFramePacket):
        rgb = np.asarray(packet.rgb_image)
        depth = np.asarray(packet.depth_image)

        if rgb.ndim == 2:
            rgb = rgb[..., None]
        if rgb.ndim != 3:
            raise ValueError(
                f"RGB image must have shape (H, W, C) or (H, W); got {rgb.shape}"
            )
        if depth.ndim != 2:
            raise ValueError(f"Depth image must have shape (H, W); got {depth.shape}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                "RGB and depth image sizes do not match: "
                f"{rgb.shape[:2]} vs {depth.shape}"
            )

        msg = rgbd_t()
        msg.timestamp = float(packet.timestamp)
        msg.height = int(packet.height) if int(packet.height) > 0 else int(rgb.shape[0])
        msg.width = int(packet.width) if int(packet.width) > 0 else int(rgb.shape[1])
        msg.num_rgb_channels = int(packet.num_rgb_channels) or int(rgb.shape[2])
        msg.rgb_channel_type = int(packet.rgb_channel_type)
        msg.rgb_image = pack_image_to_bytes(rgb, msg.rgb_channel_type)
        msg.rgb_size = len(msg.rgb_image)
        msg.depth_channel_type = int(packet.depth_channel_type)
        msg.depth_image = pack_image_to_bytes(depth, msg.depth_channel_type)
        msg.depth_size = len(msg.depth_image)
        self._ensure_lcm().publish(self.rgbd_channel, msg.encode())

    def publish_camera_info(self, packet: CameraInfoPacket):
        intrinsics = np.asarray(packet.intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Camera intrinsics must have shape (3, 3); got {intrinsics.shape}"
            )

        world_to_camera = np.asarray(packet.world_to_camera, dtype=np.float64)
        if world_to_camera.shape != (4, 4):
            raise ValueError(
                "Camera extrinsic must have shape (4, 4); "
                f"got {world_to_camera.shape}"
            )

        msg = camera_info_t()
        msg.height = int(packet.height)
        msg.width = int(packet.width)
        msg.timestamp = float(packet.timestamp)
        msg.fx = float(intrinsics[0, 0])
        msg.fy = float(intrinsics[1, 1])
        msg.cx = float(intrinsics[0, 2])
        msg.cy = float(intrinsics[1, 2])
        msg.extrinsic = tuple(
            np.asarray(world_to_camera[:3, :], dtype=np.float32).reshape(-1).tolist()
        )
        msg.depth_factor = float(packet.depth_factor)
        msg.fixed = bool(packet.fixed)
        msg.attached_body = str(packet.attached_body)
        self._ensure_lcm().publish(self.camera_info_channel, msg.encode())


class NamedVecListLcmPublisher:
    def __init__(
        self,
        channel: str,
        pub_hz: float = 60.0,
        lcm_factory: Callable[[], object] | None = None,
        verbose: bool = False,
    ):
        self.channel = str(channel)
        self.pub_hz = max(1.0, float(pub_hz))
        self.verbose = bool(verbose)
        self._lcm_factory = lcm_factory or _default_lcm_factory

        self._lcm = None
        self._stop_event = threading.Event()
        self._payload_queue: queue.Queue[NamedVecListPayload] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run, name="NamedVecListLcmPublisher"
        )

    def start(self):
        if self._thread.is_alive():
            return
        self._lcm = self._lcm_factory()
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(self, payload: NamedVecListPayload):
        payload_copy = payload.copy()
        while True:
            try:
                self._payload_queue.put_nowait(payload_copy)
                return
            except queue.Full:
                try:
                    self._payload_queue.get_nowait()
                except queue.Empty:
                    continue

    def _run(self):
        timeout_s = 1.0 / self.pub_hz
        while not self._stop_event.is_set():
            try:
                payload = self._payload_queue.get(timeout=timeout_s)
            except queue.Empty:
                continue
            self._publish_payload(payload)

    def _publish_payload(self, payload: NamedVecListPayload):
        arr = np.asarray(payload.vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        msg = vec_list_t()
        msg.timestamp = float(payload.timestamp)
        msg.num_vecs = int(arr.shape[0])
        msg.vec_dim = int(arr.shape[1]) if arr.size else 0
        msg.name_list = [str(name) for name in payload.names]
        msg.vec_list = arr.tolist()
        self._lcm.publish(payload.channel, msg.encode())


class NamedVecListLcmSubscriber:
    def __init__(
        self,
        channel: str,
        sub_poll_hz: float = 500.0,
        lcm_factory: Callable[[], object] | None = None,
        verbose: bool = False,
    ):
        self.channel = str(channel)
        self.sub_poll_hz = max(1.0, float(sub_poll_hz))
        self.verbose = bool(verbose)
        self._lcm_factory = lcm_factory or _default_lcm_factory

        self._lcm = None
        self._lock = threading.Lock()
        self._latest_payload: NamedVecListPayload | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"NamedVecListLcmSubscriber[{self.channel}]"
        )

    def start(self):
        if self._thread.is_alive():
            return
        self._lcm = self._lcm_factory()
        self._lcm.subscribe(self.channel, self._handle_message)
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        timeout_ms = max(1, int(round(1000.0 / self.sub_poll_hz)))
        while not self._stop_event.is_set():
            self._lcm.handle_timeout(timeout_ms)

    def _handle_message(self, channel: str, data: bytes):
        msg = vec_list_t.decode(data)
        arr = np.asarray(msg.vec_list, dtype=np.float32)
        if arr.size == 0:
            arr = arr.reshape((0, int(msg.vec_dim)))
        payload = NamedVecListPayload(
            channel=str(channel),
            timestamp=float(msg.timestamp),
            names=[str(name) for name in msg.name_list],
            vecs=arr.copy(),
        )
        with self._lock:
            self._latest_payload = payload

    def get_latest(self) -> NamedVecListPayload | None:
        with self._lock:
            return None if self._latest_payload is None else self._latest_payload.copy()

    def pop_latest(self) -> NamedVecListPayload | None:
        with self._lock:
            if self._latest_payload is None:
                return None
            payload = self._latest_payload.copy()
            self._latest_payload = None
            return payload

    def has_payload(self) -> bool:
        with self._lock:
            return self._latest_payload is not None
