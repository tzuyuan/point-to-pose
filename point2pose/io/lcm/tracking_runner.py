from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from point2pose.data_types.frame import Frame
from point2pose.io.lcm.data_models import NamedVecListPayload
from point2pose.io.lcm.pose_export import (
    build_bbox_pose_vector,
    build_mesh_pose_vector,
    object_name_from_index,
)
from point2pose.io.lcm.runtime import NamedVecListLcmPublisher, RgbdLcmSubscriber
from point2pose.utils.transform import inverse_SE3


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
from point2pose.utils.visualization import (
    draw_points_on_image,
    draw_posed_3d_box,
    draw_xyz_axis,
    get_n_uncertainty_colors,
)


class LcmTrackingRunner:
    """
    LCM-driven twin of examples/realsense_tracking/realsense_tracking.py.

    The prompt collection UI (including the SAM2 mask preview), the pipeline
    calls and the tracking visualization are deliberately kept identical to that
    demo; only the frame source (LCM instead of a directly opened RealSense) and
    the pose publishing are different.
    """

    def __init__(self, config_path: str = "configs/pipeline/lcm_tracking.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self._lcm_cfg = self.cfg.get("lcm", {})

        self._rgbd_channel = str(self._lcm_cfg.get("rgbd_channel", "d455_1"))
        self._camera_info_channel = str(
            self._lcm_cfg.get("camera_info_channel", f"{self._rgbd_channel}_info")
        )
        self._obj_pose_channel = str(
            self._lcm_cfg.get("obj_pose_bb2world_channel", "hw_obj_pose")
        )
        self._obj_mesh_pose_channel = str(
            self._lcm_cfg.get("obj_pose_mesh2world_channel", "hw_obj_mesh_pose")
        )
        self._window_name = str(
            self._lcm_cfg.get("window_name", "Point2Pose LCM Tracker")
        )
        self._sub_poll_hz = float(self._lcm_cfg.get("sub_poll_hz", 500.0))
        self._pub_hz = float(self._lcm_cfg.get("pub_hz", 60.0))
        self._drop_stale_frames = bool(self._lcm_cfg.get("drop_stale_frames", True))
        self._max_frame_drain = max(1, int(self._lcm_cfg.get("max_frame_drain", 8)))
        self._verbose = bool(self._lcm_cfg.get("verbose", False))
        # Communication-delay debugging: print one timing line per tracked frame
        # (and a periodic receive summary while collecting points).
        self._comm_debug = bool(self._lcm_cfg.get("comm_debug", False))
        self._comm_debug_every = max(1, int(self._lcm_cfg.get("comm_debug_every", 1)))
        self._comm_last_report = 0.0

        self._visualize_points = self.cfg.visualization.params.visualize_points
        # Draw the posed 3D box + xyz axis per object (set false to see only masks).
        self._visualize_bbox = bool(
            self.cfg.visualization.params.get("visualize_bbox", True)
        )
        # Print (and overlay) the age of each processed frame: process time now
        # minus the rgb timestamp from the sensor.
        self._print_frame_age = bool(self._lcm_cfg.get("print_frame_age", False))
        self._sub_queue_capacity = int(self._lcm_cfg.get("sub_queue_capacity", 0))
        self._timing = {}  # stage timestamps of the current loop iteration
        self._t_loop_prev = None
        self._points_vis_method = self.cfg.visualization.params.points_vis_method
        self._save_images = self.cfg.visualization.params.save_images
        self._output_image_dir = Path(self.cfg.visualization.params.output_image_dir)

        # Create output directory if saving images is enabled
        if self._save_images:
            self._output_image_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Image saving enabled. Images will be saved to: {self._output_image_dir}"
            )

        # LCM transport replaces the demo's direct RealSense handle.
        self.subscriber = RgbdLcmSubscriber(
            rgbd_channel=self._rgbd_channel,
            camera_info_channel=self._camera_info_channel,
            sub_poll_hz=self._sub_poll_hz,
            drop_stale_frames=self._drop_stale_frames,
            max_frame_drain=self._max_frame_drain,
            verbose=self._verbose,
            sub_queue_capacity=self._sub_queue_capacity,
        )
        self.publisher = NamedVecListLcmPublisher(
            channel=self._obj_pose_channel,
            pub_hz=self._pub_hz,
            verbose=self._verbose,
        )
        self.mesh_pose_publisher = NamedVecListLcmPublisher(
            channel=self._obj_mesh_pose_channel,
            pub_hz=self._pub_hz,
            verbose=self._verbose,
        )

        # Per-object prompt-point storage. Each entry holds one object's clicked
        # points/labels. Start with a single (empty) object; press 'n' to begin the
        # next object.
        self.object_points = [[]]
        self.object_labels = [[]]
        self.current_obj = 0
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # SAM2 mask preview (recomputed only when the point set changes)
        self._preview_masks = None
        self._preview_dirty = False
        # Distinct BGR colors used to draw points/overlays per object
        self._obj_palette = [
            (0, 255, 0),
            (255, 128, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (0, 128, 255),
            (255, 255, 0),
            (128, 255, 0),
        ]

        # Cache for consistent, distinctive frame-based point colors
        self._frame_color_lookup = {}
        self._frame_color_used_hsv = set()

        # Latest camera info, needed for intrinsics and for lifting poses to world
        self._latest_camera_info = None

        # Create window and set mouse callback
        cv2.startWindowThread()
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self.mouse_callback)

        if self.cfg.pipeline.type != "modular":
            raise ValueError(
                f"Only 'modular' pipeline is supported, got: {self.cfg.pipeline.type}"
            )
        self._show_status_screen("Loading SAM2 / tracker checkpoints...")
        self.pipeline = self._build_pipeline()

        print("Instructions:")
        print("- Left click:  Add positive point to the CURRENT object")
        print("- Right click: Add negative point to the CURRENT object")
        print("- Press 'n':   Start a NEW object (query the next object)")
        print("- Press 's':   Start tracking")
        print("- Press 'r':   Reset points")
        print("- Press 'q':   Quit")

    def _build_pipeline(self):
        # Imported here rather than at module scope: the tracker/segmenter backends
        # pull in torchvision, which spins forever when imported before the first
        # cv2.namedWindow call (the window is created in __init__).
        from point2pose.pipeline.modular_pipeline import ModularPipeline

        return ModularPipeline(self.cfg)

    def mouse_callback(self, event, x, y, _flags, _param):
        """Collect prompt points for the object currently being annotated."""
        if self.tracking_started:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point for the current object
            self.object_points[self.current_obj].append([x, y])
            self.object_labels[self.current_obj].append(1)
            self._preview_dirty = True
            print(f"[obj {self.current_obj}] +positive point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point for the current object
            self.object_points[self.current_obj].append([x, y])
            self.object_labels[self.current_obj].append(0)
            self._preview_dirty = True
            print(f"[obj {self.current_obj}] -negative point: ({x}, {y})")

    def next_object(self):
        """Finish the current object and start collecting points for a new one."""
        if len(self.object_points[self.current_obj]) == 0:
            print(f"[obj {self.current_obj}] no points yet; click before pressing 'n'.")
            return
        if not any(lbl == 1 for lbl in self.object_labels[self.current_obj]):
            print(
                f"[obj {self.current_obj}] add at least one positive (left-click) "
                "point first."
            )
            return
        self.object_points.append([])
        self.object_labels.append([])
        self.current_obj += 1
        self._preview_dirty = True
        print(
            f"Started object {self.current_obj}. Click its points, "
            "or press 's' to start tracking."
        )

    def _draw_mask_overlay(self, display_bgr, masks):
        """Overlay per-object SAM2 masks (logits [N,1,H,W], >0 = fg) onto a BGR image."""
        if masks is None or len(masks) == 0:
            return display_bgr

        height, width = display_bgr.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[..., 1] = 255  # green base (HSV)
        any_mask = False

        num_obj = len(masks)
        for i in range(num_obj):
            obj_mask = masks[i, 0] > 0.0
            if hasattr(obj_mask, "cpu"):
                obj_mask = obj_mask.cpu().numpy()
            obj_mask = np.asarray(obj_mask)
            if obj_mask.shape != (height, width):
                obj_mask = cv2.resize(
                    obj_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if np.any(obj_mask):
                any_mask = True
                hue = int((i + 3) / (num_obj + 3) * 255)
                overlay[obj_mask, 0] = hue
                overlay[obj_mask, 2] = 255

        if not any_mask:
            return display_bgr

        overlay = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(display_bgr, 1, overlay, 0.5, 0)

    def reset_points(self):
        """Reset collected points and restart tracking"""
        self.object_points = [[]]
        self.object_labels = [[]]
        self.current_obj = 0
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None
        self._preview_masks = None
        self._preview_dirty = False

        # Reset pipeline
        if self.cfg.pipeline.type != "modular":
            raise ValueError(
                f"Only 'modular' pipeline is supported, got: {self.cfg.pipeline.type}"
            )
        self._show_status_screen("Resetting tracking pipeline...")
        self.pipeline = self._build_pipeline()

        print("Points reset. Click to add new points.")

    def _generate_next_frame_color(self):
        """Generate a new distinctive HSV-based color and convert it to BGR."""
        golden_ratio_conjugate = 0.6180339887498949
        base_index = len(self._frame_color_lookup)
        saturation_cycle = (255, 230, 200, 180)
        value_cycle = (255, 235, 215)

        attempt = 0
        while True:
            idx = base_index + attempt
            hue = int(round(((idx * golden_ratio_conjugate) % 1.0) * 179)) % 180
            saturation = saturation_cycle[idx % len(saturation_cycle)]
            value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]

            hsv_tuple = (hue, saturation, value)
            if hsv_tuple not in self._frame_color_used_hsv:
                self._frame_color_used_hsv.add(hsv_tuple)
                hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

            attempt += 1

    def _colors_for_frame_ids(self, frame_ids: np.ndarray) -> np.ndarray:
        """Return consistent BGR colors for the provided frame ids."""
        if frame_ids.size == 0:
            return np.empty((0, 3), dtype=np.uint8)

        colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
        unique_ids = np.unique(frame_ids.astype(np.int64))

        for fid in unique_ids:
            fid_int = int(fid)
            if fid_int not in self._frame_color_lookup:
                self._frame_color_lookup[fid_int] = self._generate_next_frame_color()

            colors[frame_ids == fid] = self._frame_color_lookup[fid_int]

        return colors

    def start_tracking(self):
        """Initialize pipeline tracking with collected points"""
        # Keep only objects that actually have points.
        groups = [
            (pts, lbls)
            for pts, lbls in zip(self.object_points, self.object_labels)
            if len(pts) > 0
        ]
        if len(groups) == 0:
            print("No points collected! Please click on objects first.")
            return False

        # Get current frame for initialization
        frame = self.create_frame_from_lcm(0)
        if frame is None:
            print("Waiting for both RGBD and camera info before tracking can start.")
            return False

        # Add user points to pipeline, grouped per object
        objects_points = [pts for pts, _ in groups]
        objects_labels = [lbls for _, lbls in groups]
        self.pipeline.add_user_points(objects_points, objects_labels)

        # Initialize pipeline with first frame
        self.current_poses = self.pipeline.step(frame)
        self._publish_current_objects(frame.timestamp)

        self.tracking_started = True
        self.frame_count = 1

        total_points = sum(len(pts) for pts in objects_points)
        print(
            f"Started tracking with {len(objects_points)} object(s), "
            f"{total_points} points"
        )
        print(f"Number of objects: {len(self.current_poses)}")

        return True

    def create_frame_from_lcm(self, frame_id, rgbd_packet=None):
        """Create Frame object from the newest LCM RGBD packet + camera info."""
        if rgbd_packet is None:
            rgbd_packet = self.subscriber.peek_latest_rgbd()
        camera_info = self._latest_camera_info
        if camera_info is None:
            camera_info = self.subscriber.get_latest_camera_info()

        if rgbd_packet is None or camera_info is None:
            return None

        self._latest_camera_info = camera_info

        # The publisher sends RGB, so no BGR->RGB conversion is needed here.
        frame_rgb = self._normalize_rgb(rgbd_packet.rgb_image)
        frame_depth = np.asarray(rgbd_packet.depth_image).astype(np.float32)

        return Frame(
            id=frame_id,
            rgb=frame_rgb,
            depth=frame_depth,
            intrinsics=np.asarray(camera_info.intrinsics, dtype=np.float64).copy(),
            depth_factor=float(camera_info.depth_factor),
            timestamp=float(rgbd_packet.timestamp),
        )

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        elif rgb.ndim == 3 and rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        elif rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[..., :3]

        if rgb.dtype == np.uint8:
            return rgb.copy()
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 255.0)
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
            return rgb.astype(np.uint8)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _mask_fallback_object_ids(self) -> set:
        """Objects whose pose came from the SAM mask fallback on the last step."""
        hist = getattr(self.pipeline, "hist_fe_results", None)
        if not hist:
            return set()
        triggered = getattr(hist[-1], "mask_fallback_triggered", None) or {}
        return {obj_id for obj_id, hit in triggered.items() if hit}

    def _publish_current_objects(self, timestamp: float):
        if self.pipeline is None or not getattr(self.pipeline, "objects", None):
            return
        if self._latest_camera_info is None:
            return
        camera_to_world = inverse_SE3(
            np.asarray(self._latest_camera_info.world_to_camera, dtype=np.float64)
        )
        bbox_vectors = []
        mesh_vectors = []
        names = []
        for obj_idx, obj in enumerate(self.pipeline.objects):
            if getattr(obj, "pose", None) is None:
                continue
            bbox_vectors.append(
                build_bbox_pose_vector(obj, camera_to_world=camera_to_world)
            )
            mesh_vectors.append(
                build_mesh_pose_vector(obj, camera_to_world=camera_to_world)
            )
            names.append(object_name_from_index(obj_idx))

        if not bbox_vectors:
            return

        bbox_payload = NamedVecListPayload(
            channel=self._obj_pose_channel,
            timestamp=float(timestamp),
            names=names,
            vecs=np.stack(bbox_vectors, axis=0).astype(np.float32),
        )
        mesh_payload = NamedVecListPayload(
            channel=self._obj_mesh_pose_channel,
            timestamp=float(timestamp),
            names=names,
            vecs=np.stack(mesh_vectors, axis=0).astype(np.float32),
        )
        self.publisher.submit(bbox_payload)
        self.mesh_pose_publisher.submit(mesh_payload)

    def _print_comm_summary(self, rgbd_packet):
        st = self.subscriber.comm_stats()
        now = time.time()
        print(
            "[comm] rx={rx} dropped={dr} queued={q} | "
            "sensor->rx {lat:6.1f} ms | decode {dec:5.1f} ms | "
            "inter-arrival {gap:6.1f} ms | newest age {age:6.1f} ms".format(
                rx=st["rx_count"],
                dr=st["rx_dropped"],
                q=st["queued"],
                lat=1e3 * st["rx_latency_s"],
                dec=1e3 * st["rx_decode_s"],
                gap=1e3 * st["rx_gap_s"],
                age=1e3 * (now - rgbd_packet.timestamp),
            ),
            flush=True,
        )

    def _print_timing_table(self, fid: int):
        """Per-frame timeline of one loop iteration, in ms.

        sensor->rx   sensor stamp -> decoded in the LCM handler thread (wire + any
                     starvation of that thread).  Compare with the probe: if the
                     probe shows ~2 ms and this shows more, the handler thread is
                     being starved by this process (GIL) or LCM queue backlog.
        rx->pop      sat in our deque waiting for the main loop.
        frame        create_frame_from_lcm (copies, normalisation).
        step         pipeline.step incl. CUDA sync.
        publish      pose export + submit.
        vis          visualize_tracking_results (mask .cpu(), drawing).
        show         cv2.imshow + waitKey(1).
        AGE@show     now - rgb timestamp when the image hits the screen; this is
                     the delay you see in the window.
        loop         previous show -> this show (1/loop = effective fps).
        """
        T = self._timing
        st = self.subscriber.comm_stats(reset_window=True)
        ms = lambda a, b: 1e3 * (T[b] - T[a])  # noqa: E731
        loop_ms = (
            1e3 * (T["show"] - self._t_loop_prev) if self._t_loop_prev else float("nan")
        )
        self._t_loop_prev = T["show"]
        mt = getattr(self.pipeline, "last_module_times", None) or {}
        mods = " ".join(f"{k[:5]}={1e3 * v:.0f}" for k, v in mt.items() if v >= 0.0005)
        print(
            "[timing] f{fid:<5d}"
            " sensor->rx {a:6.1f} | rx->pop {b:6.1f} | frame {c:5.1f} | step {d:6.1f}"
            " | publish {e:4.1f} | vis {f:5.1f} | show {g:5.1f} || AGE@show {age:6.1f}"
            " | loop {loop:6.1f} ms".format(
                fid=fid,
                a=1e3 * (T["rx"] - T["sensor"]),
                b=1e3 * (T["pop"] - T["rx"]),
                c=ms("pop", "frame"),
                d=ms("frame", "step"),
                e=ms("step", "publish"),
                f=ms("publish", "vis"),
                g=ms("vis", "show"),
                age=1e3 * (T["show"] - T["sensor"]),
                loop=loop_ms,
            ),
            flush=True,
        )
        print(
            "         rx-thread: got {n} msgs since last, max sensor->rx {mx:6.1f} ms,"
            " lcm-backlog hits {bl}, dropped total {dr}, queued {q}"
            "{mods}".format(
                n=st["rx_since"],
                mx=1e3 * st["rx_max_latency_since_s"],
                bl=st["rx_backlog_hits"],
                dr=st["rx_dropped"],
                q=st["queued"],
                mods=(" | step modules(ms): " + mods) if mods else "",
            ),
            flush=True,
        )

    def _show_status_screen(self, message: str):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lines = [
            message,
            "L-click +, R-click -, 'n' next obj, 's' start, 'r' reset",
        ]
        y = 180
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y += 40
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        """Visualize tracking results on the frame"""
        display_frame = frame.rgb.copy()
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        height, width = display_frame.shape[:2]
        camera_intrinsics = frame.intrinsics
        fallback_ids = self._mask_fallback_object_ids()

        # Draw segmentation masks if available
        if hasattr(frame, "mask") and frame.mask is not None:
            mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            mask_overlay[..., 1] = 255  # Green base

            for i in range(len(frame.mask)):
                obj_mask = frame.mask[i, 0] > 0.0
                ## TODO: optimize this by removing the cpu().numpy()
                obj_mask = obj_mask.cpu().numpy()
                if np.any(obj_mask):
                    # Color each object differently
                    hue = (i + 3) / (len(frame.mask) + 3) * 255
                    mask_overlay[obj_mask, 0] = hue
                    mask_overlay[obj_mask, 2] = 255

            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
            display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)

        if self._visualize_points:
            if self._points_vis_method == "uncertainty":

                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue

                    uncertainty_color = get_n_uncertainty_colors(
                        self.pipeline.track_table.uncertainty[
                            self.pipeline.track_table.obj2track_map[i]
                        ]
                    )
                    draw_points_on_image(
                        display_frame,
                        self.pipeline.track_table.track_2d[
                            self.pipeline.track_table.obj2track_map[i]
                        ],
                        uncertainty_color,
                    )
            elif self._points_vis_method == "visible":
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue

                    # Generate N by 3 array with (0,255,0) for each row
                    track_2d_points = self.pipeline.track_table.track_2d[
                        self.pipeline.track_table.obj2track_map[i]
                    ]
                    N = len(track_2d_points)
                    visible_color = np.full((N, 3), (0, 0, 255), dtype=np.uint8)
                    visible_color[
                        self.pipeline.track_table.visible[
                            self.pipeline.track_table.obj2track_map[i]
                        ]
                    ] = (0, 255, 0)
                    draw_points_on_image(
                        display_frame,
                        track_2d_points,
                        visible_color,
                    )
            elif self._points_vis_method == "visible_uncertainty":
                # Plot only visible points, colored by their uncertainty colors
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue

                    track_idx = self.pipeline.track_table.obj2track_map[i]
                    track_2d_points = self.pipeline.track_table.track_2d[track_idx]
                    visible_mask = self.pipeline.track_table.visible[track_idx]

                    if np.any(visible_mask):
                        uncertainty_color = get_n_uncertainty_colors(
                            self.pipeline.track_table.uncertainty[track_idx]
                        )

                        draw_points_on_image(
                            display_frame,
                            track_2d_points[visible_mask],
                            uncertainty_color[visible_mask],
                        )
            elif self._points_vis_method == "frame_id":
                # Color each point based on the frame id it was first seen (object.key_point_frames)
                for i, obj in enumerate(objects):
                    if i not in self.pipeline.track_table.obj2track_map:
                        continue

                    track_idx = self.pipeline.track_table.obj2track_map[i]
                    track_2d_points = self.pipeline.track_table.track_2d[track_idx]
                    visible_mask = self.pipeline.track_table.visible[track_idx]

                    # Only proceed if there are visible points
                    if not np.any(visible_mask):
                        continue
                    if obj.key_point_frames.shape[0] == 0:
                        continue
                    # Align per-object track order with object's key point order
                    # Assume key_point_frames order corresponds to obj2track_map order
                    num_tracks_for_obj = len(track_idx)
                    kp_frames_for_obj = obj.key_point_frames[
                        :num_tracks_for_obj
                    ].astype(np.int32)

                    # Frame ids for visible points; replace unknown -1 with current frame id if available
                    frame_ids = kp_frames_for_obj[visible_mask]
                    if frame_id is not None:
                        frame_ids = frame_ids.copy()
                        frame_ids[frame_ids == -1] = int(frame_id)

                    # Use cached, distinctive colors per frame id
                    colors_bgr = self._colors_for_frame_ids(frame_ids)

                    # Draw only visible points for this object, using aligned colors
                    draw_points_on_image(
                        display_frame,
                        track_2d_points[visible_mask],
                        colors_bgr,
                    )

        # Draw pose information
        for i, obj in enumerate(objects):
            if self._visualize_bbox and obj.pose is not None:
                pose = obj.pose @ obj.init_pose
                half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
                bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

                # Orange box while the pose is coming from the SAM mask fallback
                # rather than from point-track registration.
                line_color = (0, 165, 255) if i in fallback_ids else (0, 255, 0)
                display_frame = draw_posed_3d_box(
                    camera_intrinsics,
                    display_frame,
                    pose,
                    bbox_min_max_local,
                    line_color=line_color,
                )
                display_frame = draw_xyz_axis(
                    image=display_frame, ob_in_cam=pose, K=camera_intrinsics
                )

        if fallback_ids:
            cv2.putText(
                display_frame,
                "SAM mask fallback: obj "
                + ",".join(str(i) for i in sorted(fallback_ids)),
                (10, height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        # Save image if flag is enabled and frame_id is provided
        if self._save_images and frame_id is not None:
            image_filename = self._output_image_dir / f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(image_filename), display_frame)

        return display_frame

    def run(self):
        """Main tracking loop"""
        self.subscriber.start()
        self.publisher.start()
        self.mesh_pose_publisher.start()

        try:
            while True:
                self._latest_camera_info = (
                    self.subscriber.get_latest_camera_info() or self._latest_camera_info
                )

                if not self.tracking_started:
                    # Get frames for point collection visualization
                    rgbd_packet = self.subscriber.peek_latest_rgbd()

                    if rgbd_packet is None:
                        self._show_status_screen("Waiting for LCM RGBD...")
                        key = cv2.waitKey(1) & 0xFF
                        if not self._handle_key(key):
                            break
                        continue

                    if self._comm_debug and time.time() - self._comm_last_report > 1.0:
                        self._comm_last_report = time.time()
                        self._print_comm_summary(rgbd_packet)

                    # Convert to display format (BGR)
                    rgb = self._normalize_rgb(rgbd_packet.rgb_image)
                    display_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    # Recompute the SAM2 preview mask only when the point set changed.
                    if self._preview_dirty:
                        try:
                            self._preview_masks = self.pipeline.preview_user_masks(
                                rgb, self.object_points, self.object_labels
                            )
                        except Exception as exc:  # preview is best-effort
                            print(f"[preview] SAM2 preview failed: {exc}")
                            self._preview_masks = None
                        self._preview_dirty = False

                    # Overlay the SAM2 mask preview for the clicked points
                    display_frame = self._draw_mask_overlay(
                        display_frame, self._preview_masks
                    )

                    # Show collected points, colored per object
                    total_points = 0
                    for obj_idx, (pts, lbls) in enumerate(
                        zip(self.object_points, self.object_labels)
                    ):
                        obj_color = self._obj_palette[obj_idx % len(self._obj_palette)]
                        for point, label in zip(pts, lbls):
                            total_points += 1
                            px, py = int(point[0]), int(point[1])
                            if label == 1:
                                # positive: filled circle in the object's color
                                cv2.circle(display_frame, (px, py), 5, obj_color, -1)
                            else:
                                # negative: red cross
                                cv2.drawMarker(
                                    display_frame,
                                    (px, py),
                                    (0, 0, 255),
                                    cv2.MARKER_TILTED_CROSS,
                                    12,
                                    2,
                                )

                    # Show instructions + status
                    cv2.putText(
                        display_frame,
                        "L-click +, R-click -, 'n' next obj, 's' start, 'r' reset",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    num_objects = sum(1 for pts in self.object_points if pts)
                    cv2.putText(
                        display_frame,
                        f"Object {self.current_obj} | objects: {num_objects} | "
                        f"points: {total_points}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self._obj_palette[self.current_obj % len(self._obj_palette)],
                        2,
                    )
                    if self._latest_camera_info is None:
                        cv2.putText(
                            display_frame,
                            "Waiting for LCM camera info...",
                            (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                    # Save image if flag is enabled (for point collection phase)
                    if self._save_images:
                        image_filename = (
                            self._output_image_dir
                            / f"point_collection_{self.frame_count:06d}.png"
                        )
                        cv2.imwrite(str(image_filename), display_frame)

                else:
                    # Track the live stream rather than a backlog.
                    T = self._timing
                    T.clear()
                    T["loop_start"] = time.time()
                    if self._drop_stale_frames:
                        rgbd_packet = self.subscriber.pop_latest_rgbd()
                    else:
                        rgbd_packet = self.subscriber.pop_oldest_rgbd()
                    T["pop"] = time.time()

                    # Create frame for pipeline
                    frame = (
                        None
                        if rgbd_packet is None
                        else self.create_frame_from_lcm(
                            self.frame_count, rgbd_packet=rgbd_packet
                        )
                    )
                    if frame is None:
                        key = cv2.waitKey(1) & 0xFF
                        if not self._handle_key(key):
                            break
                        continue
                    T["frame"] = time.time()
                    T["sensor"] = float(rgbd_packet.timestamp)
                    T["rx"] = float(rgbd_packet.recv_time) or T["pop"]

                    # Run pipeline step
                    self.pipeline.step(frame)
                    if self._comm_debug:
                        _cuda_sync()  # torch work is async; make 'step' honest
                    T["step"] = time.time()
                    self._publish_current_objects(frame.timestamp)
                    T["publish"] = time.time()
                    if self._print_frame_age:
                        print(
                            f"[age] frame {self.frame_count}: now - rgb timestamp = "
                            f"{1e3 * (T['publish'] - T['sensor']):.1f} ms",
                            flush=True,
                        )

                    # Visualize results
                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects, self.frame_count
                    )
                    T["vis"] = time.time()
                    frame_age_ms = 1e3 * (T["vis"] - T["sensor"])

                    # Show tracking info
                    height, _ = display_frame.shape[:2]
                    cv2.putText(
                        display_frame,
                        f"Frame: {self.frame_count}",
                        (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Objects: {len(self.current_poses) if self.current_poses is not None else 0}",
                        (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if self._print_frame_age:
                        cv2.putText(
                            display_frame,
                            f"Age: {frame_age_ms:.0f} ms",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                    self.frame_count += 1

                # Display the frame
                cv2.imshow(self._window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if self.tracking_started and self._timing:
                    self._timing["show"] = time.time()
                    if (
                        self._comm_debug
                        and (self.frame_count - 1) % self._comm_debug_every == 0
                    ):
                        self._print_timing_table(self.frame_count - 1)
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            # Cleanup
            self.subscriber.stop()
            self.publisher.stop()
            self.mesh_pose_publisher.stop()
            cv2.destroyAllWindows()

    def _handle_key(self, key) -> bool:
        """Apply one keypress. Returns False when the loop should exit."""
        if key == ord("q"):
            return False
        if key == ord("s") and not self.tracking_started:
            if self.start_tracking():
                print("Pipeline tracking started!")
            else:
                print("Failed to start pipeline tracking!")
        elif key == ord("n") and not self.tracking_started:
            self.next_object()
        elif key == ord("r"):
            self.reset_points()
        return True
