from __future__ import annotations

import cv2
import numpy as np

from point2pose.io.lcm.data_models import NamedVecListPayload
from point2pose.io.lcm.obstacle_tracker import Sam2ObstacleTracker
from point2pose.io.lcm.runtime import NamedVecListLcmPublisher
from point2pose.io.lcm.tracking_runner import LcmTrackingRunner
from point2pose.utils.transform import inverse_SE3


class LcmObstacleTrackingRunner(LcmTrackingRunner):
    """
    LcmTrackingRunner plus a SAM2-only tracked obstacle.

    Everything the base runner does (object prompts, 6D pipeline, the two pose
    channels, the visualization) is unchanged. On top of it:

      * press 'o' to toggle between prompting OBJECTS and prompting the OBSTACLE
        (clicks, 'n' and the mask preview follow the active mode);
      * the obstacle is segmented by its own SAM2 predictor every frame - it never
        enters the point-track / registration pipeline;
      * its mask + depth are turned into a sphere (see Sam2ObstacleTracker) and
        published on lcm.obstacle_pose_channel as [x y z r] in the world frame,
        one row per obstacle, named obstacle_0, obstacle_1, ...

    The obstacle is optional: press 's' without any obstacle points and the run
    behaves exactly like the base runner (the obstacle channel stays silent).
    """

    OBSTACLE_COLOR = (0, 0, 255)  # BGR red for prompts / overlay

    def __init__(self, config_path: str = "configs/pipeline/lcm_obstacle_tracking.yaml"):
        super().__init__(config_path)

        self._obstacle_cfg = self.cfg.get("obstacle", {})
        self._obstacle_channel = str(
            self._lcm_cfg.get("obstacle_pose_channel", "hw_obstacle_pose")
        )
        self.obstacle_publisher = NamedVecListLcmPublisher(
            channel=self._obstacle_channel,
            pub_hz=self._pub_hz,
            verbose=self._verbose,
        )

        # Obstacle prompt storage mirrors the object one; 'o' switches modes.
        self.obstacle_points = [[]]
        self.obstacle_labels = [[]]
        self.current_obstacle = 0
        self.obstacle_mode = False
        self._obstacle_preview_masks = None
        self._obstacle_preview_dirty = False

        segmenter_cfg = self._obstacle_cfg.get("segmenter", None)
        if segmenter_cfg is None:
            segmenter_cfg = self.cfg.segmenter
        seg_params = segmenter_cfg.get("params", segmenter_cfg)
        self._show_status_screen("Loading obstacle SAM2 checkpoint...")
        self.obstacle_tracker = Sam2ObstacleTracker(
            obstacle_cfg=self._obstacle_cfg,
            segmenter_cfg=seg_params,
            pipeline_params=self.cfg.pipeline.get("params", {}),
        )
        self.obstacle_tracker.ensure_segmenter()

        print("- Press 'o':   Toggle OBJECT / OBSTACLE prompting mode")
        print(
            f"  Obstacle sphere published on '{self._obstacle_channel}' as [x y z r], "
            f"radius_mode={self.obstacle_tracker.radius_mode}, "
            f"radius_m={self.obstacle_tracker.radius_m:.3f}"
        )

    # ------------------------------------------------------------ prompting
    def mouse_callback(self, event, x, y, _flags, _param):
        if not self.obstacle_mode:
            return super().mouse_callback(event, x, y, _flags, _param)
        if self.tracking_started:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.obstacle_points[self.current_obstacle].append([x, y])
            self.obstacle_labels[self.current_obstacle].append(1)
            self._obstacle_preview_dirty = True
            print(f"[obstacle {self.current_obstacle}] +positive point: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.obstacle_points[self.current_obstacle].append([x, y])
            self.obstacle_labels[self.current_obstacle].append(0)
            self._obstacle_preview_dirty = True
            print(f"[obstacle {self.current_obstacle}] -negative point: ({x}, {y})")

    def toggle_obstacle_mode(self):
        self.obstacle_mode = not self.obstacle_mode
        target = "OBSTACLE" if self.obstacle_mode else "OBJECT"
        print(f"Prompting mode: {target}")

    def next_object(self):
        if not self.obstacle_mode:
            return super().next_object()
        pts = self.obstacle_points[self.current_obstacle]
        lbls = self.obstacle_labels[self.current_obstacle]
        if len(pts) == 0:
            print(
                f"[obstacle {self.current_obstacle}] no points yet; click before "
                "pressing 'n'."
            )
            return
        if not any(lbl == 1 for lbl in lbls):
            print(
                f"[obstacle {self.current_obstacle}] add at least one positive "
                "(left-click) point first."
            )
            return
        self.obstacle_points.append([])
        self.obstacle_labels.append([])
        self.current_obstacle += 1
        self._obstacle_preview_dirty = True
        print(f"Started obstacle {self.current_obstacle}.")

    def reset_points(self):
        self.obstacle_points = [[]]
        self.obstacle_labels = [[]]
        self.current_obstacle = 0
        self.obstacle_mode = False
        self._obstacle_preview_masks = None
        self._obstacle_preview_dirty = False
        self.obstacle_tracker.reset()
        super().reset_points()

    def _handle_key(self, key) -> bool:
        if key == ord("o") and not self.tracking_started:
            self.toggle_obstacle_mode()
            return True
        return super()._handle_key(key)

    # ------------------------------------------------------------ tracking
    def start_tracking(self):
        groups = [
            (pts, lbls)
            for pts, lbls in zip(self.object_points, self.object_labels)
            if len(pts) > 0
        ]
        if len(groups) == 0:
            print("No object points collected! Please click on objects first.")
            return False

        frame = self.create_frame_from_lcm(0)
        if frame is None:
            print("Waiting for both RGBD and camera info before tracking can start.")
            return False

        self.pipeline.add_user_points(
            [pts for pts, _ in groups], [lbls for _, lbls in groups]
        )
        self.current_poses = self.pipeline.step(frame)

        # Obstacle: same first frame, separate SAM2 predictor.
        num_obstacles = self.obstacle_tracker.initialize(
            frame.rgb, self.obstacle_points, self.obstacle_labels
        )
        if num_obstacles > 0:
            self.obstacle_tracker.step(frame)
        self._publish_current_objects(frame.timestamp)

        self.tracking_started = True
        self.frame_count = 1
        print(
            f"Started tracking with {len(groups)} object(s) and "
            f"{num_obstacles} obstacle(s)"
        )
        return True

    def _publish_current_objects(self, timestamp: float):
        super()._publish_current_objects(timestamp)
        self._publish_obstacles(timestamp)

    def _publish_obstacles(self, timestamp: float):
        if not self.obstacle_tracker.initialized:
            return
        if self._latest_camera_info is None:
            return
        camera_to_world = inverse_SE3(
            np.asarray(self._latest_camera_info.world_to_camera, dtype=np.float64)
        )
        names, vecs = [], []
        for idx, est in enumerate(self.obstacle_tracker.estimates):
            if not self.obstacle_tracker.should_publish(est):
                continue
            vecs.append(
                self.obstacle_tracker.build_sphere_vector(
                    est, camera_to_world=camera_to_world
                )
            )
            names.append(self.obstacle_tracker.obstacle_name_from_index(idx))
        if not vecs:
            return
        self.obstacle_publisher.submit(
            NamedVecListPayload(
                channel=self._obstacle_channel,
                timestamp=float(timestamp),
                names=names,
                vecs=np.stack(vecs, axis=0).astype(np.float32),
            )
        )

    # ------------------------------------------------------------ drawing
    def _draw_obstacle_mask_overlay(self, display_bgr, masks):
        """Red tint for obstacle masks (objects use the hue palette of the base)."""
        if not masks:
            return display_bgr
        union = np.zeros(display_bgr.shape[:2], dtype=bool)
        for m in masks:
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape != union.shape:
                m = cv2.resize(
                    m.astype(np.uint8),
                    (union.shape[1], union.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            union |= m
        if not union.any():
            return display_bgr
        overlay = display_bgr.copy()
        overlay[union] = self.OBSTACLE_COLOR
        return cv2.addWeighted(display_bgr, 0.6, overlay, 0.4, 0)

    def _draw_obstacle_prompts(self, display_bgr):
        for pts, lbls in zip(self.obstacle_points, self.obstacle_labels):
            for point, label in zip(pts, lbls):
                px, py = int(point[0]), int(point[1])
                if label == 1:
                    cv2.circle(display_bgr, (px, py), 6, self.OBSTACLE_COLOR, 2)
                    cv2.circle(display_bgr, (px, py), 2, self.OBSTACLE_COLOR, -1)
                else:
                    cv2.drawMarker(
                        display_bgr,
                        (px, py),
                        (0, 0, 128),
                        cv2.MARKER_TILTED_CROSS,
                        12,
                        2,
                    )

    def _draw_obstacle_spheres(self, display_bgr, intrinsics):
        K = np.asarray(intrinsics, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for idx, est in enumerate(self.obstacle_tracker.estimates):
            if est.center_cam is None or est.center_cam[2] <= 1e-6:
                continue
            X, Y, Z = est.center_cam
            u = int(round(fx * X / Z + cx))
            v = int(round(fy * Y / Z + cy))
            r_px = max(2, int(round(est.radius_m * 0.5 * (fx + fy) / Z)))
            color = self.OBSTACLE_COLOR if est.visible else (0, 100, 200)
            cv2.circle(display_bgr, (u, v), r_px, color, 2)
            cv2.drawMarker(display_bgr, (u, v), color, cv2.MARKER_CROSS, 10, 1)
            cv2.putText(
                display_bgr,
                f"obstacle_{idx} z={Z:.2f}m r={est.radius_m:.3f}"
                + ("" if est.visible else " (lost)"),
                (u + r_px + 4, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return display_bgr

    def _draw_tracker_crop_window(self, display_bgr):
        """Outline the crop window when the point tracker is `tapir_crop`."""
        tracker = getattr(getattr(self.pipeline, "frontend", None), "tracker", None)
        box = getattr(tracker, "last_crop_box", None)
        if box is None:
            return display_bgr
        x0, y0, w, h = (int(v) for v in box)
        color = (255, 255, 0)  # cyan
        cv2.rectangle(display_bgr, (x0, y0), (x0 + w, y0 + h), color, 2)
        cv2.putText(
            display_bgr,
            f"crop {w}x{h} -> {tracker._resize_width}x{tracker._resize_height}",
            (x0 + 4, max(14, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        return display_bgr

    def visualize_tracking_results(self, frame, objects, frame_id=None):
        display = super().visualize_tracking_results(frame, objects, frame_id)
        display = self._draw_tracker_crop_window(display)
        if self.obstacle_tracker.initialized:
            display = self._draw_obstacle_mask_overlay(
                display, [e.mask for e in self.obstacle_tracker.estimates]
            )
            display = self._draw_obstacle_spheres(display, frame.intrinsics)
        return display

    # ------------------------------------------------------------ main loop
    def run(self):
        self.subscriber.start()
        self.publisher.start()
        self.mesh_pose_publisher.start()
        self.obstacle_publisher.start()

        try:
            while True:
                self._latest_camera_info = (
                    self.subscriber.get_latest_camera_info() or self._latest_camera_info
                )

                if not self.tracking_started:
                    rgbd_packet = self.subscriber.peek_latest_rgbd()
                    if rgbd_packet is None:
                        self._show_status_screen("Waiting for LCM RGBD...")
                        key = cv2.waitKey(1) & 0xFF
                        if not self._handle_key(key):
                            break
                        continue

                    rgb = self._normalize_rgb(rgbd_packet.rgb_image)
                    display_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if self._preview_dirty:
                        try:
                            self._preview_masks = self.pipeline.preview_user_masks(
                                rgb, self.object_points, self.object_labels
                            )
                        except Exception as exc:
                            print(f"[preview] SAM2 preview failed: {exc}")
                            self._preview_masks = None
                        self._preview_dirty = False

                    if self._obstacle_preview_dirty:
                        try:
                            logits = self.obstacle_tracker.preview(
                                rgb, self.obstacle_points, self.obstacle_labels
                            )
                            self._obstacle_preview_masks = (
                                self.obstacle_tracker._logits_to_bool_masks(
                                    logits, rgb.shape[:2]
                                )
                            )
                        except Exception as exc:
                            print(f"[preview] obstacle SAM2 preview failed: {exc}")
                            self._obstacle_preview_masks = None
                        self._obstacle_preview_dirty = False

                    display_frame = self._draw_mask_overlay(
                        display_frame, self._preview_masks
                    )
                    display_frame = self._draw_obstacle_mask_overlay(
                        display_frame, self._obstacle_preview_masks
                    )

                    total_points = 0
                    for obj_idx, (pts, lbls) in enumerate(
                        zip(self.object_points, self.object_labels)
                    ):
                        obj_color = self._obj_palette[obj_idx % len(self._obj_palette)]
                        for point, label in zip(pts, lbls):
                            total_points += 1
                            px, py = int(point[0]), int(point[1])
                            if label == 1:
                                cv2.circle(display_frame, (px, py), 5, obj_color, -1)
                            else:
                                cv2.drawMarker(
                                    display_frame,
                                    (px, py),
                                    (0, 0, 255),
                                    cv2.MARKER_TILTED_CROSS,
                                    12,
                                    2,
                                )
                    self._draw_obstacle_prompts(display_frame)

                    cv2.putText(
                        display_frame,
                        "L-click +, R-click -, 'n' next, 'o' obj/obstacle, "
                        "'s' start, 'r' reset",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                    num_objects = sum(1 for pts in self.object_points if pts)
                    num_obstacles = sum(1 for pts in self.obstacle_points if pts)
                    obstacle_points = sum(len(p) for p in self.obstacle_points)
                    if self.obstacle_mode:
                        status = (
                            f"[OBSTACLE mode] obstacle {self.current_obstacle} | "
                            f"obstacles: {num_obstacles} | points: {obstacle_points}"
                        )
                        status_color = self.OBSTACLE_COLOR
                    else:
                        status = (
                            f"[OBJECT mode] object {self.current_obj} | "
                            f"objects: {num_objects} | points: {total_points}"
                        )
                        status_color = self._obj_palette[
                            self.current_obj % len(self._obj_palette)
                        ]
                    cv2.putText(
                        display_frame,
                        status,
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        status_color,
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

                    if self._save_images:
                        cv2.imwrite(
                            str(
                                self._output_image_dir
                                / f"point_collection_{self.frame_count:06d}.png"
                            ),
                            display_frame,
                        )

                else:
                    if self._drop_stale_frames:
                        rgbd_packet = self.subscriber.pop_latest_rgbd()
                    else:
                        rgbd_packet = self.subscriber.pop_oldest_rgbd()

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

                    self.pipeline.step(frame)
                    self.obstacle_tracker.step(frame)
                    self._publish_current_objects(frame.timestamp)

                    display_frame = self.visualize_tracking_results(
                        frame, self.pipeline.objects, self.frame_count
                    )

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
                    n_obj = len(self.current_poses) if self.current_poses is not None else 0
                    cv2.putText(
                        display_frame,
                        f"Objects: {n_obj} | Obstacles: "
                        f"{self.obstacle_tracker.num_obstacles}",
                        (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    self.frame_count += 1

                cv2.imshow(self._window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.subscriber.stop()
            self.publisher.stop()
            self.mesh_pose_publisher.stop()
            self.obstacle_publisher.stop()
            cv2.destroyAllWindows()
