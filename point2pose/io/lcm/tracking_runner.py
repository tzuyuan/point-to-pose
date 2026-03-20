from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from omegaconf import OmegaConf

from point2pose.data_types.frame import Frame
from point2pose.io.lcm.data_models import NamedVecListPayload
from point2pose.io.lcm.pose_export import (
    build_bbox_pose_vector,
    build_mesh_pose_vector,
    object_name_from_index,
    resolve_bbox_center_pose,
    resolve_visualization_box,
)
from point2pose.io.lcm.runtime import NamedVecListLcmPublisher, RgbdLcmSubscriber
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.visualization import (
    _resolve_pose_and_bbox,
    draw_oriented_3d_box,
    draw_points_on_image,
    draw_posed_3d_box,
    draw_xyz_axis,
    get_n_uncertainty_colors,
)


class LcmTrackingRunner:
    def __init__(self, config_path: str = "configs/pipeline/pipeline_test.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self._lcm_cfg = self.cfg.get("lcm", {})
        self._visual_cfg = self.cfg.get("visualization", {}).get("params", {})

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

        self._visualize_points = bool(self._visual_cfg.get("visualize_points", True))
        self._points_vis_method = str(
            self._visual_cfg.get("points_vis_method", "visible_uncertainty")
        )
        self._save_images = bool(self._visual_cfg.get("save_images", False))
        self._output_image_dir = Path(
            self._visual_cfg.get(
                "output_image_dir", "/home/justin/code/point-to-pose/debug/output_images"
            )
        )
        if self._save_images:
            self._output_image_dir.mkdir(parents=True, exist_ok=True)

        bbox_mode = self.cfg.pipeline.params.get("bbox_estimation_mode", None)
        if bbox_mode is None:
            bbox_mode = (
                "first_frame_dense"
                if self.cfg.pipeline.params.get("estimate_init_pose", False)
                else "continuous"
            )
        self._bbox_estimation_mode = str(bbox_mode).lower()

        self.subscriber = RgbdLcmSubscriber(
            rgbd_channel=self._rgbd_channel,
            camera_info_channel=self._camera_info_channel,
            sub_poll_hz=self._sub_poll_hz,
            drop_stale_frames=self._drop_stale_frames,
            max_frame_drain=self._max_frame_drain,
            verbose=self._verbose,
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

        self.click_point_groups = []
        self.click_label_groups = []
        self.current_click_points = []
        self.current_click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.pipeline = None
        self._latest_frame_for_init = None
        self._latest_camera_info = None

        cv2.startWindowThread()
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self.mouse_callback)
        self._show_status_screen("Waiting for LCM RGBD and camera info...")

    def _initialize_pipeline(self):
        if self.pipeline is not None:
            return

        self._show_status_screen("Initializing tracking pipeline...")
        if self.cfg.pipeline.type == "single_process":
            from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess

            self.pipeline = PipelineSingleProcess(self.cfg)
        elif self.cfg.pipeline.type == "modular":
            from point2pose.pipeline.modular_pipeline import ModularPipeline

            self.pipeline = ModularPipeline(self.cfg)
        else:
            raise ValueError(f"Invalid pipeline type: {self.cfg.pipeline.type}")

    def mouse_callback(self, event, x, y, _flags, _param):
        if self.tracking_started:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_click_points.append([x, y])
            self.current_click_labels.append(1)
            print(
                f"Added positive point for object {self._active_prompt_object_index()}: ({x}, {y})"
            )
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.current_click_points.append([x, y])
            self.current_click_labels.append(0)
            print(
                f"Added negative point for object {self._active_prompt_object_index()}: ({x}, {y})"
            )

    def reset_points(self):
        self.click_point_groups = []
        self.click_label_groups = []
        self.current_click_points = []
        self.current_click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.pipeline = None
        self._show_status_screen(
            "Prompts reset. Add clicks for object 1, press 's' for next object, 'r' to run."
        )

    def _count_positive_labels(self, labels) -> int:
        return int(np.sum(np.asarray(labels, dtype=np.int32) == 1))

    def _active_prompt_object_index(self) -> int:
        return len(self.click_point_groups) + 1

    def _prompt_object_color(self, obj_idx: int):
        palette = [
            (0, 255, 0),
            (255, 200, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (0, 128, 255),
        ]
        return palette[obj_idx % len(palette)]

    def _finalize_current_object(self) -> bool:
        if len(self.current_click_points) == 0:
            print("No points collected for the current object yet.")
            return False
        if self._count_positive_labels(self.current_click_labels) == 0:
            print(
                "The current object needs at least one positive click before it can be finalized."
            )
            return False

        self.click_point_groups.append([list(pt) for pt in self.current_click_points])
        self.click_label_groups.append(
            [int(label) for label in self.current_click_labels]
        )
        obj_idx = len(self.click_point_groups)
        num_pos = self._count_positive_labels(self.current_click_labels)
        num_neg = len(self.current_click_labels) - num_pos
        self.current_click_points = []
        self.current_click_labels = []
        print(
            f"Finalized object {obj_idx} with {num_pos} positive and {num_neg} negative point(s)."
        )
        return True

    def _prompt_groups_ready_for_tracking(self) -> bool:
        return len(self.click_point_groups) > 0 or len(self.current_click_points) > 0

    def _prepare_prompt_groups_for_tracking(self):
        if len(self.current_click_points) > 0:
            if not self._finalize_current_object():
                return None, None
        if len(self.click_point_groups) == 0:
            print("No object prompts collected yet. Click points before pressing 'r'.")
            return None, None

        prompt_points = [
            [list(point) for point in point_group]
            for point_group in self.click_point_groups
        ]
        prompt_labels = [
            [int(label) for label in label_group]
            for label_group in self.click_label_groups
        ]
        return prompt_points, prompt_labels

    def _draw_prompt_group(self, image, points, labels, obj_idx, is_current=False):
        color = self._prompt_object_color(obj_idx)
        label_prefix = f"O{obj_idx + 1}"
        for point_idx, (point, label) in enumerate(zip(points, labels), start=1):
            px = int(point[0])
            py = int(point[1])
            if int(label) == 1:
                cv2.circle(image, (px, py), 6 if is_current else 5, color, -1)
                cv2.circle(image, (px, py), 8 if is_current else 7, (255, 255, 255), 1)
            else:
                cv2.drawMarker(
                    image,
                    (px, py),
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12 if is_current else 10,
                    thickness=2,
                )
            cv2.putText(
                image,
                f"{label_prefix}:{point_idx}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color if int(label) == 1 else (0, 0, 255),
                1,
            )

    def _draw_prompt_collection_overlay(self, display_frame):
        for obj_idx, (points, labels) in enumerate(
            zip(self.click_point_groups, self.click_label_groups)
        ):
            self._draw_prompt_group(display_frame, points, labels, obj_idx=obj_idx)

        if len(self.current_click_points) > 0:
            self._draw_prompt_group(
                display_frame,
                self.current_click_points,
                self.current_click_labels,
                obj_idx=len(self.click_point_groups),
                is_current=True,
            )

        lines = [
            "L:+  R:-  s:next object  r:run  c:clear  q:quit",
            f"Finalized objects: {len(self.click_point_groups)} | Active object: {self._active_prompt_object_index()}",
            (
                f"Current points: {len(self.current_click_points)}"
                f" | Ready to run: {'yes' if self._prompt_groups_ready_for_tracking() else 'no'}"
            ),
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                display_frame,
                line,
                (10, 30 + 30 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _show_status_screen(self, message: str):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lines = [
            message,
            "L:+  R:-  s:next object  r:run  c:clear  q:quit",
        ]
        if self._bbox_estimation_mode == "manual":
            lines.append("Press 'b' during tracking to estimate the bounding box")
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

    def _build_frame(self, rgbd_packet, camera_info, frame_id: int) -> Frame:
        rgb = self._normalize_rgb(rgbd_packet.rgb_image)
        depth = np.asarray(rgbd_packet.depth_image).copy()
        return Frame(
            id=frame_id,
            rgb=rgb,
            depth=depth,
            intrinsics=np.asarray(camera_info.intrinsics, dtype=np.float64).copy(),
            depth_factor=float(camera_info.depth_factor),
            timestamp=float(rgbd_packet.timestamp),
        )

    def start_tracking(self):
        prompt_points, prompt_labels = self._prepare_prompt_groups_for_tracking()
        if prompt_points is None or prompt_labels is None:
            return False

        rgbd_packet = self.subscriber.peek_latest_rgbd()
        camera_info = self.subscriber.get_latest_camera_info()
        if rgbd_packet is None or camera_info is None:
            print("Waiting for both RGBD and camera info before tracking can start.")
            return False

        self._initialize_pipeline()
        frame = self._build_frame(rgbd_packet, camera_info, frame_id=0)
        self.pipeline.add_user_points(prompt_points, prompt_labels)
        self.pipeline.step(frame)
        self._publish_current_objects(frame.timestamp, camera_info)

        self._latest_frame_for_init = frame
        self._latest_camera_info = camera_info
        self.tracking_started = True
        self.frame_count = 1

        total_points = int(sum(len(group) for group in prompt_points))
        print(
            f"Started tracking with {total_points} prompt point(s) across {len(prompt_points)} object(s)."
        )
        return True

    def _publish_current_objects(self, timestamp: float, camera_info):
        if self.pipeline is None or not getattr(self.pipeline, "objects", None):
            return
        camera_to_world = inverse_SE3(
            np.asarray(camera_info.world_to_camera, dtype=np.float64)
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

    def visualize_tracking_results(self, frame: Frame):
        display_frame = cv2.cvtColor(frame.rgb.copy(), cv2.COLOR_RGB2BGR)
        height, _ = display_frame.shape[:2]

        if hasattr(frame, "mask") and frame.mask is not None:
            mask_overlay = np.zeros((height, display_frame.shape[1], 3), dtype=np.uint8)
            mask_overlay[..., 1] = 255
            for idx in range(len(frame.mask)):
                obj_mask = (frame.mask[idx, 0] > 0.0).cpu().numpy()
                if np.any(obj_mask):
                    hue = (idx + 3) / (len(frame.mask) + 3) * 255
                    mask_overlay[obj_mask, 0] = hue
                    mask_overlay[obj_mask, 2] = 255
            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
            display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)

        if (
            self._visualize_points
            and self.pipeline is not None
            and hasattr(self.pipeline, "track_table")
            and self.pipeline.track_table is not None
        ):
            for idx, obj in enumerate(self.pipeline.objects):
                if idx not in self.pipeline.track_table.obj2track_map:
                    continue
                track_idx = self.pipeline.track_table.obj2track_map[idx]
                track_points = self.pipeline.track_table.track_2d[track_idx]
                if self._points_vis_method == "uncertainty":
                    colors = get_n_uncertainty_colors(
                        self.pipeline.track_table.uncertainty[track_idx]
                    )
                    draw_points_on_image(display_frame, track_points, colors)
                elif self._points_vis_method == "visible":
                    colors = np.full((len(track_points), 3), (0, 0, 255), dtype=np.uint8)
                    colors[self.pipeline.track_table.visible[track_idx]] = (0, 255, 0)
                    draw_points_on_image(display_frame, track_points, colors)
                elif self._points_vis_method == "visible_valid":
                    colors = np.full((len(track_points), 3), (0, 0, 255), dtype=np.uint8)
                    visible = self.pipeline.track_table.visible[track_idx]
                    valid = self.pipeline.track_table.valid[track_idx]
                    colors[visible & valid] = (0, 255, 0)
                    draw_points_on_image(display_frame, track_points, colors)
                else:
                    visible_mask = self.pipeline.track_table.visible[track_idx]
                    if np.any(visible_mask):
                        colors = get_n_uncertainty_colors(
                            self.pipeline.track_table.uncertainty[track_idx]
                        )
                        draw_points_on_image(
                            display_frame,
                            track_points[visible_mask],
                            colors[visible_mask],
                        )

        for obj in getattr(self.pipeline, "objects", []):
            if getattr(obj, "pose", None) is None:
                continue
            pose_in_cam, bbox_source, assume_pose_is_bbox_center = resolve_visualization_box(obj)
            axis_pose = resolve_bbox_center_pose(
                pose_in_cam=pose_in_cam,
                bbox_source=bbox_source,
                assume_pose_is_bbox_center=assume_pose_is_bbox_center,
            )
            if bbox_source is not None:
                pose_draw, bbox_min_max_local, bbox_corners_local = _resolve_pose_and_bbox(
                    pose_in_cam=pose_in_cam,
                    bbox_source=bbox_source,
                    bbox_frame="mesh",
                    assume_pose_is_bbox_center=assume_pose_is_bbox_center,
                )
                if bbox_corners_local is not None:
                    display_frame = draw_oriented_3d_box(
                        frame.intrinsics, display_frame, pose_draw, bbox_corners_local
                    )
                elif bbox_min_max_local is not None:
                    display_frame = draw_posed_3d_box(
                        frame.intrinsics, display_frame, pose_draw, bbox_min_max_local
                    )
            display_frame = draw_xyz_axis(
                image=display_frame, ob_in_cam=axis_pose, K=frame.intrinsics
            )

        return display_frame

    def run(self):
        self.subscriber.start()
        self.publisher.start()
        self.mesh_pose_publisher.start()

        try:
            while True:
                self._latest_camera_info = self.subscriber.get_latest_camera_info()
                if not self.tracking_started:
                    rgbd_packet = self.subscriber.peek_latest_rgbd()
                    if rgbd_packet is None:
                        self._show_status_screen("Waiting for LCM RGBD...")
                        key = cv2.waitKey(1) & 0xFF
                    elif self._latest_camera_info is None:
                        frame = cv2.cvtColor(self._normalize_rgb(rgbd_packet.rgb_image), cv2.COLOR_RGB2BGR)
                        self._draw_prompt_collection_overlay(frame)
                        cv2.putText(
                            frame,
                            "Waiting for LCM camera info...",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow(self._window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        frame = cv2.cvtColor(self._normalize_rgb(rgbd_packet.rgb_image), cv2.COLOR_RGB2BGR)
                        self._draw_prompt_collection_overlay(frame)
                        cv2.imshow(self._window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                else:
                    if self._drop_stale_frames:
                        rgbd_packet = self.subscriber.pop_latest_rgbd()
                    else:
                        rgbd_packet = self.subscriber.pop_oldest_rgbd()
                    if rgbd_packet is None:
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        camera_info = (
                            self._latest_camera_info
                            if self._latest_camera_info is not None
                            else self.subscriber.get_latest_camera_info()
                        )
                        if camera_info is None:
                            key = cv2.waitKey(1) & 0xFF
                            continue
                        frame = self._build_frame(rgbd_packet, camera_info, self.frame_count)
                        self.pipeline.step(frame)
                        self._publish_current_objects(frame.timestamp, camera_info)
                        display_frame = self.visualize_tracking_results(frame)
                        cv2.putText(
                            display_frame,
                            f"Frame: {self.frame_count}",
                            (10, display_frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            f"Objects: {len(getattr(self.pipeline, 'objects', []))}",
                            (10, display_frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow(self._window_name, display_frame)
                        if self._save_images:
                            image_path = self._output_image_dir / f"frame_{self.frame_count:06d}.png"
                            cv2.imwrite(str(image_path), display_frame)
                        self.frame_count += 1
                        key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.reset_points()
                elif key == ord("s") and not self.tracking_started:
                    self._finalize_current_object()
                elif key == ord("r") and not self.tracking_started:
                    if self.start_tracking():
                        print("Pipeline tracking started!")
                    else:
                        print("Failed to start pipeline tracking!")
                elif (
                    key == ord("b")
                    and self.tracking_started
                    and self._bbox_estimation_mode == "manual"
                ):
                    if self.pipeline is None or not hasattr(self.pipeline, "update_object_bboxes"):
                        print("Manual bbox estimation is not available in this pipeline.")
                    else:
                        updated = self.pipeline.update_object_bboxes(force=True)
                        print(f"Manual bbox estimation updated {updated} object(s).")
                time.sleep(0.001)
        finally:
            self.subscriber.stop()
            self.publisher.stop()
            self.mesh_pose_publisher.stop()
            cv2.destroyAllWindows()
