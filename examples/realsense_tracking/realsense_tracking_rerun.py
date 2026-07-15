"""
Live RealSense + SAM2 + point2pose tracking, visualized in Rerun (like ../kv_tracker's demo):
real-time camera frustum, camera trajectory, and the live 3D point map being tracked.

Two 3D panels are logged side by side:
  - "world/obj_<i>"      : object fixed at the origin, live camera frustum + trajectory moving
                            (point2pose's native object-frame convention).
  - "camframe/obj_<i>"   : camera fixed at the origin, object/points moving instead -- with
                            short trailing traces (last N frames) of the currently-visible
                            tracked points, so you can see which points are actively locked on.

Same click-to-select / 's' to start / 'q' to quit flow as realsense_tracking.py; the cv2 window
still shows the 2D RGB + tracked points, and a Rerun viewer (auto-spawned) shows the 3D scene.

Run (in the ``point2pose`` conda env):
    python examples/realsense_tracking/realsense_tracking_rerun.py \
        [--config configs/pipeline/pipeline_test2.yaml]
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr
from omegaconf import OmegaConf

from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.pipeline.components.reconstruction_exporter import ReconstructionExporter
from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3, transform_pts
from point2pose.utils.visualization import draw_points_on_image
from point2pose.utils.rerun_viz import (
    rr_init,
    rr_log_camera,
    rr_log_live_camera,
    rr_log_trajectory,
    rr_log_points3d,
    rr_log_bbox,
    rr_log_fps,
    rr_log_point_traces,
    colors_for_track_ids,
    PointTraceBuffer,
)

TRACE_LEN = 5  # frames of trailing history to keep per visible point in the camera-fixed panel

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/pipeline/pipeline_test2.yaml"
WIN = "RealSense Tracker (Rerun demo)"


class RealSenseRerunTracker:
    """Same flow as RealSensePipelineTracker, plus live Rerun 3D visualization."""

    def __init__(self, config_path=str(DEFAULT_CONFIG), export_dir=None):
        self.cfg = OmegaConf.load(config_path)
        rs_serial = self.cfg.realsense.params.rs_serial

        self._init_realsense(rs_serial)

        self.exporter = ReconstructionExporter(export_dir) if export_dir else None

        # Optional viewport crop: aim+lock a square crop of the full sensor frame before
        # tracking starts, then run SAM clicks / TAPIR / everything downstream on the
        # cropped+resized frame instead of the full 640x480 image. This matters because
        # tracker.params.resize_height/width (e.g. 256) resizes the WHOLE live frame before
        # TAPIR sees it -- if the object only occupies a small fraction of the frame, its
        # effective resolution after that resize is tiny. Cropping first keeps the object
        # filling the tracker's input. Disabled (full-frame, legacy behavior) unless
        # input.params.crop_size is set in the config -- mirrors examples/model_based_tracking/
        # realsense_tracking_model.py's pick_viewport()/_crop_resize()/_cropped_resized_K().
        ip = self.cfg.input.params if "input" in self.cfg else {}
        self.crop_size = int(ip["crop_size"]) if "crop_size" in ip else None
        self.track_res = int(ip.get("track_res", self.crop_size)) if self.crop_size else None
        self.crop_xy = None  # set by pick_viewport(); (x0,y0) top-left of the locked crop
        self._mouse_xy = (self.img_w // 2, self.img_h // 2)

        # "Live" intrinsics/resolution actually fed to the pipeline -- equal to the full
        # sensor frame's until pick_viewport() locks a crop, at which point they become the
        # cropped+resized ones. self.camera_intrinsics/img_w/img_h (set in _init_realsense)
        # stay at the FULL sensor resolution always, since _clamped_crop_xy/pick_viewport
        # need the uncropped bounds.
        self.K = self.camera_intrinsics
        self.live_w, self.live_h = self.img_w, self.img_h

        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None

        # per-object camera-center trajectory (world == object frame), for the rerun path
        self.cam_traj = {}
        # per-object rolling point traces (camera-fixed panel), keyed by object id
        self.point_traces = {}
        self._fps_hist = []

        cv2.startWindowThread()
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.mouse_callback)

        if self.cfg.pipeline.type == "single_process":
            self.pipeline = PipelineSingleProcess(self.cfg)
        elif self.cfg.pipeline.type == "modular":
            self.pipeline = ModularPipeline(self.cfg)
        else:
            raise ValueError(f"Invalid pipeline type: {self.cfg.pipeline.type}")

        rr_init("point2pose live tracking")

        print("Instructions:")
        print("- Left click: Add positive point")
        print("- Right click: Add negative point")
        print("- Press 's': Start tracking")
        print("- Press 'q': Quit")
        print("- Press 'r': Reset points")

    def _init_realsense(self, rs_serial):
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self._rs_align = rs.align(rs.stream.color)
        profile = self.rs_pipeline.start(config)

        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_intrinsics = np.array(
            [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]],
            dtype=np.float64,
        )
        self.img_w, self.img_h = intrinsics.width, intrinsics.height
        self.depth_factor = 1000.0

        print("Camera initialized:")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    def mouse_callback(self, event, x, y, _flags, _param):
        # Once a viewport crop is locked, the cv2 window shows the cropped+resized frame
        # 1:1, so click coords are already in the cropped frame's pixel space -- no
        # transform needed here (pipeline.step() only ever sees cropped frames after lock).
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_points.append([x, y])
            self.click_labels.append(1)
            print(f"Added positive point: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click_points.append([x, y])
            self.click_labels.append(0)
            print(f"Added negative point: ({x}, {y})")

    def _viewport_cb(self, event, x, y, _flags, _param):
        self._mouse_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_xy = self._clamped_crop_xy(x, y)

    def _clamped_crop_xy(self, cx, cy):
        half = self.crop_size // 2
        x0 = int(np.clip(cx - half, 0, self.img_w - self.crop_size))
        y0 = int(np.clip(cy - half, 0, self.img_h - self.crop_size))
        return (x0, y0)

    def pick_viewport(self):
        """Aim+lock a square crop of the full sensor frame before tracking. No-op if
        crop_size wasn't configured (input.params.crop_size)."""
        if self.crop_size is None:
            return
        print(f"[viewport] aim the {self.crop_size}px crop with the mouse, "
              "LEFT click to lock ('q' to quit)")
        cv2.setMouseCallback(WIN, self._viewport_cb)
        self.crop_xy = None
        while self.crop_xy is None:
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            rgb = np.asanyarray(color_frame.get_data())
            disp = rgb.copy()
            x0, y0 = self._clamped_crop_xy(*self._mouse_xy)
            cv2.rectangle(disp, (x0, y0), (x0 + self.crop_size, y0 + self.crop_size),
                          (0, 255, 255), 2)
            cv2.putText(disp, f"VIEWPORT {self.crop_size}px (click to lock)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("viewport selection cancelled")

        self.K = self._cropped_resized_K(self.camera_intrinsics, self.crop_xy,
                                         self.crop_size, self.track_res)
        self.live_w = self.live_h = self.track_res
        print(f"[viewport] locked crop_xy={self.crop_xy}, size={self.crop_size} "
              f"-> {self.track_res}px")
        cv2.setMouseCallback(WIN, self.mouse_callback)  # switch to SAM +/- click phase

    def _crop_resize(self, img, interp):
        x0, y0 = self.crop_xy
        crop = img[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        return cv2.resize(crop, (self.track_res, self.track_res), interpolation=interp)

    @staticmethod
    def _cropped_resized_K(K_full, crop_xy, crop_size, out_res):
        x0, y0 = crop_xy
        s = out_res / float(crop_size)
        K = K_full.copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] *= s
        K[1, 2] *= s
        return K

    def reset_points(self):
        self.click_points = []
        self.click_labels = []
        self.tracking_started = False
        self.frame_count = 0
        self.current_poses = None
        self.cam_traj = {}
        self.point_traces = {}

        if self.cfg.pipeline.type == "single_process":
            self.pipeline = PipelineSingleProcess(self.cfg)
        elif self.cfg.pipeline.type == "modular":
            self.pipeline = ModularPipeline(self.cfg)
        else:
            raise ValueError(f"Invalid pipeline type: {self.cfg.pipeline.type}")

        if self.exporter is not None:
            # frame_count restarts at 0 above -- reset export state too, or
            # all_frames_poses.npy/idx.npy would mix poses across separate sessions.
            self.exporter = ReconstructionExporter(self.exporter.results_path)

        print("Points reset. Click to add new points.")

    def _grab_frame(self, frame_id):
        frames = self.rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        if self.crop_xy is not None:
            rgb = self._crop_resize(rgb, cv2.INTER_LINEAR)
            depth = self._crop_resize(depth, cv2.INTER_NEAREST)

        return Frame(
            id=frame_id, rgb=rgb, depth=depth,
            intrinsics=self.K, depth_factor=self.depth_factor,
            timestamp=time.time(),
        )

    def start_tracking(self):
        if len(self.click_points) == 0:
            print("No points collected! Please click on objects first.")
            return False

        frame = self._grab_frame(0)
        if frame is None:
            print("Failed to get frames for initialization")
            return False

        self.pipeline.add_user_points(self.click_points, self.click_labels)
        self.current_poses = self.pipeline.step(frame)

        self.tracking_started = True
        self.frame_count = 1
        print(f"Started tracking with {len(self.click_points)} points")
        print(f"Number of objects: {len(self.current_poses)}")
        return True

    # ------------------------------------------------------------------ #
    # 2D overlay (cv2 window) -- tracked points only; pose boxes are shown in rerun instead.
    # Each point gets a stable per-track-id color (visible: full brightness, invisible: dim),
    # matching the 3D panels.
    def _draw_2d(self, frame):
        disp = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR).copy()
        tt = self.pipeline.track_table
        for i, obj in enumerate(self.pipeline.objects):
            if i not in tt.obj2track_map:
                continue
            idx = tt.obj2track_map[i]
            pts2d = tt.track_2d[idx]
            colors_rgb = colors_for_track_ids(idx, visible=tt.visible[idx])
            draw_points_on_image(disp, pts2d, colors_rgb[:, ::-1])  # RGB -> BGR for cv2
        return disp

    # ------------------------------------------------------------------ #
    # Reconstruction export (--export-dir): dumps RGB-D/mask/pose every frame.
    # reconstruct.py backprojects whichever frames it needs directly from these, so no
    # separate keyframe pointcloud export is needed. obj 0 only -- multi-object export
    # is not supported yet.
    def _export_step(self, frame):
        if not self.pipeline.objects:
            return
        obj = self.pipeline.objects[0]
        self.exporter.export_frame(frame, obj, self.frame_count)

    # ------------------------------------------------------------------ #
    # 3D rerun logging. Two panels per object:
    #   world/obj_<i>    -- object fixed at origin, camera moves (native convention)
    #   camframe/obj_<i> -- camera fixed at origin, object/points move, with trailing
    #                       traces of the currently-visible tracked points
    def _log_rerun(self, frame):
        tt = self.pipeline.track_table
        for i, obj in enumerate(self.pipeline.objects):
            if obj.pose is None:
                continue
            # obj.key_points live in the "map" frame (raw frame-0 camera coords, see
            # KeyFrameManager.initialize -> convert_pixel_to_world); obj.pose (the SVD/PnP
            # register() output) maps THAT frame directly to the current camera frame, so
            # key_points/camera/trajectory all use obj.pose alone.
            # obj.bbox, by contrast, is defined in a canonical LOCAL frame centered at its own
            # origin (see draw_posed_3d_box usage: bbox_min_max = [-half, +half]); obj.init_pose
            # (= [R,t] of the initial oriented bbox) places that canonical box into the map
            # frame, so the bbox alone needs the extra obj.init_pose factor.
            T_obj2cam = obj.pose
            T_bbox2cam = obj.pose @ obj.init_pose

            # ---- world panel: object fixed, camera moves ----
            wns = f"world/obj_{i}"
            rr_log_live_camera(
                f"{wns}/live_cam", T_obj2cam, self.K,
                self.live_w, self.live_h, image=frame.rgb,
                color=(255, 0, 0) if obj.lost else (0, 200, 60),
            )
            cam_center = inverse_SE3(T_obj2cam)[:3, 3]
            self.cam_traj.setdefault(i, []).append(cam_center)
            rr_log_trajectory(f"{wns}/trajectory", self.cam_traj[i])
            if getattr(obj, "bbox", None) is not None:
                rr_log_bbox(f"{wns}/bbox_local", obj.bbox.extent, (0.0, 0.0, 0.0), color=(255, 200, 0))

            # ---- all tracked map points (map frame), colored per track id, dim if invisible --
            # shared by both panels. Not every key_points row has a live track (older/dropped
            # rows), so restrict to rows with a valid track id via obj.kp_track_indices.
            vis_track_ids = np.empty((0,), dtype=np.int64)
            vis_obj_pts = np.empty((0, 3), dtype=np.float64)
            all_obj_pts = np.empty((0, 3), dtype=np.float64)
            colors_rgb = np.empty((0, 3), dtype=np.uint8)
            if i in tt.obj2track_map and len(obj.key_points):
                g_idx = tt.obj2track_map[i]
                obj_rows = obj.track_idx_2_obj_idx[g_idx]
                row_ok = obj_rows >= 0
                all_track_ids = g_idx[row_ok]
                all_obj_pts = obj.key_points[obj_rows[row_ok]]
                all_visible = tt.visible[all_track_ids]

                colors_rgb = colors_for_track_ids(all_track_ids, visible=all_visible)
                rr_log_points3d(f"{wns}/map_points", all_obj_pts, colors=colors_rgb, radii=0.003)

                if all_visible.any():
                    vis_track_ids = all_track_ids[all_visible]
                    vis_obj_pts = all_obj_pts[all_visible]

            # ---- camframe panel: camera fixed at origin, object/points move ----
            cns = f"camframe/obj_{i}"
            rr_log_camera(f"{cns}/fixed_cam", np.eye(4), self.K,
                          self.live_w, self.live_h, color=(255, 0, 0) if obj.lost else (0, 200, 60))
            if getattr(obj.bbox, "extent", None) is not None:
                rr_log_bbox(f"{cns}/bbox_cam", obj.bbox.extent, T_bbox2cam[:3, 3], color=(255, 200, 0))
            if all_obj_pts.size:
                pts_cam = transform_pts(T_obj2cam, all_obj_pts)
                rr_log_points3d(f"{cns}/map_points_cam", pts_cam, colors=colors_rgb, radii=0.003)

            if vis_track_ids.size:
                vis_pts_cam = transform_pts(T_obj2cam, vis_obj_pts)
                trace_buf = self.point_traces.setdefault(i, PointTraceBuffer(max_len=TRACE_LEN))
                trace_buf.update(self.frame_count, vis_track_ids, vis_pts_cam)
                rr_log_point_traces(f"{cns}/traces", trace_buf.strips(), color=(0, 255, 0))

    # ------------------------------------------------------------------ #
    def run_tracking(self):
        try:
            self.pick_viewport()
            while True:
                if not self.tracking_started:
                    frames = self.rs_pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    display_frame = np.asanyarray(color_frame.get_data()).copy()
                    if self.crop_xy is not None:
                        display_frame = self._crop_resize(display_frame, cv2.INTER_LINEAR)
                    for i, (point, label) in enumerate(zip(self.click_points, self.click_labels)):
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, color, -1)
                        cv2.putText(display_frame, f"{i+1}", (int(point[0]) + 10, int(point[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    cv2.putText(display_frame, "Left click: +, Right click: -, Press 's' to start",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Points: {len(self.click_points)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    t0 = time.perf_counter()
                    frame = self._grab_frame(self.frame_count)
                    if frame is None:
                        continue

                    self.pipeline.step(frame)
                    if self.exporter is not None:
                        self._export_step(frame)
                    rr.set_time("frame", sequence=self.frame_count)
                    self._log_rerun(frame)

                    display_frame = self._draw_2d(frame)
                    dt = time.perf_counter() - t0
                    self._fps_hist.append(1.0 / max(dt, 1e-6))
                    if len(self._fps_hist) > 30:
                        self._fps_hist.pop(0)
                    fps = float(np.mean(self._fps_hist))
                    rr_log_fps(self.frame_count, fps, extra_text=f"Objects: {len(self.pipeline.objects)}")

                    height, _ = display_frame.shape[:2]
                    cv2.putText(display_frame, f"Frame: {self.frame_count}  FPS: {fps:.1f}",
                                (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame,
                                f"Objects: {len(self.current_poses) if self.current_poses is not None else 0}",
                                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    self.frame_count += 1

                cv2.imshow(WIN, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and not self.tracking_started:
                    if self.start_tracking():
                        print("Pipeline tracking started!")
                    else:
                        print("Failed to start pipeline tracking!")
                elif key == ord("r"):
                    self.reset_points()

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--export-dir", default=None,
                     help="if set, export RGB-D/mask/pose (every frame) to this "
                          "directory, in a layout point2pose/model_tracking/"
                          "reconstruct.py can train from")
    args = ap.parse_args()
    try:
        tracker = RealSenseRerunTracker(args.config, export_dir=args.export_dir)
        tracker.run_tracking()
    except (RuntimeError, OSError, ImportError) as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. RealSense camera connected")
        print("2. Pipeline configuration file exists")
        print("3. Required dependencies installed (incl. rerun-sdk)")
        print("4. SAM2 and tracker checkpoints available")


if __name__ == "__main__":
    main()
