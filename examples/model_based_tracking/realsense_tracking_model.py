"""
Live model-based 6-DoF tracking on a RealSense stream.

Sequential flow (no callback-driven state machine -- callbacks only record clicks):
  Phase 1  pick_viewport()  -- aim a square crop on the full frame, click to lock.
  Phase 2  build_map()      -- render the model at the cropped intrinsics, seed TAPIR.
  Phase 3  pick_sam()       -- (only if use_segmenter) click +/- points, 's' to start.
  Phase 4  track()          -- live tracking: TAPIR -> PnP -> pose.

Everything downstream of the crop runs at ``track_res`` (a small square, e.g. 256), which MUST
equal TAPIR's img_* and the map render resolution.

Run (in the ``point2pose`` conda env):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/model_based_tracking/realsense_tracking_model.py \
        --config configs/pipeline/model_tracking.yaml [--viser]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for sub in ("LightGlue", "tapnet", "segment-anything-2-real-time"):
    p = str(Path(__file__).resolve().parents[2] / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

from point2pose.data_types.frame import Frame
from point2pose.pipeline.model_tracking_pipeline import ModelTrackingPipeline
from point2pose.utils.visualization import (
    draw_xyz_axis,
    draw_posed_3d_box,
    draw_points_on_image,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/pipeline/model_tracking.yaml"
WIN = "Model Tracker"
DEPTH_WIN = "Live Depth"


class RealSenseModelTracker:
    def __init__(self, config_path, use_viser=False):
        self.cfg = OmegaConf.load(config_path)
        self.cfg.setdefault("pipeline", OmegaConf.create({"params": {}}))
        self.use_segmenter = bool(self.cfg.pipeline.params.get("use_segmenter", False))
        self.use_viser = use_viser
        self.viser = None
        self.show_hidden_points = False

        ip = self.cfg.input.params if "input" in self.cfg else {}
        self.crop_size = int(ip.get("crop_size", 320))
        self.track_res = int(ip.get("track_res", 256))
        self.depth_viz_max_m = float(self.cfg.pipeline.params.get("max_depth", 2.0))

        rs_params = self.cfg.realsense.params if "realsense" in self.cfg else {}
        rs_serial = rs_params.get("rs_serial", None)
        self.depth_source = rs_params.get("depth_source", "realsense")
        assert self.depth_source in ("realsense", "foundation_stereo"), (
            f"unknown realsense.params.depth_source {self.depth_source!r}, "
            "expected 'realsense' or 'foundation_stereo'"
        )
        self._init_realsense(rs_serial, need_ir=(self.depth_source == "foundation_stereo"))

        self.fs_depth = None
        if self.depth_source == "foundation_stereo":
            from examples.model_based_tracking.foundation_stereo_depth import (
                FoundationStereoDepth,
            )
            print("[init] loading Fast-FoundationStereo (first run may take a while)...")
            self.fs_depth = FoundationStereoDepth(
                rs_params.foundation_stereo_model_dir,
                valid_iters=int(rs_params.get("foundation_stereo_valid_iters", 8)),
                hiera=bool(rs_params.get("foundation_stereo_hiera", False)),
            )

        self.pipeline = None
        self.crop_xy = ((self.W - self.crop_size) // 2, (self.H - self.crop_size) // 2)
        self.K = None  # cropped+resized intrinsics, set after pick_viewport

        # transient mouse state (callbacks only WRITE these; the loops read them)
        self._mouse_xy = (self.W // 2, self.H // 2)
        self._click_lock = False
        self.click_points = []
        self.click_labels = []
        self._start_sam = False

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, self.W, self.H)
        cv2.namedWindow(DEPTH_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEPTH_WIN, self.W, self.H)

    # ------------------------------------------------------------------ #
    def _init_realsense(self, rs_serial, need_ir=False):
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        if rs_serial is not None:
            config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if need_ir:
            # D4xx left/right IR sensors -- indices 1/2, same res as depth (they're what the
            # depth sensor itself stereo-matches from).
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        self._rs_align = rs.align(rs.stream.color)
        profile = self.rs_pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K_full = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64
        )
        self.depth_factor = 1000.0
        self.H, self.W = 480, 640

        if need_ir:
            left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ir_intr = left_profile.get_intrinsics()
            self.K_ir = np.array(
                [[ir_intr.fx, 0, ir_intr.ppx], [0, ir_intr.fy, ir_intr.ppy], [0, 0, 1]],
                dtype=np.float64,
            )
            extr = left_profile.get_extrinsics_to(right_profile)
            # RealSense extrinsics are stored row-major flattened; translation is left->right in
            # the left IR frame, so its norm is exactly the stereo baseline.
            self.ir_baseline_m = float(np.linalg.norm(np.array(extr.translation)))
            ir2color = left_profile.get_extrinsics_to(
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            )
            self.R_ir2color = np.array(ir2color.rotation, dtype=np.float64).reshape(3, 3).T
            self.t_ir2color = np.array(ir2color.translation, dtype=np.float64)
            print(f"[init] IR baseline: {self.ir_baseline_m * 1000:.2f}mm")

    def _grab_raw(self):
        frames = self.rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            return None, None
        rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)

        if self.depth_source == "foundation_stereo":
            left_f, right_f = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not left_f or not right_f:
                return None, None
            left_ir = np.asanyarray(left_f.get_data())
            right_ir = np.asanyarray(right_f.get_data())
            left_ir_rgb = np.tile(left_ir[..., None], (1, 1, 3))
            right_ir_rgb = np.tile(right_ir[..., None], (1, 1, 3))
            depth_ir = self.fs_depth.infer_depth(
                left_ir_rgb, right_ir_rgb, self.K_ir, self.ir_baseline_m
            )
            from examples.model_based_tracking.foundation_stereo_depth import reproject_depth
            depth_np = reproject_depth(
                depth_ir, self.K_ir, self.K_full, self.R_ir2color, self.t_ir2color,
                dst_hw=(self.H, self.W),
            ) * self.depth_factor  # meters -> the same raw-mm convention _make_frame expects
        else:
            depth_np = np.asanyarray(depth.get_data()).astype(np.float32)
        return rgb, depth_np

    def _show_depth(self, depth_np):
        """Colorize the live RealSense depth (raw uint16, mm) and show it in DEPTH_WIN.
        Clips the colormap range to self.depth_viz_max_m so nearby objects use most of the
        color range (defaults to pipeline.params.max_depth, the same gating depth used
        elsewhere)."""
        max_depth_mm = self.depth_viz_max_m * 1000.0
        valid = depth_np > 0
        norm = np.clip(depth_np / max_depth_mm, 0.0, 1.0)
        depth_u8 = (norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0  # no-return pixels shown as black, not "near" (turbo's low end)
        cv2.imshow(DEPTH_WIN, depth_color)

    def _crop_resize(self, img, interp):
        x0, y0 = self.crop_xy
        crop = img[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        return cv2.resize(crop, (self.track_res, self.track_res), interpolation=interp)

    def _make_frame(self, rgb, depth, fid):
        return Frame(
            id=fid,
            rgb=self._crop_resize(rgb, cv2.INTER_LINEAR),
            depth=self._crop_resize(depth, cv2.INTER_NEAREST),
            intrinsics=self.K, depth_factor=self.depth_factor, timestamp=time.time(),
        )

    @staticmethod
    def _cropped_resized_K(K_full, crop_xy, crop_size, out_res):
        x0, y0 = crop_xy
        s = out_res / float(crop_size)
        K = K_full.copy()
        K[0, 2] -= x0; K[1, 2] -= y0
        K[0, 0] *= s; K[1, 1] *= s; K[0, 2] *= s; K[1, 2] *= s
        return K

    # ================================================================== #
    # Phase 1: pick the crop viewport
    # ================================================================== #
    def _viewport_cb(self, event, x, y, _flags, _param):
        self._mouse_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_lock = True

    def pick_viewport(self):
        print("[phase2] aim the crop with the mouse, LEFT click to lock ('q' to quit)")
        cv2.resizeWindow(WIN, self.W, self.H)
        cv2.setMouseCallback(WIN, self._viewport_cb)
        self._click_lock = False
        while True:
            rgb, depth = self._grab_raw()
            if rgb is None:
                continue
            self._show_depth(depth)
            disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            cx, cy = self._mouse_xy
            half = self.crop_size // 2
            x0 = int(np.clip(cx - half, 0, self.W - self.crop_size))
            y0 = int(np.clip(cy - half, 0, self.H - self.crop_size))
            cv2.rectangle(disp, (x0, y0), (x0 + self.crop_size, y0 + self.crop_size),
                          (0, 255, 255), 2)
            cv2.putText(disp, f"VIEWPORT {self.crop_size}px  (click to lock)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if self._click_lock:
                self.crop_xy = (x0, y0)
                self.K = self._cropped_resized_K(self.K_full, self.crop_xy,
                                                 self.crop_size, self.track_res)
                print(f"[phase2] locked crop_xy={self.crop_xy}, size={self.crop_size} "
                      f"-> {self.track_res}px")
                cv2.setMouseCallback(WIN, lambda *a: None)  # detach callback
                cv2.resizeWindow(WIN, self.track_res * 2, self.track_res * 2)  # square crop view
                return True

    # ================================================================== #
    # Phase 2: build the map (heavy; runs in the main flow, not a callback)
    # ================================================================== #
    def build_map(self):
        # The map is rendered with a STANDARD centered K (see MapBuilder.render_fov_deg), and
        # runtime PnP uses each live frame's own intrinsics -- so the map does NOT depend on the
        # crop/live K at all, and we build it up front. The K passed here is only stored on the
        # map for reference; pass a centered placeholder at track_res.
        print("[phase2] building model map (first run may take a while)...")
        placeholder_K = np.array(
            [[self.track_res, 0, self.track_res / 2],
             [0, self.track_res, self.track_res / 2], [0, 0, 1]], dtype=np.float64
        )
        self.pipeline = ModelTrackingPipeline(self.cfg)
        self.pipeline.build_map(placeholder_K, (self.track_res, self.track_res))
        if self.use_viser:
            from examples.model_based_tracking.live_viser_debug import LiveViserDebug
            mt = self.cfg.model_tracking.params
            self.viser = LiveViserDebug(
                mesh_path=mt.mesh_path,
                render_K=self.pipeline.model_map.render_intrinsics,
                img_hw=(self.track_res, self.track_res),
                model_map=self.pipeline.model_map,
                mesh_scale=float(mt.get("mesh_scale", 1.0)),
            )

    # ================================================================== #
    # Phase 3: pick SAM points (optional)
    # ================================================================== #
    def _sam_cb(self, event, x, y, _flags, _param):
        # window shows the crop 1:1 at track_res -> click == image coords
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_points.append([float(x), float(y)]); self.click_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click_points.append([float(x), float(y)]); self.click_labels.append(0)

    def pick_sam(self):
        """Return the frame used to init SAM, or None to skip / quit."""
        print("[phase3] click +/- (L/R) on the object, 's' to start, 'c' clear, 'q' quit")
        cv2.setMouseCallback(WIN, self._sam_cb)
        while True:
            rgb, depth = self._grab_raw()
            if rgb is None:
                continue
            self._show_depth(depth)
            frame = self._make_frame(rgb, depth, 0)
            disp = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR).copy()
            if self.click_points:
                disp = self._overlay_mask(disp, self._sam_preview(frame))
            for (x, y), lab in zip(self.click_points, self.click_labels):
                cv2.circle(disp, (int(x), int(y)), 4,
                           (0, 255, 0) if lab == 1 else (0, 0, 255), -1)
            cv2.putText(disp, "SAM: L/R +/-, 's' start", (6, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("c"):
                self.click_points.clear(); self.click_labels.clear()
            if key == ord("s") and self.click_points:
                self.pipeline.add_user_points(self.click_points, self.click_labels)
                if self.pipeline.initialize_segmenter(frame):
                    print("[phase3] segmenter initialized")
                    cv2.setMouseCallback(WIN, lambda *a: None)
                    return frame
                print("[phase3] segmenter init failed; add clearer prompts")

    def _sam_preview(self, frame):
        seg = self.pipeline.frontend.segmenter
        if seg is None or not self.click_points:
            return None
        try:
            seg.predictor.load_first_frame(frame.rgb)
            _, _, ml = seg.predictor.add_new_prompt(
                frame_idx=0, obj_id=0,
                points=np.array(self.click_points, np.float32),
                labels=np.array(self.click_labels, np.int32),
            )
            return ml
        except Exception as e:
            print(f"[phase3] SAM preview failed: {e}")
            return None

    # ================================================================== #
    # Phase 4: live tracking
    # ================================================================== #
    def track(self):
        print("[phase4] tracking. 'r' reset pose, 'q' quit")
        cv2.setMouseCallback(WIN, lambda *a: None)
        fid = 0
        while True:
            rgb, depth = self._grab_raw()
            if rgb is None:
                continue
            self._show_depth(depth)
            # When viser is paused, freeze the last frame (don't step) so the inlier slider can
            # inspect the correspondences of the frame that was on screen when you paused.
            if self.viser is not None and self.viser.paused:
                cv2.waitKey(30)
                continue

            frame = self._make_frame(rgb, depth, fid)
            self.pipeline.step(frame)
            if self.viser is not None and self.pipeline.objects:
                obj = self.pipeline.objects[0]
                fe = self.pipeline.last_fe_result
                inliers = fe.valid_key_points.get(0) if fe is not None else None
                inlier_rows = (fe.valid_stats.get(0, {}).get("inlier_rows")
                               if fe is not None else None)
                self.viser.update(obj.pose, rgb=frame.rgb, lost=obj.lost,
                                  inlier_pts_obj=inliers, inlier_rows=inlier_rows)
            cv2.imshow(WIN, self._draw_tracking(frame))
            fid += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord("r"):
                self._reset_pose()

    def _reset_pose(self):
        for obj in self.pipeline.objects:
            obj.pose = np.eye(4); obj.lost = False
        self.pipeline.frontend._pose_initialized.clear()
        print("[phase4] pose reset")

    # ------------------------------------------------------------------ #
    def _overlay_mask(self, disp, mask):
        if mask is None:
            return disp
        overlay = disp.copy()
        for i in range(len(mask)):
            m = mask[i, 0]
            m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
            if (m > 0).any():
                overlay[m > 0] = (0, 165, 255)
        return cv2.addWeighted(overlay, 0.4, disp, 0.6, 0)

    def _draw_tracking(self, frame):
        disp = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR).copy()
        disp = self._overlay_mask(disp, getattr(frame, "mask", None))
        tt = self.pipeline.track_table
        fe = self.pipeline.last_fe_result
        for obj_id, obj in enumerate(self.pipeline.objects):
            if obj_id not in tt.obj2track_map:
                continue
            # GREEN = points that actually passed the visible + low-uncertainty + in-mask
            # filter and were fed to PnP (front-end's valid_indices). These are the real
            # correspondences; drawing only these avoids the confusing "green point outside the
            # mask" (which was just a raw TAPIR-visible point that the filter later dropped).
            used = None
            if fe is not None and obj_id in fe.valid_indices:
                used = np.asarray(fe.valid_indices[obj_id], dtype=np.int64)
            if used is not None and used.size:
                draw_points_on_image(disp, tt.track_2d[used],
                                     np.full((used.size, 3), (0, 255, 0), np.uint8))
            if self.show_hidden_points:
                # Also show raw-visible-but-rejected points in red for debugging.
                idx = np.asarray(tt.obj2track_map[obj_id], dtype=np.int64)
                vis = np.asarray(tt.visible[idx], dtype=bool)
                rejected = np.setdiff1d(idx[vis], used if used is not None else [])
                if rejected.size:
                    draw_points_on_image(disp, tt.track_2d[rejected],
                                         np.full((rejected.size, 3), (0, 0, 255), np.uint8))
        for obj in self.pipeline.objects:
            if obj.pose is None or getattr(obj, "bbox", None) is None:
                continue
            pose = obj.pose @ obj.init_pose
            half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
            # Oriented boxes (e.g. Open3D's OrientedBoundingBox, used for the gsplat mapper --
            # see MapBuilder._gaussian_obb) carry their own rotation + center in the object
            # frame; fold that into the drawn pose and draw the box centered at its local
            # origin. Axis-aligned boxes (plain _AABB, no .R) keep the old center +/- half path.
            rot = getattr(obj.bbox, "R", None)
            if rot is not None:
                center = np.asarray(obj.bbox.get_center(), dtype=float)
                T_obb = np.eye(4)
                T_obb[:3, :3] = np.asarray(rot, dtype=float)
                T_obb[:3, 3] = center
                pose = pose @ T_obb
                bbox_min_max = np.vstack([-half, half])
            else:
                center = np.asarray(obj.bbox.center, dtype=float)
                bbox_min_max = np.vstack([center - half, center + half])
            lost = getattr(obj, "lost", False)
            cv2.putText(disp, f"obj {obj.id}: {'LOST' if lost else 'OK'}",
                        (6, 16 + 16 * obj.id), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255) if lost else (0, 255, 0), 1)
            if not lost:
                # Guard: if a bad pose momentarily puts the object origin behind the camera,
                # the projected axis/box points become non-finite and OpenCV raises. Skip that
                # frame's overlay instead of crashing.
                try:
                    if self._pose_in_front(pose):
                        disp = draw_posed_3d_box(self.K, disp, pose, bbox_min_max)
                        disp = draw_xyz_axis(image=disp, ob_in_cam=pose, K=self.K)
                except cv2.error as e:
                    print(f"[phase4] overlay skipped (bad projection): {e}")
        return disp

    @staticmethod
    def _pose_in_front(pose, min_z=1e-3):
        """True if the object origin is in front of the camera (positive depth)."""
        return float(pose[2, 3]) > min_z

    # ------------------------------------------------------------------ #
    def run(self):
        try:
            self.build_map()                       # phase 1: build map up front (crop-independent)
            if not self.pick_viewport():           # phase 2: choose the live crop / K
                return
            if self.use_segmenter:                 # phase 3 (optional)
                if self.pick_sam() is None:
                    return
            self.track()                           # phase 4
        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--viser", action="store_true",
                    help="viser view: mesh + live estimated camera frustum")
    ap.add_argument("--show-hidden-points", action="store_true",
                    help="also draw non-visible map points in red")
    args = ap.parse_args()
    t = RealSenseModelTracker(args.config, use_viser=args.viser)
    t.show_hidden_points = args.show_hidden_points
    t.run()


if __name__ == "__main__":
    main()
