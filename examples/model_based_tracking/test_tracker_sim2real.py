"""
Minimal sim2real tracker sanity check (no PnP, no full map).

Renders the model from a few near-frontal viewpoints, samples SuperPoint keypoints on those
renders, seeds the point tracker (TAPNext / TAPIR) with them, then tracks those points in the
LIVE RealSense stream. The window shows, side by side:

    [ seed render + its keypoints ]   |   [ live frame + tracked points ]

This isolates the tracker + the sim2real (render vs. real camera) gap: if the tracked points
stick to the object as you move the camera, the render->real matching works; if they drift, the
gap (lighting/appearance) is too large for the tracker.

Everything (renderer with envmap, sampler, tracker) is built from the SAME config as the full
pipeline, so it reflects your real settings.

Run (point2pose env):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/model_based_tracking/test_tracker_sim2real.py \
        --config configs/pipeline/model_tracking.yaml
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

# Import map_builder first: it pulls in the full module graph in the right order and avoids a
# circular import that triggers if data_types.sampler_context is imported before the modules.
from point2pose.model_tracking.map_builder import MapBuilder
from point2pose.data_types.frame import Frame
from point2pose.data_types.sampler_context import SamplerContext
from point2pose.utils.transform import inverse_SE3

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/pipeline/model_tracking.yaml"


def look_at_obj2cam(cam_pos, up=(0, 0, 1)):
    cam_pos = np.asarray(cam_pos, float)
    fwd = -cam_pos / (np.linalg.norm(cam_pos) + 1e-9)
    up = np.asarray(up, float)
    if abs(fwd @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-9
    down = np.cross(fwd, right)
    T = np.eye(4); T[:3, :3] = np.stack([right, down, fwd], 1); T[:3, 3] = cam_pos
    return inverse_SE3(T)


def frontal_poses(n_views, radius, spread_deg):
    """A few near-frontal viewpoints (camera in front of the object, small yaw spread)."""
    if n_views == 1:
        yaws = [np.pi]
    else:
        yaws = np.deg2rad(np.linspace(-spread_deg, spread_deg, n_views))
    poses = []
    for a in yaws:
        # "front" = camera on -Y looking toward origin, small yaw about +Z
        cam = radius * np.array([np.sin(a), -np.cos(a), 0.15])
        poses.append(look_at_obj2cam(cam))
    return poses


class Sim2RealTest:
    def __init__(self, config_path, n_views, spread_deg, no_camera=False):
        self._no_camera = no_camera
        self.cfg = OmegaConf.load(config_path)
        self.cfg.setdefault("pipeline", OmegaConf.create({"params": {}}))
        ip = self.cfg.input.params if "input" in self.cfg else {}
        self.track_res = int(ip.get("track_res", 256))

        rs_serial = self.cfg.realsense.params.rs_serial if "realsense" in self.cfg else None
        self._init_realsense(rs_serial)

        # Reuse the map builder to get a renderer (with envmap), sampler and tracker from config.
        self.mb = MapBuilder(self.cfg)
        self.sampler = self.mb.sampler
        self.tracker = self.mb.tracker
        self.render_K = self.mb._standard_render_K(
            self.track_res, self.track_res, self.mb.render_fov_deg
        )

        # Optional SAM2 segmenter: click the object once, then its mask blacks out the background
        # in every live frame before the tracker sees it.
        self.use_segmenter = bool(self.cfg.pipeline.params.get("use_segmenter", False))
        self.segmenter = None
        self._seg_ready = False
        self.click_points, self.click_labels = [], []
        if self.use_segmenter and not no_camera:
            from point2pose.core.build import build_from_cfg
            from point2pose.core.module_registry import SEGMENTER
            self.segmenter = build_from_cfg(self.cfg.segmenter, SEGMENTER)

        self._seed(n_views, spread_deg)

        if not self._no_camera:
            cv2.namedWindow("sim2real", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("sim2real", self.track_res * 2, self.track_res)
            cv2.setMouseCallback("sim2real", self._mouse_cb)

    def self_track_check(self):
        """
        Sanity check: feed the FIRST seed render back through the tracker and verify its own
        keypoints come back to their seeded pixels (error should be ~0). This validates the
        seed/track coordinate handling. (We only check view 0 because the tracker is a causal
        stream -- feeding view 1's render after view 0 legitimately drifts view 0's points.)
        """
        rgb, kps = self.seed_views[0]
        tr, unc, vis = self.tracker.track_once(Frame(id=0, rgb=rgb))
        err = np.linalg.norm(tr[: len(kps)] - kps, axis=1)
        print(f"[sim2real] self-track view 0: median err {float(np.median(err)):.2f}px "
              f"(expect ~0), visible {int(vis[:len(kps)].sum())}/{len(kps)}")

    # ---- SAM2 segmenter helpers ----
    def _mouse_cb(self, event, x, y, _flags, _param):
        if self.segmenter is None or self._seg_ready:
            return
        # clicks land on the RIGHT (live) panel: subtract the seed panel width
        lx = x - self.track_res
        if lx < 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_points.append([lx, y]); self.click_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click_points.append([lx, y]); self.click_labels.append(0)

    def _sam_preview(self, rgb):
        """One-shot SAM mask for the current clicks (setup phase)."""
        if self.segmenter is None or not self.click_points:
            return None
        try:
            self.segmenter.predictor.load_first_frame(rgb)
            _, _, ml = self.segmenter.predictor.add_new_prompt(
                frame_idx=0, obj_id=0,
                points=np.array(self.click_points, np.float32),
                labels=np.array(self.click_labels, np.int32),
            )
            return ml
        except Exception as e:
            print(f"[sim2real] SAM preview failed: {e}")
            return None

    def _start_segmenter(self, rgb):
        if not self.click_points:
            print("[sim2real] click the object first, then 's'")
            return
        self.segmenter.add_input_object(self.click_points, self.click_labels)
        self.segmenter.initialize(rgb)
        self._seg_ready = bool(getattr(self.segmenter, "tracking_started", True))
        print(f"[sim2real] segmenter ready: {self._seg_ready}")

    def _mask_from_logits(self, ml):
        """union bool mask (track_res) from SAM logits [N,1,H,W]."""
        if ml is None:
            return None
        union = None
        for i in range(len(ml)):
            m = ml[i, 0]
            m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
            m = m > 0
            union = m if union is None else (union | m)
        return union

    def _apply_mask_overlay(self, bgr, mask):
        if mask is None:
            return bgr
        ov = bgr.copy()
        ov[mask] = (0, 165, 255)
        return cv2.addWeighted(ov, 0.35, bgr, 0.65, 0)

    def _init_realsense(self, rs_serial):
        self.H, self.W = 480, 640
        if getattr(self, "_no_camera", False):
            self.rs_pipeline = None
            return
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        if rs_serial is not None:
            config.enable_device(str(rs_serial))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.rs_pipeline.start(config)

    # ------------------------------------------------------------------ #
    def _sample_render(self, rgb, depth, view_id):
        """Run SuperPoint on a render (restricted to the object mask)."""
        import torch

        mask = depth > 0
        m = torch.from_numpy(mask.astype(np.uint8))[None, None].to(
            getattr(self.sampler, "device", "cuda")
        )
        dummy_depth = np.where(mask, 1.0, 0.0).astype(np.float32)
        frame = Frame(id=view_id, rgb=rgb, depth=dummy_depth, mask=m,
                      intrinsics=np.eye(3), depth_factor=1.0)
        ctx = SamplerContext(frame=frame, track_table=None, min_depth=0.01, max_depth=10.0)
        pts = self.sampler.sample(ctx, obj_id=0)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def _seed(self, n_views, spread_deg):
        renderer = self.mb._make_renderer(self.render_K, self.track_res, self.track_res)
        self.seed_views = []       # list of (rgb, kps)  -- kps in track_res pixels
        pt_view = []               # per global query point -> its seed view index
        try:
            for v, T in enumerate(frontal_poses(n_views, self.mb.render_radius, spread_deg)):
                rgb, depth = renderer.render(T)
                kps = self._sample_render(rgb, depth, v)
                if kps.shape[0] == 0:
                    print(f"[sim2real] view {v}: no keypoints, skipping")
                    continue
                self.tracker.add_query_points(Frame(id=v, rgb=rgb), kps)
                pt_view.extend([len(self.seed_views)] * kps.shape[0])
                self.seed_views.append((rgb, kps))
                print(f"[sim2real] seeded view {v}: {kps.shape[0]} keypoints")
        finally:
            renderer.delete()
        if hasattr(self.tracker, "finalize_seeding"):
            self.tracker.finalize_seeding()

        self.n_seed_views = len(self.seed_views)
        self.pt_view = np.asarray(pt_view, dtype=int)
        # one distinct BGR color per view (evenly spaced hue). live tracks are drawn in the
        # color of the seed VIEW they came from -> the color mix on the live frame tells you
        # which views are currently matching.
        self.view_colors = self._view_colors(self.n_seed_views)
        self.view_filter = -1  # -1 = all views; else only show this view's points
        self.seed_panel = self._make_seed_panel()

    @staticmethod
    def _view_colors(n):
        if n == 0:
            return np.zeros((0, 3), np.uint8)
        hues = (np.arange(n) * 180.0 / max(1, n)).astype(np.uint8)
        hsv = np.stack([hues, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], 1)
        return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]  # (n,3) BGR

    def _make_seed_panel(self):
        """Grid montage of the seed renders, each view's keypoints in that view's color."""
        if not self.seed_views:
            return np.zeros((self.track_res, self.track_res, 3), np.uint8)
        n = self.n_seed_views
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        cell = max(64, self.track_res // cols)
        panel = np.zeros((cell * rows, cell * cols, 3), np.uint8)
        for i, (rgb, kps) in enumerate(self.seed_views):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            col = tuple(int(c) for c in self.view_colors[i])
            for (x, y) in kps:
                cv2.circle(img, (int(x), int(y)), 2, col, -1)
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 3)
            cv2.putText(img, f"v{i}:{len(kps)}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            r, c = divmod(i, cols)
            panel[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(img, (cell, cell))
        return cv2.resize(panel, (self.track_res, self.track_res))

    # ------------------------------------------------------------------ #
    def _grab_live(self):
        frames = self.rs_pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        bgr = np.asanyarray(color.get_data())
        # center square crop -> track_res (matches how seed renders are square)
        s = min(self.H, self.W)
        y0 = (self.H - s) // 2; x0 = (self.W - s) // 2
        crop = bgr[y0:y0 + s, x0:x0 + s]
        return cv2.resize(crop, (self.track_res, self.track_res))

    def run(self):
        if self.use_segmenter and self.segmenter is not None:
            print("[sim2real] SETUP: click the object on the RIGHT panel (L=+, R=-), 's' to start")
        print("[sim2real] keys: 'q' quit | '0'=all views | '1'..'9'=only that view | 'c' clear clicks")
        try:
            while True:
                live_bgr = self._grab_live()
                if live_bgr is None:
                    continue
                live_rgb = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2RGB)

                # ---- SETUP: collect clicks, live SAM preview, wait for 's' ----
                if self.use_segmenter and self.segmenter is not None and not self._seg_ready:
                    prev = self._mask_from_logits(self._sam_preview(live_rgb)) \
                        if self.click_points else None
                    right = self._apply_mask_overlay(live_bgr.copy(), prev)
                    for (px, py), lab in zip(self.click_points, self.click_labels):
                        cv2.circle(right, (int(px), int(py)), 4,
                                   (0, 255, 0) if lab == 1 else (0, 0, 255), -1)
                    cv2.putText(right, "SETUP: click object, 's' to start", (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imshow("sim2real", np.hstack([self.seed_panel, right]))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("c"):
                        self.click_points.clear(); self.click_labels.clear()
                    elif key == ord("s"):
                        self._start_segmenter(live_rgb)
                    continue

                # ---- TRACK: (optional) mask -> black out background -> tracker ----
                mask = None
                if self._seg_ready:
                    _, ml = self.segmenter.segment(live_rgb)
                    mask = self._mask_from_logits(ml)
                    if mask is not None:
                        live_rgb = np.where(mask[..., None], live_rgb, 0).astype(live_rgb.dtype)

                tracks, unc, vis = self.tracker.track_once(Frame(id=0, rgb=live_rgb))

                right = cv2.cvtColor(live_rgb, cv2.COLOR_RGB2BGR).copy()
                right = self._apply_mask_overlay(right, mask)
                nvis = 0
                per_view = np.zeros(self.n_seed_views, int)
                for i, ((x, y), v) in enumerate(zip(tracks, vis)):
                    if not v:
                        continue
                    view = int(self.pt_view[i]) if i < len(self.pt_view) else 0
                    if self.view_filter >= 0 and view != self.view_filter:
                        continue
                    col = tuple(int(c) for c in self.view_colors[view])
                    cv2.circle(right, (int(x), int(y)), 2, col, -1)
                    nvis += 1
                    per_view[view] += 1
                title = (f"LIVE  visible {nvis}"
                         + (f"  [view {self.view_filter} only]" if self.view_filter >= 0 else ""))
                cv2.putText(right, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                # per-view visible counts in view colors (which views are matching now)
                for vw in range(self.n_seed_views):
                    col = tuple(int(c) for c in self.view_colors[vw])
                    cv2.putText(right, f"v{vw}:{per_view[vw]}", (6, 32 + 14 * vw),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

                cv2.imshow("sim2real", np.hstack([self.seed_panel, right]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("0"):
                    self.view_filter = -1
                elif ord("1") <= key <= ord("9"):
                    v = key - ord("1")
                    self.view_filter = v if v < self.n_seed_views else self.view_filter
        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--n-views", type=int, default=1, help="number of near-frontal seed renders")
    ap.add_argument("--spread-deg", type=float, default=45.0, help="yaw spread across seed views")
    ap.add_argument("--seed-only", action="store_true",
                    help="build+seed and run a self-track sanity check, then exit (no camera)")
    args = ap.parse_args()
    t = Sim2RealTest(args.config, args.n_views, args.spread_deg, no_camera=args.seed_only)
    if args.seed_only:
        t.self_track_check()
    else:
        t.run()


if __name__ == "__main__":
    main()
