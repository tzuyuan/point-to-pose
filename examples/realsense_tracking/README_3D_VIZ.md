# 3D Visualization Plug-in for the RealSense Demo

`realsense_tracking_3d.py` runs the exact same demo as `realsense_tracking.py`
(which it subclasses — the original file is untouched) and adds a live 3D UI.

## Rerun UI (default) — kv_tracker-style viewer

The default `ui_mode: rerun` logs into the **Rerun** viewer — the same UI as
[kv_tracker](https://github.com/Marwan99/kv_tracker), and entity-compatible
with the `model_based_tracking` branch's reference viewer
(`point2pose/utils/rerun_viz.py`): `world/obj_<i>` for the object-fixed
stage, `camframe/obj_<i>` for the camera-fixed stage, per-track-id point
colors (dimmed while a point is untracked), and per-point trace buffers.

Curated layout (sent as a Rerun blueprint):

| View | Fixed frame | Shows |
|---|---|---|
| **Object frame · map** | the tracked object | keypoint map (track-id colors), **continuity-filtered SDF mesh** (logged when it changes, so scrubbing shows the reconstruction grow), camera trajectory, current camera frustum textured with the live RGB (red when lost), keyframe frustums with RGB thumbnails, bbox |
| **Camera frame · trails** | the RealSense sensor | sensor frustum, the **full keypoint map** posed in the camera frame — currently visible points light up in their own stable color, untracked ones stay dimmed — plus fading per-point traces (stale traces auto-drop) and the object bbox at its current pose |
| **RGB / Events** tabs | — | **fully annotated 2D view**: tracked points in track-id colors, SAM2 segmentation mask (labeled per object), and reprojection whiskers from each map point's predicted pixel to its tracked observation (green → red by pixel error); keyframe / lost / re-acquired events |
| **Residual / Tracking** | — | residual [mm] in its own plot with a pinned y-axis (`rerun.residual_cap_mm`: values clipped at the cap so a lost-frame spike can't flatten the history); inliers / tracked points / FPS in a second plot |

The camera frustum is green while tracking and red while lost; it turns
green again on recovery — either when the pipeline clears the lost flag
(successful f2m re-registration passing the jump guard) or as soon as the
current frame shows strong registration evidence (≥5 inliers, residual
< 20 mm), since the flag itself can lag recovery.

The camera trajectory follows the same rule as the frustum color: it pauses
while the camera is red (lost, no recovery evidence) and resumes in a new
segment once it turns green again — whether the pipeline cleared the flag
or the evidence rule kicked in — with the same color and earlier segments
preserved. A lost→re-acquired teleport is never drawn as travelled path;
motion while tracking always stays connected, however fast.

The 2D annotations live under the camera entity, so they also appear
projected on the frustum's image plane in 3D, and colors match the 3D points
(hovering highlights the same point everywhere).

The SDF mesh is cleaned with the *same* `filter_disconnected_components`
used by `scripts/debug_visualization/visualize_textured_mesh.py`
(largest-connected-components continuity filter) — tune it under
`visualization_3d.mesh_filter`.

**Show/hide buttons**: the demo's cv2 window grows a button strip during
tracking — `map · mesh · kfs · traj · bbox · traces · 2d · mask · reproj`
(bbox and mask start hidden; initial states live in
`visualization_3d.rerun.show`). Clicking a button re-sends the Rerun
blueprint, so toggles apply instantly and across the entire timeline (the
log itself is untouched). The same toggles are also available natively via
the eye icons in the viewer's entity sidebar; note that toggling from the
strip resets any manual panel rearrangement, since it replaces the blueprint.

The last strip button, **`pts:<mode>`**, cycles how points and traces are
colored (3D + 2D consistently): `track_id` (default — stable distinct color
per physical point, dimmed while not visible) → `inlier` (green =
registration inlier, red = outlier, gray = unused) → `frame_id` →
`uncertainty` → `object`. Unlike the show/hide buttons this changes what
gets *logged*, so it applies from the next frame onward.

What the Rerun viewer gives for free: **timeline scrubbing** (replay any part
of the session, pause, step frame by frame), per-view camera controls, and
screenshots. Set `visualization_3d.rerun.save_rrd: ./debug/session.rrd` to
record the whole session to a file you can re-open later with
`rerun session.rrd` — ideal for cutting demo videos offline.

The cv2 window is used for prompt clicks, the 2D overlay, and the button strip.

## Run

```bash
conda activate ms
python examples/realsense_tracking/realsense_tracking_3d.py \
    --config configs/pipeline/pipeline_test2.yaml \
    --viz-config configs/visualization/pose_3d_demo.yaml
```

Both flags are optional. Without `--viz-config`, the visualizer reads a
`visualization_3d:` section from the pipeline config if one exists, otherwise
it uses built-in defaults (see `configs/visualization/pose_3d_demo.yaml` for
the annotated full set). Keyboard controls are unchanged: click points,
`n` next object, `s` start tracking, `r` reset, `q` quit.

## Other UI modes

- `ui_mode: web` — browser viewer via viser (control panel with toggles and
  sliders, multi-client). Automatic fallback if rerun is not installed.
- `ui_mode: combined` — single SLAM-style cv2 dashboard window with clickable
  buttons and mp4 recording (left-drag rotate, shift-drag pan, right-drag or
  wheel zoom).
- `ui_mode: windows` — two separate interactive Open3D windows.

## Using the plug-in elsewhere

The visualizer is a read-only observer of `ModularPipeline` — it works with
any runner, not just the RealSense demo:

```python
from point2pose.visualization import Pose3DVisualizer

viz = Pose3DVisualizer(cfg.get("visualization_3d"))  # None -> defaults
...
pipeline.step(frame)
canvas = viz.update(pipeline, frame, overlay_2d=display_bgr)
if canvas is not None:                    # combined mode only
    cv2.imshow("my window", canvas)
...
viz.close()
```

Package layout (`point2pose/visualization/`):

- `snapshot.py` — copies pipeline state into plain numpy (`SceneSnapshot`);
  the only file that knows about pipeline internals / frame conventions.
- `rerun_dashboard.py` — Rerun UI (blueprint, stages, metrics, events).
- `web_dashboard.py` — viser web UI.
- `dashboard.py` — combined single-window cv2 UI.
- `object_frame_view.py`, `camera_frame_view.py` — Open3D views (windows
  mode; offscreen renderers for the combined dashboard).
- `geometry.py` — frustum/box/trail/colormap builders (pure functions).
- `view_base.py` — lazy Open3D window wrapper.
- `pose_3d_visualizer.py` — facade + `DEFAULT_CONFIG`.

## Notes

- Multi-object: both stages and the metrics show every object; the RGB feed,
  mesh, and keyframe thumbnails follow `object_view.obj_id` (default 0).
- `show_bbox`/bbox display needs `estimate_init_pose: true` in the pipeline
  config (already set in `pipeline_test2.yaml`).
- Color modes (`rerun.map_color_mode` / `rerun.point_color_mode`):
  `track_id` (default; stable per physical point, dimmed when untracked),
  `frame_id`, `uncertainty`, `object`.
- Dependency note: `rerun-sdk` 0.36 requires numpy >= 2; this env is pinned
  to numpy 2.1.3, which also satisfies numba (< 2.3) and tensorflow (< 2.2).
