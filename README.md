<div align="center">

# Point2Pose

### Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects via 2D Point Trackers

**European Conference on Computer Vision (ECCV) 2026**

[Tzu-Yuan Lin](https://tzuyuan.github.io)<sup>1</sup> &nbsp;·&nbsp;
[Ho Jae Lee](https://hojae-io.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Kevin Doherty](https://keevindoherty.github.io/)<sup>2,§</sup> &nbsp;·&nbsp;
[Yonghyeon Lee](https://www.gabe-yhlee.com/)<sup>1</sup> &nbsp;·&nbsp;
[Sangbae Kim](https://meche.mit.edu/people/faculty/SANGBAE@MIT.EDU)<sup>1</sup>

<sup>1</sup>Massachusetts Institute of Technology &nbsp;&nbsp; <sup>2</sup>Boston Dynamics

<sup>§</sup>Work conducted in personal time and independently of the author's affiliated organization.

[![Project Page](https://img.shields.io/badge/Project-Page-1f8acb?style=for-the-badge)](https://point2pose.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.10415-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2604.10415)
[![PDF](https://img.shields.io/badge/Paper-PDF-4c1?style=for-the-badge)](https://arxiv.org/pdf/2604.10415)
[![Video](https://img.shields.io/badge/Video-YouTube-red?style=for-the-badge)](https://youtu.be/NRfGyx1nes4)

<img src="https://point2pose.github.io/static/images/teaser.jpg" width="92%" alt="Point2Pose teaser: multi-object 6D pose tracking and reconstruction">

</div>

---

**Point2Pose** is a *model-free* method for causal 6D pose tracking of **multiple rigid objects** from monocular RGB-D video. It is initialized from nothing but a few clicked image points — no CAD model, no category prior, no per-object training. A long-range 2D point tracker provides persistent correspondences, so an object that is **fully occluded or leaves the frame is re-localized the instant it comes back**. While tracking, the system incrementally fuses an online TSDF and reconstructs a textured mesh of each target.

## Disclaimer
**The readme are AI-generated. Please submit an issue if you find any problem.**

## ✨ Highlights

- **Model-free & category-agnostic** — click a few points, start tracking. No CAD model or object-specific training.
- **Multi-object** — several objects tracked simultaneously through mutual occlusion and interaction.
- **Instant recovery from complete occlusion** — long-range point identities survive the object disappearing entirely.
- **Simultaneous 3D reconstruction** — online TSDF fusion produces a colored/textured mesh of each tracked object.
- **Modular by construction** — segmenter, point tracker, sampler, registration, optimizer, and criterion are swappable via a registry and one YAML file.
- **Live demo** — RealSense RGB-D demo with an interactive [Rerun](https://rerun.io) 3D viewer (map, keyframes, mesh growth, trajectory, metrics).
- **LCM bridge** — take RGB-D frames from an [LCM](https://lcm-proj.github.io/) bus and publish per-frame object poses back onto it, so the tracker drops into an existing robot stack.
- **Mask-based pose fallback** — when the point tracks stop supporting a reliable registration, translation is re-derived from the SAM2 mask instead of freezing at the last good pose.
- **New dataset** — `YCBMultiTrack`, a dynamic multi-object RGB-D benchmark with motion-capture ground truth (synthetic + real).

Runtime is **2–10 Hz** depending on tracker resolution and number of tracked points; the 2D point tracker is the dominant cost.

## 📰 News and Updates

- **[ECCV 2026]** Point2Pose is accepted to **ECCV 2026**! 🎉
- **[2026-09]** **LCM bridge** — subscribe to RGB-D over LCM and publish tracked object poses back onto the bus, plus a **mask-based pose fallback** for frames where the point tracks go unreliable. See [LCM Bridge](#-lcm-bridge).
- **[2026-08]** Live 3D visualization plug-in for the RealSense demo (Rerun / viser / Open3D UIs) — see [3D viewer](#3d-visualization-rerun).
- **[2026-08]** Three additional point-tracker backends — **TAPNext++**, **Track-On2**, and **LiteTracker** — plus a benchmark harness to compare trackers on the same sequence. See [Swappable point trackers](#swappable-point-trackers).
- **[2026-06]** [Project page](https://point2pose.github.io/) is live, with videos of real-world multi-object tracking and occlusion recovery.
- **[2026-04]** Paper released on [arXiv](https://arxiv.org/abs/2604.10415).
- **[Coming soon]** `YCBMultiTrack` dataset release (synthetic + real sequences with mocap ground truth).

## 📑 Table of Contents

- [Installation](#-installation)
- [RealSense Live Demo](#-realsense-live-demo)
- [LCM Bridge](#-lcm-bridge)
- [Running on Datasets](#-running-on-datasets)
- [Configuration & Architecture](#-configuration--architecture)
- [Outputs and Logging](#-outputs-and-logging)
- [Benchmarking Point Trackers](#-benchmarking-point-trackers)
- [Repository Structure](#-repository-structure)
- [Known Issues](#-known-issues)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
- [Citation](#-citation)

---

## 🛠 Installation

Tested on Ubuntu 22.04 with Python 3.11, PyTorch 2.4 + CUDA 12.1, and an NVIDIA RTX 4090.

### 1. Clone the repository

```bash
git clone --recurse-submodules git@github.com:tzuyuan/point-to-pose.git
cd point-to-pose
```

(Already cloned? `git submodule update --init --recursive`.)

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate point2pose
```

That is the whole setup — there is no package to install. Every entry-point script
adds the repository root to `sys.path`, so **run everything from the repository
root** and imports resolve on their own:

```bash
python examples/realsense_tracking/realsense_tracking.py
python experiments/ho3d/run_ho3d_single.py -v AP12 ...
```

The pins in [environment.yml](environment.yml) are the exact versions the paper
results were produced with (Ubuntu 22.04 · Python 3.11 · CUDA 12.1 · RTX 4090).
Three extras are commented out at the bottom of the file — uncomment what you
need: `pycuda` (CUDA TSDF fusion, needs `nvcc` at install time), `transformers`
(Track-On2 backend), `lcm` (the [LCM bridge](#-lcm-bridge)).

<details>
<summary>Using pip / venv instead of conda</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# other CUDA build: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

[requirements.txt](requirements.txt) carries the same pins as `environment.yml`.
</details>

### 3. Third-party components

Three components are not on PyPI and must be installed from source. The quick route:

```bash
pip install --no-build-isolation -r requirements-third-party.txt
```

(`--no-build-isolation` matters — these packages import torch at build time.) Or clone them individually, which is preferable if you want to read or patch their code:

<table>
<tr><th>Component</th><th>Used for</th><th>Install</th></tr>
<tr><td><a href="https://github.com/Gy920/segment-anything-2-real-time">SAM2 real-time</a></td><td>Segmentation (required)</td><td><code>git clone git@github.com:Gy920/segment-anything-2-real-time.git && cd segment-anything-2-real-time && pip install -e .</code></td></tr>
<tr><td><a href="https://github.com/google-deepmind/tapnet">tapnet</a> (BootsTAPIR)</td><td>Default point tracker (required)</td><td><code>git clone https://github.com/deepmind/tapnet.git && cd tapnet && pip install .</code></td></tr>
<tr><td><a href="https://github.com/cvg/LightGlue">LightGlue</a></td><td>SuperPoint keypoint sampling (required)</td><td><code>git clone https://github.com/cvg/LightGlue.git && cd LightGlue && pip install -e .</code></td></tr>
</table>

### 4. Download checkpoints

```bash
# SAM2 (from inside the segment-anything-2-real-time clone)
cd checkpoints && ./download_ckpts.sh
# then copy/symlink sam2.1_hiera_large.pt into point-to-pose/checkpoints/sam2.1/

# BootsTAPIR (default tracker)
wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

Expected layout (paths are configurable in the YAML configs):

```
checkpoints/
├── sam2.1/    sam2.1_hiera_large.pt          # segmentation
├── tapir/     causal_bootstapir_checkpoint.pt # default point tracker
├── tapnext/   tapnextpp_ckpt.pt              # optional tracker
└── trackon/   trackon2_dinov2_checkpoint.pt  # optional tracker
```

> ⚠️ **Update the paths in the configs.** The YAML files under [configs/](configs/) currently contain absolute paths (`/home/justin/code/point-to-pose/...`, `/home/justin/data/...`). Point `checkpoint_path`, `debug_dir`, and `pose_save_path` at your own locations before running.

### Swappable point trackers

The default tracker is BootsTAPIR (`type: tapir`). Four alternatives ship with the repo — all implement the same `Tracker` interface (`initialize`, `add_query_points`, `track_once`) and are selected purely by the `tracker:` block of the pipeline config. Example blocks for each are in [configs/pipeline/pipeline_test2.yaml](configs/pipeline/pipeline_test2.yaml).

| `type` | Method | Latency* | Notes |
|---|---|---|---|
| `tapir` | [BootsTAPIR](https://github.com/google-deepmind/tapnet) | ~18 ms | Default; used for all paper results |
| `tapnext` | [TAPNext++](https://arxiv.org/abs/2604.10582) | ~12 ms | Causal SSM state; strongest occlusion re-detection on single-object scenes |
| `trackon` | [Track-On2 / Track-On-R](https://arxiv.org/abs/2509.19115) | ~22 ms | Global patch-classification re-detection with a FIFO point memory |
| `litetracker` | [LiteTracker](https://arxiv.org/abs/2504.09904) | ~6 ms | Training-free causal CoTracker3; fastest, but local search only |
| `cotracker3_online` | [CoTracker3](https://github.com/facebookresearch/co-tracker) | ~41 ms | Reference baseline |

\* Tracker forward pass only, RTX 4090, at each tracker's benchmark resolution.

<details>
<summary><b>TAPNext++ setup</b> (<code>type: tapnext</code>)</summary>

Lives in `tapnet/tapnext/` of the tapnet repo, which must be recent enough to include it (commit `7f13cb6`, Apr 2026 or later):

```bash
cd tapnet && git pull   # or: git checkout origin/main -- tapnet/tapnext tapnet/tapnextpp
wget -P checkpoints/tapnext https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```

A 512-resolution fine-tuned checkpoint also exists (`https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt`, use with `input_resolution: 512`).

*Caveat:* TAPNext queries are position-only. Points added mid-stream are injected on the next processed frame; anchor-frame (past keyframe) queries are injected by position alone, since the recurrent state cannot be rewound. Its fixed 256×256 input also starves small objects when several share a frame, so it underperforms TAPIR on multi-object scenes.
</details>

<details>
<summary><b>Track-On2 setup</b> (<code>type: trackon</code>)</summary>

```bash
cd third_party
git clone https://github.com/gorkaydemir/track_on.git
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "transformers>=4.56.1"
```

The mmcv wheel URL must match your torch/CUDA version; see the [track_on README](https://github.com/gorkaydemir/track_on) for building from source.

```bash
# DINOv2 backbone (default, ungated; ViT backbone auto-downloads from HF)
wget -O checkpoints/trackon/trackon2_dinov2_checkpoint.pt "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov2_checkpoint.pt?download=true"

# DINOv3 variants (better real-world numbers, esp. Track-On-R)
wget -O checkpoints/trackon/trackon2_dinov3_checkpoint.pt "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov3_checkpoint.pt?download=true"
wget -O checkpoints/trackon/track_on_r.pt "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true"
```

DINOv3 variants require access to [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) (gated Meta license) plus `huggingface-cli login`, and `vit_backbone: dinov3_s_plus` in the config. Accuracy of the DINOv2 checkpoint is comparable per the authors, and it needs no login. mmcv ops are fp32-only.
</details>

<details>
<summary><b>LiteTracker setup</b> (<code>type: litetracker</code>)</summary>

```bash
cd third_party
git clone https://github.com/ImFusionGmbH/lite-tracker.git
```

No extra Python dependencies. Point `checkpoint_path` at the CoTracker3 [scaled_online.pth](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth) weights (CC BY-NC — non-commercial). Like CoTracker3, localization is a local search around the previous position: robust for smooth motion, but it cannot re-detect a point that moved far while occluded.
</details>

---

## 🎥 RealSense Live Demo

Click a few points on any object in the live feed and Point2Pose starts tracking its 6D pose and reconstructing its mesh — no CAD model, no training.

### Requirements

- Intel RealSense RGB-D camera (tested on D435i / D455), USB 3.0
- NVIDIA GPU with CUDA (≥8 GB recommended)
- `pyrealsense2`, SAM2 and TAPIR checkpoints in place

### 1. Set your camera serial

```bash
rs-enumerate-devices -s     # find your serial
```

Set it in the config you plan to use:

```yaml
# configs/pipeline/pipeline_test2.yaml
realsense:
  params:
    rs_serial: 941322070969
```

Other keys worth checking in the same file: `pipeline.params.max_num_obj` (how many objects to track), `estimate_init_pose`, `debug_level`, `save_pose` / `pose_save_path`, and the `tracker:` block.

### 2. Run

```bash
conda activate point2pose
python examples/realsense_tracking/realsense_tracking.py     # 2D overlay only
```

### 3. Controls

| Key / mouse | Action |
|---|---|
| **Left click** | Add a *positive* prompt point to the current object |
| **Right click** | Add a *negative* prompt point (background / exclusion) |
| **`n`** | Finish this object and start prompting the **next** object |
| **`s`** | Start tracking with the collected prompts |
| **`r`** | Reset all prompt points |
| **`q`** | Quit |

Workflow: click 1–3 points on object #1 → press `n` → click points on object #2 → … → press `s`. A live SAM2 mask preview updates as you click, so you can verify the segmentation before committing. Once tracking starts, the window shows the masks, the tracked points, the estimated pose axes/box, and the frame counter.

### 3D visualization (Rerun)

`realsense_tracking_3d.py` runs the exact same demo and adds a live 3D UI:

```bash
python examples/realsense_tracking/realsense_tracking_3d.py \
    --config configs/pipeline/pipeline_test2.yaml \
    --viz-config configs/visualization/pose_3d_demo.yaml
```

Both flags are optional (without `--viz-config`, a `visualization_3d:` section in the pipeline config is used, otherwise built-in defaults). The Rerun viewer shows, on a scrubbable timeline:

- **Object frame · map** — the keypoint map, the growing TSDF mesh, camera trajectory, live-textured camera frustum, and keyframe frustums with RGB thumbnails.
- **Camera frame · trails** — the full map posed in the camera frame with fading per-point traces; the frustum turns **red when tracking is lost** and green again on recovery.
- **RGB / Events** — tracked points, SAM2 masks, and reprojection whiskers colored by pixel error.
- **Residual / Tracking** — residual (mm), inliers, tracked points, and FPS plots.

A button strip in the cv2 window toggles layers (`map · mesh · kfs · traj · bbox · traces · 2d · mask · reproj`) and cycles point coloring (`track_id → inlier → frame_id → uncertainty → object`). Set `visualization_3d.rerun.save_rrd: ./debug/session.rrd` to record the whole session and replay it later with `rerun session.rrd` — handy for cutting demo videos offline.

Other UI modes via `ui_mode`: `web` (viser, browser-based), `combined` (single cv2 dashboard with mp4 recording), `windows` (two Open3D windows). Full details: [examples/realsense_tracking/README_3D_VIZ.md](examples/realsense_tracking/README_3D_VIZ.md).

### Recording sequences

To capture RGB-D for offline runs (saved in the `YCBMultiTrack` layout: `rgb/`, `depth/` uint16 mm, `cam_K.txt`):

```bash
python examples/realsense_tracking/record_rgbd.py --out ~/data/my_take01 [--serial N]
# keys: r / space = start-stop recording, q / esc = quit
```

### Demo troubleshooting

| Symptom | Fix |
|---|---|
| Camera not found | Check `rs_serial` in the config and USB 3.0 connection |
| CUDA OOM | Use `sam2.1_hiera_small.pt`, lower the tracker resolution, or reduce `sampler.params.num_points` |
| Object flagged "lost" and never recovers | RealSense stereo depth residuals are ~3 mm; keep `register.params.residual_thres` and `map_growth_max_mean_residual` at ~0.006 (already set in `pipeline_test2.yaml`) |
| Pose rejected during normal handheld motion | Relax `pose_jump_guard_trans_thres` / `pose_jump_guard_rot_deg_thres` |
| Poor tracking | Better lighting, more textured surfaces, add negative prompt points to exclude background |

---

## 🔌 LCM Bridge

The same tracker, wired to an [LCM](https://lcm-proj.github.io/) bus instead of a camera: it **subscribes** to RGB-D frames and camera info, and **publishes** one pose message per tracked frame. Use it to drop Point2Pose into an existing robot stack, or to run the camera and the tracker on different machines.

Needs the optional `lcm` dependency (`pip install lcm==1.5.1`, or uncomment it in [environment.yml](environment.yml)).

### Channels

| Direction | Channel (config key) | Message | Payload |
|---|---|---|---|
| in | `lcm.rgbd_channel` | `rgbd_t` | RGB + depth images, timestamp |
| in | `lcm.camera_info_channel` | `camera_info_t` | `fx fy cx cy`, 3×4 extrinsic mapping world → camera, depth factor |
| out | `lcm.obj_pose_bb2world_channel` | `vec_list_t` | `[x y z qw qx qy qz sx sy sz]` — oriented box pose **and extents**, sorted descending |
| out | `lcm.obj_pose_mesh2world_channel` | `vec_list_t` | `[x y z qw qx qy qz]` — object/mesh frame pose |

Both outputs are in the **world frame** implied by `camera_info_t.extrinsic` (the tracker inverts it to lift camera-frame poses into world); send identity to get camera-frame poses. Each `vec_list_t` carries one row per object, named `obj_0`, `obj_1`, … The wire formats live in [point2pose/io/lcm/messages/](point2pose/io/lcm/messages/) and are reimplemented here, so no external LCM type package is needed.

### Run it

```bash
# Terminal 1 — any RGB-D source on the bus (this one drives a RealSense)
python examples/lcm_tracking/realsense_lcm_publisher.py \
    --config configs/pipeline/lcm_tracking.yaml [--preview]

# Terminal 2 — the tracker
python examples/lcm_tracking/point2pose_lcm_tracking.py \
    --config configs/pipeline/lcm_tracking.yaml
```

Run both from the repository root — some checkpoint paths are resolved relative to the working directory.

Terminal 2 opens a window on the incoming stream and waits for prompts. Controls differ slightly from the RealSense demo:

| Key / mouse | Action |
|---|---|
| **Left click** | Add a *positive* prompt point to the current object |
| **Right click** | Add a *negative* prompt point |
| **`s`** | Finish this object, start prompting the **next** one |
| **`r`** | **Run** — start tracking with the collected prompts |
| **`c`** | Clear all prompts and reset the pipeline |
| **`q`** | Quit |

Poses are published from the first tracked frame onward.

### Consuming the poses

```python
from point2pose.io.lcm import NamedVecListLcmSubscriber

sub = NamedVecListLcmSubscriber(channel="hw_obj_pose")
sub.start()
payload = sub.pop_latest()          # None until the first message arrives
if payload is not None:
    for name, vec in zip(payload.names, payload.vecs):
        xyz, quat_wxyz, extent = vec[:3], vec[3:7], vec[7:]
        print(name, xyz, quat_wxyz, extent)
```

`RgbdLcmPublisher` / `RgbdLcmSubscriber` are available the same way if you want to feed frames from your own source — see [examples/lcm_tracking/realsense_lcm_publisher.py](examples/lcm_tracking/realsense_lcm_publisher.py) for a complete producer.

### Latency

Subscriber and publisher each run on their own thread, and both coalesce to the newest message: with `lcm.drop_stale_frames: true` the tracker skips frames that piled up while the previous step was running, so it stays locked to the live stream rather than falling behind on a backlog. Set it to `false` to process every frame instead (the queue depth is `lcm.max_frame_drain`).


---

## 📊 Running on Datasets

Point2Pose is evaluated on [HO3D-v3](https://www.tugraz.at/index.php?id=40231), [YCBInEOAT](https://github.com/wenbowen123/iros20-6d-pose-tracking), and our own **YCBMultiTrack** (synthetic + real). Every runner takes `--data_path`, `--out_dir`, and `--config_path`; the paper settings live in `configs/ho3d_exp/eccv_final.yaml`, `configs/ycbineoat/eccv_final.yaml`, and `configs/ycbinisaac/eccv_final.yaml`.

```bash
# HO3D — single sequence / all 13 evaluation sequences
python experiments/ho3d/run_ho3d_single.py -v AP12 \
    --data_path /path/to/HO3D_V3 --out_dir results/ho3d_single \
    -c configs/ho3d_exp/eccv_final.yaml
python experiments/ho3d/run_ho3d_all.py \
    --data_path /path/to/HO3D_V3 --out_dir results/ho3d_all \
    -c configs/ho3d_exp/eccv_final.yaml

# YCBInEOAT
python experiments/ycbineoat/run_ycbineoat_all.py \
    --data_path /path/to/YCBInEOAT -m /path/to/YCB_models_with_ply \
    --out_dir results/ycbineoat_all -c configs/ycbineoat/eccv_final.yaml

# YCBMultiTrack (synthetic + real)
python experiments/ycbinisaac/run_ycbinisaac_all.py \
    --data_path /path/to/YCBMultiTrack -m /path/to/YCB_models \
    --out_dir results/ycbinisaac_all -c configs/ycbinisaac/eccv_final.yaml
```

Each runner writes per-sequence poses, ADD / ADD-S AUC tables, error-vs-time plots, and exported meshes (Chamfer distance against the ground-truth mesh where available) into `--out_dir`. Ablations from the paper are driven by [experiments/ho3d/run_ho3d_ablation.py](experiments/ho3d/run_ho3d_ablation.py), which sweeps every `configs/ho3d_exp/eccv_abla_*.yaml` config into its own output folder (`--data_path`, `--config_glob`, `--output_root`).

**Dataset layout.** `YCBInIsaacReader` / `YcbineoatReader` expect, per sequence: `rgb/` (or `jpg/`), `depth/`, `cam_K.txt`, plus `masks/<object>/` and `annotated_poses/<object>/` for evaluation; `Ho3dReader` reads the standard HO3D `evaluation/<seq>/` layout. See [point2pose/io/sources/dataset/datareader.py](point2pose/io/sources/dataset/datareader.py).

---

## 🧩 Configuration & Architecture

The pipeline is a registry of interchangeable modules assembled from one YAML file. Every block has a `type` (registry key) and a `params` dict, so swapping a component never requires touching code.


<div align="center">
<img src="https://point2pose.github.io/static/images/pipeline.png" width="88%" alt="Point2Pose pipeline overview">
</div>

| Block | Registry keys |
|---|---|
| `segmenter` | `sam2`, `dummy` |
| `tracker` | `tapir`, `tapnext`, `trackon`, `litetracker`, `cotracker3_online`, `cotracker3_offline` |
| `sampler` | `super_point_balanced`, `super_point_fps`, `super_point`, `uniform_fps`, `random`, `orb` |
| `register` | `svd_residual_outlier`, `svd_cluster_ransac`, `svd_cluster_sdf_refine`, `svd_cluster`, `svd_ransac`, `svd`, `svd_outlier_sdf`, `svd_uncertainty_irls`, `svd_uncertainty_outlier`, `pnp_cluster_ransac`, `open3d_icp`, `teaserpp` |
| `local_optimizer` / `global_optimizer` | `lm_graph`, `lm_graph_reproj`, `lm_graph_sdf`, `isam2` |
| `criterion` | `rotation_threshold`, `rotation_threshold_and_min_num`, `rotation_threshold_and_min_num_spread`, `rotation_grid`, `registration_residual`, `uncertainty_ratio`, `uncertainty_number`, `mask_area`, `iteration` |
| `reconstructor` | `sdf_builder` |

Key pipeline parameters: `max_num_obj`, `frame_reg_mode` (`f2f` / `f2m` / `hybrid`), `estimate_init_pose`, `use_graph_optimization`, and the pose-jump-guard / map-growth gates. [configs/pipeline/pipeline_test2.yaml](configs/pipeline/pipeline_test2.yaml) is the annotated reference config.

Adding a new module is three steps: subclass the base class in [point2pose/core/](point2pose/core/), decorate it with `@TRACKER.register_module("my_tracker")` (or the relevant registry), and point the config's `type` at the new key.

### Mask-based pose fallback

Point tracks are the primary signal, but they degrade before SAM2 does: under fast motion, motion blur, or heavy partial occlusion the registration can be left with too few inliers to trust while the mask is still clean. The default behaviour is to reject the estimate and freeze the pose, which shows up as the box visibly lagging the object.

`MaskPoseFallbackManager` fills that gap. When a frame's registration is **weak** it discards the registered translation and re-derives it from the mask: back-project the mask center to the mask's median depth, and place the object's box center there. **Rotation is never touched** — a silhouette carries no reliable orientation — so the fallback holds the last good rotation until the tracks recover.

A frame counts as weak if *any* of these hold (`pipeline.params.mask_pose_fallback_*`):

| Condition | Key |
|---|---|
| Too few valid correspondences | `weak_min_valid_points` |
| Too few registration inliers | `weak_min_inliers` |
| Mean residual above threshold | `weak_mean_residual` |
| The pose-jump guard rejected this frame | `use_on_jump_reject` |
| The object is flagged lost | `use_on_lost` |

The correction is deliberately conservative: `gain` scales how far toward the mask estimate to move, `max_translation_step` caps the per-frame jump, and `depth_blend` trades off the previous depth against the mask's median depth (mask depth picks up the occluder whenever the mask bleeds past the object, so `0.5` is the default rather than `1.0`). `center_mode: bbox` uses the mask's bounding-box center, which is more stable under partial occlusion than the `centroid`.

Enable it with `mask_pose_fallback_enable: true`; [configs/pipeline/lcm_tracking.yaml](configs/pipeline/lcm_tracking.yaml) has the full annotated block. To tune the thresholds before letting it act, set `mask_pose_fallback_compute_only: true` — every frame then reports what the fallback *would* have done, in `FrontEndResult.mask_fallback_stats`, without changing the pose. `mask_pose_fallback_debug: true` prints each application.

Per-frame diagnostics land on `FrontEndResult`: `mask_fallback_triggered`, `mask_fallback_pose_before` / `_after`, and `mask_fallback_stats` (which carries the weak reason, mask area and center, depth estimate and its source, and the applied translation delta).

---

## 💾 Outputs and Logging

Set in the pipeline config:

```yaml
pipeline:
  params:
    save_pose: true
    pose_save_path: /path/to/poses
    debug_level: 1                 # 0-2
    debug_dir: /path/to/debug
```

| File | Contents |
|---|---|
| `obj_<i>_pose.txt` | Per-object pose in **TUM format**: `timestamp tx ty tz qx qy qz qw` (meters) |
| `registration_stats.txt` | Per-frame registration diagnostics: iterations, threshold, residual mean/median/max, inlier counts |
| `<debug_dir>/output_images/` | Annotated frames (points, masks, pose box) when `visualization.params.save_images: true` |
| exported meshes | Reconstructed TSDF meshes (`.ply`, optionally textured `.glb`) written by the dataset runners |

Full description: [doc/pose_logging.md](doc/pose_logging.md).

---

## 📁 Repository Structure

```
point2pose/
├── core/           base classes + module registry
├── data_types/     Frame, KeyFrame, PointTrackTable, results
├── io/             dataset readers, RealSense source, LCM bridge, pose/point-cloud logging
├── modules/        segmenter · tracker · sampler · register · optimizer · criterion · reconstruction
├── pipeline/       ModularPipeline and its components
├── visualization/  Rerun / viser / Open3D dashboards
└── utils/          transforms, Lie algebra, evaluation, mesh metrics

configs/            per-dataset and per-experiment YAML (eccv_final.yaml = paper settings)
environment.yml     conda environment (requirements.txt carries the same pins for pip/venv)
examples/           RealSense live demo (2D, 3D viz, recorder), LCM bridge
experiments/        dataset runners, ablations, tracker sweep
scripts/            benchmarks, debug visualization, paper/poster figures
test/               pytest unit tests (`pytest`)
doc/                pose logging and RealSense tracker docs
```

---

## ⚠️ Known Issues

- **OpenCV window hangs when importing torchvision first.** With torchvision 0.19 + opencv-python 4.11, importing torchvision *before* the first `cv2.namedWindow` call makes that call spin forever. The tracker modules therefore defer heavy imports until construction — when writing new scripts with an OpenCV UI, **create the window before constructing `ModularPipeline`** (the RealSense demo already does this).
- **Global bf16 autocast.** The SAM2 segmenter module enables global bf16 autocast at import time; be aware if you mix in fp32-only ops (e.g. mmcv used by Track-On).
- **numpy pinning.** `rerun-sdk` ≥ 0.36 needs numpy ≥ 2, while numba (< 2.3) and tensorflow (< 2.2) impose upper bounds — numpy 2.1.3 satisfies all three.
- **Absolute paths in configs.** The shipped YAML files reference the authors' machine paths; update them for your setup.

---

## 🙏 Acknowledgements

This work builds on excellent open-source projects: [SAM2](https://github.com/facebookresearch/sam2) and its [real-time fork](https://github.com/Gy920/segment-anything-2-real-time), [TAPIR / BootsTAPIR and TAPNext](https://github.com/google-deepmind/tapnet), [Track-On2](https://github.com/gorkaydemir/track_on), [LiteTracker](https://github.com/ImFusionGmbH/lite-tracker), [CoTracker3](https://github.com/facebookresearch/co-tracker), [LightGlue / SuperPoint](https://github.com/cvg/LightGlue), [GTSAM](https://github.com/borglab/gtsam), [Open3D](https://www.open3d.org/), and [Rerun](https://rerun.io). We also thank the authors of [BundleTrack](https://github.com/wenbowen123/BundleTrack), [BundleSDF](https://github.com/NVlabs/BundleSDF), and [FoundationPose](https://github.com/NVlabs/FoundationPose) for their datasets and baselines.

## 📄 License

Released under the [BSD 3-Clause License](LICENSE). Third-party components keep their own licenses — note in particular that CoTracker3 weights (used by `cotracker3_online` and `litetracker`) are **CC BY-NC** (non-commercial).

## 📚 Citation

If you find Point2Pose useful in your research, please cite:

```bibtex
@inproceedings{lin2026point2pose,
  title     = {Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction
               for Multiple Unknown Objects via 2D Point Trackers},
  author    = {Lin, Tzu-Yuan and Lee, Ho Jae and Doherty, Kevin and Lee, Yonghyeon and Kim, Sangbae},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
