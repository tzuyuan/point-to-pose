# point-to-pose




## Dependencies

### Segment-Anything-2-Real-Time
We use a [modified version](https://github.com/Gy920/segment-anything-2-real-time/tree/main) of SAM2 to perform realtime segmentation. To install, do
```
git clone git@github.com:Gy920/segment-anything-2-real-time.git

cd segment-anything-2-real-time

pip install -e .
```
Then, to download the checkpoints, inside `segment-anything-2-real-time`, 
```
cd checkpoints
./download_ckpts.sh
```
Then put the check point under `./checkpoints/sam2.1/` or modify the config file to point to the file. 

### Track Any Points 

To use the BoostTAPIR for pose tracking, we use the official implementation of [TAPIR](https://github.com/google-deepmind/tapnet).

```
git clone https://github.com/deepmind/tapnet.git

cd tapnet

pip install .
```

Then, download the OnlineBoostTAPIR [checkpoints](https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy) and put it under `./checkpoints/tapir/`.

### TAPNext++ (tracker type: `tapnext`)

TAPNext++ ([arXiv 2604.10582](https://arxiv.org/abs/2604.10582), Apache 2.0) is the
successor of BootsTAPIR in the same `tapnet` repository: purely causal per-frame
tracking with an SSM state, much better occlusion re-detection, and latency nearly
independent of the number of points (~10 ms/frame on RTX 4090).

The PyTorch implementation lives in `tapnet/tapnext/` of the tapnet repo, which must
be recent enough to include it (commit `7f13cb6`, Apr 2026, or later):

```
cd tapnet && git pull   # or: git checkout origin/main -- tapnet/tapnext tapnet/tapnextpp
```

Download the checkpoint (standard 256 resolution):
```
wget -P checkpoints/tapnext https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```
A 512-resolution fine-tuned checkpoint also exists
(`https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt`, use with
`input_resolution: 512`).

Note: TAPNext queries are position-only. New points added mid-stream are injected
on the next processed frame; anchor-frame (past keyframe) queries are injected by
position only, since the recurrent state cannot be rewound.

### Track-On2 / Track-On-R (tracker type: `trackon`)

Track-On2 ([arXiv 2509.19115](https://arxiv.org/abs/2509.19115), MIT) is a strictly
online per-frame tracker with a FIFO point memory. It localizes by patch
classification over the whole frame, so it can re-detect points after occlusion.
~21 ms/frame at 640x480 / 24 points on RTX 4090 (fp32; mmcv ops do not support half
precision).

```
cd third_party
git clone https://github.com/gorkaydemir/track_on.git
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "transformers>=4.56.1"
```
The mmcv wheel URL must match your torch/CUDA version (cu121/torch2.4 above); see the
[track_on README](https://github.com/gorkaydemir/track_on) for building from source on
other setups.

Checkpoints (model weights only — the ViT backbone is pulled from Hugging Face at
first run):
```
# DINOv2 backbone (default, no gating; backbone auto-downloads)
wget -O checkpoints/trackon/trackon2_dinov2_checkpoint.pt "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov2_checkpoint.pt?download=true"

# DINOv3 variants (better real-world numbers, esp. Track-On-R):
wget -O checkpoints/trackon/trackon2_dinov3_checkpoint.pt "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov3_checkpoint.pt?download=true"
wget -O checkpoints/trackon/track_on_r.pt "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true"
```
To use the DINOv3 variants you must request access to
[facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
on Hugging Face (gated Meta license), run `huggingface-cli login`, and set
`vit_backbone: dinov3_s_plus` in the tracker config. The DINOv2 checkpoint has
comparable accuracy per the authors and needs no login.

### LiteTracker (tracker type: `litetracker`)

LiteTracker ([arXiv 2504.09904](https://arxiv.org/abs/2504.09904), MICCAI 2025) is a
training-free causal variant of CoTracker3 online: per-frame tracking (~6 ms/frame on
RTX 4090) using a temporal memory buffer and EMA motion prior. It reuses the standard
CoTracker3 online weights (which are CC BY-NC licensed — non-commercial).

```
cd third_party
git clone https://github.com/ImFusionGmbH/lite-tracker.git
```
No extra Python dependencies. Point `checkpoint_path` at the CoTracker3
[scaled_online.pth](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth)
weights you already use for the `cotracker3_online` tracker.

Note: like CoTracker3, localization is a local search around the previous position —
robust and precise for smooth motion, but it cannot re-detect a point after it moved
far while occluded (TAPNext++/Track-On handle that case better).

### Switching trackers

All trackers implement the same `Tracker` interface (`initialize`,
`add_query_points`, `track_once`) and are selected via the `tracker:` block of the
pipeline config (`type: tapir | tapnext | trackon | litetracker |
cotracker3_online`). See `configs/pipeline/pipeline_test2.yaml` for example
parameter blocks, `scripts/benchmark_trackers.py` to compare trackers on a
recorded sequence (latency + mask-inlier quality + overlay videos), and
`scripts/benchmark_trackers_gt.py` for GT-pose reprojection accuracy (e2d).

Note for GUI scripts: importing torchvision before the first `cv2.namedWindow`
call makes that call hang forever (torchvision 0.19 / opencv-python 4.11
conflict). The tracker modules therefore defer their heavy imports until the
tracker is constructed — when writing new scripts with an OpenCV UI, create the
window before constructing `ModularPipeline` (the realsense example already
does this).

### LightGlue (For Super Points)
We use the [LightGlue](https://github.com/cvg/lightglue) impelmentation of the SuperPoints. 
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

## Dependencies via pip install
```bash
pip install -r requirements.txt
```

## Dependencies for examples below
```bash
pip install -r requirements-demo.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

## Examples: 3D Reconstruction

See [examples/realsense_tracking/reconstruction/README.md](examples/realsense_tracking/reconstruction/README.md) for a full example of capturing and training a 2DGS/3DGS.

## Examples: Model-based tracking

See [examples/model_based_tracking/README.md](examples/model_based_tracking/README.md) for a full example of model-based tracking from a textured mesh or a trained gaussian splat.