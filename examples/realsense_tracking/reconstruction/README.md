## Examples: 3D Reconstruction using Point-to-Pose

Capture a live SLAM tracking session (SDF map, no pre-existing mesh needed --
model-based tracking is not used here since it requires a mesh) and train a
2DGS/3DGS reconstruction from it.

<video src="../../assets/videos/point-to-pose-reconstruct-example.mp4" controls playsinline style="max-width: 100%;"></video>

```
gsplat
pytorch-msssim
```

`gsplat` JIT-compiles a CUDA extension on first run. If it fails with
`cuda_runtime_api.h: No such file or directory`, point the compiler at your
conda env's own CUDA headers:
```bash
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
```

Capture + train in one command:
```bash
python examples/realsense_tracking/reconstruction/run.py --name my_capture
```

Or run each step separately:
```bash
python examples/realsense_tracking/reconstruction/capture.py \
    --config configs/pipeline/reconstruction_capture.yaml --export-dir debug/my_capture

python examples/realsense_tracking/reconstruction/train_gaussians.py \
    --config configs/reconstruct/export_test.yaml --results_path debug/my_capture
```

`configs/pipeline/reconstruction_capture.yaml` is a copy of `pipeline_test2.yaml`
(SLAM/`ModularPipeline`, no pre-existing mesh needed) with paths made
repo-relative; set your camera's `rs_serial` there.

Add `--viewer` (either command) for a live viser preview during training
(free-viewpoint render, updates as you move the camera):
```bash
python examples/realsense_tracking/reconstruction/run.py --name my_capture --viewer
```

Output: `debug/my_capture/gaussians.pt`. View it later:
```bash
python examples/realsense_tracking/reconstruction/view_gaussians.py \
    --checkpoint debug/my_capture/gaussians.pt
```
