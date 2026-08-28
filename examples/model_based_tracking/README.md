## Examples: Model-based tracking

Once you have a textured mesh or a trained gaussian splat, you can use it for model-based tracking. 
Instead of online point map building, this model-based tracking renders the mesh or gaussian splat and sample points from it to track.

### Model-based tracking from a textured mesh
```bash
python examples/model_based_tracking/realsense_tracking_model.py \
    --config configs/pipeline/model_tracking.yaml
```

### Model-based tracking from a trained gaussian splat

Once you have a `gaussians.pt`, use it as the model-based tracking map instead of a
textured mesh with `configs/pipeline/model_tracking_gsplat.yaml` (same as
`model_tracking.yaml` but `renderer: gsplat`; set `model_tracking.params.gsplat_path`
to your checkpoint):
```bash
python examples/model_based_tracking/realsense_tracking_model.py \
    --config configs/pipeline/model_tracking_gsplat.yaml
```

<video src="../../assets/videos/point-to-pose-gsplat-example.webm" controls playsinline style="max-width: 100%;"></video>