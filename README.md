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

### LightGlue (For Super Points)
We use the [LightGlue](https://github.com/cvg/lightglue) impelmentation of the SuperPoints. 
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

## Dependencies via pip install (TODO: Make requirements.txt)
### cupoch (TODO: remove this dependency)
```
pip install cupoch
```

### tensorflow
```
pip install tensorflow
pip install tensorflow-datasets
```

### LCM (Optional)
```
pip install lcm
```

### Open3d
```
pip install open3d 
```

### gtsam
```
pip install gtsam-develop
```

### For SDF
```
pip install numba
pip install trimesh
pip install pycuda
```

### skimage
```
pip install scikit-image
```