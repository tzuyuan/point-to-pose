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
