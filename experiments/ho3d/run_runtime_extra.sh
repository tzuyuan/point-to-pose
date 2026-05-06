#!/usr/bin/env bash
# Chain three additional runtime experiments after the TAPIR n=30 baseline:
#   1. cotracker3 on AP10
#   2. cotracker3 on MPM12
#   3. TAPIR on YCBMultiTrack 006_mustard_bottle (single object, real)
# Each run writes timings.csv + quality.json into a clean per-tracker subdir.

set -e

source /home/justin/anaconda3/etc/profile.d/conda.sh
conda activate ms

REPO=/home/justin/code/point-to-pose
RESULTS_ROOT=$REPO/results/runtime_analysis_20260505

CT3_CONFIG=$REPO/configs/ho3d_exp/eccv_final_cotracker3.yaml
TAPIR_CONFIG=$REPO/configs/ho3d_exp/eccv_final.yaml

CT3_OUT=$RESULTS_ROOT/cotracker3
YCB_OUT=$RESULTS_ROOT/ycb_tapir

mkdir -p "$CT3_OUT" "$YCB_OUT"

echo "===== [1/3] cotracker3 on AP10 ====="
python $REPO/experiments/ho3d/run_ho3d_runtime.py \
  --video_name AP10 --num_points 30 --config_path "$CT3_CONFIG" \
  --out_dir "$CT3_OUT"

echo "===== [2/3] cotracker3 on MPM12 ====="
python $REPO/experiments/ho3d/run_ho3d_runtime.py \
  --video_name MPM12 --num_points 30 --config_path "$CT3_CONFIG" \
  --out_dir "$CT3_OUT"

echo "===== [3/3] TAPIR on YCBMultiTrack 006_mustard_bottle ====="
python $REPO/experiments/ycbinisaac/run_ycbinisaac_runtime.py \
  --video_name 006_mustard_bottle --config_path "$TAPIR_CONFIG" \
  --out_dir "$YCB_OUT" --no_quality_eval

echo "===== ALL DONE ====="
