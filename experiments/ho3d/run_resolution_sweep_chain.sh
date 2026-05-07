#!/usr/bin/env bash
# Resolution sweep — each (resolution, sequence) runs in its own Python subprocess
# so a teardown segfault in one cannot kill the rest of the chain.
# Skips 256x256 (already complete from the in-process driver run).

set +e  # don't abort on a single subprocess failure
source /home/justin/anaconda3/etc/profile.d/conda.sh
conda activate ms

REPO=/home/justin/code/point-to-pose
OUT_ROOT=$REPO/results/runtime_analysis_20260505/resolution_sweep/tapir
HO3D_CFG=$REPO/configs/ho3d_exp/eccv_final.yaml
HO3D_DATA=/home/justin/data/HO3D_V3
YCB_DATA=/home/justin/data/YCBMultiTrack_new
YCB_MODEL=/home/justin/data/HO3D_V3/models

mkdir -p "$OUT_ROOT"
LOG=$OUT_ROOT/resolution_sweep_chain.log
echo "=== chain start $(date -Iseconds) ===" >> "$LOG"

run_ho3d () {
  local H=$1; local W=$2; local VID=$3
  local OUT="$OUT_ROOT/${H}x${W}"
  echo "===== ho3d $VID @ ${H}x${W} =====" | tee -a "$LOG"
  python "$REPO/experiments/ho3d/run_ho3d_runtime.py" \
    -v "$VID" --num_points 30 \
    --resize_height "$H" --resize_width "$W" \
    --config_path "$HO3D_CFG" \
    --out_dir "$OUT" --no_mesh_eval \
    --data_path "$HO3D_DATA" >>"$LOG" 2>&1
  echo "==> exit=$? (ho3d $VID @ ${H}x${W})" | tee -a "$LOG"
}

run_ycb () {
  local H=$1; local W=$2; local VID=$3
  local OUT="$OUT_ROOT/${H}x${W}"
  echo "===== ycb $VID @ ${H}x${W} =====" | tee -a "$LOG"
  python "$REPO/experiments/ycbinisaac/run_ycbinisaac_runtime.py" \
    -v "$VID" \
    --resize_height "$H" --resize_width "$W" \
    --config_path "$HO3D_CFG" \
    --out_dir "$OUT" \
    --data_path "$YCB_DATA" \
    --model_path "$YCB_MODEL" >>"$LOG" 2>&1
  echo "==> exit=$? (ycb $VID @ ${H}x${W})" | tee -a "$LOG"
}

for H in 384 480; do
  W=$H
  run_ho3d "$H" "$W" AP10
  run_ho3d "$H" "$W" MPM12
  run_ycb  "$H" "$W" 006_mustard_bottle
done

echo "=== chain end $(date -Iseconds) ===" >> "$LOG"
echo "ALL DONE"
