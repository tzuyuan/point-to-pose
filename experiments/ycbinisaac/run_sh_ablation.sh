#!/bin/bash
# SH(K) ablation driver: run {full, sh10, sh1} on {006_mustard_bottle, 008_pudding_box}.
# Logs each run to results/abla/logs/<method>__<seq>.log.
set -u

DATA=/home/justin/data/YCBMultiTrack_new
MODELS=/home/justin/data/HO3D_V3/models
RESULTS_ROOT=/home/justin/code/point-to-pose/results/abla
LOG_DIR=$RESULTS_ROOT/logs
mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  ["full"]="./configs/ycbinisaac/eccv_final.yaml"
  ["sh10"]="./configs/ycbinisaac_exp/abla_sh10.yaml"
  ["sh1"]="./configs/ycbinisaac_exp/abla_sh1.yaml"
)
SEQS=(006_mustard_bottle 008_pudding_box)
METHODS=(full sh10 sh1)

cd /home/justin/code/point-to-pose

for METHOD in "${METHODS[@]}"; do
  CFG=${CONFIGS[$METHOD]}
  for SEQ in "${SEQS[@]}"; do
    OUT="$RESULTS_ROOT/$METHOD"
    LOG="$LOG_DIR/${METHOD}__${SEQ}.log"
    echo "[`date +%H:%M:%S`] === START $METHOD on $SEQ -> $OUT ==="
    echo "  log: $LOG"
    # use conda env "ms" via conda run
    conda run -n ms --no-capture-output python experiments/ycbinisaac/run_ycbinisaac_single.py \
      --data_path "$DATA" \
      --video_name "$SEQ" \
      --out_dir "$OUT" \
      --config_path "$CFG" \
      --model_path "$MODELS" \
      > "$LOG" 2>&1
    RC=$?
    echo "[`date +%H:%M:%S`] === END   $METHOD on $SEQ (rc=$RC) ==="
    if [ $RC -ne 0 ]; then
      echo "  RUN FAILED — see $LOG. Continuing to next."
    fi
  done
done

echo "[`date +%H:%M:%S`] === ALL RUNS DONE ==="
