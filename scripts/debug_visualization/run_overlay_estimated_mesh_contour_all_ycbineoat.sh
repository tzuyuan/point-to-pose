#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-/home/justin/results/eccv_point2pose/final_results/ycbineoat_all_final}"
DATA_ROOT="${2:-/home/justin/data/YCBInEOAT}"
MODEL_ROOT="${3:-/home/justin/data/HO3D_V3/models}"
EXTRA_ARGS=("${@:4}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ALL_SCRIPT="${REPO_ROOT}/scripts/debug_visualization/run_overlay_estimated_mesh_contour_all.py"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "[Error] RESULTS_ROOT does not exist: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${RUN_ALL_SCRIPT}" ]]; then
    echo "[Error] Batch overlay script not found: ${RUN_ALL_SCRIPT}" >&2
    exit 1
fi

python3 "${RUN_ALL_SCRIPT}" \
    --dataset ycbineoat \
    --run_root "${RESULTS_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --model_root "${MODEL_ROOT}" \
    --mesh_source dataset \
    --show_axis \
    --axis_scale 0.08 \
    --axis_thickness 5 \
    --line_thickness 5 \
    "${EXTRA_ARGS[@]}"
