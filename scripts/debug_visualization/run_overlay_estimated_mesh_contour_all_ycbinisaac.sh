#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-/home/justin/code/point-to-pose/results/ycb_multi_track_finished}"
DATA_ROOT="${2:-/home/justin/data/YCBMultiTrack_new}"
MODEL_ROOT="${3:-/home/justin/data/HO3D_V3/models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OVERLAY_SCRIPT="${REPO_ROOT}/scripts/debug_visualization/overlay_estimated_mesh_contour.py"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "[Error] RESULTS_ROOT does not exist: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${OVERLAY_SCRIPT}" ]]; then
    echo "[Error] Overlay script not found: ${OVERLAY_SCRIPT}" >&2
    exit 1
fi

mapfile -t RUN_DIRS < <(find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "[Error] No sequence directories found under: ${RESULTS_ROOT}" >&2
    exit 1
fi

echo "[Info] Results root: ${RESULTS_ROOT}"
echo "[Info] Data root:    ${DATA_ROOT}"
echo "[Info] Model root:   ${MODEL_ROOT}"
echo "[Info] Num sequences: ${#RUN_DIRS[@]}"

num_ok=0
num_fail=0

for run_dir in "${RUN_DIRS[@]}"; do
    seq_name="$(basename "${run_dir}")"
    echo
    echo "[Run] ${seq_name}"

    if python3 "${OVERLAY_SCRIPT}" \
        --dataset ycbinisaac \
        --run_dir "${run_dir}" \
        --data_root "${DATA_ROOT}" \
        --mesh_source dataset \
        --model_root "${MODEL_ROOT}"; then
        num_ok=$((num_ok + 1))
        echo "[OK] ${seq_name}"
    else
        num_fail=$((num_fail + 1))
        echo "[Fail] ${seq_name}" >&2
    fi
done

echo
echo "[Done] success=${num_ok} failed=${num_fail}"

if [[ "${num_fail}" -gt 0 ]]; then
    exit 1
fi
