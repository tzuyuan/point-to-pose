#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_PY="${PROJECT_ROOT}/experiments/ho3d/run_ho3d_all.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH="${1:-/home/justin/data/HO3D_V3/}"
OUTPUT_ROOT="${2:-/home/justin/results/eccv_point2pose/ablation}"

# Explicit ablation list (intentionally not using a glob).
CONFIGS=(\
  "configs/ho3d_exp/eccv_abla_no_kf_graph.yaml"
  "configs/ho3d_exp/eccv_abla_cotracker.yaml"
  "configs/ho3d_exp/eccv_abla_sp_only.yaml"
  "configs/ho3d_exp/eccv_abla_uniform_samp.yaml"
)

mkdir -p "${OUTPUT_ROOT}"

echo "HO3D ablation runs"
echo "Project root : ${PROJECT_ROOT}"
echo "Runner       : ${RUNNER_PY}"
echo "Python       : ${PYTHON_BIN}"
echo "Data path    : ${DATA_PATH}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Failure mode : continue (never stop on failure)"
echo
echo "Configs (explicit list):"
for cfg in "${CONFIGS[@]}"; do
  echo "  - ${cfg}"
done
echo

SUMMARY_CSV="${OUTPUT_ROOT}/ablation_runs_summary.csv"
printf "config_name,status,return_code,config_path,run_out_dir,log_path\n" > "${SUMMARY_CSV}"

for cfg_rel in "${CONFIGS[@]}"; do
  cfg_abs="${PROJECT_ROOT}/${cfg_rel}"
  if [[ ! -f "${cfg_abs}" ]]; then
    echo "Missing config: ${cfg_abs}"
    printf "%s,%s,%s,%s,%s,%s\n" \
      "$(basename "${cfg_rel}" .yaml)" "missing" "127" "${cfg_abs}" "" "" >> "${SUMMARY_CSV}"
    continue
  fi

  cfg_name="$(basename "${cfg_rel}" .yaml)"
  run_dir="${OUTPUT_ROOT}/${cfg_name}"
  log_path="${run_dir}/ablation_run.log"

  mkdir -p "${run_dir}"
  cp -f "${cfg_abs}" "${run_dir}/config_used.yaml"

  echo "Running ${cfg_name}"
  echo "  config : ${cfg_abs}"
  echo "  output : ${run_dir}"
  echo "  log    : ${log_path}"

  if "${PYTHON_BIN}" "${RUNNER_PY}" \
    --data_path "${DATA_PATH}" \
    --out_dir "${run_dir}" \
    --config_path "${cfg_abs}" > "${log_path}" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  if [[ "${rc}" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  printf "%s,%s,%s,%s,%s,%s\n" \
    "${cfg_name}" "${status}" "${rc}" "${cfg_abs}" "${run_dir}" "${log_path}" >> "${SUMMARY_CSV}"

  echo "  status : ${status} (rc=${rc})"
  echo
done

echo "Done."
echo "Summary CSV: ${SUMMARY_CSV}"
