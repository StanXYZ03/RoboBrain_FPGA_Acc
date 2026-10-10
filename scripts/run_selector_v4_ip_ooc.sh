#!/usr/bin/env bash
# run_selector_v4_ip_ooc.sh - reproducible driver for the Selector v4 HLS-IP OOC
# export + implementation flow.
#
# It performs a df audit, runs `scripts/export_selector_v4_ip_ooc.tcl` (Vitis HLS
# csynth + export Verilog IP catalog + Vivado OOC synth/opt/place/route) with all
# large files under /tmp, then collects only the small reports/summaries into
# `artifacts/token_selector/2026-09-28-selector-ooc-v4`.
#
# No GPU, no board, no driver, no bitstream, no selection-maths change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${RUN_DIR:-/tmp/robobrain_selector_export_run_v4}"
PROJECT_DIR="${PROJECT_DIR:-/tmp/robobrain_token_selector_hls_export_v4}"
IP_DIR="${IP_DIR:-/tmp/robobrain_selector_ip_ooc_v4/ip}"
IP_ZIP="${IP_ZIP:-${IP_DIR}/token_selector_top_packed_ip.zip}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/token_selector/2026-09-28-selector-ooc-v4}"
HLS_LOG="${RUN_DIR}/console_export_v4.log"

VITIS_HLS="${VITIS_HLS:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
PYTHON="${PYTHON:-/home/zhaoxiangyu/conda-envs/robobrain25/bin/python}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"

mkdir -p "${RUN_DIR}" "${IP_DIR}" "${ARTIFACT_DIR}"

echo "=== df audit (before) ==="
df -h /tmp /home | tee "${RUN_DIR}/df_before.txt"

export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"
echo "=== running Vitis HLS export+OOC impl: $(date -Is) ==="
echo "command: ${VITIS_HLS} -f ${SCRIPT_DIR}/export_selector_v4_ip_ooc.tcl"
set +e
timeout "${TIMEOUT_SECONDS}" "${VITIS_HLS}" -f "${SCRIPT_DIR}/export_selector_v4_ip_ooc.tcl" 2>&1 | tee "${HLS_LOG}"
rc=${PIPESTATUS[0]}
set -e
echo "=== Vitis HLS exit code: ${rc} ==="

echo "=== df audit (after) ==="
df -h /tmp /home | tee "${RUN_DIR}/df_after.txt"

echo "=== summarizing OOC reports ==="
"${PYTHON}" "${SCRIPT_DIR}/summarize_selector_ooc.py" \
    --project-dir "${PROJECT_DIR}" \
    --ip-zip "${IP_ZIP}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --hls-log "${HLS_LOG}" \
    --df-before "${RUN_DIR}/df_before.txt" \
    --df-after "${RUN_DIR}/df_after.txt" \
    --repo-root "${REPO_ROOT}"

echo "=== done: $(date -Is) ; hls_exit=${rc} ==="
exit "${rc}"
