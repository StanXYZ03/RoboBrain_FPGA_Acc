#!/usr/bin/env bash
# run_ooc.sh - single-job driver for the compact integer Selector IP export and
# ZCU102 out-of-context (OOC) route.
#
# Everything large lives under the private /tmp of THIS process (Vitis HLS
# project, Vivado project, IP repo, checkpoints).  Only bounded text/JSON
# evidence is copied into the repository artifact directory.
#
# Usage:
#   bash fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_ooc.sh
#
# Nothing on the board, no bitstream, no source/interface change.
set -uo pipefail

VITIS_HLS="${VITIS_HLS:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
VIVADO="${VIVADO:-/home/xilinx/Vivado/2023.2/bin/vivado}"
PYTHON="${PYTHON:-/home/zhaoxiangyu/conda-envs/robobrain25/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OOC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${OOC_DIR}/../.." && pwd)"
ART="${REPO_ROOT}/artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route"

RUN_DIR=/tmp/robobrain_integer_selector_compact_ooc_run
HLS_PROJ=/tmp/robobrain_integer_selector_compact_ooc_hls
IP_REPO=/tmp/robobrain_integer_selector_compact_ooc_ip
VIV_WORK=/tmp/robobrain_integer_selector_compact_ooc_vivado
IP_NAME=integer_selector_pipeline_compact
TOP=integer_selector_pipeline_compact
HLS_TIMEOUT="${HLS_TIMEOUT:-7200}"
VIVADO_TIMEOUT="${VIVADO_TIMEOUT:-5400}"   # 90 min total guard for the task

mkdir -p "${RUN_DIR}" "${ART}"/{hls,ip,vivado,logs,scripts}
cp -f "${OOC_DIR}"/scripts/*.sh "${OOC_DIR}"/scripts/*.tcl "${OOC_DIR}"/scripts/*.py "${ART}/scripts/" 2>/dev/null || true
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"
export ISPC_PROJECT_DIR="${HLS_PROJ}"
export ISPC_IP_REPO_DIR="${IP_REPO}"
export ISPC_VIVADO_WORK="${VIV_WORK}"

HLS_RC=-1; VIV_RC=-1
HLS_T0=$(date +%s); HLS_T1=$HLS_T0
VIV_T0=0; VIV_T1=0
export ISPC_IP_DIR=""

cd "${RUN_DIR}"

echo "=== tool versions ===" | tee "${ART}/tool_versions.txt"
"${VITIS_HLS}" -version 2>&1 | head -4 | tee -a "${ART}/tool_versions.txt"
"${VIVADO}" -version 2>&1 | head -4 | tee -a "${ART}/tool_versions.txt"
echo "=== df (before) ===" | tee "${RUN_DIR}/df_before.txt"
df -h /tmp /home | tee -a "${RUN_DIR}/df_before.txt"

########################################################################
# Stage 1: Vitis HLS csynth_design + export ip_catalog (-flow syn)
########################################################################
echo "=== HLS start: $(date -Is) ==="
HLS_T0=$(date +%s)
timeout -k 60 "${HLS_TIMEOUT}" "${VITIS_HLS}" -f "${OOC_DIR}/scripts/run_hls_export.tcl" \
    > "${RUN_DIR}/hls_export.log" 2>&1
HLS_RC=$?
HLS_T1=$(date +%s)
echo "HLS_EXIT=${HLS_RC} HLS_SECONDS=$((HLS_T1-HLS_T0))"
cp -f "${RUN_DIR}/hls_export.log" "${ART}/hls/hls_export.log"
if [ -f "${HLS_PROJ}/solution1/syn/report/csynth.rpt" ]; then
    cp -f "${HLS_PROJ}/solution1/syn/report/csynth.rpt" "${ART}/hls/csynth.rpt"
fi
if [ -f "${HLS_PROJ}/solution1/syn/report/csynth.xml" ]; then
    cp -f "${HLS_PROJ}/solution1/syn/report/csynth.xml" "${ART}/hls/csynth.xml"
fi
# Locate the exported IP component.  With `export_design -output <dir> -flow syn`
# Vitis HLS writes `<dir>/export.zip`; unpack it to get component.xml + hdl/ and
# constraints/.  `<dir>/<ipname>/` and `<dir>/` are also accepted.
IP_DIR=""
if [ -f "${IP_REPO}/${IP_NAME}/component.xml" ]; then
    IP_DIR="${IP_REPO}/${IP_NAME}"
elif [ -f "${IP_REPO}/component.xml" ]; then
    IP_DIR="${IP_REPO}"
else
    EXPORT_ZIP="$(find "${IP_REPO}" -maxdepth 2 -name '*.zip' -print -quit 2>/dev/null)"
    if [ -n "${EXPORT_ZIP}" ]; then
        echo "EXPORT_ZIP=${EXPORT_ZIP}"
        echo "EXPORT_ZIP_BYTES=$(stat -c %s "${EXPORT_ZIP}")"
        echo "EXPORT_ZIP_SHA256=$(sha256sum "${EXPORT_ZIP}" | awk '{print $1}')"
        cp -f "${EXPORT_ZIP}" "${ART}/ip/$(basename "${EXPORT_ZIP}")"
        mkdir -p "${IP_REPO}/extracted"
        unzip -o -q "${EXPORT_ZIP}" -d "${IP_REPO}/extracted"
    fi
    IP_DIR="$(dirname "$(find "${IP_REPO}/extracted" -name component.xml -print -quit 2>/dev/null)")"
fi
echo "IP_DIR=${IP_DIR}"
export ISPC_IP_DIR="${IP_DIR}"
if [ -n "${IP_DIR}" ] && [ -f "${IP_DIR}/component.xml" ]; then
    cp -f "${IP_DIR}/component.xml" "${ART}/ip/component.xml"
fi

# Exact top-level generated port list (interface inventory evidence).
if [ -f "${IP_DIR}/hdl/verilog/${TOP}.v" ]; then
    sed -n "/^module ${TOP}/,/^);/p" "${IP_DIR}/hdl/verilog/${TOP}.v" \
        > "${ART}/ip/top_module_ports.txt"
fi

# Exported OOC XDC (clock assumption evidence).
for f in "${IP_DIR}"/constraints/*.xdc; do
    [ -f "$f" ] && cp -f "$f" "${ART}/ip/$(basename "$f")"
done

HLS_ERR=$(grep -c -E '^ERROR:|ERROR: \[HLS' "${RUN_DIR}/hls_export.log" 2>/dev/null || true)
echo "HLS_ERROR_LINES=${HLS_ERR:-0}"

if [ "${HLS_RC}" -ne 0 ]; then
    echo "HLS stage failed (rc=${HLS_RC}); skipping Vivado."
else
########################################################################
# Stage 2: Vivado OOC synth + opt + place + phys_opt + route
########################################################################
echo "=== Vivado start: $(date -Is) ==="
VIV_T0=$(date +%s)
timeout -k 60 "${VIVADO_TIMEOUT}" "${VIVADO}" -mode batch \
    -source "${OOC_DIR}/scripts/run_vivado_ooc.tcl" \
    -log "${RUN_DIR}/vivado_ooc.log" \
    -journal "${RUN_DIR}/vivado_ooc.jou" \
    > "${RUN_DIR}/vivado_console.log" 2>&1
VIV_RC=$?
VIV_T1=$(date +%s)
echo "VIVADO_EXIT=${VIV_RC} VIVADO_SECONDS=$((VIV_T1-VIV_T0))"
cp -f "${RUN_DIR}/vivado_ooc.log" "${ART}/vivado/vivado_ooc.log" 2>/dev/null || true
cp -f "${RUN_DIR}/vivado_console.log" "${ART}/vivado/vivado_console.log" 2>/dev/null || true

# Copy bounded reports only (text).
for f in "${VIV_WORK}"/reports/*.rpt "${VIV_WORK}"/reports/*.txt; do
    [ -f "$f" ] && cp -f "$f" "${ART}/vivado/$(basename "$f")"
done
[ -f "${VIV_WORK}/ooc_stage_status.tsv" ] && cp -f "${VIV_WORK}/ooc_stage_status.tsv" "${ART}/vivado/ooc_stage_status.tsv"
[ -f "${VIV_WORK}/qor_suggestions_post_route.rqs" ] && cp -f "${VIV_WORK}/qor_suggestions_post_route.rqs" "${ART}/vivado/qor_suggestions_post_route.rqs"

# Routed checkpoint: keep ONLY if <= 200 MiB, else record hash/size only.
if [ -f "${VIV_WORK}/post_route.dcp" ]; then
    DCP_BYTES=$(stat -c %s "${VIV_WORK}/post_route.dcp")
    DCP_SHA=$(sha256sum "${VIV_WORK}/post_route.dcp" | awk '{print $1}')
    echo "POST_ROUTE_DCP_BYTES=${DCP_BYTES}"
    echo "POST_ROUTE_DCP_SHA256=${DCP_SHA}"
    if [ "${DCP_BYTES}" -le 209715200 ]; then
        cp -f "${VIV_WORK}/post_route.dcp" "${ART}/vivado/post_route.dcp"
        echo "POST_ROUTE_DCP_COPIED=1"
    else
        echo "POST_ROUTE_DCP_COPIED=0 (>200MiB; hash/size recorded only)"
    fi
fi
fi

echo "=== df (after) ===" | tee "${RUN_DIR}/df_after.txt"
df -h /tmp /home | tee -a "${RUN_DIR}/df_after.txt"
cp -f "${RUN_DIR}/df_before.txt" "${ART}/logs/df_before.txt"
cp -f "${RUN_DIR}/df_after.txt" "${ART}/logs/df_after.txt"

{
    echo "HLS_EXIT=${HLS_RC}"
    echo "HLS_SECONDS=$((HLS_T1-HLS_T0))"
    echo "VIVADO_EXIT=${VIV_RC}"
    echo "VIVADO_SECONDS=$((VIV_T1-VIV_T0))"
    echo "IP_DIR=${IP_DIR:-}"
    echo "FINISHED_UTC=$(date -u -Is)"
} > "${ART}/logs/run_status.txt"
cat "${ART}/logs/run_status.txt"

echo "=== run_ooc.sh finished: $(date -Is) ==="
exit 0
