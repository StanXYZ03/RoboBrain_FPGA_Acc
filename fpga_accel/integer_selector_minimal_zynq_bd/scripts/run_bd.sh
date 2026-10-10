#!/usr/bin/env bash
# run_bd.sh - single-job driver for the minimal ZCU102 Zynq MPSoC block design
# around the accepted compact integer Selector IP.
#
# Everything large lives under the private /tmp of THIS process (unpacked IP,
# Vivado project, generated products, synthesis checkpoint).  Only bounded
# text/Tcl evidence is copied into the repository artifact directory.
#
# No bitstream, no board access, no source/interface change.
#
# Usage:
#   bash fpga_accel/integer_selector_minimal_zynq_bd/scripts/run_bd.sh
set -uo pipefail

VIVADO="${VIVADO:-/home/xilinx/Vivado/2023.2/bin/vivado}"
PYTHON="${PYTHON:-/home/zhaoxiangyu/conda-envs/robobrain25/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BD_DIR}/../.." && pwd)"
ART="${REPO_ROOT}/artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd"
ACCEPTED_IP="${REPO_ROOT}/artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route/ip/export.zip"

WORK=/tmp/robobrain_integer_selector_bd
IP_REPO=${WORK}/ip_repo
IP_DIR=${IP_REPO}/integer_selector_pipeline_compact
PROJ=${WORK}/vivado
RPT=${WORK}/reports
RUN_LOG=${WORK}/logs
DESIGN=integer_selector_minimal_zynq_bd

BD_TIMEOUT="${BD_TIMEOUT:-3600}"        # 60 min for build+validate+generate+wrapper
SYNTH_TIMEOUT="${SYNTH_TIMEOUT:-4200}"  # 70 min for bounded synthesis

mkdir -p "${WORK}" "${RUN_LOG}" "${ART}"/{ip,bd,vivado,logs,scripts}
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"
export ISPC_BD_IP_REPO="${IP_REPO}"
export ISPC_BD_WORK="${PROJ}"
export ISPC_BD_REPORTS="${RPT}"

cp -f "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.tcl "${SCRIPT_DIR}"/*.py "${ART}/scripts/" 2>/dev/null || true

########################################################################
# Protected SHA snapshot (before)
########################################################################
"${PYTHON}" "${SCRIPT_DIR}/protected_sha.py" snapshot --repo "${REPO_ROOT}" \
    --out "${ART}/protected_sha_before.json" | tee "${ART}/logs/protected_sha_before.txt"

########################################################################
# Tool versions / disk
########################################################################
{
    echo "=== tool versions ==="
    "${VIVADO}" -version 2>&1 | head -4
    "${PYTHON}" --version 2>&1
} | tee "${ART}/tool_versions.txt"
df -h /tmp /home | tee "${ART}/logs/df_before.txt"

########################################################################
# Unpack the accepted IP archive into the /tmp IP repository
########################################################################
rm -rf "${IP_REPO}"
mkdir -p "${IP_DIR}"
sha256sum "${ACCEPTED_IP}" | tee "${ART}/ip/export.zip.sha256"
stat -c '%s' "${ACCEPTED_IP}" | tee "${ART}/ip/export.zip.bytes"
unzip -o -q "${ACCEPTED_IP}" -d "${IP_DIR}"
if [ -f "${IP_DIR}/component.xml" ]; then
    cp -f "${IP_DIR}/component.xml" "${ART}/ip/component.xml"
fi
if [ -f "${IP_DIR}/hdl/verilog/integer_selector_pipeline_compact.v" ]; then
    sed -n "/^module integer_selector_pipeline_compact/,/^);/p" \
        "${IP_DIR}/hdl/verilog/integer_selector_pipeline_compact.v" \
        > "${ART}/ip/top_module_ports.txt"
fi
ls -la "${IP_DIR}" | tee "${ART}/ip/unpacked_listing.txt"

########################################################################
# Stage A: build / validate / address / wrapper
########################################################################
echo "=== Stage A start: $(date -Is) ==="
A_T0=$(date +%s)
timeout -k 60 "${BD_TIMEOUT}" "${VIVADO}" -mode batch \
    -source "${SCRIPT_DIR}/build_bd.tcl" \
    -log "${RUN_LOG}/build_bd.log" \
    -journal "${RUN_LOG}/build_bd.jou" \
    > "${RUN_LOG}/build_bd_console.log" 2>&1
BD_RC=$?
A_T1=$(date +%s)
echo "BUILD_RC=${BD_RC} BUILD_SECONDS=$((A_T1-A_T0))"

cp -f "${RUN_LOG}/build_bd.log" "${ART}/vivado/build_bd.log" 2>/dev/null || true
cp -f "${RUN_LOG}/build_bd_console.log" "${ART}/vivado/build_bd_console.log" 2>/dev/null || true

# bounded BD evidence dumps (all under $RPT)
for f in "${RPT}"/*.tsv "${RPT}"/*.txt; do
    [ -f "$f" ] && cp -f "$f" "${ART}/bd/$(basename "$f")"
done
[ -f "${RPT}/${DESIGN}.tcl" ] && cp -f "${RPT}/${DESIGN}.tcl" "${ART}/bd/${DESIGN}.tcl"
BD_SRC="${PROJ}/project/${DESIGN}.srcs/sources_1/bd/${DESIGN}/${DESIGN}.bd"
[ -f "${BD_SRC}" ] && cp -f "${BD_SRC}" "${ART}/bd/${DESIGN}.bd"

########################################################################
# Stage B: bounded synthesis (no implementation, no bitstream)
########################################################################
SYNTH_RC=-1
SYNTH_PROC_RC=-1
SYNTH_T0=0; SYNTH_T1=0
if [ "${BD_RC}" -eq 0 ]; then
    echo "=== Stage B (synthesis) start: $(date -Is) ==="
    SYNTH_T0=$(date +%s)
    timeout -k 60 "${SYNTH_TIMEOUT}" "${VIVADO}" -mode batch \
        -source "${SCRIPT_DIR}/synth_bd.tcl" \
        -log "${RUN_LOG}/synth_bd.log" \
        -journal "${RUN_LOG}/synth_bd.jou" \
        > "${RUN_LOG}/synth_bd_console.log" 2>&1
    SYNTH_PROC_RC=$?
    SYNTH_T1=$(date +%s)
    # Authoritative synthesis result from the stage status written by the Tcl.
    SYNTH_RC=$(awk -F'\t' '$1=="synth_design"{print $2}' "${RPT}/synth_stage_status.tsv" 2>/dev/null | tail -1)
    [ -z "${SYNTH_RC}" ] && SYNTH_RC=-1
    echo "SYNTH_PROC_RC=${SYNTH_PROC_RC} SYNTH_RC=${SYNTH_RC} SYNTH_SECONDS=$((SYNTH_T1-SYNTH_T0))"
    cp -f "${RUN_LOG}/synth_bd.log" "${ART}/vivado/synth_bd.log" 2>/dev/null || true
    cp -f "${RUN_LOG}/synth_bd_console.log" "${ART}/vivado/synth_bd_console.log" 2>/dev/null || true
    for f in "${RPT}"/*post_synth_bd*.rpt "${RPT}/synth_stage_status.tsv"; do
        [ -f "$f" ] && cp -f "$f" "${ART}/vivado/$(basename "$f")"
    done
else
    echo "Stage A failed (rc=${BD_RC}); skipping synthesis."
fi

########################################################################
# Protected SHA snapshot (after) + compare
########################################################################
"${PYTHON}" "${SCRIPT_DIR}/protected_sha.py" snapshot --repo "${REPO_ROOT}" \
    --out "${ART}/protected_sha_after.json" | tee "${ART}/logs/protected_sha_after.txt"
"${PYTHON}" "${SCRIPT_DIR}/protected_sha.py" compare \
    --before "${ART}/protected_sha_before.json" \
    --after "${ART}/protected_sha_after.json" \
    --out "${ART}/protected_sha_compare.json" | tee "${ART}/logs/protected_sha_compare.txt"

df -h /tmp /home | tee "${ART}/logs/df_after.txt"

BITSTREAM_FILES=$(find "${PROJ}" -name '*.bit' 2>/dev/null | wc -l | tr -d ' ')
{
    echo "BUILD_RC=${BD_RC}"
    echo "BUILD_SECONDS=$((A_T1-A_T0))"
    echo "SYNTH_RC=${SYNTH_RC}"
    echo "SYNTH_PROC_RC=${SYNTH_PROC_RC}"
    echo "SYNTH_SECONDS=$((SYNTH_T1-SYNTH_T0))"
    echo "BITSTREAM_FILES=${BITSTREAM_FILES}"
    echo "FINISHED_UTC=$(date -u -Is)"
} | tee "${ART}/logs/run_status.txt"

########################################################################
# Reports (JSON + Markdown + harness result)
########################################################################
set +e
"${PYTHON}" "${SCRIPT_DIR}/summarize_bd.py" \
    --reports "${RPT}" \
    --art "${ART}" \
    --repo "${REPO_ROOT}" \
    --json-out "${REPO_ROOT}/reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.json" \
    --md-out "${REPO_ROOT}/reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.md" \
    --harness-out "${REPO_ROOT}/reports/token_selector/logs/2026-10-03-integer-selector-minimal-zynq-bd-harness-result.json" \
    2>&1 | tee "${ART}/logs/summarize_bd.txt"
set -e

echo "=== run_bd.sh finished: $(date -Is) ==="
exit 0
