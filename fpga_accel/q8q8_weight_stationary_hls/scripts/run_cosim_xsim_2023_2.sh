#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vitis_hls_bin=${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}
xsim_bin=${XSIM_BIN:-/home/xilinx/Vivado/2023.2/bin/xsim}
sim_dir="$project_dir/build/q8q8_weight_stationary_prj/solution1/sim/verilog"
sim_root="$project_dir/build/q8q8_weight_stationary_prj/solution1/sim"
post_dir="$project_dir/build/q8q8_weight_stationary_prj/solution1/sim/wrapc_pc"
artifact_dir="$project_dir/artifacts"
mkdir -p "$artifact_dir"

if [[ ${1:-} != "--finish-only" ]]; then
    set +e
    "$vitis_hls_bin" -f "$project_dir/scripts/run_hls.tcl" cosim \
        2>&1 | tee "$artifact_dir/q8q8_weight_stationary_native_cosim.log"
    hls_status=${PIPESTATUS[0]}
    set -e
    if [[ $hls_status -eq 0 ]]; then
        echo "COSIM_RESULT=PASS path=native"
        exit 0
    fi
    if ! grep -q "unexpected exception" \
            "$sim_dir/run_xsim.log" \
            "$artifact_dir/q8q8_weight_stationary_native_cosim.log" \
            2>/dev/null; then
        echo "Native co-simulation failed for an unknown reason." >&2
        exit "$hls_status"
    fi
fi

xsim_script="$sim_dir/xsim.dir/q8q8_gemm/xsim_script.tcl"
if [[ ! -f "$xsim_script" || ! -x "$post_dir/cosim.pc.exe" ]]; then
    echo "Co-simulation snapshot or post-check executable is missing." >&2
    exit 2
fi
wave_cfg="$sim_dir/q8q8_gemm_dataflow_ana.wcfg"
if [[ -f "$wave_cfg" ]]; then mv "$wave_cfg" "$wave_cfg.disabled"; fi
sed -i -e 's/ -autoloadwcfg//' -e 's/ -nosignalhandlers//' "$xsim_script"

# XSim 2023.2 can throw an undocumented "unexpected exception" when its user
# state directory is not writable (for example, inside a restricted sandbox).
# A normal host shell with a writable Vivado user directory is required.  A
# short temporary copy also keeps the fallback independent of workspace paths;
# copy only the generated RTL results back for the HLS checker.
short_root=$(mktemp -d /tmp/q8XXXXXX)
trap 'rm -rf -- "$short_root"' EXIT
mkdir -p "$short_root/sim"
cp -a "$sim_root/tv" "$short_root/sim/"
cp -a "$sim_root/verilog" "$short_root/sim/"
run_sim_dir="$short_root/sim/verilog"
# In batch-only environments XSim 2023.2 may fail before creating its default
# settings file. Supplying the deterministic defaults avoids that launch bug.
cp "$project_dir/scripts/xsimSettings.ini" \
    "$run_sim_dir/xsim.dir/q8q8_gemm/xsimSettings.ini"
xsim_status=1
for attempt in 1 2 3; do
    set +e
    (
        cd "$run_sim_dir"
        env -u Q8Q8_XSIM_NO_SIGNAL_HANDLERS \
            "$xsim_bin" q8q8_gemm -tclbatch q8q8_gemm.tcl
    ) 2>&1 | tee \
        "$artifact_dir/q8q8_weight_stationary_rtl_xsim_attempt${attempt}.log"
    xsim_status=${PIPESTATUS[0]}
    set -e
    if [[ $xsim_status -eq 0 ]]; then
        cp "$artifact_dir/q8q8_weight_stationary_rtl_xsim_attempt${attempt}.log" \
            "$artifact_dir/q8q8_weight_stationary_rtl_xsim.log"
        break
    fi
done
if [[ $xsim_status -ne 0 ]]; then
    echo "XSim failed after three no-wave attempts." >&2
    exit "$xsim_status"
fi
mkdir -p "$sim_root/tv/rtldatafile"
cp -a "$short_root/sim/tv/rtldatafile/." "$sim_root/tv/rtldatafile/"
(
    cd "$post_dir"
    ./cosim.pc.exe --rtl-smoke
) 2>&1 | tee "$artifact_dir/q8q8_weight_stationary_rtl_postcheck.log"
grep -q "TB_RESULT=PASS mode=rtl_smoke" \
    "$artifact_dir/q8q8_weight_stationary_rtl_postcheck.log"
echo "COSIM_RESULT=PASS path=xsim-2023.2-no-wave-workaround"
