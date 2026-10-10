#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vitis_hls_bin=${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}
vivado_bin=${VIVADO_BIN:-/home/xilinx/Vivado/2023.2/bin/vivado}
sim_dir="$project_dir/build/q8q8_dataflow_prj/solution1/sim/verilog"
post_dir="$project_dir/build/q8q8_dataflow_prj/solution1/sim/wrapc_pc"
artifact_dir="$project_dir/artifacts"
mkdir -p "$artifact_dir"

if [[ ${1:-} != "--finish-only" ]]; then
    set +e
    "$vitis_hls_bin" -f "$project_dir/scripts/run_hls.tcl" cosim \
        2>&1 | tee "$artifact_dir/q8q8_dataflow_native_cosim.log"
    hls_status=${PIPESTATUS[0]}
    set -e

    if [[ $hls_status -eq 0 ]]; then
        echo "COSIM_RESULT=PASS path=native"
        exit 0
    fi
    # Depending on the 2023.2 patch level, the exception is written either to
    # run_xsim.log or only to the parent Vitis HLS console log captured above.
    if ! grep -q "unexpected exception" \
            "$sim_dir/run_xsim.log" \
            "$artifact_dir/q8q8_dataflow_native_cosim.log" 2>/dev/null; then
        echo "Native co-simulation failed for an unknown reason; refusing workaround." >&2
        exit "$hls_status"
    fi
fi

xsim_script="$sim_dir/xsim.dir/q8q8_gemm/xsim_script.tcl"
if [[ ! -f "$xsim_script" || ! -x "$post_dir/cosim.pc.exe" ]]; then
    echo "Co-simulation snapshot or post-check executable is missing." >&2
    exit 2
fi

# Vivado 2023.2 can crash while loading the automatically generated DATAFLOW
# GUI waveform. The waveform is not part of the DUT: preserve it under a new
# name and execute the same elaborated snapshot without automatic wave loading.
wave_cfg="$sim_dir/q8q8_gemm_dataflow_ana.wcfg"
if [[ -f "$wave_cfg" ]]; then
    mv "$wave_cfg" "$wave_cfg.disabled"
fi
sed -i \
    -e 's/ -autoloadwcfg//' \
    -e 's/ -nosignalhandlers//' \
    "$xsim_script"

xsim_status=1
for attempt in 1 2 3; do
    no_signal_handlers=0
    [[ $attempt -eq 2 ]] && no_signal_handlers=1
    set +e
    (
        cd "$sim_dir"
        Q8Q8_XSIM_NO_SIGNAL_HANDLERS=$no_signal_handlers \
            "$vivado_bin" -mode batch \
            -source "$project_dir/scripts/run_xsim_no_wave.tcl"
    ) 2>&1 | tee "$artifact_dir/q8q8_dataflow_rtl_xsim_attempt${attempt}.log"
    xsim_status=${PIPESTATUS[0]}
    set -e
    if [[ $xsim_status -eq 0 ]]; then
        cp "$artifact_dir/q8q8_dataflow_rtl_xsim_attempt${attempt}.log" \
            "$artifact_dir/q8q8_dataflow_rtl_xsim.log"
        break
    fi
done
if [[ $xsim_status -ne 0 ]]; then
    echo "XSim failed after three no-wave attempts." >&2
    exit "$xsim_status"
fi

(
    cd "$post_dir"
    ./cosim.pc.exe --rtl-smoke
) 2>&1 | tee "$artifact_dir/q8q8_dataflow_rtl_postcheck.log"

grep -q "TB_RESULT=PASS mode=rtl_smoke" \
    "$artifact_dir/q8q8_dataflow_rtl_postcheck.log"
echo "COSIM_RESULT=PASS path=xsim-2023.2-no-wave-workaround"
