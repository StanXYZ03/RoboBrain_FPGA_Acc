#!/bin/sh
# Versioned, board-side DATAFLOW measurement protocol. It never resets or
# programs the board; the host must first complete the separately gated JTAG
# volatile boot and capture UART from before that boot begins.
set -eu

SELFTEST=${Q8Q8_SELFTEST_BIN:-/opt/robobrain/q8q8-selftest}
BENCHMARK=${Q8Q8_BENCHMARK_BIN:-/opt/robobrain/q8q8-benchmark}
DEVMEM=${Q8Q8_DEVMEM_BIN:-devmem}
LOCK=${Q8Q8_MEASURE_LOCK:-/run/q8q8-dataflow-measure.lock}
DRY_RUN=0

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "DATAFLOW_MEASURE_FAIL stage=arguments" >&2; exit 2 ;;
esac

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "DATAFLOW_MEASURE_FAIL stage=lock reason=already_running" >&2
    exit 3
fi

finish() {
    status=$1
    trap - EXIT HUP INT TERM
    if [ "$DRY_RUN" -eq 0 ]; then
        control=$($DEVMEM 0xA0000000 32 2>/dev/null || echo unreadable)
        echo "DATAFLOW_MEASURE_FINAL_CONTROL value=$control"
    fi
    rmdir "$LOCK" 2>/dev/null || true
    if [ "$status" -eq 0 ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "DATAFLOW_MEASURE_DRY_RUN_PASS"
        else
            echo "DATAFLOW_MEASURE_PASS protocol=1"
        fi
    else
        echo "DATAFLOW_MEASURE_FAIL stage=command rc=$status" >&2
    fi
    exit "$status"
}
trap 'finish $?' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

require_executable() {
    [ -x "$1" ] || {
        echo "DATAFLOW_MEASURE_FAIL stage=prerequisite missing=$1" >&2
        exit 4
    }
}

check_idle() {
    control=$($DEVMEM 0xA0000000 32)
    case "$control" in
        0x*|0X*) ;;
        *) echo "DATAFLOW_MEASURE_FAIL stage=idle invalid_control=$control" >&2; exit 5 ;;
    esac
    [ $((control & 4)) -ne 0 ] || {
        echo "DATAFLOW_MEASURE_FAIL stage=idle control=$control" >&2
        exit 5
    }
    echo "DATAFLOW_MEASURE_IDLE control=$control"
}

run_command() {
    label=$1
    shift
    echo "DATAFLOW_MEASURE_STAGE label=$label state=begin"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DATAFLOW_MEASURE_COMMAND label=$label argv=$*"
    else
        "$@"
        check_idle
    fi
    echo "DATAFLOW_MEASURE_STAGE label=$label state=done"
}

echo "DATAFLOW_MEASURE_BEGIN protocol=1 dry_run=$DRY_RUN"
if [ "$DRY_RUN" -eq 0 ]; then
    require_executable "$SELFTEST"
    require_executable "$BENCHMARK"
    command -v "$DEVMEM" >/dev/null 2>&1 || {
        echo "DATAFLOW_MEASURE_FAIL stage=prerequisite missing=$DEVMEM" >&2
        exit 4
    }
fi

# Independent pre-check. Linux PID 1 has already run one boot self-test, but
# this marker brackets the raw measurement protocol itself.
run_command pre_selftest "$SELFTEST" --timeout-ms 5000

# Frozen M=512 shape: isolates DATAFLOW hardware from the prior PE16 kernel.
run_command kernel_language_prefill_fixed512 "$BENCHMARK" \
    --case language_prefill --chunk-m 512 --repeat 3

# Like-for-like chunk-policy A/B at identical full workload shapes.
run_command visual_prefill_fixed512 "$BENCHMARK" \
    --case visual_prefill --chunk-m 512 --repeat 3
run_command visual_prefill_auto "$BENCHMARK" \
    --case visual_prefill --auto-chunk --repeat 3
run_command vision_encoder_fixed512 "$BENCHMARK" \
    --case vision_encoder --chunk-m 512 --repeat 3
run_command vision_encoder_auto "$BENCHMARK" \
    --case vision_encoder --auto-chunk --repeat 3
run_command decoder_gemv_fixed "$BENCHMARK" \
    --case decoder_gemv --chunk-m 1 --repeat 3
run_command decoder_gemv_auto "$BENCHMARK" \
    --case decoder_gemv --auto-chunk --repeat 3

# APM is collected on the selected auto policy. Each profile is independent;
# all counters are reset and overflow-gated by q8q8-benchmark per launch.
for workload in visual_prefill vision_encoder decoder_gemv; do
    run_command "${workload}_auto_apm_traffic" "$BENCHMARK" \
        --case "$workload" --auto-chunk --repeat 3 --apm traffic
    run_command "${workload}_auto_apm_latency" "$BENCHMARK" \
        --case "$workload" --auto-chunk --repeat 3 --apm latency
done

run_command post_selftest "$SELFTEST" --timeout-ms 5000
