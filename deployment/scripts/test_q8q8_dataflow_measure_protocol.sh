#!/bin/sh
# Offline structural test for the board-side protocol. No UART or board access.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PROTOCOL="$ROOT/deployment/ram_linux/q8q8_dataflow_measure.sh"
MOCK_DEVMEM="$ROOT/deployment/scripts/testdata/mock_devmem_idle.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

fail() {
    echo "DATAFLOW_MEASURE_PROTOCOL_TEST_FAIL: $*" >&2
    exit 1
}

sh -n "$PROTOCOL" || fail "shell syntax"
Q8Q8_MEASURE_LOCK="$TMP/lock" sh "$PROTOCOL" --dry-run > "$TMP/protocol.log"

grep -q '^DATAFLOW_MEASURE_BEGIN protocol=1 dry_run=1$' "$TMP/protocol.log" \
    || fail "begin marker"
grep -q '^DATAFLOW_MEASURE_DRY_RUN_PASS$' "$TMP/protocol.log" \
    || fail "completion marker"
[ "$(grep -c ' state=begin$' "$TMP/protocol.log")" -eq 15 ] \
    || fail "expected 15 begin stages"
[ "$(grep -c ' state=done$' "$TMP/protocol.log")" -eq 15 ] \
    || fail "expected 15 completed stages"
[ "$(grep -c -- '--case visual_prefill --auto-chunk --repeat 3' "$TMP/protocol.log")" -eq 3 ] \
    || fail "visual prefill auto execution/APM coverage"
[ "$(grep -c -- '--case vision_encoder --auto-chunk --repeat 3' "$TMP/protocol.log")" -eq 3 ] \
    || fail "vision encoder auto execution/APM coverage"
[ "$(grep -c -- '--case decoder_gemv --auto-chunk --repeat 3' "$TMP/protocol.log")" -eq 3 ] \
    || fail "decoder auto execution/APM coverage"
[ "$(grep -c -- '--case visual_prefill --chunk-m 512 --repeat 3$' "$TMP/protocol.log")" -eq 1 ] \
    || fail "visual prefill fixed-512 arm"
[ "$(grep -c -- '--case vision_encoder --chunk-m 512 --repeat 3$' "$TMP/protocol.log")" -eq 1 ] \
    || fail "vision encoder fixed-512 arm"
[ "$(grep -c -- 'q8q8-selftest --timeout-ms 5000$' "$TMP/protocol.log")" -eq 2 ] \
    || fail "pre/post self-tests"
! grep -q 'Grounding\|Pointing\|Trajectory\|General' "$TMP/protocol.log" \
    || fail "task name leaked into hardware protocol"

Q8Q8_MEASURE_LOCK="$TMP/live-lock" \
Q8Q8_SELFTEST_BIN=/bin/true \
Q8Q8_BENCHMARK_BIN=/bin/true \
Q8Q8_DEVMEM_BIN="$MOCK_DEVMEM" \
    sh "$PROTOCOL" > "$TMP/mock-live.log"
grep -q '^DATAFLOW_MEASURE_PASS protocol=1$' "$TMP/mock-live.log" \
    || fail "mock live completion marker"
[ "$(grep -c '^DATAFLOW_MEASURE_IDLE control=0x00000004$' "$TMP/mock-live.log")" -eq 15 ] \
    || fail "per-command idle gates"
grep -q '^DATAFLOW_MEASURE_FINAL_CONTROL value=0x00000004$' "$TMP/mock-live.log" \
    || fail "final idle gate"

FROZEN="$ROOT/deployment/artifacts/q8q8_pe16_dataflow_measure_protocol_dry_run.log"
if [ -e "$FROZEN" ]; then
    cmp -s "$TMP/protocol.log" "$FROZEN" || fail "frozen dry-run log changed"
else
    cp "$TMP/protocol.log" "$FROZEN"
fi
echo "DATAFLOW_MEASURE_PROTOCOL_TEST_PASS stages=15 idle_gates=15 task_fields=0"
