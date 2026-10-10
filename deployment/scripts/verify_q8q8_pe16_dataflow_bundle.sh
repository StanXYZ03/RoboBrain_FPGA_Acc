#!/bin/sh
# Offline-only DATAFLOW bitstream, dynamic-DDR FSBL and RAM-Linux gate.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
HLS="$ROOT/fpga_accel/q8q8_pe16_dataflow_hls/artifacts"
ACCEL_ROOT="$ROOT/fpga_accel/q8q8_pe16_dataflow_zcu102_vivado"
ACCEL="$ACCEL_ROOT/artifacts"
BIT_RUN="$ACCEL/runs/20260829T141332Z_bitstream"
RUNTIME="$ROOT/fpga_accel/q8q8_runtime"
RUNTIME_ARTIFACTS="$RUNTIME/artifacts/dataflow_measurement_20260830"
RAM="$ROOT/deployment/artifacts/ram_linux_pe16_dataflow_measurement_20260830"
RAM_BUILD="$ROOT/build/ram-linux-pe16-dataflow-measurement-20260830"
QEMU_LOG="$ROOT/deployment/reports/2026-08-30-ram-linux-pe16-dataflow-measurement-qemu.log"
MEASURE_PROTOCOL="$ROOT/deployment/ram_linux/q8q8_dataflow_measure.sh"
MEASURE_PROTOCOL_TEST="$ROOT/deployment/scripts/test_q8q8_dataflow_measure_protocol.sh"
SUMMARIZER_TEST="$ROOT/deployment/scripts/test_summarize_q8q8_pe16_dataflow_board.py"
UART_CAPTURE_TEST="$ROOT/deployment/scripts/test_capture_q8q8_dataflow_uart.py"
JTAG="$ROOT/deployment/scripts/run_q8q8_accel_jtag.tcl"
BASELINE_FSBL_BUILD="$ROOT/fpga_accel/q8q8_pe16_zcu102_vivado/build/q8q8_pe16_fsbl"
VITIS_ROOT=${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}
FDT_BIN="$VITIS_ROOT/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin"
READELF="$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-readelf"
XSCT="$VITIS_ROOT/bin/xsct"

fail() {
    echo "Q8Q8_PE16_DATAFLOW_BUNDLE_FAIL: $*" >&2
    exit 1
}

for file in \
    "$HLS/q8q8_pe16_dataflow_ip.zip" \
    "$BIT_RUN/q8q8_pe16_dataflow_zcu102.bit" \
    "$BIT_RUN/q8q8_pe16_dataflow_zcu102.xsa" \
    "$BIT_RUN/bitstream_summary.txt" \
    "$ACCEL/q8q8_pe16_dataflow_fsbl.elf" \
    "$ACCEL/q8q8_pe16_dataflow_fsbl_metadata.txt" \
    "$RUNTIME_ARTIFACTS/q8q8-selftest-arm64" \
    "$RUNTIME_ARTIFACTS/q8q8-benchmark-arm64" \
    "$RUNTIME_ARTIFACTS/runtime_unit_test.log" \
    "$RUNTIME_ARTIFACTS/runtime_sanitizer_test.log" \
    "$RUNTIME_ARTIFACTS/benchmark_unit_test.log" \
    "$RUNTIME_ARTIFACTS/benchmark_sanitizer_test.log" \
    "$RUNTIME_ARTIFACTS/apm_unit_test.log" \
    "$RUNTIME_ARTIFACTS/dt_overlay_test.log" \
    "$RUNTIME_ARTIFACTS/register_map_pe16_dataflow_test.log" \
    "$RUNTIME_ARTIFACTS/dispatch_unit_test.log" \
    "$RUNTIME_ARTIFACTS/dispatch_sanitizer_test.log" \
    "$RAM/Image" "$RAM/system-16gb.dtb" "$RAM/rootfs.cpio.gz" \
    "$RAM/boot.scr" "$RAM/SHA256SUMS" "$RAM/NETWORK_CONFIG" \
    "$MEASURE_PROTOCOL" "$MEASURE_PROTOCOL_TEST" "$SUMMARIZER_TEST" \
    "$ROOT/deployment/scripts/capture_q8q8_dataflow_uart.py" "$UART_CAPTURE_TEST" \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_measure_protocol_test.log" \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_summarizer_test.log" \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_uart_capture_test.log" \
    "$QEMU_LOG" "$JTAG"; do
    [ -r "$file" ] || fail "missing artifact: $file"
done
for command in sha256sum unzip gzip cpio grep awk mktemp cmp; do
    command -v "$command" >/dev/null 2>&1 || fail "missing host tool: $command"
done
[ -x "$FDT_BIN/fdtget" ] || fail "fdtget missing"
[ -x "$READELF" ] || fail "readelf missing"
[ -x "$XSCT" ] || fail "XSCT missing"

grep -qx 'WNS_NS=0.078' "$BIT_RUN/bitstream_summary.txt" || fail "setup timing mismatch"
grep -qx 'WHS_NS=0.010' "$BIT_RUN/bitstream_summary.txt" || fail "hold timing mismatch"
grep -qx 'UNROUTED_NETS=0' "$BIT_RUN/bitstream_summary.txt" || fail "unrouted nets remain"
grep -qx 'BITSTREAM_GENERATED=YES' "$BIT_RUN/bitstream_summary.txt" || fail "bitstream gate failed"
grep -qx 'BOARD_ACCESSED=NO' "$BIT_RUN/bitstream_summary.txt" || fail "unexpected board access marker"
grep -qx 'BOARD_PROGRAMMED=NO' "$BIT_RUN/bitstream_summary.txt" || fail "unexpected board marker"

bit_hash=$(sha256sum "$BIT_RUN/q8q8_pe16_dataflow_zcu102.bit" | awk '{print $1}')
xsa_bit_hash=$(unzip -p "$BIT_RUN/q8q8_pe16_dataflow_zcu102.xsa" \
    q8q8_pe16_dataflow_zcu102.bit | sha256sum | awk '{print $1}')
[ "$bit_hash" = "$xsa_bit_hash" ] || fail "standalone and XSA bitstreams differ"

(cd "$ACCEL" && tail -n 1 q8q8_pe16_dataflow_fsbl_metadata.txt | \
    sha256sum -c - >/dev/null) || fail "FSBL checksum mismatch"
grep -qx 'DDR_INIT_MODE=SPD_DYNAMIC_DUAL_RANK' \
    "$ACCEL/q8q8_pe16_dataflow_fsbl_metadata.txt" || fail "dynamic DDR mode missing"
grep -qx 'BOOT_PATH=PARTITION_LOAD_EXCLUDED_JTAG' \
    "$ACCEL/q8q8_pe16_dataflow_fsbl_metadata.txt" || fail "JTAG-only FSBL mode missing"
grep -qx 'SOURCE_XSA_SHA256=6a1fd18fde062ebc192c37b1288ca0116c8013cfe9caa1ca8800705482b43f04' \
    "$ACCEL/q8q8_pe16_dataflow_fsbl_metadata.txt" || fail "FSBL/XSA provenance mismatch"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"$READELF" -sW "$ACCEL/q8q8_pe16_dataflow_fsbl.elf" > "$tmp/fsbl.symbols"
for symbol in XFsbl_DdrInit XFsbl_DdrcPhyTraining XFsbl_Exit XFsbl_Loop; do
    grep -q " $symbol$" "$tmp/fsbl.symbols" || fail "FSBL symbol missing: $symbol"
done
grep -q ' psu_ps_pl_isolation_removal_data' "$tmp/fsbl.symbols" \
    || fail "PS-PL isolation helper missing"
grep -q ' psu_ps_pl_reset_config_data' "$tmp/fsbl.symbols" \
    || fail "PS-PL reset helper missing"
grep -q '0x53504434U' "$ACCEL_ROOT/build/q8q8_pe16_dataflow_fsbl/xfsbl_ddr_init.c" \
    || fail "SPD evidence patch missing"
grep -q 'AddrMapCsBit0 = PDimmPtr->NumBgAddr' \
    "$ACCEL_ROOT/build/q8q8_pe16_dataflow_fsbl/xfsbl_ddr_init.c" \
    || fail "dual-rank address patch missing"
cmp -s "$ACCEL_ROOT/build/q8q8_pe16_dataflow_fsbl/psu_init.c" \
    "$BASELINE_FSBL_BUILD/psu_init.c" || fail "PS initialization changed from board baseline"
cmp -s "$ACCEL_ROOT/build/q8q8_pe16_dataflow_fsbl/xfsbl_ddr_init.c" \
    "$BASELINE_FSBL_BUILD/xfsbl_ddr_init.c" || fail "patched DDR initialization changed"

grep -q 'TEST_RESULT=PASS cases=4' "$RUNTIME_ARTIFACTS/runtime_unit_test.log" \
    || fail "runtime unit test missing"
grep -q 'TEST_RESULT=PASS cases=4' "$RUNTIME_ARTIFACTS/runtime_sanitizer_test.log" \
    || fail "runtime sanitizer test missing"
grep -q 'BENCH_SELFTEST_PASS' "$RUNTIME_ARTIFACTS/benchmark_unit_test.log" \
    || fail "benchmark unit test missing"
grep -q 'BENCH_SELFTEST_PASS' "$RUNTIME_ARTIFACTS/benchmark_sanitizer_test.log" \
    || fail "benchmark sanitizer test missing"
grep -q 'APM_UNIT_PASS' "$RUNTIME_ARTIFACTS/apm_unit_test.log" || fail "APM test missing"
grep -q 'DT_TEST_PASS' "$RUNTIME_ARTIFACTS/dt_overlay_test.log" || fail "DT test missing"
grep -q 'REGISTER_MAP_PASS registers=10 source=q8q8_pe16_dataflow_ip.zip' \
    "$RUNTIME_ARTIFACTS/register_map_pe16_dataflow_test.log" \
    || fail "DATAFLOW register map mismatch"
grep -q 'DISPATCH_TEST_RESULT=PASS cases=5 task_fields=0' \
    "$RUNTIME_ARTIFACTS/dispatch_unit_test.log" || fail "dispatch unit test missing"
grep -q 'DISPATCH_TEST_RESULT=PASS cases=5 task_fields=0' \
    "$RUNTIME_ARTIFACTS/dispatch_sanitizer_test.log" \
    || fail "dispatch sanitizer test missing"

(cd "$RAM" && sha256sum -c SHA256SUMS >/dev/null) || fail "RAM-Linux checksums differ"
grep -qx 'pl_accel_mode=1' "$RAM/NETWORK_CONFIG" || fail "accelerator mode disabled"
grep -qx 'CONFIG_UIO=y' "$RAM_BUILD/linux/.config" || fail "UIO not built in"
grep -qx 'CONFIG_UIO_PDRV_GENIRQ=y' "$RAM_BUILD/linux/.config" \
    || fail "generic UIO not built in"
grep -q 'RAM_LINUX_QEMU_PASS' "$QEMU_LOG" \
    || fail "RAM Linux QEMU smoke evidence missing"

memory_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" /memory reg)
arena_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" \
    /reserved-memory/q8q8-dma@b7c000000 reg)
uio_reg=$("$FDT_BIN/fdtget" -t x "$RAM/system-16gb.dtb" \
    /q8q8-gemm@a0000000 reg)
[ "$memory_reg" = '0 0 0 7ff00000 8 0 3 80000000' ] || fail "16GB map mismatch"
[ "$arena_reg" = 'b 7c000000 0 4000000' ] || fail "DMA arena mismatch"
[ "$uio_reg" = '0 a0000000 0 10000 b 7c000000 0 4000000' ] || fail "UIO map mismatch"

mkdir "$tmp/rootfs"
(cd "$tmp/rootfs" && gzip -dc "$RAM/rootfs.cpio.gz" | cpio -id --quiet \
    opt/robobrain/q8q8-selftest opt/robobrain/q8q8-benchmark \
    opt/robobrain/q8q8-dataflow-measure init) \
    || fail "cannot extract runtime payload"
for name in q8q8-selftest q8q8-benchmark; do
    [ -s "$tmp/rootfs/opt/robobrain/$name" ] || fail "payload missing: $name"
    "$READELF" -h "$tmp/rootfs/opt/robobrain/$name" | grep -q 'Machine:.*AArch64' \
        || fail "$name is not AArch64"
done
[ "$(sha256sum "$tmp/rootfs/opt/robobrain/q8q8-selftest" | awk '{print $1}')" = \
  "$(sha256sum "$RUNTIME_ARTIFACTS/q8q8-selftest-arm64" | awk '{print $1}')" ] \
    || fail "staged self-test mismatch"
[ "$(sha256sum "$tmp/rootfs/opt/robobrain/q8q8-benchmark" | awk '{print $1}')" = \
  "$(sha256sum "$RUNTIME_ARTIFACTS/q8q8-benchmark-arm64" | awk '{print $1}')" ] \
    || fail "staged benchmark mismatch"
grep -q 'RAM_LINUX_PL_SELFTEST_PASS' "$tmp/rootfs/init" \
    || fail "init does not run PL self-test"
grep -q -- '--auto-chunk' "$tmp/rootfs/opt/robobrain/q8q8-benchmark" \
    || fail "staged benchmark lacks automatic DMA chunking"
grep -q 'visual_prefill' "$tmp/rootfs/opt/robobrain/q8q8-benchmark" \
    || fail "staged benchmark lacks full M=1080 visual prefill case"
cmp -s "$tmp/rootfs/opt/robobrain/q8q8-dataflow-measure" \
    "$MEASURE_PROTOCOL" || fail "staged DATAFLOW measurement protocol mismatch"
protocol_test=$($MEASURE_PROTOCOL_TEST)
echo "$protocol_test" | grep -q \
    'DATAFLOW_MEASURE_PROTOCOL_TEST_PASS stages=15 idle_gates=15 task_fields=0' \
    || fail "DATAFLOW measurement protocol test missing"
summarizer_test=$($SUMMARIZER_TEST)
echo "$summarizer_test" | grep -q \
    'DATAFLOW_BOARD_SUMMARIZER_TEST_PASS cases=4 latency_formula=total_over_total' \
    || fail "DATAFLOW board summarizer test missing"
grep -qx 'DATAFLOW_MEASURE_PROTOCOL_TEST_PASS stages=15 idle_gates=15 task_fields=0' \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_measure_protocol_test.log" \
    || fail "frozen protocol-test evidence mismatch"
grep -qx 'DATAFLOW_BOARD_SUMMARIZER_TEST_PASS cases=4 latency_formula=total_over_total' \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_summarizer_test.log" \
    || fail "frozen summarizer-test evidence mismatch"
uart_capture_test=$($UART_CAPTURE_TEST)
echo "$uart_capture_test" | grep -q \
    'DATAFLOW_UART_CAPTURE_TEST_PASS command_after_boot=1 append_only=1' \
    || fail "DATAFLOW UART capture test missing"
grep -qx 'DATAFLOW_UART_CAPTURE_TEST_PASS command_after_boot=1 append_only=1' \
    "$ROOT/deployment/artifacts/q8q8_pe16_dataflow_uart_capture_test.log" \
    || fail "frozen UART-capture evidence mismatch"

for variable in Q8Q8_RAM_ARTIFACT_DIR Q8Q8_ACCEL_FSBL \
        Q8Q8_ACCEL_FSBL_METADATA Q8Q8_ACCEL_BITSTREAM; do
    grep -q "$variable" "$JTAG" || fail "JTAG override missing: $variable"
done
tcl_complete=$("$XSCT" -nodisp -quiet -eval \
    "set fd [open {$JTAG} r]; set script [read \$fd]; close \$fd; puts [info complete \$script]")
[ "$tcl_complete" = 1 ] || fail "JTAG Tcl is incomplete"

echo "Q8Q8_PE16_DATAFLOW_BUNDLE_PASS bit_sha256=$bit_hash memory_bytes=17178820608 dma_bytes=67108864 control=0xA0000000 board_accessed=no board_programmed=no"
