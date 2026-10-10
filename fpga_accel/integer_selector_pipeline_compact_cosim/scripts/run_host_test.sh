#!/usr/bin/env bash
# Focused host compilation + execution of the dedicated RTL-co-simulation
# testbench against the five frozen cases, BEFORE any C/RTL co-simulation runs.
#
# Scope: host g++ only.  No dlopen, no Vitis, no GPU, no network, no Vivado,
# no board.  The accepted compact source/header and the accepted fixture
# payloads are referenced read-only.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$proj_dir/../.." && pwd)"

fixture_dir="$repo_root/artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures"
case_list="$proj_dir/cases/rtl_cosim_cases.txt"
artifact_dir="$repo_root/artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim"
build_dir="${ISPC_COSIM_HOST_BUILD:-/tmp/robobrain_integer_selector_compact_cosim_host}"

src="$repo_root/fpga_accel/integer_selector_pipeline_compact_hls/src/integer_selector_pipeline_compact.cpp"
tb="$proj_dir/tb/tb_integer_selector_pipeline_compact_cosim.cpp"

inc_flags=(
  "-I$repo_root/fpga_accel/integer_selector_pipeline_compact_hls/include"
  "-I$repo_root/fpga_accel/integer_selector_scorer_hls/include"
  "-I$repo_root/fpga_accel/token_selector_hls/include"
  "-I$repo_root/fpga_accel/integer_selector_pipeline_hls/include"
)

mkdir -p "$build_dir" "$artifact_dir/logs" "$artifact_dir/cases"

# 0) Resolve the five required representatives against the accepted manifest.
python3 "$proj_dir/tools/validate_cases.py" \
  --fixtures "$fixture_dir" \
  --cases "$case_list" \
  --out "$artifact_dir/cases/rtl_cosim_cases_resolved.json" \
  > "$artifact_dir/logs/00_case_resolution.log"
echo "case resolution OK -> $artifact_dir/cases/rtl_cosim_cases_resolved.json"

# 1) Plain host build + run.
g++ -std=c++14 -O2 -Wno-unknown-pragmas "${inc_flags[@]}" \
  "$src" "$tb" -o "$build_dir/host_tb"
"$build_dir/host_tb" "$fixture_dir" "$case_list" \
  | tee "$artifact_dir/logs/01_host_test.log"

# 2) ASan + UBSan host build + run (canary / bounds hardening).
g++ -std=c++14 -O1 -g -Wno-unknown-pragmas -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${inc_flags[@]}" \
  "$src" "$tb" -o "$build_dir/host_tb_asan"
"$build_dir/host_tb_asan" "$fixture_dir" "$case_list" \
  | tee "$artifact_dir/logs/02_host_test_asan_ubsan.log"

echo "host focused test: PASS"
