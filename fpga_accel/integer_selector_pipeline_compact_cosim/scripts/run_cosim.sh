#!/usr/bin/env bash
# Controlled C/RTL co-simulation driver for the accepted compact integer
# Selector pipeline.
#
#   scripts/run_cosim.sh
#
# Runs exactly `csynth_design` + `cosim_design -rtl verilog -tool xsim
# -trace_level none` for the five frozen representative cases.  The HLS project
# stays under /tmp; only compact logs / reports / transaction+performance
# summaries / generated metadata are copied back into the repository *within
# this same process* (the /tmp project itself is never archived).
#
# A watchdog enforces the bounded-stress clause: once the console log shows
# `[COSIM 212-15] Starting XSIM`, if XSim runs longer than 60 minutes the whole
# task process group is terminated, the log is preserved, and
# `logs/03_cosim_xsim_timeout.flag` is raised.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$proj_dir/../.." && pwd)"
artifact_dir="$repo_root/artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim"

vitis_hls_bin="${VITIS_HLS_BIN:-/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls}"
project_dir="${ISPC_COSIM_PROJECT_DIR:-/tmp/robobrain_integer_selector_compact_cosim}"
log_path="$artifact_dir/logs/03_cosim.log"
timeout_flag="$artifact_dir/logs/03_cosim_xsim_timeout.flag"
mkdir -p "$artifact_dir/logs" "$artifact_dir/cosim"
rm -f "$timeout_flag"

test -x "$vitis_hls_bin" || { echo "missing Vitis HLS: $vitis_hls_bin" >&2; exit 3; }

# SuSE libtinfo.so.5 needed by the 2023.2 Tcl shell on modern hosts.
export LD_LIBRARY_PATH="/home/xilinx/xic/lib/lnx64.o/SuSE:${LD_LIBRARY_PATH:-}"

rm -rf "$project_dir"
session_start="$(date +%s)"
echo "[run_cosim] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$log_path"

setsid "$vitis_hls_bin" -f "$proj_dir/scripts/run_cosim.tcl" >> "$log_path" 2>&1 &
vitis_pid=$!
sleep 2
vitis_pgid="$(ps -o pgid= -p "$vitis_pid" 2>/dev/null | tr -d ' ')"
[ -n "$vitis_pgid" ] || vitis_pgid="$vitis_pid"

(
  xsim_start_epoch=""
  while kill -0 "$vitis_pid" 2>/dev/null; do
    now="$(date +%s)"
    if [ -z "$xsim_start_epoch" ] && grep -q "COSIM 212-15" "$log_path" 2>/dev/null; then
      xsim_start_epoch="$now"
      echo "[run_cosim] watchdog: XSIM started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log_path"
    fi
    killed=0
    if [ -n "$xsim_start_epoch" ] && [ "$((now - xsim_start_epoch))" -gt 3600 ]; then
      echo "[run_cosim] watchdog: XSIM exceeded 3600 s after start; terminating task process group $vitis_pgid" >> "$log_path"
      touch "$timeout_flag"
      killed=1
    elif [ "$((now - session_start))" -gt 7200 ]; then
      echo "[run_cosim] watchdog: total wall clock exceeded 7200 s; terminating task process group $vitis_pgid" >> "$log_path"
      killed=1
    fi
    if [ "$killed" -eq 1 ]; then
      kill -TERM -"$vitis_pgid" 2>/dev/null || kill -TERM "$vitis_pid" 2>/dev/null || true
      sleep 15
      kill -KILL -"$vitis_pgid" 2>/dev/null || kill -KILL "$vitis_pid" 2>/dev/null || true
      break
    fi
    sleep 10
  done
) &
watchdog_pid=$!

set +e
wait "$vitis_pid"
rc=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true

end_epoch="$(date +%s)"
elapsed=$((end_epoch - session_start))
echo "[run_cosim] vitis_hls exit=$rc elapsed_seconds=$elapsed xsim_timeout_flag=$([ -f "$timeout_flag" ] && echo yes || echo no)" >> "$log_path"

# ---------------------------------------------------------------------------
# Archive only compact summaries from the /tmp project (still inside this
# process; /tmp is not part of the repository).
# ---------------------------------------------------------------------------
mkdir -p "$artifact_dir/cosim/syn_report" "$artifact_dir/cosim/sim_report"
: > "$artifact_dir/cosim/archive_index.txt"

archive() {
  local src="$1" dst="$2"
  if [ -f "$src" ]; then
    local sz
    sz="$(stat -c%s "$src")"
    if [ "$sz" -le 8388608 ]; then
      cp -f "$src" "$dst/$(basename "$src")"
      printf '%s\t%s\n' "$(basename "$src")" "$sz" >> "$artifact_dir/cosim/archive_index.txt"
    fi
  fi
}

# Top-level synthesis report (interface table, latency, resources).
for ext in rpt xml; do
  archive "$project_dir/solution1/syn/report/integer_selector_pipeline_compact_csynth.$ext" \
          "$artifact_dir/cosim/syn_report"
done

# Co-simulation report files (transaction / performance / options).
if [ -d "$project_dir/solution1/sim/report" ]; then
  for f in "$project_dir/solution1/sim/report"/*; do
    [ -f "$f" ] || continue
    archive "$f" "$artifact_dir/cosim/sim_report"
  done
fi
if [ -d "$project_dir/solution1/sim/report/verilog" ]; then
  # keep only tiny transaction/performance metadata files, never the RTL tree
  for f in "$project_dir/solution1/sim/report/verilog"/*.rpt \
           "$project_dir/solution1/sim/report/verilog"/*.log; do
    [ -f "$f" ] || continue
    archive "$f" "$artifact_dir/cosim/sim_report"
  done
fi
while IFS= read -r f; do
  archive "$f" "$artifact_dir/cosim"
done < <(find "$project_dir" -name "cosim_options.xml" -type f 2>/dev/null)

# ---------------------------------------------------------------------------
# /tmp project size + waveform/trace dump audit (proves nothing was copied).
# ---------------------------------------------------------------------------
{
  echo "project_dir=$project_dir"
  echo "project_bytes=$(du -sb "$project_dir" 2>/dev/null | awk '{print $1}')"
  echo "waveform_files_wdb=$(find "$project_dir" -name '*.wdb' -type f 2>/dev/null | wc -l)"
  echo "waveform_files_vcd=$(find "$project_dir" -name '*.vcd' -type f 2>/dev/null | wc -l)"
  echo "waveform_files_saif=$(find "$project_dir" -name '*.saif' -type f 2>/dev/null | wc -l)"
  echo "waveform_files_fsdb=$(find "$project_dir" -name '*.fsdb' -type f 2>/dev/null | wc -l)"
} > "$artifact_dir/cosim/tmp_project_audit.txt"

echo "[run_cosim] archived compact summaries -> $artifact_dir/cosim"
exit "$rc"
