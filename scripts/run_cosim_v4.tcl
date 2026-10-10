# run_cosim_v4.tcl - controlled Vitis HLS C simulation + C/RTL co-simulation for
# the packed v4 token selector (`token_selector_top_packed`).
#
# Scope: ONLY csim + csynth + cosim for the 3 dedicated representative cases.
# No IP export, no bitstream, no board access.  The RTL language is Verilog and
# the co-simulation trace level is at least `port`.
#
# IMPORTANT: this project directory is DIFFERENT from the csynth-only v4 project
# (`/tmp/robobrain_token_selector_hls_csynth_v4`) so cosim and csynth never
# share a /tmp project/solution directory.
#
# The co-simulation testbench is NOT the 184-fixture g++ differential bench; it
# replays exactly three frozen fixtures generated from the Python reference by
# tools/generate_cosim_v4_fixtures.py:
#   1. cosim_identity_4x4_K48
#   2. cosim_16x16_K64_alltie          (context + all scores tied)
#   3. cosim_32x32_K96_mixedextreme    (context + saturated token/score extremes)
#
# Reproduce:
#   mkdir -p /tmp/robobrain_selector_cosim_run_v4
#   cd /tmp/robobrain_selector_cosim_run_v4
#   export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
#   timeout 5400 /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
#       /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_cosim_v4.tcl

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ..]]
set src_dir    [file join $repo_root fpga_accel token_selector_hls]
set src_file   [file join $src_dir src token_selector_hls.cpp]
set tb_file    [file join $src_dir tb tb_token_selector_cosim_v4.cpp]
set inc_dir    [file join $src_dir include]
set generator  [file join $src_dir tools generate_cosim_v4_fixtures.py]

# Dedicated /tmp project directory (not shared with the csynth-only v4 project).
set project_dir   /tmp/robobrain_token_selector_hls_cosim_v4
set solution_name solution1

# Frozen 3-case co-simulation fixtures (generated from the Python golden).
set fixture_dir   /tmp/robobrain_token_selector_cosim_fixtures_v4
set python_bin    /home/zhaoxiangyu/conda-envs/robobrain25/bin/python

# Host-compiler workaround (Ubuntu 24.04 / glibc 2.39):
#   * the binutils 2.37 bundled with Vitis HLS 2023.2 cannot read the
#     `.relr.dyn` section of the system libm.so.6, so the C testbench link
#     fails.  Force the system `ld` through COMPILER_PATH.
#   * do NOT switch to the system GCC: its libstdc++ headers cannot be parsed by
#     the HLS clang-3.9 `xilinx-tb-process` instrumentation, so the
#     top-function call would not be rewritten and co-simulation would abort
#     with `COSIM 212-330`.  Keep the bundled GCC 8.3.0 for the host build.
#   * the bundled GCC does not know the Ubuntu multiarch crt directory, so point
#     LIBRARY_PATH at it.
# These only affect the host C testbench build; HLS synthesis, the RTL and the
# DUT are unchanged.
set bundled_gcc /home/xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
if {[file exists $bundled_gcc/gcc] && [file exists $bundled_gcc/g++]} {
  set ::env(AP_GCC_PATH) $bundled_gcc
}
set sysld_dir /tmp/robobrain_selector_cosim_sysld_v4
file mkdir $sysld_dir
if {![file exists $sysld_dir/ld]} {
  file link -symbolic $sysld_dir/ld /usr/bin/ld
}
set ::env(COMPILER_PATH) $sysld_dir
set ::env(LIBRARY_PATH) "/usr/lib/x86_64-linux-gnu"

puts "INFO: run_cosim_v4.tcl repo_root=$repo_root"
puts "INFO: run_cosim_v4.tcl project_dir=$project_dir"
puts "INFO: run_cosim_v4.tcl fixture_dir=$fixture_dir"

file mkdir $fixture_dir
if {[catch {exec $python_bin $generator --out-dir $fixture_dir} fixture_log]} {
  puts "ERROR: cosim fixture generation failed: $fixture_log"
  exit 1
}
puts "INFO: run_cosim_v4.tcl fixtures: $fixture_log"

open_project -reset $project_dir
set_top token_selector_top_packed
add_files -cflags "-I$inc_dir" $src_file
add_files -tb -cflags "-I$inc_dir" $tb_file
open_solution -reset $solution_name
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

# Keep the host link compatible with the HLS co-simulation wrapper objects.
config_csim -ldflags "-no-pie"

# 1) C simulation of the 3 frozen cases.
csim_design -argv "$fixture_dir"

# 2) C synthesis (needed for C/RTL co-simulation; independent of the
#    csynth-only v4 project directory).
csynth_design

# 3) C/RTL co-simulation: Verilog RTL, port-level trace, same 3 cases.
#    -no-pie keeps the host link compatible with the HLS co-simulation wrapper
#    objects.  Intentional timeout around the whole tool invocation is applied
#    by the caller (see the header comment).
cosim_design -rtl verilog -trace_level port \
    -ldflags "-no-pie" \
    -argv "$fixture_dir"

puts "INFO: run_cosim_v4.tcl csim+csynth+cosim completed"
exit
