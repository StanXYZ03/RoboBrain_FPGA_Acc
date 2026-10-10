# run_cosim.tcl - bounded Vitis HLS 2023.2 C/RTL co-simulation of the accepted
# compact integer Selector pipeline top ``integer_selector_pipeline_compact``.
#
# Scope: csynth_design + cosim_design ONLY.
#   * RTL language : verilog
#   * simulator    : xsim
#   * trace        : none (no VCD/WDB/SAIF waveform dumping)
#   * NO export_design, NO Vivado implementation, NO bitstream, NO board, NO GPU,
#     NO network.
#
# The HLS project lives under /tmp; only small logs / reports / transaction and
# performance summaries are copied back into the repository by
# scripts/run_cosim.sh.
#
# The dedicated testbench reads the frozen expected arenas / scalars directly
# from the accepted fixture directory (read-only) and has no dlopen.
#
# Reproduce (from anywhere):
#   /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
#     <repo>/fpga_accel/integer_selector_pipeline_compact_cosim/scripts/run_cosim.tcl

set script_dir [file normalize [file dirname [info script]]]
set proj_dir   [file normalize [file join $script_dir ".."]]
set repo_root  [file normalize [file join $proj_dir ".." ".."]]

set src         [file join $repo_root fpga_accel integer_selector_pipeline_compact_hls src integer_selector_pipeline_compact.cpp]
set tb          [file join $proj_dir tb tb_integer_selector_pipeline_compact_cosim.cpp]
set inc_a       [file join $repo_root fpga_accel integer_selector_pipeline_compact_hls include]
set inc_b       [file join $repo_root fpga_accel integer_selector_scorer_hls include]
set inc_c       [file join $repo_root fpga_accel token_selector_hls include]
set inc_d       [file join $repo_root fpga_accel integer_selector_pipeline_hls include]
set fixture_dir [file join $repo_root artifacts token_selector 2026-10-03-integer-selector-compact-interface-hls fixtures]
set case_list   [file join $proj_dir cases rtl_cosim_cases.txt]

# Dedicated /tmp project directory (never shared with the csynth-only project).
set project_dir /tmp/robobrain_integer_selector_compact_cosim
if {[info exists ::env(ISPC_COSIM_PROJECT_DIR)]} {
    set project_dir $::env(ISPC_COSIM_PROJECT_DIR)
}

# Ubuntu 24.04 host workarounds, reused verbatim from the accepted pipeline /
# scorer / qkv-adapter / cosim-v4 runs:
#   * force the system `ld` through COMPILER_PATH (bundled binutils cannot read
#     the modern libm.so.6 .relr.dyn section);
#   * keep the bundled GCC 8.3.0 (system GCC headers break xilinx-tb-process);
#   * point LIBRARY_PATH at the multiarch crt directory;
#   * -no-pie keeps the host link compatible with the HLS wrapper objects.
set bundled_gcc /home/xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
if {[file exists $bundled_gcc/gcc] && [file exists $bundled_gcc/g++]} {
    set ::env(AP_GCC_PATH) $bundled_gcc
}
set sysld_dir /tmp/robobrain_integer_selector_compact_cosim_sysld
file mkdir $sysld_dir
if {![file exists $sysld_dir/ld]} {
    file link -symbolic $sysld_dir/ld /usr/bin/ld
}
set ::env(COMPILER_PATH) $sysld_dir
set ::env(LIBRARY_PATH) "/usr/lib/x86_64-linux-gnu"

# The dedicated testbench resolves its inputs from these environment variables
# (and also accepts them as argv, in case -argv splitting differs).
set ::env(ISPC_COSIM_FIXTURE_DIR) $fixture_dir
set ::env(ISPC_COSIM_CASES) $case_list

set cflags "-std=c++14 -I$inc_a -I$inc_b -I$inc_c -I$inc_d"

puts "INFO: run_cosim.tcl repo_root=$repo_root"
puts "INFO: run_cosim.tcl project_dir=$project_dir"
puts "INFO: run_cosim.tcl fixture_dir=$fixture_dir"
puts "INFO: run_cosim.tcl case_list=$case_list"

open_project -reset $project_dir
set_top integer_selector_pipeline_compact
add_files $src -cflags $cflags
add_files -tb $tb -cflags $cflags
open_solution -reset solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

# Keep the host link compatible with the HLS co-simulation wrapper objects.
config_csim -ldflags "-no-pie"

# 1) C synthesis (also produces the interface report consumed by the audit).
csynth_design

# 2) C/RTL co-simulation: Verilog RTL, XSim, waveform/trace dumping disabled.
cosim_design -rtl verilog -tool xsim -trace_level none \
    -ldflags "-no-pie" \
    -argv "$fixture_dir $case_list"

puts "INFO: run_cosim.tcl csynth+cosim completed"
exit
