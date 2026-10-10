# Vitis HLS C-simulation for the standalone integer Selector scorer.
#
# ONLY `csim_design` is invoked here: no csynth, no cosim, no IP export, no
# Vivado, no bitstream, no board access.  The HLS project stays under /tmp; the
# raw log is copied back into the repository artifact directory by
# scripts/run_stage.sh.
#
# The testbench reads its fixture directory from the IS_FIXTURE_DIR environment
# variable (set by scripts/run_stage.sh) so no command-line argument is needed.
#
# Ubuntu 24.04 host workaround, reused verbatim from the existing
# selector_qkv_adapter run_hls.tcl:
#   * force the system `ld` through COMPILER_PATH (bundled binutils cannot read
#     the modern libm.so.6 .relr.dyn section);
#   * keep the bundled GCC 8.3.0 (system GCC headers break xilinx-tb-process);
#   * point LIBRARY_PATH at the multiarch crt directory;
#   * -no-pie keeps the host link compatible with the HLS wrapper objects.

set script_dir [file normalize [file dirname [info script]]]
set root_dir   [file normalize [file join $script_dir ".."]]
cd $root_dir

set project_dir "/tmp/robobrain_integer_selector_scorer_hls_csim"
if {[info exists ::env(IS_HLS_PROJECT_DIR)]} {
    set project_dir $::env(IS_HLS_PROJECT_DIR)
}

set bundled_gcc /home/xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
if {[file exists $bundled_gcc/gcc] && [file exists $bundled_gcc/g++]} {
    set ::env(AP_GCC_PATH) $bundled_gcc
}
set sysld_dir /tmp/robobrain_integer_selector_sysld
file mkdir $sysld_dir
if {![file exists $sysld_dir/ld]} {
    file link -symbolic $sysld_dir/ld /usr/bin/ld
}
set ::env(COMPILER_PATH) $sysld_dir
set ::env(LIBRARY_PATH) "/usr/lib/x86_64-linux-gnu"

open_project -reset $project_dir
set_top integer_selector_scorer
add_files src/integer_selector_scorer.cpp -cflags "-std=c++14 -Iinclude"
add_files -tb tb/tb_integer_selector_scorer.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

config_csim -ldflags "-no-pie"
csim_design
exit
