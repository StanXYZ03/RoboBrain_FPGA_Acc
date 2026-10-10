# HLS C-simulation project for the Selector -> LayerNorm-Q8_0 adapter and the
# independent QKV bias-add kernel. Only `csim_design` is ever invoked here:
# no csynth, no cosim, no IP export, no Vivado, no board.
#
# The large HLS project lives under /tmp; only small raw logs are copied into
# the repository artifact directory.

set mode "csim_adapter"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {csim_adapter csim_bias}} {
        set mode $requested
    }
}
set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir

set project_dir "/tmp/robobrain_selector_qkv_adapter_hls_prj"
if {[info exists ::env(SQA_HLS_PROJECT_DIR)]} {
    set project_dir $::env(SQA_HLS_PROJECT_DIR)
}

# Host-compiler workaround (Ubuntu 24.04 / glibc 2.39), reused verbatim from the
# existing selector cosim script scripts/run_cosim_v4.tcl:
#   * the binutils 2.37 bundled with Vitis HLS 2023.2 cannot read the
#     `.relr.dyn` section of the system libm.so.6, so the C testbench link
#     fails. Force the system `ld` through COMPILER_PATH.
#   * do NOT switch to system GCC: the HLS clang-3.9 xilinx-tb-process cannot
#     parse its libstdc++ headers. Keep the bundled GCC 8.3.0 for the host build.
#   * the bundled GCC does not know the Ubuntu multiarch crt directory, so point
#     LIBRARY_PATH at it.
#   * -no-pie keeps the host link compatible with the HLS wrapper objects.
# These affect only the host C testbench build; the DUT and the algorithm are
# unchanged, and no csynth/cosim/export is run.
set bundled_gcc /home/xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
if {[file exists $bundled_gcc/gcc] && [file exists $bundled_gcc/g++]} {
    set ::env(AP_GCC_PATH) $bundled_gcc
}
set sysld_dir /tmp/robobrain_selector_qkv_adapter_sysld
file mkdir $sysld_dir
if {![file exists $sysld_dir/ld]} {
    file link -symbolic $sysld_dir/ld /usr/bin/ld
}
set ::env(COMPILER_PATH) $sysld_dir
set ::env(LIBRARY_PATH) "/usr/lib/x86_64-linux-gnu"

open_project -reset $project_dir
if {$mode eq "csim_bias"} {
    set_top qkv_bias_add
} else {
    set_top selector_ln_q8_adapter
}
add_files src/selector_qkv_adapter.cpp -cflags "-std=c++14 -Iinclude"
add_files -tb tb/tb_selector_qkv_adapter.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

config_csim -ldflags "-no-pie"
csim_design
exit
