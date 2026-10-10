# Vitis HLS C-simulation for the compact integer Selector pipeline.
#
# ONLY `csim_design` is invoked: no csynth, no cosim, no IP export, no Vivado,
# no bitstream, no board access.  The host testbench additionally dlopen()s a
# shared object built from the accepted, unmodified `integer_selector_pipeline`
# so the compact arena can be compared byte-for-byte with the accepted top.
#
# The testbench reads its fixture directory from ISPC_FIXTURE_DIR and the
# reference .so from ISPC_REF_SO (both set by scripts/run_stage.sh).
#
# Ubuntu 24.04 host workaround, reused verbatim from the accepted pipeline /
# scorer / selector_qkv_adapter runs:
#   * force the system `ld` through COMPILER_PATH (bundled binutils cannot read
#     the modern libm.so.6 .relr.dyn section);
#   * keep the bundled GCC 8.3.0 (system GCC headers break xilinx-tb-process);
#   * point LIBRARY_PATH at the multiarch crt directory;
#   * -no-pie keeps the host link compatible with the HLS wrapper objects;
#   * -ldl for the reference shared object.

set script_dir [file normalize [file dirname [info script]]]
set root_dir   [file normalize [file join $script_dir ".."]]
cd $root_dir

set project_dir "/tmp/robobrain_integer_selector_compact_hls_csim"
if {[info exists ::env(ISPC_HLS_PROJECT_DIR)]} {
    set project_dir $::env(ISPC_HLS_PROJECT_DIR)
}

set bundled_gcc /home/xilinx/Vitis_HLS/2023.2/tps/lnx64/gcc-8.3.0/bin
if {[file exists $bundled_gcc/gcc] && [file exists $bundled_gcc/g++]} {
    set ::env(AP_GCC_PATH) $bundled_gcc
}
set sysld_dir /tmp/robobrain_integer_selector_compact_sysld
file mkdir $sysld_dir
if {![file exists $sysld_dir/ld]} {
    file link -symbolic $sysld_dir/ld /usr/bin/ld
}
set ::env(COMPILER_PATH) $sysld_dir
set ::env(LIBRARY_PATH) "/usr/lib/x86_64-linux-gnu"

set cflags "-std=c++14 -Iinclude -I../integer_selector_scorer_hls/include -I../token_selector_hls/include -I../integer_selector_pipeline_hls/include"

open_project -reset $project_dir
set_top integer_selector_pipeline_compact
add_files src/integer_selector_pipeline_compact.cpp -cflags $cflags
add_files -tb tb/tb_integer_selector_pipeline_compact.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

config_csim -ldflags "-no-pie -ldl"
csim_design
exit
