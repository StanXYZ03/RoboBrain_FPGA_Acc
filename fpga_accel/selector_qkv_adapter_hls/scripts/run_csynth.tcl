# Controlled Vitis HLS C-synthesis for the Selector->LayerNorm-Q8_0 adapter and
# the independent QKV bias-add kernel.
#
# ONLY `csynth_design` is invoked: no C/RTL cosimulation, no IP export, no
# Vivado, no bitstream, no board access.  part xczu9eg-ffvb1156-2-e, 5.0 ns
# clock and the tool default uncertainty (not overridden / not relaxed) are
# kept.  The large HLS project stays under /tmp; only small reports are copied
# back into the repository by scripts/run_csynth.sh.
#
# Usage:
#   vitis_hls -f scripts/run_csynth.tcl adapter
#   vitis_hls -f scripts/run_csynth.tcl bias

set mode "adapter"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {adapter bias}} {
        set mode $requested
    }
}
set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set src_file [file join $root_dir src selector_qkv_adapter.cpp]
set inc_dir  [file join $root_dir include]

set project_dir "/tmp/robobrain_selector_qkv_adapter_csynth_$mode"
if {[info exists ::env(SQA_HLS_CSYNTH_PROJECT_DIR)]} {
    set project_dir $::env(SQA_HLS_CSYNTH_PROJECT_DIR)
}

puts "INFO: run_csynth.tcl mode=$mode project_dir=$project_dir"

open_project -reset $project_dir
if {$mode eq "bias"} {
    set_top qkv_bias_add
} else {
    set_top selector_ln_q8_adapter
}
add_files $src_file -cflags "-std=c++14 -I$inc_dir"
open_solution -reset solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

csynth_design

puts "INFO: run_csynth.tcl csynth_design completed for $mode"
exit
