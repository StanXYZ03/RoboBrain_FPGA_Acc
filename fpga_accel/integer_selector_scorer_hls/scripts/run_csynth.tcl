# Controlled Vitis HLS C-synthesis for the standalone integer Selector scorer.
#
# ONLY `csynth_design` is invoked: no C/RTL co-simulation, no IP export, no
# Vivado, no bitstream, no board access.  part xczu9eg-ffvb1156-2-e, 5.0 ns
# clock and the tool default uncertainty (never overridden / never relaxed) are
# kept.  The large HLS project stays under /tmp; only small reports are copied
# back into the repository by scripts/run_stage.sh.

set script_dir [file normalize [file dirname [info script]]]
set root_dir   [file normalize [file join $script_dir ".."]]
cd $root_dir

set project_dir "/tmp/robobrain_integer_selector_scorer_hls_csynth"
if {[info exists ::env(IS_HLS_PROJECT_DIR)]} {
    set project_dir $::env(IS_HLS_PROJECT_DIR)
}

open_project -reset $project_dir
set_top integer_selector_scorer
add_files src/integer_selector_scorer.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

csynth_design

puts "INFO: run_csynth.tcl csynth_design completed"
exit
