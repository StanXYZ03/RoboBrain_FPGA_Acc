# run_hls_export.tcl - Vitis HLS 2023.2 csynth_design + IP-catalog export for the
# accepted compact integer Selector top `integer_selector_pipeline_compact`.
#
# This file only *runs the tools*; it never edits accepted source.  The single
# source file read is the accepted
#   fpga_accel/integer_selector_pipeline_compact_hls/src/integer_selector_pipeline_compact.cpp
# which read-only #includes the two accepted upstream translation units.
#
# Frozen assumptions (identical to the accepted compact-interface csynth):
#   * top          : integer_selector_pipeline_compact
#   * part         : xczu9eg-ffvb1156-2-e
#   * clock        : create_clock -period 5.0 -name default
#   * uncertainty  : Vitis HLS default (never passed on the command line)
#   * flow_target  : vivado, exactly like the accepted compact-interface run
#
# The exported component is packaged as an IP catalog entry (-format ip_catalog)
# with -flow syn, i.e. IP packaging + the HLS-generated OOC synthesis.  The
# explicit Vivado out-of-context synthesis + opt/place/phys_opt/route for this
# task is run separately by run_vivado_ooc.tcl on the exported RTL and on the
# HLS-generated `constraints/<top>_ooc.xdc` (ap_clk = 5.000 ns).
#
# Everything large stays under /tmp.  Only small logs/reports are copied back by
# scripts/run_ooc.sh.

set script_dir [file normalize [file dirname [info script]]]
set ooc_dir    [file normalize [file join $script_dir ".."]]
set root_dir   [file normalize [file join $ooc_dir ".." ".."]]
set src_dir    [file join $root_dir fpga_accel integer_selector_pipeline_compact_hls]

set project_dir "/tmp/robobrain_integer_selector_compact_ooc_hls"
if {[info exists ::env(ISPC_PROJECT_DIR)]} {
    set project_dir $::env(ISPC_PROJECT_DIR)
}

set ip_repo_dir "/tmp/robobrain_integer_selector_compact_ooc_ip"
if {[info exists ::env(ISPC_IP_REPO_DIR)]} {
    set ip_repo_dir $::env(ISPC_IP_REPO_DIR)
}

set cflags "-std=c++14 -I$src_dir/include -I$root_dir/fpga_accel/integer_selector_scorer_hls/include -I$root_dir/fpga_accel/token_selector_hls/include -I$root_dir/fpga_accel/integer_selector_pipeline_hls/include"

file mkdir $ip_repo_dir

puts "INFO: run_hls_export.tcl root_dir=$root_dir"
puts "INFO: run_hls_export.tcl src=$src_dir/src/integer_selector_pipeline_compact.cpp"
puts "INFO: run_hls_export.tcl project_dir=$project_dir"
puts "INFO: run_hls_export.tcl ip_repo_dir=$ip_repo_dir"

open_project -reset $project_dir
set_top integer_selector_pipeline_compact
add_files $src_dir/src/integer_selector_pipeline_compact.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

# 1) C synthesis (same frozen setup as the accepted compact-interface csynth).
csynth_design

# 2) Package the synthesized design as an IP catalog component.  No source or
#    interface change; the 3x128-bit m_axi + 1 AXI-Lite control contract is
#    whatever the accepted source produces.
export_design -rtl verilog -format ip_catalog -flow syn \
    -vendor robobrain.local -library hls \
    -ipname integer_selector_pipeline_compact -version 1.0 \
    -display_name "RoboBrain2.5 integer_selector_pipeline_compact" \
    -description "Compact AXI envelope (3x128-bit m_axi input_arena/wv_q/output_arena + 1 AXI-Lite control) over the accepted fused integer Selector pipeline; source, maths and interface unchanged" \
    -output $ip_repo_dir

puts "INFO: run_hls_export.tcl export_design completed ip_repo_dir=$ip_repo_dir"
exit
