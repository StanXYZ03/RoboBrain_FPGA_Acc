set script_dir [file dirname [file normalize [info script]]]
set smoke_dir [file dirname $script_dir]
set artifact_dir [file normalize "$smoke_dir/artifacts"]
set project_dir "$artifact_dir/project"

file mkdir $artifact_dir
create_project -force pl_uart_smoke $project_dir -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property target_language Verilog [current_project]

add_files -norecurse [list \
    "$smoke_dir/rtl/pl_uart_smoke_core.sv" \
    "$smoke_dir/rtl/pl_uart_smoke_top.sv"]
add_files -fileset constrs_1 -norecurse "$smoke_dir/constraints/zcu102_pl_uart_smoke.xdc"
set_property top pl_uart_smoke_top [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "PL_BUILD_SYNTH_STATUS=$synth_status"
if {![string match "*Complete*" $synth_status]} {
    error "PL synthesis failed: $synth_status"
}

open_run synth_1
report_utilization -file "$artifact_dir/synth_utilization.rpt"
close_design

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "PL_BUILD_IMPL_STATUS=$impl_status"
if {![string match "*Complete*" $impl_status]} {
    error "PL implementation failed: $impl_status"
}

open_run impl_1
report_utilization -file "$artifact_dir/impl_utilization.rpt"
report_timing_summary -delay_type max -report_unconstrained -file "$artifact_dir/impl_timing_summary.rpt"
report_drc -file "$artifact_dir/impl_drc.rpt"
close_design

set bitstreams [glob -nocomplain "$project_dir/pl_uart_smoke.runs/impl_1/*.bit"]
if {[llength $bitstreams] != 1} {
    error "Expected exactly one bitstream, found [llength $bitstreams]"
}

file copy -force [lindex $bitstreams 0] "$artifact_dir/pl_uart_smoke.bit"
puts "PL_BUILD_BITSTREAM=$artifact_dir/pl_uart_smoke.bit"
puts "PL_BUILD_RESULT=PASS"
exit 0
