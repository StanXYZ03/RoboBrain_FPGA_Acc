set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set artifact_dir [file join $project_dir artifacts]
set xpr [file join $project_dir build vivado q8q8_zcu102_system.xpr]

if {![file exists $xpr]} {
    error "Missing implemented Vivado project: $xpr"
}
file mkdir $artifact_dir
open_project $xpr
set impl_status [get_property STATUS [get_runs impl_1]]
puts "SYSTEM_IMPL_STATUS=$impl_status"
set routed_dcp [file join [get_property DIRECTORY [get_runs impl_1]] \
    q8q8_system_wrapper_routed.dcp]
if {![file exists $routed_dcp]} {
    error "Implementation is not complete: $impl_status"
}
open_checkpoint $routed_dcp

report_utilization -hierarchical -file [file join $artifact_dir impl_utilization.rpt]
report_timing_summary -max_paths 20 \
    -file [file join $artifact_dir impl_timing_summary.rpt]
report_route_status -file [file join $artifact_dir impl_route_status.rpt]
report_drc -file [file join $artifact_dir impl_drc.rpt]
report_methodology -file [file join $artifact_dir impl_methodology.rpt]
report_power -file [file join $artifact_dir impl_power.rpt]

set setup_paths [get_timing_paths -quiet -setup -max_paths 1]
set hold_paths [get_timing_paths -quiet -hold -max_paths 1]
set wns "NA"
set whs "NA"
if {[llength $setup_paths] > 0} {
    set wns [get_property SLACK [lindex $setup_paths 0]]
}
if {[llength $hold_paths] > 0} {
    set whs [get_property SLACK [lindex $hold_paths 0]]
}
set unrouted [llength [get_nets -quiet -hierarchical \
    -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]

set summary [open [file join $artifact_dir implementation_summary.txt] w]
puts $summary "TARGET_MHZ=200"
puts $summary "ACTUAL_MHZ=199.98001"
puts $summary "WNS_NS=$wns"
puts $summary "WHS_NS=$whs"
puts $summary "UNROUTED_NETS=$unrouted"
puts $summary "BITSTREAM_GENERATED=NO"
puts $summary "BOARD_PROGRAMMED=NO"
close $summary

write_checkpoint -force [file join $artifact_dir q8q8_zcu102_routed.dcp]
puts "SYSTEM_IMPL_WNS_NS=$wns"
puts "SYSTEM_IMPL_WHS_NS=$whs"
puts "SYSTEM_IMPL_UNROUTED_NETS=$unrouted"
puts "SYSTEM_BITSTREAM_GENERATED=NO"
puts "SYSTEM_BOARD_PROGRAMMED=NO"
close_project
exit
