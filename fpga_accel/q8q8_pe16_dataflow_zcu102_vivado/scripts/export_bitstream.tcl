if {$argc != 1} {
    error "usage: export_bitstream.tcl artifact_dir"
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set artifact_dir [file normalize [lindex $argv 0]]
set xpr [file join $project_dir build vivado \
    q8q8_pe16_dataflow_zcu102_system.xpr]
set routed_dcp [file join $project_dir build vivado \
    q8q8_pe16_dataflow_zcu102_system.runs impl_1 \
    q8q8_system_wrapper_routed.dcp]

if {![file readable $xpr] || ![file readable $routed_dcp]} {
    error "Passing routed project is missing; run ./scripts/run_stage.sh impl first"
}
file mkdir $artifact_dir
open_project $xpr
open_checkpoint $routed_dcp

set setup_paths [get_timing_paths -quiet -setup -max_paths 1]
set hold_paths [get_timing_paths -quiet -hold -max_paths 1]
if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} {
    error "Routed timing paths are missing"
}
set wns [get_property SLACK [lindex $setup_paths 0]]
set whs [get_property SLACK [lindex $hold_paths 0]]
set unrouted [llength [get_nets -quiet -hierarchical \
    -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]
if {$wns < 0.0 || $whs < 0.0 || $unrouted != 0} {
    error "Routed gate failed: WNS=$wns WHS=$whs unrouted=$unrouted"
}

set xsa [file join $artifact_dir q8q8_pe16_dataflow_zcu102.xsa]
set bit [file join $artifact_dir q8q8_pe16_dataflow_zcu102.bit]
write_hw_platform -fixed -include_bit -force -file $xsa
exec unzip -p $xsa [file tail $bit] > $bit
if {![file exists $xsa] || ![file exists $bit]} {
    error "Bitstream/XSA export did not complete"
}

set summary [open [file join $artifact_dir bitstream_summary.txt] w]
puts $summary "WNS_NS=$wns"
puts $summary "WHS_NS=$whs"
puts $summary "UNROUTED_NETS=$unrouted"
puts $summary "BITSTREAM_GENERATED=YES"
puts $summary "BOARD_ACCESSED=NO"
puts $summary "BOARD_PROGRAMMED=NO"
close $summary

puts "DATAFLOW_BITSTREAM=$bit"
puts "DATAFLOW_XSA=$xsa"
puts "DATAFLOW_BITSTREAM_GENERATED=YES"
puts "DATAFLOW_BOARD_ACCESSED=NO"
puts "DATAFLOW_BOARD_PROGRAMMED=NO"
close_project
exit
