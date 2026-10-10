set script_dir [file dirname [file normalize [info script]]]
set smoke_dir [file dirname $script_dir]
set artifact_dir [file normalize "$smoke_dir/artifacts/jtag"]
set project_dir "$artifact_dir/project"

file mkdir $artifact_dir
create_project -force pl_jtag_smoke $project_dir -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property target_language Verilog [current_project]

create_ip -name vio -vendor xilinx.com -library ip -module_name vio_0
set_property -dict [list \
    CONFIG.C_NUM_PROBE_IN {2} \
    CONFIG.C_NUM_PROBE_OUT {1} \
    CONFIG.C_PROBE_IN0_WIDTH {32} \
    CONFIG.C_PROBE_IN1_WIDTH {32} \
    CONFIG.C_PROBE_OUT0_WIDTH {32} \
    CONFIG.C_PROBE_OUT0_INIT_VAL {0x00000000}] [get_ips vio_0]
generate_target all [get_ips vio_0]

add_files -norecurse [list \
    "$smoke_dir/rtl/pl_jtag_smoke_core.sv" \
    "$smoke_dir/rtl/pl_jtag_smoke_top.sv"]
add_files -fileset constrs_1 -norecurse \
    "$smoke_dir/constraints/jtag_smoke_internal_clock.xdc"
set_property top pl_jtag_smoke_top [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "PL_JTAG_BUILD_SYNTH_STATUS=$synth_status"
if {![string match "*Complete*" $synth_status]} {
    error "PL JTAG synthesis failed: $synth_status"
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "PL_JTAG_BUILD_IMPL_STATUS=$impl_status"
if {![string match "*Complete*" $impl_status]} {
    error "PL JTAG implementation failed: $impl_status"
}

open_run impl_1
report_utilization -file "$artifact_dir/impl_utilization.rpt"
report_timing_summary -delay_type max -report_unconstrained -file "$artifact_dir/impl_timing_summary.rpt"
report_drc -file "$artifact_dir/impl_drc.rpt"
close_design

set bitstreams [glob -nocomplain "$project_dir/pl_jtag_smoke.runs/impl_1/*.bit"]
set probes [glob -nocomplain \
    "$project_dir/pl_jtag_smoke.runs/impl_1/pl_jtag_smoke_top.ltx"]
if {[llength $bitstreams] != 1 || [llength $probes] != 1} {
    error "Expected one bitstream and one probes file; bit=[llength $bitstreams] ltx=[llength $probes]"
}

file copy -force [lindex $bitstreams 0] "$artifact_dir/pl_jtag_smoke.bit"
file copy -force [lindex $probes 0] "$artifact_dir/pl_jtag_smoke.ltx"
puts "PL_JTAG_BUILD_BITSTREAM=$artifact_dir/pl_jtag_smoke.bit"
puts "PL_JTAG_BUILD_PROBES=$artifact_dir/pl_jtag_smoke.ltx"
puts "PL_JTAG_BUILD_RESULT=PASS"
exit 0
