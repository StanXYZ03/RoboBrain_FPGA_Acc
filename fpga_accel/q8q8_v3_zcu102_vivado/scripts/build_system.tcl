set mode "validate"
if {$argc >= 1} {
    set mode [lindex $argv 0]
}
if {$mode ni {validate synth impl bitstream}} {
    error "usage: build_system.tcl ?validate|synth|impl|bitstream?"
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set accel_dir [file normalize [file join $project_dir ..]]
set build_dir [file join $project_dir build]
set artifact_dir [file join $project_dir artifacts]
set ip_zip [file join $accel_dir q8q8_output_burst_hls artifacts q8q8_output_burst_ip.zip]
set ip_repo_dir [file join $build_dir ip_repo]
set vivado_dir [file join $build_dir vivado]

if {![file exists $ip_zip]} {
    error "Missing packaged HLS IP: $ip_zip"
}
file mkdir $build_dir
file mkdir $artifact_dir
file delete -force $ip_repo_dir
file mkdir $ip_repo_dir
exec unzip -q -o $ip_zip -d $ip_repo_dir

create_project -force q8q8_v3_zcu102_system $vivado_dir \
    -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths $ip_repo_dir [current_project]
update_ip_catalog

create_bd_design q8q8_system
set ps [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] $ps

set accel [create_bd_cell -type ip \
    -vlnv robobrain.local:hls:q8q8_output_burst:3.0 q8q8_output_burst_0]
set output_width [get_property CONFIG.DATA_WIDTH \
    [get_bd_intf_pins $accel/m_axi_gmem_y]]
puts "SYSTEM_OUTPUT_AXI_WIDTH=$output_width"
if {$output_width != 128} {
    error "Expected 128-bit output AXI interface, got $output_width"
}

set rst [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:proc_sys_reset:* proc_sys_reset_0]
set clk_wiz [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:clk_wiz:* clk_wiz_200m]
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {99.990005} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.PRIM_SOURCE {No_buffer} \
    CONFIG.USE_RESET {true} \
    CONFIG.RESET_TYPE {ACTIVE_LOW}] $clk_wiz

set ctrl_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* ctrl_smartconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ctrl_sc
set wq_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* wq_smartconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $wq_sc
set xq_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* xq_smartconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $xq_sc
set aux_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* aux_smartconnect]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] $aux_sc

# A53 control path.
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins $ctrl_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_sc/M00_AXI] \
    [get_bd_intf_pins $accel/s_axi_control]

# Independent high-bandwidth Q8 streams.
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_wq] \
    [get_bd_intf_pins $wq_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $wq_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_xq] \
    [get_bd_intf_pins $xq_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $xq_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HPC1_FPD]

# FP16 scales and packed FP32 output share HP0. SmartConnect adapts the
# accelerator's 128-bit output master to the PS-side port as required.
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_ws] \
    [get_bd_intf_pins $aux_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_xs] \
    [get_bd_intf_pins $aux_sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_y] \
    [get_bd_intf_pins $aux_sc/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins $aux_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

# Generate a true 200 MHz accelerator clock from the PS reference.
connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $clk_wiz/clk_in1]
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $clk_wiz/resetn]
set pl_clk [get_bd_pins $clk_wiz/clk_out1]
connect_bd_net $pl_clk \
    [get_bd_pins $ps/maxihpm0_fpd_aclk] \
    [get_bd_pins $ps/saxihpc0_fpd_aclk] \
    [get_bd_pins $ps/saxihpc1_fpd_aclk] \
    [get_bd_pins $ps/saxihp0_fpd_aclk] \
    [get_bd_pins $accel/ap_clk] \
    [get_bd_pins $rst/slowest_sync_clk] \
    [get_bd_pins $ctrl_sc/aclk] \
    [get_bd_pins $wq_sc/aclk] \
    [get_bd_pins $xq_sc/aclk] \
    [get_bd_pins $aux_sc/aclk]

connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
connect_bd_net [get_bd_pins $clk_wiz/locked] [get_bd_pins $rst/dcm_locked]
set pl_resetn [get_bd_pins $rst/peripheral_aresetn]
connect_bd_net $pl_resetn \
    [get_bd_pins $accel/ap_rst_n] \
    [get_bd_pins $ctrl_sc/aresetn] \
    [get_bd_pins $wq_sc/aresetn] \
    [get_bd_pins $xq_sc/aresetn] \
    [get_bd_pins $aux_sc/aresetn]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design
write_bd_tcl -force [file join $artifact_dir q8q8_system_bd.tcl]

set address_log [open [file join $artifact_dir address_map.txt] w]
foreach space [lsort [concat [get_bd_addr_spaces $accel/*] \
        [get_bd_addr_spaces $ps/*]]] {
    puts $address_log "ADDRESS_SPACE=[get_property NAME $space]"
    foreach seg [lsort [get_bd_addr_segs -of_objects $space]] {
        puts $address_log [format "  SEGMENT=%s OFFSET=%s RANGE=%s" \
            [get_property NAME $seg] [get_property OFFSET $seg] \
            [get_property RANGE $seg]]
    }
}
close $address_log

set bd_file [get_files q8q8_system.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top q8q8_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "SYSTEM_VALIDATE=PASS"
puts "SYSTEM_MODE=$mode"
if {$mode eq "validate"} {
    close_project
    exit
}

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "System synthesis did not complete"
}
open_run synth_1
report_utilization -file [file join $artifact_dir synth_utilization.rpt]
report_timing_summary -file [file join $artifact_dir synth_timing_summary.rpt]
close_design
puts "SYSTEM_SYNTH=PASS"
if {$mode eq "synth"} {
    close_project
    exit
}

set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
set routed_dcp [file join [get_property DIRECTORY [get_runs impl_1]] \
    q8q8_system_wrapper_routed.dcp]
if {![file exists $routed_dcp]} {
    error "System implementation did not complete: $impl_status"
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
write_checkpoint -force [file join $artifact_dir q8q8_zcu102_routed.dcp]
puts "SYSTEM_IMPL_WNS_NS=$wns"
puts "SYSTEM_IMPL_WHS_NS=$whs"
puts "SYSTEM_IMPL_UNROUTED_NETS=$unrouted"

if {$unrouted != 0} {
    error "System implementation has $unrouted unrouted nets"
}
if {$wns ne "NA" && $wns < 0.0} {
    error "System implementation misses timing: WNS=$wns ns"
}
if {$whs ne "NA" && $whs < 0.0} {
    error "System implementation misses hold timing: WHS=$whs ns"
}

set bitstream_generated "NO"
if {$mode eq "bitstream"} {
    set bitstream [file join $artifact_dir q8q8_zcu102.bit]
    set xsa [file join $artifact_dir q8q8_zcu102.xsa]
    write_hw_platform -fixed -include_bit -force -file $xsa
    exec unzip -p $xsa [file tail $bitstream] > $bitstream
    if {![file exists $bitstream] || ![file exists $xsa]} {
        error "Bitstream or XSA export did not complete"
    }
    set bitstream_generated "YES"
    puts "SYSTEM_BITSTREAM=$bitstream"
    puts "SYSTEM_XSA=$xsa"
}

set summary [open [file join $artifact_dir implementation_summary.txt] w]
puts $summary "TARGET_MHZ=200"
puts $summary "ACTUAL_MHZ=199.98001"
puts $summary "OUTPUT_AXI_WIDTH=128"
puts $summary "WNS_NS=$wns"
puts $summary "WHS_NS=$whs"
puts $summary "UNROUTED_NETS=$unrouted"
puts $summary "BITSTREAM_GENERATED=$bitstream_generated"
puts $summary "BOARD_PROGRAMMED=NO"
close $summary

puts "SYSTEM_BITSTREAM_GENERATED=$bitstream_generated"
puts "SYSTEM_BOARD_PROGRAMMED=NO"
close_project
exit
