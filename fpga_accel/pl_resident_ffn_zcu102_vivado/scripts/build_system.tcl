set mode "validate"
if {$argc >= 1} {
    set mode [lindex $argv 0]
}
if {$mode ni {validate synth impl}} {
    error "usage: build_system.tcl ?validate|synth|impl? ?artifact_dir?"
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set accel_dir [file normalize [file join $project_dir ..]]
set build_dir [file join $project_dir build]
if {$argc >= 2} {
    set artifact_dir [file normalize [lindex $argv 1]]
} else {
    set artifact_dir [file join $project_dir artifacts manual_$mode]
}
set gemm_zip [file join $accel_dir q8q8_pe16_dataflow_hls artifacts \
    q8q8_pe16_dataflow_ip.zip]
set vector_zip [file join $accel_dir pl_resident_ffn_hls artifacts \
    pl_resident_ffn_vector_ip.zip]
set ip_repo_dir [file join $build_dir ip_repo]
set gemm_repo [file join $ip_repo_dir gemm]
set vector_repo [file join $ip_repo_dir vector]
set vivado_dir [file join $build_dir vivado]

foreach required [list $gemm_zip $vector_zip] {
    if {![file exists $required]} {
        error "Missing packaged HLS IP: $required"
    }
}
file mkdir $build_dir
file mkdir $artifact_dir
file delete -force $ip_repo_dir
file mkdir $gemm_repo
file mkdir $vector_repo
exec unzip -q -o $gemm_zip -d $gemm_repo
exec unzip -q -o $vector_zip -d $vector_repo

create_project -force pl_resident_ffn_zcu102_system $vivado_dir \
    -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $gemm_repo $vector_repo] [current_project]
update_ip_catalog

create_bd_design pl_resident_system
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

set gemm [create_bd_cell -type ip \
    -vlnv robobrain.local:hls:q8q8_pe16_dataflow:1.0 \
    q8q8_pe16_dataflow_0]
set vector [create_bd_cell -type ip \
    -vlnv robobrain.local:hls:pl_resident_ffn_vector:1.0 \
    pl_resident_ffn_vector_0]

set gemm_output_width [get_property CONFIG.DATA_WIDTH \
    [get_bd_intf_pins $gemm/m_axi_gmem_y]]
set vector_output_width [get_property CONFIG.DATA_WIDTH \
    [get_bd_intf_pins $vector/m_axi_gmem2]]
puts "GEMM_OUTPUT_AXI_WIDTH=$gemm_output_width"
puts "VECTOR_OUTPUT_AXI_WIDTH=$vector_output_width"
if {$gemm_output_width != 128} {
    error "Expected 128-bit GEMM output AXI, got $gemm_output_width"
}
if {$vector_output_width != 256} {
    error "Expected 256-bit vector output AXI, got $vector_output_width"
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
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $ctrl_sc
set hpc0_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* hpc0_smartconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $hpc0_sc
set hpc1_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* hpc1_smartconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $hpc1_sc
set hp0_sc [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:* hp0_smartconnect]
set_property -dict [list CONFIG.NUM_SI {4} CONFIG.NUM_MI {1}] $hp0_sc

# PS control master reaches two independent AXI-Lite slaves.
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins $ctrl_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_sc/M00_AXI] \
    [get_bd_intf_pins $gemm/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $ctrl_sc/M01_AXI] \
    [get_bd_intf_pins $vector/s_axi_control]

# HPC0 arbitrates GEMM weights and vector source0/parameter preload.
connect_bd_intf_net [get_bd_intf_pins $gemm/m_axi_gmem_wq] \
    [get_bd_intf_pins $hpc0_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $vector/m_axi_gmem0] \
    [get_bd_intf_pins $hpc0_sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $hpc0_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]

# HPC1 arbitrates GEMM activations and vector source1.
connect_bd_intf_net [get_bd_intf_pins $gemm/m_axi_gmem_xq] \
    [get_bd_intf_pins $hpc1_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $vector/m_axi_gmem1] \
    [get_bd_intf_pins $hpc1_sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $hpc1_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HPC1_FPD]

# HP0 carries GEMM scales/output and all vector results.
connect_bd_intf_net [get_bd_intf_pins $gemm/m_axi_gmem_ws] \
    [get_bd_intf_pins $hp0_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $gemm/m_axi_gmem_xs] \
    [get_bd_intf_pins $hp0_sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $gemm/m_axi_gmem_y] \
    [get_bd_intf_pins $hp0_sc/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins $vector/m_axi_gmem2] \
    [get_bd_intf_pins $hp0_sc/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins $hp0_sc/M00_AXI] \
    [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $clk_wiz/clk_in1]
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $clk_wiz/resetn]
set pl_clk [get_bd_pins $clk_wiz/clk_out1]
connect_bd_net $pl_clk \
    [get_bd_pins $ps/maxihpm0_fpd_aclk] \
    [get_bd_pins $ps/saxihpc0_fpd_aclk] \
    [get_bd_pins $ps/saxihpc1_fpd_aclk] \
    [get_bd_pins $ps/saxihp0_fpd_aclk] \
    [get_bd_pins $gemm/ap_clk] \
    [get_bd_pins $vector/ap_clk] \
    [get_bd_pins $rst/slowest_sync_clk] \
    [get_bd_pins $ctrl_sc/aclk] \
    [get_bd_pins $hpc0_sc/aclk] \
    [get_bd_pins $hpc1_sc/aclk] \
    [get_bd_pins $hp0_sc/aclk]

connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
connect_bd_net [get_bd_pins $clk_wiz/locked] [get_bd_pins $rst/dcm_locked]
set pl_resetn [get_bd_pins $rst/peripheral_aresetn]
connect_bd_net $pl_resetn \
    [get_bd_pins $gemm/ap_rst_n] \
    [get_bd_pins $vector/ap_rst_n] \
    [get_bd_pins $ctrl_sc/aresetn] \
    [get_bd_pins $hpc0_sc/aresetn] \
    [get_bd_pins $hpc1_sc/aresetn] \
    [get_bd_pins $hp0_sc/aresetn]

assign_bd_address
set ps_data_space [get_bd_addr_spaces $ps/Data]
assign_bd_address -force -target_address_space $ps_data_space \
    -offset 0xA0000000 -range 0x00010000 \
    [get_bd_addr_segs $gemm/s_axi_control/Reg]
assign_bd_address -force -target_address_space $ps_data_space \
    -offset 0xA0010000 -range 0x00010000 \
    [get_bd_addr_segs $vector/s_axi_control/Reg]
set gemm_control_mapping [get_bd_addr_segs \
    $ps/Data/SEG_q8q8_pe16_dataflow_0_Reg]
set vector_control_mapping [get_bd_addr_segs \
    $ps/Data/SEG_pl_resident_ffn_vector_0_Reg]
set gemm_control_offset [get_property OFFSET $gemm_control_mapping]
set vector_control_offset [get_property OFFSET $vector_control_mapping]
puts "GEMM_CONTROL_OFFSET=$gemm_control_offset"
puts "VECTOR_CONTROL_OFFSET=$vector_control_offset"
if {$gemm_control_offset != "0x00A0000000"} {
    error "GEMM control ABI moved: $gemm_control_offset"
}
if {$vector_control_offset != "0x00A0010000"} {
    error "Vector control address mismatch: $vector_control_offset"
}
regenerate_bd_layout
validate_bd_design
save_bd_design
write_bd_tcl -force [file join $artifact_dir pl_resident_system_bd.tcl]

set address_log [open [file join $artifact_dir address_map.txt] w]
foreach space [lsort [concat [get_bd_addr_spaces $gemm/*] \
        [get_bd_addr_spaces $vector/*] [get_bd_addr_spaces $ps/*]]] {
    puts $address_log "ADDRESS_SPACE=[get_property NAME $space]"
    foreach seg [lsort [get_bd_addr_segs -of_objects $space]] {
        puts $address_log [format "  SEGMENT=%s OFFSET=%s RANGE=%s" \
            [get_property NAME $seg] [get_property OFFSET $seg] \
            [get_property RANGE $seg]]
    }
}
close $address_log

set bd_file [get_files pl_resident_system.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top pl_resident_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "SYSTEM_VALIDATE=PASS"
puts "SYSTEM_MODE=$mode"
puts "SYSTEM_ARTIFACT_DIR=$artifact_dir"
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
    pl_resident_system_wrapper_routed.dcp]
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
set dcp [file join $artifact_dir pl_resident_ffn_zcu102_routed.dcp]
write_checkpoint -force $dcp
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

set summary [open [file join $artifact_dir implementation_summary.txt] w]
puts $summary "TARGET_MHZ=200"
puts $summary "WNS_NS=$wns"
puts $summary "WHS_NS=$whs"
puts $summary "UNROUTED_NETS=$unrouted"
puts $summary "BITSTREAM_GENERATED=NO"
puts $summary "BOARD_ACCESSED=NO"
puts $summary "BOARD_PROGRAMMED=NO"
puts $summary "SD_MODIFIED=NO"
close $summary

puts "SYSTEM_BITSTREAM_GENERATED=NO"
puts "SYSTEM_BOARD_ACCESSED=NO"
puts "SYSTEM_BOARD_PROGRAMMED=NO"
puts "SYSTEM_SD_MODIFIED=NO"
close_project
exit
