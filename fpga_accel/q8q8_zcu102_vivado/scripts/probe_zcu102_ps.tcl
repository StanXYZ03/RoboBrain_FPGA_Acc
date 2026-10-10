set part_name "xczu9eg-ffvb1156-2-e"
set probe_dir [file normalize "fpga_accel/q8q8_zcu102_vivado/build/probe"]
file mkdir $probe_dir
create_project -force zcu102_ps_probe $probe_dir -part $part_name

set candidates [get_board_parts -quiet "xilinx.com:zcu102:part0:*"]
puts "BOARD_PART_CANDIDATES=$candidates"
if {[llength $candidates] > 0} {
    set selected [lindex $candidates end]
    set_property board_part $selected [current_project]
    puts "BOARD_PART_SELECTED=$selected"
}

create_bd_design ps_probe
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0]
if {[llength $candidates] > 0} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200}] $ps

puts "PS_INTERFACE_PINS_BEGIN"
foreach pin [lsort [get_bd_intf_pins $ps/*]] {
    puts "[get_property NAME $pin] MODE=[get_property MODE $pin] VLNV=[get_property VLNV $pin]"
}
puts "PS_INTERFACE_PINS_END"

puts "PS_CLOCK_RESET_PINS_BEGIN"
foreach pin [lsort [get_bd_pins $ps/* -filter {TYPE == clk || TYPE == rst}]] {
    puts "[get_property NAME $pin] TYPE=[get_property TYPE $pin] DIR=[get_property DIR $pin]"
}
puts "PS_CLOCK_RESET_PINS_END"

puts "PS_RELEVANT_PROPERTIES_BEGIN"
foreach prop [lsort [list_property $ps]] {
    if {[regexp {PSU__USE__M_AXI|PSU__USE__S_AXI|PL0_REF_CTRL__FREQMHZ} $prop]} {
        puts "$prop=[get_property $prop $ps]"
    }
}
puts "PS_RELEVANT_PROPERTIES_END"

close_project
exit
