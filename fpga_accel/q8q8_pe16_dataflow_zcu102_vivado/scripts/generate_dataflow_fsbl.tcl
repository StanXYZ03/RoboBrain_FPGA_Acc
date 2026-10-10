set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set xsa [file join $project_dir artifacts runs 20260829T141332Z_bitstream q8q8_pe16_dataflow_zcu102.xsa]
set output [file join $project_dir build q8q8_pe16_dataflow_fsbl]

if {![file readable $xsa]} {
    error "Missing PE16 DATAFLOW accelerator XSA: $xsa"
}
file delete -force $output
file mkdir [file dirname $output]
hsi open_hw_design $xsa
hsi create_sw_design q8q8_pe16_dataflow_fsbl_bsp -proc psu_cortexa53_0 -os standalone
hsi add_library xilffs
hsi add_library xilsecure
hsi add_library xilpm
hsi generate_app -app zynqmp_fsbl -dir $output
hsi close_hw_design [hsi current_hw_design]
puts "Q8Q8_PE16_DATAFLOW_FSBL_SOURCE=$output"
exit
