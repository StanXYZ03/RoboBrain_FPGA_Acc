set root [file normalize [file dirname [file dirname [info script]]]]
set output [file normalize "$root/artifacts/ddr16_test/fsbl"]
set xsa [file normalize "$root/artifacts/ddr_smoke/zcu102_ddr_smoke.xsa"]

file mkdir [file dirname $output]
hsi open_hw_design $xsa
hsi create_sw_design ddr16_fsbl_bsp -proc psu_cortexa53_0 -os standalone
hsi add_library xilffs
hsi add_library xilsecure
hsi add_library xilpm
hsi generate_app -app zynqmp_fsbl -dir $output
puts "DDR16_FSBL=$output"
hsi close_hw_design [hsi current_hw_design]
exit
