set root [file normalize [file dirname [file dirname [info script]]]]
set output [file normalize "$root/artifacts/ddr_smoke/bsp"]
set xsa [file normalize "$root/artifacts/ddr_smoke/zcu102_ddr_smoke.xsa"]

hsi open_hw_design $xsa
hsi create_sw_design ddr_smoke_bsp -proc psu_cortexa53_0 -os standalone
hsi generate_bsp -dir $output
puts "DDR_SMOKE_BSP=$output"
hsi close_hw_design [hsi current_hw_design]
exit
