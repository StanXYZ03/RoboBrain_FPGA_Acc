set root [file normalize [file dirname [file dirname [info script]]]]
set output [file normalize "$root/artifacts/ddr_smoke/official_dram_test"]
set xsa [file normalize "$root/artifacts/ddr_smoke/zcu102_ddr_smoke.xsa"]

hsi open_hw_design $xsa
hsi create_sw_design ddr_smoke_bsp -proc psu_cortexa53_0 -os standalone
hsi generate_app -app zynqmp_dram_test -dir $output -no_bsp
puts "DDR_OFFICIAL_TEST=$output"
hsi close_hw_design [hsi current_hw_design]
exit
