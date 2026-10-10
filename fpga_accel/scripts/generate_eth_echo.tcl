set root [file normalize [file dirname [file dirname [info script]]]]
set output [file normalize "$root/artifacts/eth_test/app"]
set xsa [file normalize "$root/artifacts/ddr_smoke/zcu102_ddr_smoke.xsa"]

file mkdir [file dirname $output]
hsi open_hw_design $xsa
hsi create_sw_design eth_echo_bsp -proc psu_cortexa53_0 -os standalone
hsi add_library lwip213
hsi set_property CONFIG.lwip_dhcp false [hsi get_libs lwip213]
hsi set_property CONFIG.api_mode RAW_API [hsi get_libs lwip213]
hsi generate_app -app lwip_echo_server -dir $output
puts "ETH_ECHO_APP=$output"
hsi close_hw_design [hsi current_hw_design]
exit
