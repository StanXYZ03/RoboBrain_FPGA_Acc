open_hw_manager
connect_hw_server -url localhost:3121
open_hw_target
set device [lindex [get_hw_devices -quiet xczu9_0] 0]
current_hw_device $device
set probes_file [file normalize [file join [file dirname [file normalize [info script]]] .. artifacts jtag pl_jtag_smoke.ltx]]
set_property PROBES.FILE $probes_file $device
set_property FULL_PROBES.FILE $probes_file $device
refresh_hw_device $device
puts "VIO_LIST=[get_hw_vios -quiet]"
puts "PROBE_LIST=[get_hw_probes -quiet]"
foreach vio [get_hw_vios -quiet] {
    puts "VIO_OBJECT=$vio"
    puts [report_property -return_string $vio]
}
foreach probe [get_hw_probes -quiet] {
    puts "PROBE_OBJECT=$probe"
    puts [report_property -return_string $probe]
}
close_hw_target
disconnect_hw_server
close_hw_manager
exit 0
