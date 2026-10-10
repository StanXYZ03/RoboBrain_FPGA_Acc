connect -url tcp:127.0.0.1:3121
after 500

puts "PL_POWER_TARGETS_BEGIN"
puts [targets]
puts "PL_POWER_TARGETS_END"

targets -set -filter {name =~ "PSU"}

set pwr_state [mrd -force -value 0xFFD80100]
set jtag_sec [mrd -force -value 0xFFCA0038]
set pl_up [expr {($pwr_state >> 23) & 1}]

puts [format "PL_POWER_PWR_STATE=0x%08X" $pwr_state]
puts "PL_POWER_PLPD_BIT23=$pl_up"
puts [format "PL_POWER_JTAG_SEC=0x%08X" $jtag_sec]

disconnect
exit [expr {$pl_up ? 0 : 2}]
