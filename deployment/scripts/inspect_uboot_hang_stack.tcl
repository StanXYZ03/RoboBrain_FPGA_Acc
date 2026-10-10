# Snapshot the current pre-relocation U-Boot hang stack, then resume.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}
puts "UBOOT_HANG_STACK_BEGIN"
puts [rrd pc]
puts [rrd sp]
puts [rrd r29]
puts [rrd r30]
puts [mrd -force 0x10239D80 32]
puts "UBOOT_INITCALL_TABLE"
puts [mrd -force 0x1015DFF8 8]
puts "UBOOT_GLOBAL_DATA_FDT"
puts [mrd -force 0x10239E80 32]
puts "UBOOT_HANG_STACK_END"
con
disconnect
