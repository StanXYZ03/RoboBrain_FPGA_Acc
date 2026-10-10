connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
puts [format "BOOT_MODE_USER=0x%08X" [mrd -force -value 0xFF5E0200]]
disconnect
