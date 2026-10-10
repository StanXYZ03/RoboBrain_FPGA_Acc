# Reset only. No download and no persistent-media operation.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
rst -system
puts "BOARD_SYSTEM_RESET_PASS"
disconnect
