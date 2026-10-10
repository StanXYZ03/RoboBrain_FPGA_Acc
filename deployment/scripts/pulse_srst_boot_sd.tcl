# Pulse the physical SRST signal through the JTAG cable. No download/write.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PS TAP"}
rst -srst
after 1000
puts "BOARD_SRST_PULSE_PASS"
disconnect
