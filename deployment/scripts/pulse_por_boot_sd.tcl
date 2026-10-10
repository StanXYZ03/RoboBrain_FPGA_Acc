# Pulse the physical POR signal through the JTAG cable. No download/write.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PS TAP"}
rst -por
after 2000
puts "BOARD_POR_PULSE_PASS"
disconnect
