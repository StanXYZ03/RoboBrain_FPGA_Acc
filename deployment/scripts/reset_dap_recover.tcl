# Recover only the ARM debug access port after a system reset.
connect -url tcp:127.0.0.1:3121
targets -set 4
rst -dap
after 1000
puts "BOARD_DAP_RESET_PASS"
puts [targets]
disconnect
