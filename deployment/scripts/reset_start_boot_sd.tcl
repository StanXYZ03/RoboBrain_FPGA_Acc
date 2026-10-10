# Explicit system-reset-and-start sequence for physical SD boot. No downloads.
connect -url tcp:127.0.0.1:3121
targets -set 4
rst -system -start
after 2000
puts "BOARD_SYSTEM_RESET_START_PASS"
disconnect
