# Read the volatile headless-runtime progress mailbox without changing board state.
connect -url tcp:127.0.0.1:3121
# Use the PSU system target so the address is treated as physical even when
# Linux has enabled the A53 MMU (or has stopped in panic()).
targets -set -filter {name =~ "PSU"}
set value [mrd -force -value 0x7FF00000]
puts [format "RUNTIME_MAILBOX value=0x%08X" $value]
disconnect
