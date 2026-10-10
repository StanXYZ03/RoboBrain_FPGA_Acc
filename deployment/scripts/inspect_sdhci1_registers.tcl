# Read-only SDHCI1 register snapshot (ZCU102 SD slot at 0xFF170000).
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
puts "SDHCI1_REGISTERS_BEGIN"
puts [mrd -force 0xFF170000 64]
puts "SDHCI1_REGISTERS_END"
disconnect
