connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
stop
puts [rrd -defs]
con
disconnect
