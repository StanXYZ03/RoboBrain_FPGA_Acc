# Briefly halt a running A53 to capture the BL31 panic context, then resume it.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
stop
puts "BL31_PANIC_REGS_BEGIN"
foreach reg {pc sp r0 r1 r2 r3 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 cpsr sys} {
    if {[catch {rrd $reg} value]} {
        puts "$reg=UNAVAILABLE"
    } else {
        puts "$reg=$value"
    }
}
puts "BL31_PANIC_REGS_END"
puts "BL31_PANIC_STACK_BEGIN"
catch {puts [mrd -force 0xFFFF6400 64]}
puts "BL31_PANIC_STACK_END"
con
disconnect
