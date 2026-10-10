# Briefly halt A53 #0 to snapshot a headless runtime, then always resume it.
connect -url tcp:127.0.0.1:3121
puts "A53_RUNTIME_CONNECTED"
if {[catch {targets -set -filter {name =~ "Cortex-A53 #0"}} select_message]} {
    puts "A53_RUNTIME_SELECT_ERROR=$select_message"
    disconnect
    exit 1
}
puts "A53_RUNTIME_SELECTED=[targets -target-properties]"
if {[catch {stop} stop_message]} {
    if {![string match "*Already stopped*" $stop_message]} {
        puts "A53_RUNTIME_STOP_ERROR=$stop_message"
        disconnect
        exit 1
    }
}
puts "A53_RUNTIME_BEGIN"
foreach reg {pc r0 r1 r2 r3 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 sp cpsr} {
    catch {puts "$reg=[rrd $reg]"}
}
catch {puts [format "BOOT_MAILBOX=0x%08X" [mrd -force -value 0x7FF00000]]}
catch {puts [format "UBOOT_SCRIPT_STAGE=0x%08X" [mrd -force -value 0x03000100]]}
puts "A53_RUNTIME_END"
con
disconnect
