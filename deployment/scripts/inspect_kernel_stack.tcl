proc register_value {name} {
    set line [rrd $name]
    if {![regexp {([[:xdigit:]]{16})} $line -> value]} {
        error "cannot parse $name from: $line"
    }
    return [expr 0x$value]
}

proc read_u64 {address} {
    set lo [mrd -force -value $address]
    set hi [mrd -force -value [expr {$address + 4}]]
    return [expr {($hi << 32) | $lo}]
}

connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}
puts "KERNEL_STACK_REGISTERS_BEGIN"
foreach reg {pc sp r29 r30 r0 r1 r2 r3 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28} {
    puts [rrd $reg]
}
puts "KERNEL_STACK_REGISTERS_END"
set frame [register_value r29]
puts [format "KERNEL_FRAME_POINTER=0x%016X" $frame]
puts "KERNEL_FRAME_CHAIN_BEGIN"
for {set depth 0} {$depth < 20 && $frame != 0} {incr depth} {
    if {[catch {set next_frame [read_u64 $frame]} chain_message] ||
            [catch {set saved_lr [read_u64 [expr {$frame + 8}]]}]} {
        puts "KERNEL_FRAME_CHAIN_FAIL depth=$depth message=$chain_message"
        break
    }
    puts [format "KERNEL_FRAME depth=%u fp=0x%016X lr=0x%016X next=0x%016X" \
        $depth $frame $saved_lr $next_frame]
    if {$next_frame <= $frame} { break }
    set frame $next_frame
}
puts "KERNEL_FRAME_CHAIN_END"
if {[catch {mrd -force $frame 64} frame_message]} {
    puts "KERNEL_FRAME_READ_FAIL=$frame_message"
} else {
    puts "KERNEL_FRAME_MEMORY_BEGIN"
    puts $frame_message
    puts "KERNEL_FRAME_MEMORY_END"
}
disconnect
