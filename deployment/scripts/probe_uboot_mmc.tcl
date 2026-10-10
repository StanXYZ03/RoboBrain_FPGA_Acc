# Execute read-only U-Boot MMC commands and inspect block 0 without UART.
set main_loop 0x7FCB1BDC
set command_address 0x03000000
set buffer_address 0x21000000

proc write_c_string {address value} {
    append value "\0"
    set length [string length $value]
    for {set offset 0} {$offset < $length} {incr offset 4} {
        set word 0
        for {set byte_index 0} {$byte_index < 4} {incr byte_index} {
            set index [expr {$offset + $byte_index}]
            if {$index < $length} {
                scan [string index $value $index] %c byte_value
                set word [expr {$word | (($byte_value & 0xFF) << (8 * $byte_index))}]
            }
        }
        mwr [expr {$address + $offset}] $word
    }
}

proc call_at_main_loop {address arg0 arg1} {
    global main_loop
    bpadd -addr $main_loop -type hw -temp
    rwr r0 $arg0
    rwr r1 $arg1
    rwr r30 $main_loop
    rwr pc $address
    if {[catch {con -block -timeout 20} message]} {
        error "U-Boot call at $address failed: $message"
    }
}

proc run_command_capture {command} {
    global command_address
    write_c_string $command_address $command
    # Invalidate the DAP-written command string before U-Boot parses it.
    call_at_main_loop 0x7FC866B4 $command_address [expr {$command_address + 0x1000}]
    call_at_main_loop 0x7FCBD0D4 $command_address 0
    puts "UBOOT_COMMAND=$command"
    puts "UBOOT_COMMAND_RETURN=[rrd r0]"
}

connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}

foreach device {0 1} {
    run_command_capture "mmc dev $device"
    run_command_capture "mmc rescan"
    run_command_capture [format "mw.l 0x%X 0 0x80" $buffer_address]
    run_command_capture [format "mmc read 0x%X 0 1" $buffer_address]
    puts "MMC${device}_BLOCK0_BEGIN"
    puts [mrd -force $buffer_address 128]
    puts "MMC${device}_BLOCK0_END"
}

con
disconnect
