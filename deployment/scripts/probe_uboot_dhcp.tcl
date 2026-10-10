# Run a volatile, read-only DHCP/ping probe in an already-running pinned U-Boot.
set main_loop 0x7FCB1BDC
set command_address 0x03000000

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

proc call_at_main_loop {address arg0 arg1 timeout_seconds} {
    global main_loop
    bpadd -addr $main_loop -type hw -temp
    rwr r0 $arg0
    rwr r1 $arg1
    rwr r30 $main_loop
    rwr pc $address
    if {[catch {con -block -timeout $timeout_seconds} message]} {
        error "U-Boot call at $address failed: $message"
    }
}

proc run_command_capture {command timeout_seconds} {
    global command_address
    write_c_string $command_address $command
    call_at_main_loop 0x7FC866B4 $command_address [expr {$command_address + 0x1000}] 20
    call_at_main_loop 0x7FCBD0D4 $command_address 0 $timeout_seconds
    set result [rrd r0]
    puts "UBOOT_COMMAND=$command"
    puts "UBOOT_COMMAND_RETURN=$result"
}

connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}
run_command_capture "setenv autoload no; dhcp" 60
run_command_capture "ping 192.168.50.1" 30
con
disconnect
