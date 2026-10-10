# Call U-Boot MMC APIs directly and capture pointers/error codes without UART.
set main_loop 0x7FCB1BDC

proc call_uboot {address arg0 arg1} {
    global main_loop
    bpadd -addr $main_loop -type hw -temp
    rwr r0 $arg0
    rwr r1 $arg1
    rwr r30 $main_loop
    rwr pc $address
    if {[catch {con -block -timeout 20} message]} {
        error "U-Boot API call at $address failed: $message"
    }
    set result_line [rrd r0]
    if {![regexp {([[:xdigit:]]{16})} $result_line -> result_hex]} {
        error "cannot parse U-Boot API result: $result_line"
    }
    return [expr 0x$result_hex]
}

connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}

foreach device_number {0 1} {
    set device [call_uboot 0x7FCD2B8C $device_number 0]
    puts [format "MMC_API device_number=%u pointer=0x%016X" $device_number $device]
    if {$device != 0} {
        set card_detect [call_uboot 0x7FCD2604 $device 0]
        set init_result [call_uboot 0x7FCD1F08 $device 0]
        puts [format "MMC_API device_number=%u card_detect=0x%016X init_result=0x%016X" \
            $device_number $card_detect $init_result]
    }
}

con
disconnect
