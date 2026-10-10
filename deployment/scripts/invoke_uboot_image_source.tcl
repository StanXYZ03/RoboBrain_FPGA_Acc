# Retry only the U-Boot legacy-image handoff while preserving validated images.
set root [file normalize [file join [file dirname [info script]] ../..]]
set boot_script [file join $root deployment/artifacts/ram_linux/boot.scr]
set main_loop 0x7FCB1BDC

connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
stop

if {[mrd -force -value 0x00100000] != 0xEDFE0DD0 ||
        ([mrd -force -value 0x04000000] & 0x00FFFFFF) != 0x00088B1F ||
        [mrd -force -value 0x18000038] != 0x644D5241} {
    disconnect
    error "validated DTB/rootfs/Image are no longer present"
}

dow -data $boot_script 0x20000000
mwr 0x03000100 0
if {[mrd -force -value 0x20000000] != 0x56190527} {
    disconnect
    error "boot.scr reload gate failed"
}

# Make the DAP write visible to the CPU before image_source_script reads it.
bpadd -addr $main_loop -type hw -temp
rwr r0 0x20000000
rwr r1 0x20001000
rwr r30 $main_loop
rwr pc 0x7FC866B4
if {[catch {con -block -timeout 20} message]} {
    catch {stop}
    disconnect
    error "script cache invalidate failed: $message"
}

proc call_image_check {label address return_address} {
    bpadd -addr $return_address -type hw -temp
    rwr r0 0x20000000
    rwr r30 $return_address
    rwr pc $address
    if {[catch {con -block -timeout 10} check_message]} {
        error "$label call failed: $check_message"
    }
    puts "$label=[rrd r0]"
}

# Board-side legacy-image gates: format must be 1 and both CRC predicates 1.
call_image_check UBOOT_IMAGE_FORMAT 0x7FC8E100 $main_loop
call_image_check UBOOT_IMAGE_HCRC 0x7FC8D8C8 $main_loop
call_image_check UBOOT_IMAGE_DCRC 0x7FC8D918 $main_loop

rwr r0 0x20000000
rwr r1 0
rwr r30 $main_loop
rwr pc 0x7FC951A8
puts "UBOOT_IMAGE_SOURCE_INVOKE=START"
bpadd -addr $main_loop -type hw -temp
if {[catch {con -block -timeout 10} source_message]} {
    puts "UBOOT_IMAGE_SOURCE_RETURN=NO_RETURN message=$source_message"
    disconnect
    exit 0
}
puts "UBOOT_IMAGE_SOURCE_RETURN=[rrd r0]"
puts [format "UBOOT_IMAGE_SOURCE_MAGIC=0x%08X" [mrd -force -value 0x20000000]]
puts [format "UBOOT_IMAGE_SOURCE_STAGE=0x%08X" [mrd -force -value 0x03000100]]
con
disconnect
puts "UBOOT_IMAGE_SOURCE_INVOKE=HANDOFF"
