# Invoke a validated U-Boot command without UART. No persistent-media access.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
stop

if {[mrd -force -value 0x10084A48] != 0x52800000 ||
        [mrd -force -value 0x10084A4C] != 0xD65F03C0 ||
        [mrd -force -value 0xFFFEB0DC] != 0xD503201F} {
    disconnect
    error "expected RAM-only U-Boot/BL31 patches are absent"
}
if {[mrd -force -value 0x20000000] != 0x56190527} {
    disconnect
    error "boot.scr magic is absent from DDR"
}

# "source 0x20000000\0" in the 1 MiB mailbox region excluded from Linux RAM.
set command_address 0x7FF00100
mwr $command_address 0x72756F73
mwr [expr {$command_address + 4}] 0x30206563
mwr [expr {$command_address + 8}] 0x30303278
mwr [expr {$command_address + 12}] 0x30303030
mwr [expr {$command_address + 16}] 0x00000030

# Pinned U-Boot: run_command link 0x100B80D4 + reloc_off 0x6FC05000.
# If it returns, resume at main_loop; on successful booti it never returns.
mwr 0x7FF00000 0x5542434D
rwr r0 $command_address
rwr r1 0
rwr r30 0x7FCB1BDC
rwr pc 0x7FCBD0D4
puts "UBOOT_SOURCE_INVOKE=START command=source_0x20000000"
con
disconnect
puts "UBOOT_SOURCE_INVOKE=HANDOFF"
