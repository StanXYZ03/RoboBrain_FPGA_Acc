connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}
puts "UBOOT_RELOCATION_REGS_BEGIN"
foreach reg {pc r18 sp r29 r30} {
    catch {puts "$reg=[rrd $reg]"}
}
puts "UBOOT_RELOCATION_REGS_END"
puts "UBOOT_GD_BEGIN"
# x18 is U-Boot's global_data pointer. Read enough words to locate ram_top,
# relocaddr and reloc_off without relying on UART.
if {![catch {rrd r18} gd_line] && [regexp {([[:xdigit:]]{16})} $gd_line -> gd_hex]} {
    puts [mrd -force [expr 0x$gd_hex] 64]
}
puts "UBOOT_GD_END"
disconnect
