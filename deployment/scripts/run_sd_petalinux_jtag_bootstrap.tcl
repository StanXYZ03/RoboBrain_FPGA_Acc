# Headless SD-PetaLinux bootstrap for the upgraded 16 GiB ZCU102.
# JTAG initializes PS/DDR and preloads firmware/U-Boot into volatile RAM only;
# unmodified U-Boot then follows the physical SD boot mode. No media writes.

set project_root /home/zhaoxiangyu/RoboBrain2.5
set fsbl "$project_root/fpga_accel/artifacts/ddr16_test/fsbl.elf"
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set pmufw "$platform_boot/pmufw.elf"
set atf "$platform_boot/bl31.elf"
set uboot "$platform_boot/u-boot.elf"
set control_fdt "$project_root/deployment/artifacts/ram_linux/system-16gb.dtb"

foreach file [list $fsbl $pmufw $atf $uboot $control_fdt] {
    if {![file readable $file]} { error "missing SD bootstrap artifact: $file" }
}

connect -url tcp:127.0.0.1:3121
after 1000
targets -set -filter {name =~ "PSU"}
puts "SD_BOOT_STAGE=SYSTEM_RESET"
rst -system
after 1000
rst -dap
after 1000
targets -set -filter {name =~ "PSU"}
mwr 0xFFCA0038 0x1FF
after 500

targets -set -filter {name =~ "MicroBlaze PMU"}
dow $pmufw
con
after 500

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
puts "SD_BOOT_STAGE=DYNAMIC_DDR_FSBL"
dow $fsbl
bpadd -addr 0xFFFC943C -type hw -temp
if {[catch {con -block -timeout 20} fsbl_message]} {
    catch {stop}
    disconnect
    error "dynamic DDR FSBL did not reach clean exit: $fsbl_message"
}

set magic [mrd -force -value 0xFFFEF000]
set ranks [mrd -force -value 0xFFFEF004]
set rank_lo [mrd -force -value 0xFFFEF008]
set rank_hi [mrd -force -value 0xFFFEF00C]
set rank_bytes [expr {($rank_hi << 32) | $rank_lo}]
set total_bytes [expr {$rank_bytes * $ranks}]
puts [format "SD_BOOT_DDR magic=0x%08X ranks=%u total_bytes=%s" $magic $ranks $total_bytes]
if {$magic != 0x53504434 || $ranks != 2 || $total_bytes != 17179869184} {
    disconnect
    error "16 GiB DDR initialization gate failed"
}

targets -set -filter {name =~ "PSU"}
set physical_boot_mode [mrd -force -value 0xFF5E0200]
puts [format "SD_BOOT_MODE physical=0x%08X" $physical_boot_mode]
if {$physical_boot_mode != 0xE} {
    disconnect
    error "board is not physically strapped for SD boot"
}

targets -set -filter {name =~ "Cortex-A53 #0"}
dow $uboot
dow $atf
# This U-Boot build expects an external control-FDT at 0x00100000 when its
# ELF-embedded fallback is unavailable through the manual JTAG load.
dow -data $control_fdt 0x00100000
if {[mrd -force -value 0x00100000] != 0xEDFE0DD0} {
    disconnect
    error "U-Boot control-FDT read-back gate failed"
}

# Manual JTAG preload has no FSBL partition-handoff table. Force only BL31's
# adjacent preloaded-BL33 path. U-Boot itself remains unpatched and sees SD.
if {[mrd -force -value 0xFFFEB0DC] != 0x350002E0} {
    disconnect
    error "unexpected pinned BL31 boot-mode branch"
}
mwr 0xFFFEB0DC 0xD503201F
if {[mrd -force -value 0xFFFEB0DC] != 0xD503201F} {
    disconnect
    error "volatile BL31 preloaded-BL33 patch failed"
}

# A manually preloaded U-Boot has no SD-FSBL control-FDT handoff. Select its
# proven JTAG control-FDT path during early init; standard distro boot targets
# remain enabled and will scan MMC/SD after main_loop starts.
if {[mrd -force -value 0x10084A48] != 0xA9BE7BFD ||
        [mrd -force -value 0x10084A4C] != 0x52804000} {
    disconnect
    error "unexpected pinned U-Boot boot-mode code"
}
mwr 0x10084A48 0x52800000
mwr 0x10084A4C 0xD65F03C0
if {[mrd -force -value 0x10084A48] != 0x52800000 ||
        [mrd -force -value 0x10084A4C] != 0xD65F03C0} {
    disconnect
    error "volatile U-Boot JTAG control-FDT patch failed"
}

mwr 0x7FF00000 0x53444230
set relocated_main_loop 0x7FCB1BDC
bpadd -addr $relocated_main_loop -type hw -temp
puts "SD_BOOT_STAGE=START_ATF_UBOOT"
if {[catch {con -block -timeout 20} uboot_message]} {
    catch {stop}
    disconnect
    error "U-Boot did not reach relocated main_loop: $uboot_message"
}
puts "SD_BOOT_UBOOT_MAIN_LOOP=PASS"
con
disconnect
puts "SD_BOOT_JTAG_HANDOFF_PASS target=sd_distro_boot media_writes=none"
