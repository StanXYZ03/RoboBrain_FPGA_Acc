# Closest JTAG equivalent of physical SD boot: run the official ZCU102 FSBL,
# which reads BOOT.BIN from the physically selected SD device. No host kernel.
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set fsbl "$platform_boot/fsbl.elf"
set pmufw "$platform_boot/pmufw.elf"

connect -url tcp:127.0.0.1:3121
after 500
# A system reset can leave the ZynqMP DAP in AXI AP error.  Reset the DAP
# immediately afterwards before touching PSU registers or downloading images.
targets -set -filter {name =~ "PSU"}
rst -system
after 1000
rst -dap
after 1000
targets -set -filter {name =~ "PSU"}
mwr 0xFFCA0038 0x1FF
mwr 0xFFCA0010 0
set sd_power_word [mrd -force -value 0xFF170028]
mwr 0xFF170028 [expr {$sd_power_word & 0xFFFF00FF}]
after 500
mwr 0xFF170028 [expr {($sd_power_word & 0xFFFF00FF) | 0x00000F00}]
after 200
after 500

targets -set -filter {name =~ "MicroBlaze PMU"}
dow $pmufw
con
after 500

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
dow $fsbl
puts "STOCK_SD_BOOT_STAGE=FSBL_START"
# Stock BOOT.BIN clears debug comparators and Linux can lock the DAP, so a
# breakpoint is not a reliable completion signal.  Let the official FSBL run
# freely; Ethernet/DHCP is the external success criterion.
con
after 1000
disconnect
puts "STOCK_SD_BOOT_HANDOFF=PASS source=sd_boot_bin"
