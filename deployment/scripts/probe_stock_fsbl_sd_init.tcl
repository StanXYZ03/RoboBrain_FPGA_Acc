# Validate stock FSBL's read-only SD/FAT init after validated 16 GiB DDR init.
set project_root /home/zhaoxiangyu/RoboBrain2.5
set dynamic_fsbl "$project_root/fpga_accel/artifacts/ddr16_test/fsbl.elf"
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set stock_fsbl "$platform_boot/fsbl.elf"
set pmufw "$platform_boot/pmufw.elf"

connect -url tcp:127.0.0.1:3121
after 500
targets -set -filter {name =~ "PSU"}
rst -system
after 1000
mwr 0xFFCA0038 0x1FF
after 500

targets -set -filter {name =~ "MicroBlaze PMU"}
dow $pmufw
con
after 500

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
dow $dynamic_fsbl
bpadd -addr 0xFFFC943C -type hw -temp
if {[catch {con -block -timeout 20} ddr_message]} {
    catch {stop}
    disconnect
    error "dynamic DDR FSBL failed: $ddr_message"
}
if {[mrd -force -value 0xFFFEF000] != 0x53504434 ||
        [mrd -force -value 0xFFFEF004] != 2} {
    disconnect
    error "16 GiB DDR gate failed"
}
puts "STOCK_SD_PROBE_DDR=PASS"

# Load, but do not run, stock FSBL. Call only its FAT/SD initialization entry.
dow $stock_fsbl
rwr sp 0xFFFE3D80
set return_address 0xFFFC3424
bpadd -addr $return_address -type hw -temp
rwr r0 0x10
rwr r30 $return_address
rwr pc 0xFFFCFA28
if {[catch {con -block -timeout 30} sd_message]} {
    catch {stop}
    disconnect
    error "stock XFsbl_SdInit call failed: $sd_message"
}
puts "STOCK_SD_PROBE_RESULT=[rrd r0]"
puts "STOCK_SD_PROBE_PC=[rrd pc]"
disconnect
