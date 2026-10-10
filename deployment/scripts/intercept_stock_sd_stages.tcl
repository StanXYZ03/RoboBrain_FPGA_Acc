# Catch either the known platform U-Boot main_loop or common Linux entries.
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
connect -url tcp:127.0.0.1:3121
after 500
if {[catch {targets -set -filter {name =~ "PSU"}}]} {
    targets -set 4
    rst -system
    after 1000
    rst -dap
    after 1500
    disconnect
    connect -url tcp:127.0.0.1:3121
    targets -set -filter {name =~ "PSU"}
}
rst -system
after 1000
mwr 0xFFCA0038 0x1FF
after 500
targets -set -filter {name =~ "MicroBlaze PMU"}
dow "$platform_boot/pmufw.elf"
con
after 500
targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
dow "$platform_boot/fsbl.elf"
foreach address {0x7FCB1BDC 0x18000000 0x00200000 0x00080000} {
    bpadd -addr $address -type hw
}
puts "SD_STAGE_INTERCEPT_WAIT"
if {[catch {con -block -timeout 120} message]} {
    puts "SD_STAGE_INTERCEPT_TIMEOUT=$message"
    disconnect
    exit 1
}
puts "SD_STAGE_INTERCEPT_PC=[rrd pc]"
puts "SD_STAGE_INTERCEPT_X0=[rrd r0]"
puts "SD_STAGE_INTERCEPT_X1=[rrd r1]"
puts "SD_STAGE_INTERCEPT_X2=[rrd r2]"
puts "SD_STAGE_INTERCEPT_X3=[rrd r3]"
disconnect
puts "SD_STAGE_INTERCEPT_PASS state=halted"
