# Boot official SD FSBL and halt at the expected arm64 Image entry so its
# loaded DTB can be inspected/patched in volatile RAM before Linux starts.
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set fsbl "$platform_boot/fsbl.elf"
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
dow $fsbl
bpadd -addr 0x18000000 -type hw -temp
puts "SD_KERNEL_INTERCEPT_WAIT address=0x18000000"
if {[catch {con -block -timeout 120} kernel_message]} {
    puts "SD_KERNEL_INTERCEPT_TIMEOUT=$kernel_message"
    disconnect
    exit 1
}
puts "SD_KERNEL_INTERCEPT_PC=[rrd pc]"
puts "SD_KERNEL_INTERCEPT_DTB=[rrd r0]"
puts "SD_KERNEL_INTERCEPT_X1=[rrd r1]"
puts "SD_KERNEL_INTERCEPT_X2=[rrd r2]"
puts "SD_KERNEL_INTERCEPT_X3=[rrd r3]"
disconnect
puts "SD_KERNEL_INTERCEPT_PASS state=halted"
