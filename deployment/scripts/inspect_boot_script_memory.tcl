connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
puts "BOOT_SCRIPT_MEMORY_BEGIN"
puts [format "DTB_MAGIC=0x%08X" [mrd -force -value 0x00100000]]
puts [format "ROOTFS_MAGIC=0x%08X" [mrd -force -value 0x04000000]]
puts [format "IMAGE_MAGIC=0x%08X" [mrd -force -value 0x18000038]]
puts [mrd -force 0x20000000 16]
puts "BOOT_SCRIPT_MEMORY_END"
puts "UBOOT_COMMAND_MEMORY_BEGIN"
puts [mrd -force 0x03000000 8]
puts [mrd -force 0x03000100 1]
puts "UBOOT_COMMAND_MEMORY_END"
disconnect
