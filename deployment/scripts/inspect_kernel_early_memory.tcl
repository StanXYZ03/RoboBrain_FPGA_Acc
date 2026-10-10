# Physical-memory snapshot of arm64 head.S progress; no core halt required.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
puts "KERNEL_EARLY_MEMORY_BEGIN"
puts [format "BOOT_MAILBOX=0x%08X" [mrd -force -value 0x7FF00000]]
puts "BOOT_ARGS_PHYS_0x19315000"
puts [mrd -force 0x19315000 8]
puts "RELOCATED_FDT_PHYS_0x74994000"
puts [mrd -force 0x74994000 32]
puts "INIT_IDMAP_PG_DIR_PHYS_0x19180000"
puts [mrd -force 0x19180000 16]
puts "INIT_PG_DIR_PHYS_0x1944A000"
puts [mrd -force 0x1944A000 16]
puts "FDT_POINTER_PHYS_0x191863B8"
puts [mrd -force 0x191863B8 2]
puts "BOOT_CPU_MODE_PHYS_0x19326460"
puts [mrd -force 0x19326460 2]
puts "KERNEL_EARLY_MEMORY_END"
disconnect
