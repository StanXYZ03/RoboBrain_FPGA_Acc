# Read-only probe for arm64 Image headers at common PetaLinux load addresses.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
puts "SD_KERNEL_COMMON_SCAN_BEGIN"
foreach base {
    0x00080000 0x00100000 0x00200000 0x00400000
    0x00800000 0x01000000 0x02000000 0x04000000
    0x08000000 0x10000000 0x14000000 0x18000000
    0x20000000 0x28000000 0x30000000 0x40000000
} {
    set head0 [mrd -force -value $base]
    set head4 [mrd -force -value [expr {$base + 4}]]
    set magic [mrd -force -value [expr {$base + 0x38}]]
    puts [format "SD_KERNEL_CANDIDATE base=0x%08X head0=0x%08X head4=0x%08X magic=0x%08X" \
        $base $head0 $head4 $magic]
}
puts "SD_KERNEL_COMMON_SCAN_END"
disconnect
