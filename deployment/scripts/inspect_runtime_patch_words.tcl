connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
foreach {label address} {
    UBOOT_BOOTMODE_0 0x10084A48
    UBOOT_BOOTMODE_1 0x10084A4C
    UBOOT_ENABLE_CACHES 0x100819F4
    BL31_MODE_BRANCH 0xFFFEB0DC
} {
    puts [format "%s=0x%08X" $label [mrd -force -value $address]]
}
puts [format "BOOT_MAILBOX=0x%08X" [mrd -force -value 0x7FF00000]]
disconnect
