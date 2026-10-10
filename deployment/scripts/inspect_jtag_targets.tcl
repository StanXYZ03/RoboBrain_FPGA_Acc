# Read-only snapshot of JTAG target state for headless boot diagnosis.
connect -url tcp:127.0.0.1:3121
puts "JTAG_TARGETS_BEGIN"
puts [targets]
puts "JTAG_TARGETS_END"
if {![catch {targets -set -filter {name =~ "PSU"}}]} {
    catch {puts [format "BOOT_MAILBOX=0x%08X" [mrd -force -value 0x7FF00000]]}
    catch {puts [format "UBOOT_SCRIPT_STAGE=0x%08X" [mrd -force -value 0x03000100]]}
}
if {![catch {targets -set -filter {name =~ "Cortex-A53 #0"}}]} {
    for {set sample 0} {$sample < 5} {incr sample} {
        catch {puts "A53_PC_$sample=[rrd pc]"}
        after 200
    }
}
disconnect
