set root [file normalize [file dirname [file dirname [info script]]]]
set init_file [file normalize "$root/artifacts/ddr_smoke/psu_init.tcl"]
set elf_file [file normalize "$root/artifacts/ddr_smoke/ddr_status_test.elf"]
set log_file [file normalize "$root/reports/ddr_status_test.log"]
set status_base 0xFFFEF000
set log [open $log_file w]

proc emit {channel message} {
    puts $message
    puts $channel $message
    flush $channel
}

connect -url tcp:127.0.0.1:3121
after 1000
targets -set -filter {name =~ "PSU"}
emit $log "DDR_ELF_STAGE=RESET"
rst -system
after 1000

emit $log "DDR_ELF_STAGE=PS_INIT"
source $init_file
psu_init
psu_ps_pl_isolation_removal
psu_ps_pl_reset_config

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
emit $log "DDR_ELF_STAGE=DOWNLOAD_OCM"
dow $elf_file
con

set terminal 0
set phase 0
for {set attempt 0} {$attempt < 100} {incr attempt} {
    after 100
    set magic [mrd -force -value $status_base]
    set phase [mrd -force -value [expr {$status_base + 4}]]
    if {$magic == 0x44445234 && ($phase == 0x600D || $phase == 0xDEAD)} {
        set terminal 1
        break
    }
}

targets -set -filter {name =~ "Cortex-A53 #0"}
catch {stop}
set magic [mrd -force -value $status_base]
set checks [mrd -force -value [expr {$status_base + 8}]]
set failures [mrd -force -value [expr {$status_base + 12}]]
set address_lo [mrd -force -value [expr {$status_base + 16}]]
set address_hi [mrd -force -value [expr {$status_base + 20}]]
set expected [mrd -force -value [expr {$status_base + 24}]]
set actual [mrd -force -value [expr {$status_base + 28}]]

emit $log [format "DDR_ELF_STATUS magic=0x%08X phase=0x%08X checks=%u failures=%u" \
    $magic $phase $checks $failures]
emit $log [format "DDR_ELF_LAST address=0x%08X%08X expected=0x%08X actual=0x%08X" \
    $address_hi $address_lo $expected $actual]

if {$terminal && $phase == 0x600D && $failures == 0} {
    emit $log "DDR_ELF_RESULT=PASS"
    set result 0
} else {
    if {!$terminal} {
        emit $log "DDR_ELF_ERROR=TIMEOUT_OR_CPU_ABORT"
    }
    emit $log "DDR_ELF_RESULT=FAIL"
    set result 1
}

close $log
disconnect
exit $result
