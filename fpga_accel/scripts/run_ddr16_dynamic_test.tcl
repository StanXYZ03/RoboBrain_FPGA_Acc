set root [file normalize [file dirname [file dirname [info script]]]]
set fsbl_file [file normalize "$root/artifacts/ddr16_test/fsbl.elf"]
set test_file [file normalize "$root/artifacts/ddr16_test/ddr16_status_test.elf"]
set log_file [file normalize "$root/reports/ddr16_dynamic_test.log"]
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
emit $log "DDR16_STAGE=SYSTEM_RESET"
rst -system
after 1000

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
emit $log "DDR16_STAGE=DYNAMIC_FSBL"
dow $fsbl_file
bpadd -addr &XFsbl_Exit -type hw -temp
if {[catch {con -block -timeout 20} fsbl_message]} {
    catch {stop}
    emit $log "DDR16_FSBL_RUN=$fsbl_message"
}

set spd_magic [mrd -force -value $status_base]
set ranks [mrd -force -value [expr {$status_base + 4}]]
set rank_lo [mrd -force -value [expr {$status_base + 8}]]
set rank_hi [mrd -force -value [expr {$status_base + 12}]]
set bus_width [mrd -force -value [expr {$status_base + 16}]]
set dram_width [mrd -force -value [expr {$status_base + 20}]]
set rows [mrd -force -value [expr {$status_base + 24}]]
set columns [mrd -force -value [expr {$status_base + 28}]]
set banks [mrd -force -value [expr {$status_base + 32}]]
set bank_groups [mrd -force -value [expr {$status_base + 36}]]
set rank_bytes [expr {($rank_hi << 32) | $rank_lo}]
set total_bytes [expr {$rank_bytes * $ranks}]
set part ""
for {set word_index 0} {$word_index < 5} {incr word_index} {
    set word [mrd -force -value [expr {$status_base + 64 + 4 * $word_index}]]
    for {set byte_index 0} {$byte_index < 4} {incr byte_index} {
        set byte [expr {($word >> (8 * $byte_index)) & 0xFF}]
        if {$byte >= 32 && $byte <= 126} {
            append part [format %c $byte]
        }
    }
}
set part [string trim $part]

set ddrc_mstr [mrd -force -value 0xFD070000]
set ddrc_addrmap0 [mrd -force -value 0xFD070200]
set active_ranks [expr {($ddrc_mstr >> 24) & 0x3}]
set rank_address_bit [expr {$ddrc_addrmap0 & 0x1F}]

emit $log [format "DDR16_SPD magic=0x%08X ranks=%u rank_bytes=%s total_bytes=%s" \
    $spd_magic $ranks $rank_bytes $total_bytes]
emit $log [format "DDR16_GEOMETRY bus_width=%u dram_width=%u rows=%u columns=%u bank_bits=%u bank_group_bits=%u" \
    $bus_width $dram_width $rows $columns $banks $bank_groups]
emit $log [format "DDR16_DDRC mstr=0x%08X active_ranks=0x%X addrmap0=0x%08X rank_address_bit=0x%02X" \
    $ddrc_mstr $active_ranks $ddrc_addrmap0 $rank_address_bit]

if {$spd_magic != 0x53504434 || $total_bytes != 17179869184 ||
        $active_ranks != 3 || $rank_address_bit != 0x18} {
    emit $log "DDR16_ERROR=SPD_OR_RANK_MAPPING_INVALID"
    emit $log "DDR16_RESULT=FAIL"
    close $log
    disconnect
    exit 1
}

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
emit $log "DDR16_STAGE=DOWNLOAD_OCM_TEST"
dow $test_file
con

set terminal 0
set phase 0
for {set attempt 0} {$attempt < 200} {incr attempt} {
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

emit $log [format "DDR16_STATUS magic=0x%08X phase=0x%08X checks=%u failures=%u" \
    $magic $phase $checks $failures]
emit $log [format "DDR16_LAST address=0x%08X%08X expected=0x%08X actual=0x%08X" \
    $address_hi $address_lo $expected $actual]

if {$terminal && $phase == 0x600D && $checks == 16 && $failures == 0} {
    emit $log "DDR16_RESULT=PASS"
    set result 0
} else {
    if {!$terminal} {
        emit $log "DDR16_ERROR=TIMEOUT_OR_CPU_ABORT"
    }
    emit $log "DDR16_RESULT=FAIL"
    set result 1
}

close $log
disconnect
exit $result
