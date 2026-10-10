set root [file normalize [file dirname [file dirname [info script]]]]
set fsbl_file [file normalize "$root/artifacts/ddr16_test/fsbl.elf"]
set eth_file [file normalize "$root/artifacts/eth_test/eth_echo_test.elf"]
set log_file [file normalize "$root/reports/eth_minimal_test.log"]
set status_base 0xFFFEF100
set log [open $log_file w]

proc emit {channel message} {
    puts $message
    puts $channel $message
    flush stdout
    flush $channel
}

proc read_status {base} {
    set values {}
    for {set index 0} {$index < 8} {incr index} {
        lappend values [mrd -force -value [expr {$base + 4 * $index}]]
    }
    return $values
}

proc read_gem_stats {base} {
    set tx_count [mrd -force -value [expr {$base + 0x108}]]
    set tx_broadcast [mrd -force -value [expr {$base + 0x10C}]]
    set tx_multicast [mrd -force -value [expr {$base + 0x110}]]
    set rx_count [mrd -force -value [expr {$base + 0x158}]]
    set rx_broadcast [mrd -force -value [expr {$base + 0x15C}]]
    set rx_multicast [mrd -force -value [expr {$base + 0x160}]]
    set rx_fcs_error [mrd -force -value [expr {$base + 0x190}]]
    set rx_ip_error [mrd -force -value [expr {$base + 0x1A8}]]
    set rx_tcp_error [mrd -force -value [expr {$base + 0x1AC}]]
    set rx_udp_error [mrd -force -value [expr {$base + 0x1B0}]]
    set tx_unicast [expr {$tx_count - $tx_broadcast - $tx_multicast}]
    set rx_unicast [expr {$rx_count - $rx_broadcast - $rx_multicast}]
    return [list $tx_count $tx_unicast $rx_count $rx_unicast \
        $rx_fcs_error $rx_ip_error $rx_tcp_error $rx_udp_error]
}

connect -url tcp:127.0.0.1:3121
after 1000
targets -set -filter {name =~ "PSU"}
emit $log "ETH_STAGE=SYSTEM_RESET"
rst -system
after 1000

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
emit $log "ETH_STAGE=DYNAMIC_DDR_FSBL"
dow $fsbl_file
bpadd -addr &XFsbl_Exit -type hw -temp
if {[catch {con -block -timeout 20} fsbl_message]} {
    catch {stop}
    emit $log "ETH_FSBL_RUN=$fsbl_message"
}

set spd_magic [mrd -force -value 0xFFFEF000]
set ranks [mrd -force -value 0xFFFEF004]
set rank_lo [mrd -force -value 0xFFFEF008]
set rank_hi [mrd -force -value 0xFFFEF00C]
set rank_bytes [expr {($rank_hi << 32) | $rank_lo}]
set total_bytes [expr {$rank_bytes * $ranks}]
emit $log [format "ETH_DDR magic=0x%08X ranks=%u total_bytes=%s" \
    $spd_magic $ranks $total_bytes]
if {$spd_magic != 0x53504434 || $total_bytes != 17179869184} {
    emit $log "ETH_ERROR=DDR_INITIALIZATION_INVALID"
    emit $log "ETH_RESULT=FAIL"
    close $log
    disconnect
    exit 1
}

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
emit $log "ETH_STAGE=DOWNLOAD_ECHO_ELF"
dow $eth_file
con

set ready 0
for {set attempt 0} {$attempt < 300} {incr attempt} {
    after 100
    set state [read_status $status_base]
    set magic [lindex $state 0]
    set phase [lindex $state 1]
    if {$magic == 0x45544831 && ($phase == 3 || $phase == 0x600D || $phase == 0xDEAD)} {
        set ready 1
        break
    }
}

if {!$ready || $phase == 0xDEAD} {
    set error [expr {$ready ? [lindex $state 6] : 0xFFFFFFFF}]
    emit $log [format "ETH_READY magic=0x%08X phase=0x%08X error=0x%08X" \
        $magic $phase $error]
    emit $log "ETH_RESULT=FAIL"
    close $log
    disconnect
    exit 1
}

emit $log [format "ETH_SERVER_READY magic=0x%08X phase=0x%08X link=%u board=172.21.127.254:7008 udp_broadcast_port=7007" \
    $magic $phase [lindex $state 2]]

set gem_base 0xFF0E0000
set gem_start [read_gem_stats $gem_base]
set tx_unicast_start [lindex $gem_start 1]
set rx_unicast_start [lindex $gem_start 3]
emit $log [format "ETH_ICMP_READY command=ping_-I_enp4s0_172.21.127.254 tx_unicast=%u rx_unicast=%u" \
    $tx_unicast_start $rx_unicast_start]

set completed 0
set completion_mode "none"
set gem_state $gem_start
set tx_unicast_delta 0
set rx_unicast_delta 0
for {set attempt 0} {$attempt < 1200} {incr attempt} {
    after 100
    set state [read_status $status_base]
    set phase [lindex $state 1]
    set gem_state [read_gem_stats $gem_base]
    set tx_unicast_delta [expr {[lindex $gem_state 1] - $tx_unicast_start}]
    set rx_unicast_delta [expr {[lindex $gem_state 3] - $rx_unicast_start}]
    if {$phase == 0x600D} {
        set completed 1
        set completion_mode "udp_echo"
        break
    }
    if {$rx_unicast_delta >= 1 && $tx_unicast_delta >= 1} {
        set completed 1
        set completion_mode "icmp_gem_counters"
        break
    }
    if {$phase == 0xDEAD} {
        break
    }
}

set magic [lindex $state 0]
set link [lindex $state 2]
set valid_rx [lindex $state 3]
set echo_tx [lindex $state 4]
set sequence [lindex $state 5]
set error [lindex $state 6]
set ip_word [lindex $state 7]
emit $log [format "ETH_STATUS magic=0x%08X phase=0x%08X link=%u valid_rx=%u echo_tx=%u sequence=0x%08X error=0x%08X ip_word=0x%08X" \
    $magic $phase $link $valid_rx $echo_tx $sequence $error $ip_word]

set tx_count [lindex $gem_state 0]
set rx_count [lindex $gem_state 2]
set rx_fcs_error [lindex $gem_state 4]
set rx_ip_error [lindex $gem_state 5]
set rx_tcp_error [lindex $gem_state 6]
set rx_udp_error [lindex $gem_state 7]
emit $log [format "ETH_GEM mode=%s tx_count=%u tx_unicast_delta=%u rx_count=%u rx_unicast_delta=%u rx_fcs_error=%u rx_ip_error=%u rx_tcp_error=%u rx_udp_error=%u" \
    $completion_mode $tx_count $tx_unicast_delta $rx_count $rx_unicast_delta \
    $rx_fcs_error $rx_ip_error $rx_tcp_error $rx_udp_error]

set udp_pass [expr {$completion_mode eq "udp_echo" &&
    $phase == 0x600D && $valid_rx >= 1 && $echo_tx >= 1 &&
    $sequence == 0x10203040 && $error == 0}]
set icmp_pass [expr {$completion_mode eq "icmp_gem_counters" &&
    $rx_unicast_delta >= 1 && $tx_unicast_delta >= 1 &&
    $rx_fcs_error == 0 && $rx_ip_error == 0}]
if {$completed && $magic == 0x45544831 && $link == 1 &&
        ($udp_pass || $icmp_pass)} {
    emit $log "ETH_RESULT=PASS"
    set result 0
} else {
    if {!$completed} {
        emit $log "ETH_ERROR=NO_VALID_HOST_TRAFFIC"
    }
    emit $log "ETH_RESULT=FAIL"
    set result 1
}

close $log
disconnect
exit $result
