set script_dir [file dirname [file normalize [info script]]]
set smoke_dir [file dirname $script_dir]
set bitstream [file normalize "$smoke_dir/artifacts/jtag/pl_jtag_smoke.bit"]
set probes_file [file normalize "$smoke_dir/artifacts/jtag/pl_jtag_smoke.ltx"]

foreach required_file [list $bitstream $probes_file] {
    if {![file exists $required_file]} {
        error "Missing PL JTAG smoke-test artifact: $required_file"
    }
}

proc parse_hex {value} {
    set normalized [string trim $value]
    if {[string match "0x*" $normalized]} {
        return [expr {$normalized}]
    }
    scan $normalized %x parsed
    return $parsed
}

set rc 0
if {[catch {
    open_hw_manager
    connect_hw_server -url localhost:3121
    open_hw_target

    set devices [get_hw_devices -quiet xczu9_0]
    if {[llength $devices] != 1} {
        error "Expected exactly one xczu9_0 device, found [llength $devices]"
    }
    set device [lindex $devices 0]
    current_hw_device $device
    refresh_hw_device $device

    puts "PL_JTAG_BOARD_STAGE=PROGRAM"
    set_property PROGRAM.FILE $bitstream $device
    set_property PROBES.FILE $probes_file $device
    set_property FULL_PROBES.FILE $probes_file $device
    program_hw_devices $device
    refresh_hw_device $device

    set done_status "unavailable"
    catch {set done_status [get_property REGISTER.STATUS.BIT00_DONE $device]}
    puts "PL_JTAG_BOARD_DONE=$done_status"

    set vios [get_hw_vios -quiet -of_objects $device]
    if {[llength $vios] != 1} {
        error "Expected exactly one VIO core after programming, found [llength $vios]"
    }
    set vio [lindex $vios 0]
    puts "PL_JTAG_BOARD_VIO=$vio"

    set output_probes [get_hw_probes -quiet -of_objects $vio -filter {TYPE == vio_output}]
    set input_probes [get_hw_probes -quiet -of_objects $vio -filter {TYPE == vio_input}]
    if {[llength $output_probes] != 1 || [llength $input_probes] != 2} {
        error "Unexpected VIO probe layout: outputs=[llength $output_probes] inputs=[llength $input_probes]"
    }

    set challenge_probe [lindex $output_probes 0]
    set response_probe ""
    set heartbeat_probe ""
    foreach probe $input_probes {
        set probe_name [get_property NAME $probe]
        puts "PL_JTAG_BOARD_INPUT_PROBE=$probe_name"
        if {[string match "*response" $probe_name]} {
            set response_probe $probe
        } elseif {[string match "*heartbeat" $probe_name]} {
            set heartbeat_probe $probe
        }
    }
    if {$response_probe eq "" || $heartbeat_probe eq ""} {
        error "Could not identify response and heartbeat probes"
    }

    # Vivado VIO expects exactly eight hexadecimal characters for a 32-bit
    # probe when OUTPUT_VALUE_RADIX is HEX (no 0x prefix).
    set challenges {11223344 C001D00D}
    foreach challenge $challenges {
        set_property OUTPUT_VALUE $challenge $challenge_probe
        commit_hw_vio $vio
        after 100
        refresh_hw_vio $vio
        set response_text [get_property INPUT_VALUE $response_probe]
        set expected [expr {[parse_hex $challenge] ^ 0xA5C35A7E}]
        set actual [parse_hex $response_text]
        puts [format "PL_JTAG_BOARD_VECTOR challenge=0x%08X response=0x%08X expected=0x%08X" \
            [parse_hex $challenge] $actual $expected]
        if {$actual != $expected} {
            error [format "Challenge-response mismatch: got 0x%08X expected 0x%08X" $actual $expected]
        }
    }

    refresh_hw_vio $vio
    set heartbeat_before [parse_hex [get_property INPUT_VALUE $heartbeat_probe]]
    after 250
    refresh_hw_vio $vio
    set heartbeat_after [parse_hex [get_property INPUT_VALUE $heartbeat_probe]]
    puts [format "PL_JTAG_BOARD_HEARTBEAT before=0x%08X after=0x%08X" \
        $heartbeat_before $heartbeat_after]
    if {$heartbeat_before == $heartbeat_after} {
        error "PL heartbeat did not advance"
    }

    puts "PL_JTAG_BOARD_RESULT=PASS"
} message options]} {
    puts stderr "PL_JTAG_BOARD_RESULT=FAIL"
    puts stderr "PL_JTAG_BOARD_ERROR=$message"
    set rc 1
}

catch {close_hw_target}
catch {disconnect_hw_server}
catch {close_hw_manager}
exit $rc
