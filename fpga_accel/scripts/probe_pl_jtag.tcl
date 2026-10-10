set rc 0

if {[catch {
    open_hw_manager
    connect_hw_server -url localhost:3121
    open_hw_target

    set devices [get_hw_devices]
    puts "PL_PROBE_DEVICE_COUNT=[llength $devices]"

    foreach device $devices {
        puts "PL_PROBE_DEVICE=$device"
        foreach property {PART REGISTER.IR REGISTER.STATUS.BIT00_DONE REGISTER.STATUS.BIT01_INIT REGISTER.STATUS.BIT03_SECURITY REGISTER.STATUS.BIT05_GTS_CFG_B REGISTER.STATUS.BIT06_GWE REGISTER.STATUS.BIT07_GHIGH_B} {
            if {![catch {set value [get_property $property $device]}]} {
                puts "PL_PROBE_${property}=$value"
            }
        }
    }

    close_hw_target
    disconnect_hw_server
    close_hw_manager
} message]} {
    puts stderr "PL_PROBE_ERROR=$message"
    set rc 1
}

exit $rc
