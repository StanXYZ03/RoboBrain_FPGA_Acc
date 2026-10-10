set script_dir [file dirname [file normalize [info script]]]
set init_file [file normalize "$script_dir/../artifacts/ddr_smoke/psu_init.tcl"]

connect -url tcp:127.0.0.1:3121
after 1000
targets -set -filter {name =~ "PSU"}
puts "DDR_SMOKE_STAGE=RESET"
rst -system
after 1000

puts "DDR_SMOKE_STAGE=PS_INIT"
source $init_file
if {[catch {psu_init} message options]} {
    puts "DDR_SMOKE_RESULT=FAIL"
    puts "DDR_SMOKE_ERROR=PS_INIT: $message"
    disconnect
    exit 1
}
if {[catch {psu_post_config} message options]} {
    puts "DDR_SMOKE_RESULT=FAIL"
    puts "DDR_SMOKE_ERROR=PS_POST_CONFIG: $message"
    disconnect
    exit 1
}
psu_ps_pl_isolation_removal
psu_ps_pl_reset_config

after 500
# Keep the PSU/DAP selected. Forced memory accesses do not require an A53 core
# to be released from reset, which makes this a smaller DDR-only test.
targets -set -filter {name =~ "PSU"}

# ZynqMP maps the first 2 GiB below 0x80000000 and the remainder in the
# high DDR window beginning at 0x8_0000_0000.
set tests {
    {0x0000000010000000 0x13579BDF}
    {0x0000000050000000 0x2468ACE0}
    {0x0000000800000000 0xA5A55A5A}
    {0x0000000840000000 0x5A5AA5A5}
}

puts "DDR_SMOKE_STAGE=WRITE"
foreach test $tests {
    lassign $test address pattern
    mwr -force $address $pattern
}

puts "DDR_SMOKE_STAGE=READ_VERIFY"
set failures 0
foreach test $tests {
    lassign $test address expected
    set actual [mrd -force -value $address]
    puts [format "DDR_SMOKE_CHECK address=%s expected=0x%08X actual=0x%08X" \
        $address $expected $actual]
    if {$actual != $expected} {
        incr failures
    }
}

# Re-read after every location has been written to catch address aliasing.
puts "DDR_SMOKE_STAGE=ALIAS_VERIFY"
foreach test $tests {
    lassign $test address expected
    set actual [mrd -force -value $address]
    if {$actual != $expected} {
        puts [format "DDR_SMOKE_ALIAS_FAIL address=%s expected=0x%08X actual=0x%08X" \
            $address $expected $actual]
        incr failures
    }
}

if {$failures == 0} {
    puts "DDR_SMOKE_RESULT=PASS"
} else {
    puts "DDR_SMOKE_RESULT=FAIL"
    puts "DDR_SMOKE_FAILURES=$failures"
}
disconnect
exit $failures
