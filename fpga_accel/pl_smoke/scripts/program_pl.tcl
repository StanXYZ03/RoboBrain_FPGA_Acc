set script_dir [file dirname [file normalize [info script]]]
set smoke_dir [file dirname $script_dir]
set bitstream [file normalize "$smoke_dir/artifacts/pl_uart_smoke.bit"]

if {![file exists $bitstream]} {
    error "PL smoke-test bitstream is missing: $bitstream"
}

connect -url tcp:127.0.0.1:3121
after 500
targets -set -filter {name == "PL"}

puts "PL_PROGRAM_STAGE=DOWNLOAD"
fpga -file $bitstream
after 1000
puts "PL_PROGRAM_BITSTREAM=$bitstream"
puts "PL_PROGRAM_RESULT=PASS"

disconnect
exit 0
