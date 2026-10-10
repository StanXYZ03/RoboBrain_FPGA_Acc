set mode "csim"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {csim csynth cosim export export_impl}} {
        set mode $requested
    }
}
set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir
open_project -reset build/q8q8_gemv_stream6_prj
set_top q8q8_gemv_stream6
add_files src/q8q8_gemv_stream6.cpp -cflags "-std=c++14 -Iinclude"
add_files -tb tb/tb_q8q8_gemv_stream6.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default
csim_design
if {$mode in {csynth cosim export export_impl}} {
    csynth_design
}
if {$mode eq "cosim"} {
    cosim_design -rtl verilog
}
if {$mode eq "export"} {
    export_design -format ip_catalog -rtl verilog
}
if {$mode eq "export_impl"} {
    export_design -rtl verilog -format ip_catalog -flow impl \
            -vendor robobrain.local -library hls \
            -ipname q8q8_gemv_stream6 -version 1.0 \
            -display_name "RoboBrain Q8Q8 Decoder GEMV Six Stream" \
            -description "Six-stream Q8_0 batch-one Decoder GEMV candidate for ZCU102" \
            -output [file join $root_dir artifacts q8q8_gemv_stream6_ip.zip]
}
exit
