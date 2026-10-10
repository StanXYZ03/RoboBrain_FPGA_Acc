set mode "all"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {csim csynth cosim export_impl all}} {
        set mode $requested
    }
}

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir

open_project -reset build/q8q8_dataflow_prj
set_top q8q8_gemm
add_files src/q8q8_gemm.cpp -cflags "-std=c++14 -Iinclude"
add_files -tb tb/tb_q8q8_gemm.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

if {$mode eq "csim"} {
    csim_design
} elseif {$mode eq "csynth"} {
    csim_design
    csynth_design
} elseif {$mode eq "cosim"} {
    csim_design
    csynth_design
    cosim_design -rtl verilog -argv "--rtl-smoke"
} elseif {$mode eq "export_impl"} {
    csim_design
    csynth_design
    export_design -rtl verilog -format ip_catalog -flow impl \
            -vendor robobrain.local -library hls -ipname q8q8_dataflow \
            -version 1.0 -display_name "RoboBrain Q8Q8 DATAFLOW" \
            -description "Q8_0 x Q8_0 DATAFLOW GEMM for ZCU102" \
            -output [file join $root_dir artifacts q8q8_dataflow_ip.zip]
} elseif {$mode eq "all"} {
    csim_design
    csynth_design
    cosim_design -rtl verilog -argv "--rtl-smoke"
} else {
    error "unknown mode '$mode': expected csim, csynth, cosim, export_impl, or all"
}

exit
