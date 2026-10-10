set mode "all"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {csim csynth cosim all}} {
        set mode $requested
    }
}

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir

open_project -reset build/q8q8_gemm_prj
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
} elseif {$mode eq "all"} {
    csim_design
    csynth_design
    cosim_design -rtl verilog -argv "--rtl-smoke"
} else {
    error "unknown mode '$mode': expected csim, csynth, cosim, or all"
}

exit

