set mode "csynth"
if {$argc > 0} {
    set requested [lindex $argv end]
    if {$requested in {csim csynth export_impl}} {
        set mode $requested
    }
}
set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir
open_project -reset build/pl_resident_ffn_vector_prj
set_top ffn_vector
add_files src/ffn_vector.cpp -cflags "-std=c++14 -Iinclude"
add_files -tb tb/tb_ffn_vector.cpp -cflags "-std=c++14 -Iinclude"
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

if {$mode eq "csim"} {
    csim_design
} elseif {$mode eq "csynth"} {
    csim_design
    csynth_design
} elseif {$mode eq "export_impl"} {
    csim_design
    csynth_design
    export_design -rtl verilog -format ip_catalog -flow impl \
        -vendor robobrain.local -library hls \
        -ipname pl_resident_ffn_vector -version 1.0 \
        -display_name "RoboBrain PL Resident FFN Vector Engine" \
        -description "Unified Norm activation residual and Q8 requantization engine" \
        -output [file join $root_dir artifacts pl_resident_ffn_vector_ip.zip]
} else {
    error "unknown mode '$mode'"
}
exit
