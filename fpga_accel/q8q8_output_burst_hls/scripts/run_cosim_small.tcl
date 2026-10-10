set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
cd $root_dir

open_project -reset build/q8q8_output_burst_cosim_prj
set_top q8q8_gemm
set small_flags "-std=c++14 -Iinclude -DQ8Q8_MAX_K_VALUE=64"
add_files src/q8q8_gemm.cpp -cflags $small_flags
add_files -tb tb/tb_q8q8_gemm.cpp -cflags $small_flags
open_solution -reset solution1 -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.0 -name default

# The deployment project is synthesized separately with MAX_K=9728.  This
# test-only project reduces only cache depth so XSim can exercise the complete
# AXI, compute and 128-bit output path on a 32 GiB host.
csim_design -argv "--rtl-smoke"
csynth_design
cosim_design -rtl verilog -argv "--rtl-smoke"
exit
