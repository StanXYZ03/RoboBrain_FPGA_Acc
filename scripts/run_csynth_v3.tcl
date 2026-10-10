# run_csynth_v3.tcl - controlled Vitis HLS C-synthesis only (v3 packed kernel).
#
# Same flow as scripts/run_csynth_v2.tcl: ONLY `csynth_design`.  No C/RTL
# co-simulation, no IP export, no bitstream, no board access.  The synthesis
# top is the 128-bit packed token datapath `token_selector_top_packed`; the
# part, 5.0 ns clock and default (1.35 ns) uncertainty are unchanged.  All
# project and intermediate files go under /tmp/robobrain_token_selector_hls_csynth_v3
# and only the small reports are copied back into the repository.
#
# Reproduce (run from a scratch directory so tool logs also stay in /tmp):
#   mkdir -p /tmp/robobrain_selector_csynth_run_v3
#   cd /tmp/robobrain_selector_csynth_run_v3
#   export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
#   /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
#       /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/run_csynth_v3.tcl

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ..]]
set src_dir    [file join $repo_root fpga_accel token_selector_hls]
set src_file   [file join $src_dir src token_selector_hls.cpp]
set inc_dir    [file join $src_dir include]

set project_dir   /tmp/robobrain_token_selector_hls_csynth_v3
set solution_name solution1

puts "INFO: run_csynth_v3.tcl repo_root=$repo_root"
puts "INFO: run_csynth_v3.tcl project_dir=$project_dir"

open_project $project_dir
set_top token_selector_top_packed
add_files -cflags "-I$inc_dir" $src_file
open_solution $solution_name
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

csynth_design

puts "INFO: run_csynth_v3.tcl csynth_design completed"
exit
