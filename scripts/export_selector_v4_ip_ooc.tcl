# export_selector_v4_ip_ooc.tcl - export the packed v4 token selector as a
# Verilog IP catalog and run the Vivado out-of-context (OOC) implementation
# (synth + opt + place + route) that Vitis HLS 2023.2 generates for `-flow impl`.
#
# This mirrors the already-successful HLS-IP OOC route flow in
# `fpga_accel/q8q8_pe16_hls/scripts/run_hls.tcl` (`export_design ... -flow impl`),
# which produced the routed reports in
# `fpga_accel/q8q8_pe16_hls/build/q8q8_pe16_prj/solution1/impl/verilog/`.
#
# Frozen inputs / gates (unchanged by this script):
#   * top          : token_selector_top_packed
#   * part         : xczu9eg-ffvb1156-2-e
#   * clock        : create_clock -period 5.0 -name default
#   * uncertainty  : Vitis HLS default (do NOT pass -uncertainty); the default
#                    flow_target for the solution is 'vivado'.
# The clock is never relaxed: the exported OOC XDC keeps ap_clk = 5.000 ns.
#
# Everything large stays under /tmp:
#   * HLS project / Vivado project : /tmp/robobrain_token_selector_hls_export_v4
#   * exported Verilog IP zip      : /tmp/robobrain_selector_ip_ooc_v4/ip/
# A collector script (`scripts/summarize_selector_ooc.py`) copies only the small
# text/JSON reports back into the repository.
#
# Reproduce (from a scratch directory so the tool logs also stay in /tmp):
#   mkdir -p /tmp/robobrain_selector_export_run_v4
#   cd /tmp/robobrain_selector_export_run_v4
#   export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH
#   timeout 21600 /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f \
#       /home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102/scripts/export_selector_v4_ip_ooc.tcl

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ..]]
set src_dir    [file join $repo_root fpga_accel token_selector_hls]
set src_file   [file join $src_dir src token_selector_hls.cpp]
set inc_dir    [file join $src_dir include]

set project_dir  /tmp/robobrain_token_selector_hls_export_v4
set solution_name solution1
set ip_dir       /tmp/robobrain_selector_ip_ooc_v4/ip
set ip_zip       [file join $ip_dir token_selector_top_packed_ip.zip]

file mkdir $ip_dir

puts "INFO: export_selector_v4_ip_ooc.tcl repo_root=$repo_root"
puts "INFO: export_selector_v4_ip_ooc.tcl project_dir=$project_dir"
puts "INFO: export_selector_v4_ip_ooc.tcl ip_zip=$ip_zip"

open_project -reset $project_dir
set_top token_selector_top_packed
add_files -cflags "-I$inc_dir" $src_file
open_solution -reset $solution_name
set_part xczu9eg-ffvb1156-2-e
create_clock -period 5.0 -name default

# 1) C synthesis (pre-place estimate; same frozen setup as scripts/run_csynth_v4.tcl).
csynth_design

# 2) Export the Verilog IP catalog and run the Vivado OOC implementation
#    (synth_design -mode out_of_context + opt_design + place_design + route_design)
#    that HLS generates for `-flow impl`.
export_design -rtl verilog -format ip_catalog -flow impl \
    -vendor robobrain.local -library hls \
    -ipname token_selector_top_packed -version 1.0 \
    -display_name "RoboBrain2.5 token_selector_top_packed (v4)" \
    -description "Packed 128-bit fixed-point hardware-regular token selector v4 (multi-pass max-select); selection maths, host 5-segment layout and default HLS uncertainty unchanged" \
    -output $ip_zip

puts "INFO: export_selector_v4_ip_ooc.tcl export_design completed"
puts "INFO: export_selector_v4_ip_ooc.tcl ip_zip=$ip_zip"
exit
