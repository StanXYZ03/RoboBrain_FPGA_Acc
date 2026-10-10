# run_vivado_ooc.tcl - Vivado 2023.2 out-of-context synthesis + implementation of
# the exported compact integer Selector IP.
#
# Inputs (all produced by run_hls_export.tcl, all under /tmp):
#   * exported IP RTL           : $ISPC_IP_REPO_DIR/integer_selector_pipeline_compact/hdl/verilog/*.v
#   * exported HLS OOC XDC      : $ISPC_IP_REPO_DIR/integer_selector_pipeline_compact/constraints/*_ooc.xdc
#                                 (create_clock -name ap_clk -period 5.000)
#
# Flow (explicit, out-of-context):
#   synth_design -mode out_of_context -> opt_design -> place_design ->
#   phys_opt_design (legal on UltraScale+) -> route_design
#
# No bitstream is written and the board is never touched.  All large files stay
# under /tmp; only small text reports are copied back by scripts/run_ooc.sh.

set script_dir [file normalize [file dirname [info script]]]
set ooc_dir    [file normalize [file join $script_dir ".."]]

set ip_repo_dir "/tmp/robobrain_integer_selector_compact_ooc_ip"
if {[info exists ::env(ISPC_IP_REPO_DIR)]} {
    set ip_repo_dir $::env(ISPC_IP_REPO_DIR)
}
set ip_name "integer_selector_pipeline_compact"
set part    "xczu9eg-ffvb1156-2-e"
set top     "integer_selector_pipeline_compact"

set work_dir "/tmp/robobrain_integer_selector_compact_ooc_vivado"
if {[info exists ::env(ISPC_VIVADO_WORK)]} {
    set work_dir $::env(ISPC_VIVADO_WORK)
}
set report_dir [file join $work_dir reports]

file mkdir $work_dir
file mkdir $report_dir
set status_file [file join $work_dir ooc_stage_status.tsv]
set fh [open $status_file w]
puts $fh "stage\trc\telapsed_s\terror"
close $fh

proc mark {name rc t0 err} {
    global status_file
    set dt [expr {[clock seconds] - $t0}]
    set fh [open $status_file a]
    puts $fh "$name\t$rc\t$dt\t[regsub -all {\n} $err { }]"
    close $fh
    puts "STAGE $name rc=$rc elapsed_s=$dt $err"
}

set ip_dir [file join $ip_repo_dir $ip_name]
if {[info exists ::env(ISPC_IP_DIR)] && $::env(ISPC_IP_DIR) ne "" && [file exists [file join $::env(ISPC_IP_DIR) component.xml]]} {
    set ip_dir $::env(ISPC_IP_DIR)
} elseif {![file exists [file join $ip_dir hdl verilog]]} {
    # export_design -output <dir> may place the component directly in <dir>.
    if {[file exists [file join $ip_repo_dir hdl verilog]]} {
        set ip_dir $ip_repo_dir
    }
}
if {![file isdirectory $ip_dir]} {
    error "exported IP dir not found: $ip_dir"
}

set rtl_files [lsort [glob -nocomplain [file join $ip_dir hdl verilog *.v]]]
set ip_rtl_files [lsort [glob -nocomplain [file join $ip_dir hdl ip *.v]]]
set xdc_files [lsort [glob -nocomplain [file join $ip_dir constraints *.xdc]]]
if {[llength $rtl_files] == 0} {
    error "no exported Verilog RTL under $ip_dir/hdl/verilog"
}
if {[llength $xdc_files] == 0} {
    error "no exported OOC XDC under $ip_dir/constraints"
}
puts "INFO: rtl_files=[llength $rtl_files] ip_rtl_files=[llength $ip_rtl_files] xdc_files=[llength $xdc_files]"
foreach f $xdc_files { puts "INFO: xdc=$f" }

set proj_dir [file join $work_dir project]
create_project -force ooc $proj_dir -part $part
set_property target_language Verilog [current_project]
foreach f $rtl_files { read_verilog $f }
foreach f $ip_rtl_files { read_verilog $f }
foreach f $xdc_files { read_xdc $f }

# Raw timestamp of the frozen constraint file (clock/uncertainty documentation).
set cfh [open [file join $report_dir clock_constraints.txt] w]
puts $cfh "part = $part"
puts $cfh "top  = $top"
foreach f $xdc_files {
    puts $cfh "---- $f ----"
    set fh2 [open $f r]
    puts $cfh [read $fh2]
    close $fh2
}
close $cfh

# The exported HLS OOC XDC is read before synthesis so Vivado applies it during
# `synth_design` (in Non-Project mode the constraints are queued and evaluated
# once the design is elaborated).  The ap_clk check therefore runs after synth.
set t0 [clock seconds]
set rc [catch { synth_design -top $top -part $part -mode out_of_context } err]
mark "synth_design" $rc $t0 $err
if {$rc} { error "synth_design failed: $err" }

# Sanity: the exported HLS OOC XDC must yield ap_clk = 5.000 ns after synth.
set clk_ok 0
foreach c [get_clocks -quiet] {
    if {[get_property NAME $c] eq "ap_clk" && abs([get_property PERIOD $c] - 5.0) < 1e-6} {
        set clk_ok 1
    }
}
if {!$clk_ok} {
    error "ap_clk 5.000 ns clock constraint not present after synthesis"
}
catch { report_clocks -file [file join $report_dir clocks_post_synth.rpt] }

write_checkpoint -force [file join $work_dir post_synth.dcp]
report_timing_summary -file [file join $report_dir timing_summary_post_synth.rpt] -delay_type min_max -report_unconstrained -max_paths 10
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $report_dir utilization_hier_post_synth.rpt]
report_utilization -file [file join $report_dir utilization_post_synth.rpt]
report_clock_utilization -file [file join $report_dir clock_utilization_post_synth.rpt]

set t0 [clock seconds]
set rc [catch { opt_design } err]
mark "opt_design" $rc $t0 $err
if {$rc} { error "opt_design failed: $err" }
write_checkpoint -force [file join $work_dir post_opt.dcp]

set t0 [clock seconds]
set rc [catch { place_design } err]
mark "place_design" $rc $t0 $err
if {$rc} { error "place_design failed: $err" }
write_checkpoint -force [file join $work_dir post_place.dcp]
report_timing_summary -file [file join $report_dir timing_summary_post_place.rpt] -delay_type min_max -max_paths 10
report_utilization -file [file join $report_dir utilization_post_place.rpt]

set t0 [clock seconds]
set rc [catch { phys_opt_design } err]
mark "phys_opt_design" $rc $t0 $err
if {$rc} { puts "WARNING: phys_opt_design failed (continuing): $err" }

set t0 [clock seconds]
set rc [catch { route_design } err]
mark "route_design" $rc $t0 $err
if {$rc} { puts "ERROR: route_design failed: $err" }
write_checkpoint -force [file join $work_dir post_route.dcp]

# Reports are emitted even if route_design failed, so the evidence is archived.
catch { report_timing_summary -file [file join $report_dir timing_summary_post_route.rpt] -delay_type min_max -report_unconstrained -max_paths 10 }
catch { report_timing -delay_type min_max -max_paths 5 -file [file join $report_dir critical_paths_post_route.rpt] }
catch { report_utilization -hierarchical -hierarchical_depth 3 -file [file join $report_dir utilization_hier_post_route.rpt] }
catch { report_utilization -file [file join $report_dir utilization_post_route.rpt] }
catch { report_route_status -file [file join $report_dir route_status_post_route.rpt] }
catch { report_drc -file [file join $report_dir drc_post_route.rpt] }
catch { report_clock_utilization -file [file join $report_dir clock_utilization_post_route.rpt] }
catch { report_clock_interaction -file [file join $report_dir clock_interaction_post_route.rpt] }
catch { report_high_fanout_nets -file [file join $report_dir high_fanout_nets_post_route.rpt] }
catch { report_qor_suggestions -file [file join $report_dir qor_suggestions_post_route.rpt] }
catch { write_qor_suggestions -force [file join $work_dir qor_suggestions_post_route.rqs] }
catch { report_design_analysis -congestion -file [file join $report_dir design_analysis_congestion_post_route.rpt] }
catch { check_timing -file [file join $report_dir check_timing_post_route.rpt] }

puts "INFO: run_vivado_ooc.tcl finished; route_rc=$rc"
exit
