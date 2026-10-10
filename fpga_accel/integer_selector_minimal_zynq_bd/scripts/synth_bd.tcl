# synth_bd.tcl - bounded synthesis (NO implementation, NO bitstream) of the
# generated minimal ZCU102 block-design HDL wrapper.
#
# Environment:
#   ISPC_BD_WORK     Vivado scratch root (project assumed already built)
#   ISPC_BD_REPORTS  directory for bounded text synthesis reports
#
# Reads the project produced by build_bd.tcl and runs the normal Vivado
# synthesis run (`synth_1`) so that the block-design IP out-of-context synthesis
# is orchestrated correctly.  It never launches implementation and never writes
# a bitstream.

set work_dir "/tmp/robobrain_integer_selector_bd_vivado"
if {[info exists ::env(ISPC_BD_WORK)] && $::env(ISPC_BD_WORK) ne ""} { set work_dir $::env(ISPC_BD_WORK) }
set rpt_dir [file join $work_dir reports]
if {[info exists ::env(ISPC_BD_REPORTS)] && $::env(ISPC_BD_REPORTS) ne ""} { set rpt_dir $::env(ISPC_BD_REPORTS) }
file mkdir $rpt_dir

set design_name "integer_selector_minimal_zynq_bd"
set proj_dir    [file join $work_dir project]
set synth_run   "synth_1"

set status_file [file join $work_dir synth_stage_status.tsv]
set fh [open $status_file w]
puts $fh "stage\trc\telapsed_s\terror"
close $fh
proc mark {name rc t0 err} {
    global status_file
    set dt [expr {[clock seconds] - $t0}]
    set fh [open $status_file a]
    puts $fh "$name\t$rc\t$dt\t[regsub -all {\n} $err { }]"
    close $fh
    puts "SYNTHSTAGE $name rc=$rc elapsed_s=$dt $err"
}

puts "=== synth_bd.tcl start: [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] ==="

set t0 [clock seconds]
set rc [catch { open_project [file join $proj_dir ${design_name}.xpr] } err]
mark "open_project" $rc $t0 $err
if {$rc} { puts "FATAL: cannot open project $proj_dir"; exit 2 }

set top ""
catch {set top [get_property top [current_fileset]]}
puts "INFO: current_fileset top = $top"

# Ensure the standard synthesis run exists.
if {[llength [get_runs -quiet $synth_run]] == 0} {
    set t0 [clock seconds]
    set rc [catch { create_run -flow "Vivado Synthesis 2023" $synth_run } err]
    mark "create_run" $rc $t0 $err
}
catch { reset_run $synth_run }

set t0 [clock seconds]
set rc [catch { launch_runs $synth_run -jobs 8 } err]
mark "launch_runs" $rc $t0 $err
if {$rc == 0} {
    set t0 [clock seconds]
    set rc [catch { wait_on_run $synth_run } err]
    mark "wait_on_run" $rc $t0 $err
}

set status ""; set progress ""
catch {set status [get_property STATUS [get_runs $synth_run]]}
catch {set progress [get_property PROGRESS [get_runs $synth_run]]}
puts "INFO: synth run status='$status' progress='$progress'"

set t0 [clock seconds]
if {[string match "*Complete*" $status]} {
    set rc [catch { open_run $synth_run } err]
    mark "synth_design" $rc $t0 $err
} else {
    set rc 1
    mark "synth_design" 1 $t0 "run status='$status' progress='$progress'"
}

if {$rc == 0} {
    catch { write_checkpoint -force [file join $work_dir post_synth_bd.dcp] }
    catch { report_utilization -file [file join $rpt_dir utilization_post_synth_bd.rpt] }
    catch { report_utilization -hierarchical -hierarchical_depth 3 -file [file join $rpt_dir utilization_hier_post_synth_bd.rpt] }
    catch { report_timing_summary -delay_type min_max -report_unconstrained -max_paths 10 -file [file join $rpt_dir timing_summary_post_synth_bd.rpt] }
    catch { report_clocks -file [file join $rpt_dir clocks_post_synth_bd.rpt] }
    catch { report_drc -file [file join $rpt_dir drc_post_synth_bd.rpt] }
}
file copy -force $status_file [file join $rpt_dir synth_stage_status.tsv]
puts "=== synth_bd.tcl finished: [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] rc=$rc ==="
exit 0
