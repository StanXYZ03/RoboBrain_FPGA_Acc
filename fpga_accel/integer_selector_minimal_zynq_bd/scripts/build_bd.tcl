# build_bd.tcl - Vivado 2023.2 minimal ZCU102 Zynq UltraScale+ MPSoC block design
# for the accepted compact integer Selector IP
#   VLNV = robobrain.local:hls:integer_selector_pipeline_compact:1.0
#
# Intended PS-DDR / PL integration:
#   * Zynq UltraScale+ MPSoC PS owns control + DDR.
#   * PS master  M_AXI_HPM0_FPD (128-bit AXI4) -> AXI SmartConnect -> Selector
#     AXI4-Lite slave `s_axi_control`.
#   * Selector AXI4 masters `input_arena`, `wv_q`, `output_arena` (128-bit)
#     -> 3-input / 1-output AXI SmartConnect -> PS 128-bit HP slave
#     `S_AXI_HP0_FPD` (one enabled PS high-performance DDR port).  The three
#     masters are NOT merged at the IP boundary.
#   * PL fabric clock from PS `pl_clk0` @ 200 MHz; `proc_sys_reset` provides the
#     active-low peripheral reset.
#
# No bitstream is written and the board is never touched.  All large files stay
# under /tmp; only bounded text evidence is copied back by scripts/run_bd.sh.
#
# Environment overrides:
#   ISPC_BD_IP_REPO  IP repository root (contains the unpacked accepted IP)
#   ISPC_BD_WORK     Vivado scratch root (project + reports live here)
#   ISPC_BD_REPORTS  directory for the bounded text evidence dumps

set ip_repo "/tmp/robobrain_integer_selector_bd_ip"
if {[info exists ::env(ISPC_BD_IP_REPO)] && $::env(ISPC_BD_IP_REPO) ne ""} { set ip_repo $::env(ISPC_BD_IP_REPO) }
set work_dir "/tmp/robobrain_integer_selector_bd_vivado"
if {[info exists ::env(ISPC_BD_WORK)] && $::env(ISPC_BD_WORK) ne ""} { set work_dir $::env(ISPC_BD_WORK) }
set rpt_dir [file join $work_dir reports]
if {[info exists ::env(ISPC_BD_REPORTS)] && $::env(ISPC_BD_REPORTS) ne ""} { set rpt_dir $::env(ISPC_BD_REPORTS) }

set part         "xczu9eg-ffvb1156-2-e"
set board_part   "xilinx.com:zcu102:part0:3.4"
set selector_vlnv "robobrain.local:hls:integer_selector_pipeline_compact:1.0"
set design_name  "integer_selector_minimal_zynq_bd"
set ps_vlnv      "xilinx.com:ip:zynq_ultra_ps_e:3.5"

# ISPC_BD_QUICK=1 stops after validate/address/save (no generate_target, no
# wrapper) for fast connectivity debugging.
set quick 0
if {[info exists ::env(ISPC_BD_QUICK)] && $::env(ISPC_BD_QUICK) ne "" && $::env(ISPC_BD_QUICK) != 0} { set quick 1 }

file mkdir $work_dir
file mkdir $rpt_dir

set status_file [file join $work_dir bd_stage_status.tsv]
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
proc dump_all_props {obj fh {sep "\t"}} {
    foreach p [lsort [list_property $obj]] {
        set v ""
        catch {set v [get_property $p $obj]}
        puts $fh "$obj${sep}$p${sep}$v"
    }
}

puts "=== build_bd.tcl start: [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] ==="
puts "vivado version = [version -short]"

########################################################################
# Project + IP repository
########################################################################
set proj_dir [file join $work_dir project]
set t0 [clock seconds]
set rc [catch {
    create_project -force $design_name $proj_dir -part $part
    set_property target_language Verilog [current_project]
    set_property board_part $board_part [current_project]
    set_property ip_repo_paths $ip_repo [current_project]
    update_ip_catalog -rebuild
} err]
mark "create_project_ip_repo" $rc $t0 $err
if {$rc} { puts "FATAL: project/IP repo stage failed"; exit 2 }

set ipdefs [get_ipdefs -all $selector_vlnv]
puts "INFO: selector ipdefs = $ipdefs"
if {[llength $ipdefs] == 0} { puts "FATAL: selector VLNV $selector_vlnv not found in catalog"; exit 3 }

set eff_part   ""
catch {set eff_part [get_property part [current_project]]}
if {$eff_part eq ""} { catch {set eff_part [get_property PART [current_project]]} }
set eff_board  [get_property board_part [current_project]]
puts "INFO: effective part = $eff_part"
puts "INFO: effective board_part = $eff_board"

########################################################################
# Block design
########################################################################
create_bd_design $design_name
set bd_file [get_files -quiet ${design_name}.bd]

# ---- PS -------------------------------------------------------------
set ps [create_bd_cell -type ip -vlnv $ps_vlnv zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {0} \
    CONFIG.PSU__USE__S_AXI_GP4 {0} \
    CONFIG.PSU__USE__S_AXI_GP5 {0} \
    CONFIG.PSU__USE__S_AXI_GP6 {0} \
    CONFIG.PSU__USE__S_AXI_ACP {0} \
    CONFIG.PSU__USE__S_AXI_ACE {0} \
    CONFIG.PSU__USE__IRQ0 {0} \
    CONFIG.PSU__USE__IRQ1 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE {0} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
] $ps

# ---- Selector IP ----------------------------------------------------
set sel [create_bd_cell -type ip -vlnv $selector_vlnv integer_selector_pipeline_compact_0]

# ---- AXI interconnects ---------------------------------------------
# 3 Sel m_axi masters -> 1 PS HP DDR slave (arbitrated, not merged)
set sc_ddr [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_ddr]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {1}] $sc_ddr
# 1 PS master -> 1 Sel AXI4-Lite control slave
set sc_ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_control]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {1}] $sc_ctl

# ---- Reset ----------------------------------------------------------
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
# NOTE: C_EXT_RESET_HIGH is read-only on this IP version inside the BD
# (CRITICAL WARNING [BD 41-737]); the standard Xilinx Zynq pattern drives
# ext_reset_in directly from the active-low PS output pl_resetn0, and the IP
# emits the active-low peripheral_aresetn.  The actual generic value is dumped
# as evidence after elaboration.

########################################################################
# Connectivity
########################################################################
set t0 [clock seconds]
set rc [catch {
    # --- data path: three Selector AXI4 masters -> SmartConnect -> PS HP0 ---
    connect_bd_intf_net [get_bd_intf_pins $sel/m_axi_input_arena]  [get_bd_intf_pins $sc_ddr/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sel/m_axi_wv_q]         [get_bd_intf_pins $sc_ddr/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sel/m_axi_output_arena] [get_bd_intf_pins $sc_ddr/S02_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc_ddr/M00_AXI]         [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

    # --- control path: PS master -> SmartConnect -> Selector control ---
    connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $sc_ctl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc_ctl/M00_AXI]     [get_bd_intf_pins $sel/s_axi_control]

    # --- clocks: single 200 MHz PL fabric clock -----------------------
    connect_bd_net [get_bd_pins $ps/pl_clk0] \
        [get_bd_pins $ps/maxihpm0_fpd_aclk] \
        [get_bd_pins $ps/saxihp0_fpd_aclk] \
        [get_bd_pins $sc_ddr/aclk] \
        [get_bd_pins $sc_ctl/aclk] \
        [get_bd_pins $rst/slowest_sync_clk] \
        [get_bd_pins $sel/ap_clk]

    # --- resets: active-low peripheral reset --------------------------
    connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
    connect_bd_net [get_bd_pins $rst/peripheral_aresetn] \
        [get_bd_pins $sc_ddr/aresetn] \
        [get_bd_pins $sc_ctl/aresetn] \
        [get_bd_pins $sel/ap_rst_n]
} err]
mark "connect_bd" $rc $t0 $err
if {$rc} { puts "WARNING: a connection failed: $err" }

# The Selector interrupt output is an OPTIONAL status output; the minimal
# design deliberately does not consume it (PS IRQ0/IRQ1 remain disabled), so no
# interrupt interface is required.  Record the fact explicitly.
set interrupt_used 0

########################################################################
# Address assignment
########################################################################
set t0 [clock seconds]
set rc [catch { assign_bd_address } err]
mark "assign_bd_address" $rc $t0 $err
if {$rc} { puts "WARNING: assign_bd_address: $err" }

set t0 [clock seconds]
set rc [catch { validate_bd_design } err]
mark "validate_bd_design" $rc $t0 $err
set validate_rc $rc

set t0 [clock seconds]
set rc [catch { save_bd_design } err]
mark "save_bd_design" $rc $t0 $err

########################################################################
# Bounded evidence dumps (text)
########################################################################
set info_fh [open [file join $rpt_dir bd_design_info.txt] w]
puts $info_fh "design_name = $design_name"
puts $info_fh "part = $eff_part"
puts $info_fh "board_part = $eff_board"
puts $info_fh "board_part_status = available"
puts $info_fh "selector_vlnv = $selector_vlnv"
puts $info_fh "ip_repo = $ip_repo"
puts $info_fh "vivado_version = [version -short]"
puts $info_fh "interrupt_used = $interrupt_used"
puts $info_fh "validate_bd_design_rc = $validate_rc"
close $info_fh

# --- cells ----------------------------------------------------------
set fh [open [file join $rpt_dir bd_cells.tsv] w]
puts $fh "cell\tvlnv\ttype"
foreach c [lsort [get_bd_cells]] {
    set vlnv ""; catch {set vlnv [get_property VLNV $c]}
    set type ""; catch {set type [get_property TYPE $c]}
    puts $fh "$c\t$vlnv\t$type"
}
close $fh

# --- interface pins -------------------------------------------------
# `get_bd_intf_pins` with no argument returns nothing in this version, so walk
# the cells (and top-level ports) explicitly.
set all_intf_pins {}
foreach c [lsort [get_bd_cells]] {
    foreach p [get_bd_intf_pins -quiet -of_objects $c] { lappend all_intf_pins $p }
}
foreach p [get_bd_intf_ports -quiet] { lappend all_intf_pins $p }
set fh [open [file join $rpt_dir bd_intf_pins.tsv] w]
puts $fh "pin\tmode\tprotocol\tdata_width\tfreq_hz"
foreach p [lsort -unique $all_intf_pins] {
    set mode ""; catch {set mode [get_property MODE $p]}
    set proto ""; catch {set proto [get_property CONFIG.PROTOCOL $p]}
    set dw ""; catch {set dw [get_property CONFIG.DATA_WIDTH $p]}
    set fq ""; catch {set fq [get_property CONFIG.FREQ_HZ $p]}
    puts $fh "$p\t$mode\t$proto\t$dw\t$fq"
}
close $fh

# --- interface nets -------------------------------------------------
set all_intf_nets {}
foreach p $all_intf_pins {
    foreach n [get_bd_intf_nets -quiet -of_objects $p] { lappend all_intf_nets $n }
}
set fh [open [file join $rpt_dir bd_intf_nets.tsv] w]
puts $fh "net\tmembers"
foreach n [lsort -unique $all_intf_nets] {
    set members ""
    catch {set members [get_bd_intf_pins -of_objects $n]}
    puts $fh "$n\t$members"
}
close $fh

# --- signal pins of interest (clock/reset) --------------------------
set all_sig_pins {}
foreach c [lsort [get_bd_cells]] {
    foreach p [get_bd_pins -quiet -of_objects $c] { lappend all_sig_pins $p }
}
foreach p [get_bd_ports -quiet] { lappend all_sig_pins $p }
set fh [open [file join $rpt_dir bd_signal_pins.tsv] w]
puts $fh "pin\tdir\ttype\tnet"
foreach p [lsort -unique $all_sig_pins] {
    set nm [get_property NAME $p]
    if {[string match "*clk*" $nm] || [string match "*reset*" $nm] || [string match "*rst*" $nm]} {
        set dir ""; catch {set dir [get_property DIR $p]}
        set typ ""; catch {set typ [get_property TYPE $p]}
        set net ""; catch {set net [get_bd_nets -of_objects $p]}
        puts $fh "$p\t$dir\t$typ\t$net"
    }
}
close $fh

# --- clock/reset nets -----------------------------------------------
set all_sig_nets {}
foreach p $all_sig_pins {
    foreach n [get_bd_nets -quiet -of_objects $p] { lappend all_sig_nets $n }
}
set fh [open [file join $rpt_dir bd_signal_nets.tsv] w]
puts $fh "net\tmembers"
foreach n [lsort -unique $all_sig_nets] {
    set nm [get_property NAME $n]
    if {[string match "*clk*" $nm] || [string match "*reset*" $nm] || [string match "*rst*" $nm]} {
        set members ""
        catch {set members [get_bd_pins -of_objects $n]}
        puts $fh "$n\t$members"
    }
}
close $fh

# --- address segments ----------------------------------------------
set fh [open [file join $rpt_dir bd_addr_segs.tsv] w]
puts $fh "segment\toffset\trange\tmaster_addrspace\tslave_addrspace"
foreach s [lsort [get_bd_addr_segs]] {
    set off ""; catch {set off [get_property OFFSET $s]}
    set rng ""; catch {set rng [get_property RANGE $s]}
    set mas ""; catch {set mas [get_property MASTER_ADDRSPACE $s]}
    set sas ""; catch {set sas [get_property SLAVE_ADDRSPACE $s]}
    puts $fh "$s\t$off\t$rng\t$mas\t$sas"
}
close $fh

# --- address spaces -------------------------------------------------
set fh [open [file join $rpt_dir bd_addr_spaces.tsv] w]
puts $fh "addrspace"
foreach a [lsort [get_bd_addr_spaces]] { puts $fh "$a" }
close $fh

# --- PS configuration ----------------------------------------------
set fh [open [file join $rpt_dir bd_ps_config.tsv] w]
puts $fh "param\tvalue"
foreach p [lsort [list_property $ps]] {
    if {[string match "CONFIG.PSU__USE__*" $p] || [string match "*PL0*" $p] ||
        [string match "*DATA_WIDTH*" $p] || [string match "*IRQ*" $p] ||
        [string match "*SAXIGP2*" $p] || [string match "*MAXIGP0*" $p]} {
        set v ""; catch {set v [get_property $p $ps]}
        puts $fh "$p\t$v"
    }
}
close $fh

# --- Selector cell params ------------------------------------------
set fh [open [file join $rpt_dir bd_selector_params.tsv] w]
puts $fh "param\tvalue"
foreach p [lsort [list_property $sel]] {
    set v ""; catch {set v [get_property $p $sel]}
    puts $fh "$p\t$v"
}
close $fh

# --- reset cell params ---------------------------------------------
set fh [open [file join $rpt_dir bd_reset_params.tsv] w]
puts $fh "cell\tparam\tvalue"
foreach c [lsort [get_bd_cells -quiet -filter {VLNV =~ "*proc_sys_reset*"}]] {
    foreach p [lsort [list_property $c]] {
        if {[string match "CONFIG.*" $p] || [string match "VLNV" $p]} {
            set v ""; catch {set v [get_property $p $c]}
            puts $fh "$c\t$p\t$v"
        }
    }
}
close $fh

# --- interconnect cell params --------------------------------------
set fh [open [file join $rpt_dir bd_interconnect_params.tsv] w]
puts $fh "cell\tparam\tvalue"
foreach c [lsort [get_bd_cells -quiet -filter {VLNV =~ "*smartconnect*"}]] {
    foreach p [lsort [list_property $c]] {
        if {[string match "CONFIG.*" $p] || [string match "VLNV" $p]} {
            set v ""; catch {set v [get_property $p $c]}
            puts $fh "$c\t$p\t$v"
        }
    }
}
close $fh

# --- stage status copy ---------------------------------------------
file copy -force $status_file [file join $rpt_dir bd_stage_status.tsv]

########################################################################
# Reproducible BD Tcl + generated targets + HDL wrapper
########################################################################
if {$quick} {
    puts "INFO: ISPC_BD_QUICK=1 -> skipping write_bd_tcl/generate_target/make_wrapper"
    file copy -force $status_file [file join $rpt_dir bd_stage_status.tsv]
    puts "=== build_bd.tcl finished (quick): [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] ==="
    exit 0
}

set t0 [clock seconds]
set rc [catch {
    write_bd_tcl -force [file join $rpt_dir ${design_name}.tcl]
} err]
mark "write_bd_tcl" $rc $t0 $err

set t0 [clock seconds]
set rc [catch { generate_target all $bd_file } err]
mark "generate_target_all" $rc $t0 $err

set t0 [clock seconds]
set wrapper_files {}
set rc [catch {
    set wrapper_files [make_wrapper -files $bd_file -top]
    foreach w $wrapper_files { add_files -norecurse $w }
    set_property top ${design_name}_wrapper [current_fileset]
    update_compile_order -fileset sources_1
} err]
mark "make_wrapper" $rc $t0 $err

set fh [open [file join $rpt_dir bd_wrapper.txt] w]
puts $fh "wrapper_files = $wrapper_files"
catch {puts $fh "top = [get_property top [current_fileset]]"}
close $fh

puts "=== build_bd.tcl finished: [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] ==="
puts "INFO: validate_rc=$validate_rc connect_rc=see status"
exit 0
