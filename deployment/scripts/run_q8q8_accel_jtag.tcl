# Unattended Q8Q8-accelerated JTAG RAM boot.  This resets and configures the
# board, but never writes SD, QSPI or any other nonvolatile storage.
set root [file normalize [file dirname [file dirname [info script]]]]
set project_root [file dirname $root]
set fpga_root [file normalize "$project_root/fpga_accel"]
if {[info exists ::env(Q8Q8_RAM_ARTIFACT_DIR)]} {
    set ram [file normalize $::env(Q8Q8_RAM_ARTIFACT_DIR)]
} else {
    set ram [file normalize "$root/artifacts/ram_linux"]
}
if {[info exists ::env(Q8Q8_ACCEL_ARTIFACT_DIR)]} {
    set accel [file normalize $::env(Q8Q8_ACCEL_ARTIFACT_DIR)]
} else {
    set accel [file normalize "$fpga_root/q8q8_zcu102_vivado/artifacts"]
}
if {[info exists ::env(Q8Q8_ACCEL_FSBL)]} {
    set fsbl [file normalize $::env(Q8Q8_ACCEL_FSBL)]
} else {
    set fsbl [file normalize "$accel/q8q8_fsbl.elf"]
}
if {[info exists ::env(Q8Q8_ACCEL_BITSTREAM)]} {
    set bitstream [file normalize $::env(Q8Q8_ACCEL_BITSTREAM)]
} else {
    set bitstream [file normalize "$accel/q8q8_zcu102.bit"]
}
if {[info exists ::env(Q8Q8_ACCEL_FSBL_METADATA)]} {
    set fsbl_metadata [file normalize $::env(Q8Q8_ACCEL_FSBL_METADATA)]
} else {
    set fsbl_metadata [file normalize "$accel/q8q8_fsbl_metadata.txt"]
}
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set pmufw "$platform_boot/pmufw.elf"
set atf "$platform_boot/bl31.elf"
set uboot "$platform_boot/u-boot.elf"

foreach file [list $fsbl $fsbl_metadata $bitstream $pmufw $atf $uboot "$ram/Image" "$ram/system-16gb.dtb" "$ram/rootfs.cpio.gz" "$ram/boot.scr"] {
    if {![file readable $file]} { error "missing JTAG boot artifact: $file" }
}
set metadata_fd [open $fsbl_metadata r]
set metadata_text [read $metadata_fd]
close $metadata_fd
if {![regexp {XFSBL_LOOP=(0x[[:xdigit:]]+)} $metadata_text -> fsbl_loop]} {
    error "cannot read XFsbl_Loop address from FSBL metadata"
}
if {![regexp {PS_PL_ISOLATION_REMOVAL=(0x[[:xdigit:]]+)} $metadata_text -> ps_pl_isolation_removal]} {
    error "cannot read PS-PL isolation-removal address from FSBL metadata"
}
if {![regexp {PS_PL_RESET_CONFIG=(0x[[:xdigit:]]+)} $metadata_text -> ps_pl_reset_config]} {
    error "cannot read PS-PL reset-config address from FSBL metadata"
}

proc call_fsbl_helper {entry return_address label} {
    bpadd -addr $return_address -type hw -temp
    rwr r30 $return_address
    rwr pc $entry
    if {[catch {con -block -timeout 20} helper_message]} {
        catch {stop}
        error "$label failed: $helper_message"
    }
    set helper_pc_line [rrd pc]
    if {![regexp {([[:xdigit:]]{16})} $helper_pc_line -> helper_pc_hex] ||
            [expr 0x$helper_pc_hex] != $return_address} {
        error "$label did not return to FSBL loop: $helper_pc_line"
    }
    puts "RAM_BOOT_$label=PASS"
}

connect -url tcp:127.0.0.1:3121
after 1000
# A previous media-mode fallback can leave DAP sticky while the physical SD
# boot path races to retake the PS.  Recover and immediately claim PSU in this
# same XSCT session so there is no reconnect window for that race.
if {[catch {targets -set -filter {name =~ "PSU"}}]} {
    targets -set 4
    rst -system
    after 1000
    rst -dap
    after 1500
    disconnect
    connect -url tcp:127.0.0.1:3121
    after 500
}
targets -set -filter {name =~ "PSU"}
puts "RAM_BOOT_STAGE=SYSTEM_RESET"
rst -system
after 1000
rst -dap
after 1000
# Release the ZynqMP JTAG/security gates after a system reset.  Without this,
# hw_server can enumerate PMU but reports an invalid context on download.
mwr 0xFFCA0038 0x1FF
after 500

# The release above materializes the downloadable MicroBlaze child below the
# PMU container; select that child, not the container itself.
targets -set -filter {name =~ "MicroBlaze PMU"}
dow $pmufw
con
after 500

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
puts "RAM_BOOT_STAGE=DYNAMIC_DDR_FSBL"
dow $fsbl
# Stop only after XFsbl_Exit has invalidated TLB/I-cache, synchronized memory,
# and disabled EL3 caches.  Stopping at the function entry leaves PMU/BL31
# shared state incoherent and makes BL31 panic in pm_ipi_send_sync().
bpadd -addr $fsbl_loop -type hw -temp
if {[catch {con -block -timeout 20} message]} {
    catch {stop}
    puts "RAM_BOOT_FSBL=$message"
}

# Validate the same SPD/rank evidence used by the passing 16GB bare-metal test.
set magic [mrd -force -value 0xFFFEF000]
set ranks [mrd -force -value 0xFFFEF004]
set rank_lo [mrd -force -value 0xFFFEF008]
set rank_hi [mrd -force -value 0xFFFEF00C]
set rank_bytes [expr {($rank_hi << 32) | $rank_lo}]
set total_bytes [expr {$rank_bytes * $ranks}]
puts [format "RAM_BOOT_DDR magic=0x%08X ranks=%u total_bytes=%s" $magic $ranks $total_bytes]
if {$magic != 0x53504434 || $ranks != 2 || $total_bytes != 17179869184} {
    disconnect
    error "16GB DDR initialization gate failed"
}

# Configure the PL only after the matching FSBL has initialized PS clocks,
# resets, DDR and PS-PL AXI ports.  Check the configuration port here; Linux
# later performs the decisive AXI-Lite ap_idle and numerical checks via UIO.
targets -set -filter {name =~ "PSU"}
puts "RAM_BOOT_STAGE=PROGRAM_Q8Q8_PL"
fpga -file $bitstream
after 500
set pl_state [fpga -state]
puts "RAM_BOOT_Q8Q8_PL state={$pl_state}"
if {$pl_state ne "FPGA is configured"} {
    disconnect
    error "Q8Q8 PL configuration-state gate failed"
}
puts "RAM_BOOT_Q8Q8_PL_GATE=PASS"

# FSBL_PARTITION_LOAD_EXCLUDE deliberately omits the media partition loader.
# The standard FSBL normally performs these generated PS-PL power/isolation
# and reset operations immediately after programming a bitstream, so invoke
# those same routines explicitly before any A53 or Linux AXI access.
targets -set -filter {name =~ "Cortex-A53 #0"}
call_fsbl_helper $ps_pl_isolation_removal $fsbl_loop "PS_PL_ISOLATION_REMOVAL"
call_fsbl_helper $ps_pl_reset_config $fsbl_loop "PS_PL_RESET_RELEASE"

# Decisive pre-Linux reachability gate.  ap_ctrl_hs bit 2 must report ap_idle
# after reset; a configured FPGA with an isolated/reset AXI fabric cannot pass.
set q8q8_control [mrd -force -value 0xA0000000]
puts [format "RAM_BOOT_Q8Q8_CONTROL value=0x%08X" $q8q8_control]
if {($q8q8_control & 0x4) == 0} {
    disconnect
    error "Q8Q8 AXI-Lite ap_idle gate failed"
}
puts "RAM_BOOT_Q8Q8_AXI_GATE=PASS"

# The remote board is physically strapped for SD (0xE).  Record the immutable
# latched value here; the pinned BL31 is patched in RAM below to select its
# preloaded-BL33 path without changing switches or any boot media.
targets -set -filter {name =~ "PSU"}
set physical_boot_mode [mrd -force -value 0xFF5E0200]
puts [format "RAM_BOOT_MODE physical=0x%08X" $physical_boot_mode]

targets -set -filter {name =~ "Cortex-A53 #0"}
# `dow` of an ELF updates the selected A53 PC.  Load BL33 first and ATF last
# so execution begins at BL31, which then hands off U-Boot at EL2.
dow $uboot
dow $atf
# The pinned U-Boot uses an external control-FDT at 0x00100000 on its JTAG
# path.  Load it before BL31/U-Boot starts; relying on stale DDR made earlier
# boots nondeterministic after a DAP reset.
dow -data "$ram/system-16gb.dtb" 0x00100000
if {[mrd -force -value 0x00100000] != 0xEDFE0DD0} {
    disconnect
    error "pre-U-Boot control-FDT read-back gate failed"
}
# U-Boot also reads the immutable SD strap and otherwise selects SD-specific
# runtime behavior.  Make its pinned bootmode helper return JTAG (0) in RAM.
set uboot_bootmode_0 [mrd -force -value 0x10084A48]
set uboot_bootmode_1 [mrd -force -value 0x10084A4C]
if {$uboot_bootmode_0 != 0xA9BE7BFD || $uboot_bootmode_1 != 0x52804000} {
    disconnect
    error [format "unexpected pinned U-Boot bootmode words: 0x%08X 0x%08X" \
        $uboot_bootmode_0 $uboot_bootmode_1]
}
mwr 0x10084A48 0x52800000
mwr 0x10084A4C 0xD65F03C0
if {[mrd -force -value 0x10084A48] != 0x52800000 ||
        [mrd -force -value 0x10084A4C] != 0xD65F03C0} {
    disconnect
    error "volatile U-Boot JTAG bootmode patch verification failed"
}

puts "RAM_BOOT_UBOOT_JTAG_PATCH=PASS"
# This exact 2023.2 BL31 uses `cbnz w0, xbl_handover` at 0xFFFEB0DC.
# With physical SD straps it expects a media partition table, which a manual
# JTAG load does not have.  NOP that branch only in RAM to use the adjacent
# preloaded U-Boot path.  Both the original and patched words are gated.
set bl31_mode_branch [mrd -force -value 0xFFFEB0DC]
if {$bl31_mode_branch != 0x350002E0} {
    disconnect
    error [format "unexpected pinned BL31 boot-mode instruction: 0x%08X" $bl31_mode_branch]
}
mwr 0xFFFEB0DC 0xD503201F
set bl31_mode_patch [mrd -force -value 0xFFFEB0DC]
if {$bl31_mode_patch != 0xD503201F} {
    disconnect
    error "volatile BL31 JTAG-path patch verification failed"
}
puts "RAM_BOOT_BL31_JTAG_PATCH=PASS"
# "JTAG" until Linux PID 1 replaces it with its first status code.
mwr 0x7FF00000 0x4A544147
puts "RAM_BOOT_STAGE=START_ATF_UBOOT"

# Board global_data reports relocaddr=0x7FC85000 and reloc_off=0x6FC05000.
# Break at relocated main_loop (link address 0x100ACBDC)
# before autoboot scans the empty JTAG script address.  PC polling is too slow:
# this build can reach DHCP in less than one 50 ms polling interval.
set relocated_main_loop 0x7FCB1BDC
bpadd -addr $relocated_main_loop -type hw -temp
if {[catch {con -block -timeout 20} uboot_message]} {
    catch {stop}
    disconnect
    error "U-Boot relocated main_loop breakpoint failed: $uboot_message"
}
set pc_line [rrd pc]
if {![regexp {([[:xdigit:]]{16})} $pc_line -> pc_hex] ||
        [expr 0x$pc_hex] != $relocated_main_loop} {
    disconnect
    error "U-Boot did not stop at the pinned relocated main_loop: $pc_line"
}
puts [format "RAM_BOOT_UBOOT_MAIN_LOOP_BREAK pc=0x%016X" [expr 0x$pc_hex]]

# Now that relocation/MMU setup is complete, call U-Boot's own
# flush_dcache_all() (link 0x100816A0 + reloc_off 0x6FC05000).  This removes
# stale DDR-probe lines while leaving U-Boot's MMU/cache operating normally.
# JTAG image writes performed afterward are therefore coherent.
set relocated_flush_dcache_all 0x7FC866A0
bpadd -addr $relocated_main_loop -type hw -temp
rwr r30 $relocated_main_loop
rwr pc $relocated_flush_dcache_all
if {[catch {con -block -timeout 20} cache_message]} {
    catch {stop}
    disconnect
    error "U-Boot flush_dcache_all call failed: $cache_message"
}
set cache_pc_line [rrd pc]
if {![regexp {([[:xdigit:]]{16})} $cache_pc_line -> cache_pc_hex] ||
        [expr 0x$cache_pc_hex] != $relocated_main_loop} {
    disconnect
    error "U-Boot flush_dcache_all did not return to main_loop: $cache_pc_line"
}
puts "RAM_BOOT_UBOOT_DCACHE_FLUSH=PASS"

puts "RAM_BOOT_STAGE=LOAD_IMAGES_AFTER_DDR_PROBE"
dow -data "$ram/system-16gb.dtb" 0x00100000
dow -data "$ram/rootfs.cpio.gz" 0x04000000
dow -data "$ram/Image" 0x18000000
dow -data "$ram/boot.scr" 0x20000000

# Optional volatile early-entry probe retained only for reproducing the old
# headless-JTAG diagnosis.  Its trampoline contains the branch target of the
# exact pinned Image for which it was created, so it must never be installed
# into a rebuilt Image by default.  UART now provides non-invasive boot
# progress, and normal boots leave both Image entry words untouched.
if {[info exists ::env(RAM_KERNEL_ENTRY_PROBE)] &&
        $::env(RAM_KERNEL_ENTRY_PROBE) eq "1"} {
    if {[mrd -force -value 0x18000000] != 0xFA405A4D ||
            [mrd -force -value 0x18000004] != 0x14441920} {
        disconnect
        error "entry probe supports only the pinned legacy arm64 Image"
    }
    set kernel_probe_words {
        0xD2800009 0xF2AFFE09 0x5289CA8A 0x72A968AA
        0xB900012A 0xD5033F9F 0xD5033FDF 0xFA405A4D
        0xD28C9089 0xF2A32209 0xD61F0120
    }
    set probe_address 0x14000000
    foreach word $kernel_probe_words {
        mwr $probe_address $word
        incr probe_address 4
    }
    mwr 0x18000000 0x17000000
    if {[mrd -force -value 0x14000000] != 0xD2800009 ||
            [mrd -force -value 0x18000000] != 0x17000000} {
        disconnect
        error "kernel entry probe read-back gate failed"
    }
    puts "RAM_BOOT_KERNEL_ENTRY_PROBE=PASS mailbox_magic=0x4B454E54"
} else {
    puts "RAM_BOOT_KERNEL_ENTRY_PROBE=SKIP reason=uart_available"
}

# Read-back gates for FDT, gzip initramfs, arm64 Image and U-Boot script.
set dtb_magic [mrd -force -value 0x00100000]
set rootfs_magic [mrd -force -value 0x04000000]
set image_magic [mrd -force -value 0x18000038]
set script_magic [mrd -force -value 0x20000000]
puts [format "RAM_BOOT_IMAGE_MAGIC dtb=0x%08X rootfs=0x%08X image=0x%08X script=0x%08X" \
    $dtb_magic $rootfs_magic $image_magic $script_magic]
if {$dtb_magic != 0xEDFE0DD0 || ($rootfs_magic & 0x00FFFFFF) != 0x00088B1F ||
        $image_magic != 0x644D5241 || $script_magic != 0x56190527} {
    disconnect
    error "post-U-Boot image read-back gate failed"
}
puts "RAM_BOOT_POST_UBOOT_IMAGE_GATE=PASS"

# Bypass automatic boot-target scanning: this U-Boot later reuses scriptaddr
# memory while probing other targets.  Invoke the validated script immediately
# from the halted main_loop, before any target can overwrite it.
# U-Boot deliberately leaves the top-of-low-DDR mailbox reservation unmapped.
# Keep the command in ordinary low DDR, between the DTB and initramfs, so XSCT
# can write it while U-Boot's MMU is enabled.
set command_address 0x03000000
mwr 0x03000100 0
mwr $command_address 0x72756F73
mwr [expr {$command_address + 4}] 0x30206563
mwr [expr {$command_address + 8}] 0x30303278
mwr [expr {$command_address + 12}] 0x30303030
mwr [expr {$command_address + 16}] 0x00000030

# The CPU may retain cache lines from U-Boot's pre-relocation RAM probing even
# though DAP read-back sees the new bytes. Invalidate every JTAG-loaded low-DDR
# range before the command parser reads it. The stack and relocated U-Boot live
# above this range, so no live U-Boot state is discarded.
set relocated_invalidate_dcache_range 0x7FC866B4
bpadd -addr $relocated_main_loop -type hw -temp
rwr r0 0x00100000
rwr r1 0x21000000
rwr r30 $relocated_main_loop
rwr pc $relocated_invalidate_dcache_range
if {[catch {con -block -timeout 20} invalidate_message]} {
    catch {stop}
    disconnect
    error "U-Boot invalidate_dcache_range call failed: $invalidate_message"
}
set invalidate_pc_line [rrd pc]
if {![regexp {([[:xdigit:]]{16})} $invalidate_pc_line -> invalidate_pc_hex] ||
        [expr 0x$invalidate_pc_hex] != $relocated_main_loop} {
    disconnect
    error "U-Boot invalidate_dcache_range did not return to main_loop: $invalidate_pc_line"
}
puts "RAM_BOOT_UBOOT_DCACHE_INVALIDATE=PASS"

rwr r0 0x20000000
rwr r1 0
rwr r30 $relocated_main_loop
# image_source_script link 0x100901A8 + reloc_off 0x6FC05000. Calling this
# exported one-argument image entry avoids synthesizing a shell-parser call.
rwr pc 0x7FC951A8
puts "RAM_BOOT_UBOOT_SOURCE=START entry=image_source_script address=0x20000000"

if {[info exists ::env(RAM_KERNEL_TRACE)] && $::env(RAM_KERNEL_TRACE) eq "1"} {
    # Pinned linux-xlnx System.map, with KASLR disabled.  Hardware breakpoints
    # compare the post-MMU virtual PC, so use the linked kernel addresses.
    set trace_stages {
        {start_kernel 0xFFFFFFC0091008A8}
        {rest_init 0xFFFFFFC008CFD498}
        {kernel_init 0xFFFFFFC008CFD558}
        {kernel_init_freeable 0xFFFFFFC009100FC8}
        {run_init_process 0xFFFFFFC008CE3CB8}
    }
    set panic_address 0xFFFFFFC008CE4834
    bpadd -addr $panic_address -type hw
    foreach stage $trace_stages {
        set stage_name [lindex $stage 0]
        set stage_address [lindex $stage 1]
        bpadd -addr $stage_address -type hw -temp
        if {[catch {con -block -timeout 60} trace_message]} {
            catch {stop}
            puts "RAM_KERNEL_TRACE_TIMEOUT stage=$stage_name message=$trace_message"
            catch {puts "RAM_KERNEL_TRACE_PC=[rrd pc]"}
            disconnect
            exit 1
        }
        set trace_pc_line [rrd pc]
        if {![regexp {([[:xdigit:]]{16})} $trace_pc_line -> trace_pc_hex]} {
            disconnect
            error "cannot parse kernel trace PC: $trace_pc_line"
        }
        set trace_pc [expr 0x$trace_pc_hex]
        if {$trace_pc == $panic_address} {
            puts [format "RAM_KERNEL_TRACE_PANIC while_waiting=%s pc=0x%016X r0=%s" \
                $stage_name $trace_pc [rrd r0]]
            disconnect
            exit 1
        }
        puts [format "RAM_KERNEL_TRACE_HIT stage=%s pc=0x%016X" \
            $stage_name $trace_pc]
    }
    con
    disconnect
    puts "RAM_KERNEL_TRACE_HANDOFF=PASS reached=run_init_process"
    exit 0
}
con
disconnect
puts "RAM_BOOT_Q8Q8_JTAG_HANDOFF_PASS uart=ttyPS0 mode=pl_selftest"
