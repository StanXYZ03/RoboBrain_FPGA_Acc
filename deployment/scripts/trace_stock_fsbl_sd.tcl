# Read-only trace of stock ZCU102 FSBL SD initialization and BOOT.BIN reads.
set platform_boot /home/xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/sw/xilinx_zcu102_base_202320_1/boot
set fsbl "$platform_boot/fsbl.elf"
set pmufw "$platform_boot/pmufw.elf"

proc register_value {name} {
    set line [rrd $name]
    if {![regexp {([[:xdigit:]]{16})} $line -> value]} {
        error "cannot parse $name from: $line"
    }
    return [expr 0x$value]
}

proc read_c_string {address max_bytes} {
    set result ""
    for {set offset 0} {$offset < $max_bytes} {incr offset 4} {
        set word [mrd -force -value [expr {$address + $offset}]]
        for {set index 0} {$index < 4} {incr index} {
            set byte [expr {($word >> (8 * $index)) & 0xFF}]
            if {$byte == 0} { return $result }
            append result [format %c $byte]
        }
    }
    return $result
}

proc continue_to {address label timeout_seconds} {
    bpadd -addr $address -type hw -temp
    if {[catch {con -block -timeout $timeout_seconds} message]} {
        puts "$label=TIMEOUT message=$message"
        return 0
    }
    puts [format "%s=HIT pc=0x%016X" $label [register_value pc]]
    return 1
}

connect -url tcp:127.0.0.1:3121
after 500
targets -set -filter {name =~ "PSU"}
rst -system
after 1000
rst -dap
after 1000
targets -set -filter {name =~ "PSU"}
mwr 0xFFCA0038 0x1FF
# System reset does not clear CSU multiboot.  Previous failed retries can make
# FSBL look for BOOT000N.BIN instead of BOOT.BIN, so emulate the POR value.
mwr 0xFFCA0010 0
puts [format "STOCK_FSBL_MULTIBOOT_RESET=0x%08X" \
    [mrd -force -value 0xFFCA0010]]
# A JTAG system reset does not remove SD-card power.  Force a clean, volatile
# SDHCI bus-power cycle before FSBL starts issuing CMD0/CMD8/CMD55.
set sd_power_word [mrd -force -value 0xFF170028]
mwr 0xFF170028 [expr {$sd_power_word & 0xFFFF00FF}]
after 500
mwr 0xFF170028 [expr {($sd_power_word & 0xFFFF00FF) | 0x00000F00}]
after 200
puts [format "STOCK_FSBL_SD_POWER_CYCLE control=0x%08X" \
    [mrd -force -value 0xFF170028]]

targets -set -filter {name =~ "MicroBlaze PMU"}
dow $pmufw
con
after 500

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor -clear-registers
dow $fsbl

if {![continue_to 0xFFFCFA28 STOCK_FSBL_SD_INIT_ENTRY 30]} {
    catch {stop}
    disconnect
    exit 1
}
set sd_init_return [register_value r30]
puts [format "STOCK_FSBL_SD_INIT_ARGS r0=0x%016X lr=0x%016X" \
    [register_value r0] $sd_init_return]
if {![continue_to 0xFFFCFB18 STOCK_FSBL_SD_FILENAME_READY 30]} {
    catch {stop}
    disconnect
    exit 1
}
set filename_address [register_value r0]
puts [format "STOCK_FSBL_SD_FILENAME multiboot=0x%08X address=0x%016X value=%s" \
    [mrd -force -value 0xFFCA0010] $filename_address \
    [read_c_string $filename_address 32]]

# The inlined FatFS path collapses all f_open errors to XFSBL_ERROR_SD_F_OPEN.
# Catch its lower-level SD-driver and generic volume failure branches first.
bpadd -addr 0xFFFCE990 -type hw -temp
bpadd -addr 0xFFFCFCD8 -type hw -temp
bpadd -addr 0xFFFCFB6C -type hw -temp
bpadd -addr $sd_init_return -type hw -temp
if {[catch {con -block -timeout 30} sd_path_message]} {
    catch {stop}
    disconnect
    error "stock FSBL SD path did not halt: $sd_path_message"
}
set sd_path_pc [register_value pc]
if {$sd_path_pc == 0xFFFCE990} {
    set sd_instance [register_value r13]
    set sdhci_base [mrd -force -value [expr {$sd_instance + 8}]]
    puts [format "STOCK_FSBL_CARD_OP_COND_FAIL r0=0x%016X retries_left=0x%016X lr=0x%016X sdhci=0x%08X" \
        [register_value r0] [register_value r14] [register_value r30] $sdhci_base]
    puts [format "STOCK_FSBL_SDHCI present=0x%08X response0=0x%08X int_status=0x%08X" \
        [mrd -force -value [expr {$sdhci_base + 0x24}]] \
        [mrd -force -value [expr {$sdhci_base + 0x10}]] \
        [mrd -force -value [expr {$sdhci_base + 0x30}]]]
    if {[catch {con -block -timeout 30} driver_message]} {
        catch {stop}
        disconnect
        error "stock FSBL did not reach driver failure: $driver_message"
    }
    set sd_path_pc [register_value pc]
}
if {$sd_path_pc == 0xFFFCFCD8} {
    puts [format "STOCK_FSBL_SD_DRIVER_FAIL r0=0x%016X r22=0x%016X r25=0x%016X lr=0x%016X" \
        [register_value r0] [register_value r22] [register_value r25] \
        [register_value r30]]
    if {[catch {con -block -timeout 30} next_message]} {
        catch {stop}
        disconnect
        error "stock FSBL did not reach FatFS failure/return: $next_message"
    }
    set sd_path_pc [register_value pc]
}
if {$sd_path_pc == 0xFFFCFB6C} {
    puts [format "STOCK_FSBL_FATFS_FAIL r0=0x%016X r22=0x%016X r25=0x%016X lr=0x%016X" \
        [register_value r0] [register_value r22] [register_value r25] \
        [register_value r30]]
    if {[catch {con -block -timeout 30} return_message]} {
        catch {stop}
        disconnect
        error "stock FSBL SD init did not return: $return_message"
    }
    set sd_path_pc [register_value pc]
}
puts [format "STOCK_FSBL_SD_INIT_RETURN=HIT pc=0x%016X" $sd_path_pc]
puts [format "STOCK_FSBL_SD_INIT_RESULT r0=0x%016X" [register_value r0]]

if {![continue_to 0xFFFCF1B0 STOCK_FSBL_SD_COPY_ENTRY 30]} {
    catch {stop}
    disconnect
    exit 1
}
set sd_copy_return [register_value r30]
puts [format "STOCK_FSBL_SD_COPY_ARGS r0=0x%016X r1=0x%016X r2=0x%016X lr=0x%016X" \
    [register_value r0] [register_value r1] [register_value r2] $sd_copy_return]
if {![continue_to $sd_copy_return STOCK_FSBL_SD_COPY_RETURN 30]} {
    catch {stop}
    disconnect
    exit 1
}
puts [format "STOCK_FSBL_SD_COPY_RESULT r0=0x%016X" [register_value r0]]

# If the first read succeeds, trace whether FSBL reaches its final handoff.
if {[continue_to 0xFFFD1D00 STOCK_FSBL_EXIT_ENTRY 60]} {
    puts [format "STOCK_FSBL_EXIT_ARGS r0=0x%016X" [register_value r0]]
    con
}
disconnect
