# Read PMU MicroBlaze execution state, then resume it.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "PSU"}
mwr 0xFFCA0038 0x1FF
after 200
targets -set -filter {name =~ "MicroBlaze PMU"}
catch {stop}
puts "PMUFW_REGS_BEGIN"
foreach reg {pc rmsr r1 r14 r15 r16 r17} {
    if {[catch {rrd $reg} value]} {
        puts "$reg=UNAVAILABLE"
    } else {
        puts "$reg=$value"
    }
}
puts "PMUFW_REGS_END"
catch {con}
disconnect
