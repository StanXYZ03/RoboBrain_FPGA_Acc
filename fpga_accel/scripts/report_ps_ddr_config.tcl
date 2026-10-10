set project_file "/home/zhaoxiangyu/RoboBrain2.5/fpga_accel/artifacts/vivado_zcu102_smoke/project/zcu102_minimal.xpr"
open_project $project_file
open_bd_design [get_files system.bd]
set ps [get_bd_cells zynq_ultra_ps_e_0]
puts "=== PS DDR CONFIG ==="
foreach prop [lsort [list_property $ps]] {
    if {[regexp -nocase {DDR|MEMORY|BANK|RANK|ROW|COLUMN} $prop]} {
        puts "$prop=[get_property $prop $ps]"
    }
}
close_project
exit
