set project_file "/home/zhaoxiangyu/RoboBrain2.5/fpga_accel/artifacts/vivado_zcu102_smoke/project/zcu102_minimal.xpr"
set output_file [file normalize "fpga_accel/artifacts/ddr_smoke/zcu102_ddr_smoke.xsa"]
file mkdir [file dirname $output_file]
open_project $project_file
open_bd_design [get_files system.bd]
write_hw_platform -fixed -include_bit -force -file $output_file
puts "DDR_SMOKE_XSA=$output_file"
close_project
exit
