set boards [get_board_parts -quiet xilinx.com:zcu102:part0:*]
puts "BOARD_COUNT=[llength $boards]"
foreach board $boards {
    puts "BOARD=$board"
    report_property $board
}
exit
