set_property PACKAGE_PIN AL8 [get_ports user_si570_sysclk_p]
set_property PACKAGE_PIN AL7 [get_ports user_si570_sysclk_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports {user_si570_sysclk_p user_si570_sysclk_n}]
create_clock -name user_si570_clk -period 3.333 [get_ports user_si570_sysclk_p]

set_property PACKAGE_PIN F13 [get_ports uart2_PL_TX]
set_property IOSTANDARD LVCMOS33 [get_ports uart2_PL_TX]

set_property PACKAGE_PIN AG14 [get_ports gpio_led_0]
set_property IOSTANDARD LVCMOS33 [get_ports gpio_led_0]
