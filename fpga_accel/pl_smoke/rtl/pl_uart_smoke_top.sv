`timescale 1ns/1ps

module pl_uart_smoke_top (
    input  wire user_si570_sysclk_p,
    input  wire user_si570_sysclk_n,
    output wire uart2_PL_TX,
    output wire gpio_led_0
);
    wire clk_300mhz;
    logic [25:0] reset_counter = '0;
    logic [27:0] heartbeat_counter = '0;
    wire core_reset = ~reset_counter[25];

    IBUFDS #(
        .DIFF_TERM("FALSE"),
        .IBUF_LOW_PWR("FALSE"),
        .IOSTANDARD("DIFF_SSTL12")
    ) clock_input_buffer (
        .I(user_si570_sysclk_p),
        .IB(user_si570_sysclk_n),
        .O(clk_300mhz)
    );

    always_ff @(posedge clk_300mhz) begin
        if (!reset_counter[25]) begin
            reset_counter <= reset_counter + 1'b1;
        end
        heartbeat_counter <= heartbeat_counter + 1'b1;
    end

    pl_uart_smoke_core #(
        .CLK_HZ(300_000_000),
        .BAUD(115_200),
        .GAP_CYCLES(3_000_000)
    ) smoke_core (
        .clk(clk_300mhz),
        .rst(core_reset),
        .uart_tx(uart2_PL_TX)
    );

    assign gpio_led_0 = heartbeat_counter[27];
endmodule
