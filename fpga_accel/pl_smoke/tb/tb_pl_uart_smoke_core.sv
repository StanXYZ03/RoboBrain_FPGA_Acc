`timescale 1ns/1ps

module tb_pl_uart_smoke_core;
    localparam integer CLK_HZ       = 10_000_000;
    localparam integer BAUD         = 1_000_000;
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer MESSAGE_LEN  = 14;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic uart_tx;
    logic [7:0] received;
    integer character_index;
    integer bit_index;

    always #50 clk = ~clk;

    function automatic logic [7:0] expected_byte(input integer index);
        case (index)
            0:  expected_byte = "P";
            1:  expected_byte = "L";
            2:  expected_byte = "_";
            3:  expected_byte = "S";
            4:  expected_byte = "M";
            5:  expected_byte = "O";
            6:  expected_byte = "K";
            7:  expected_byte = "E";
            8:  expected_byte = "_";
            9:  expected_byte = "P";
            10: expected_byte = "A";
            11: expected_byte = "S";
            12: expected_byte = "S";
            13: expected_byte = 8'h0A;
            default: expected_byte = 8'h3F;
        endcase
    endfunction

    pl_uart_smoke_core #(
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .GAP_CYCLES(20)
    ) dut (
        .clk(clk),
        .rst(rst),
        .uart_tx(uart_tx)
    );

    initial begin
        repeat (5) @(posedge clk);
        rst <= 1'b0;

        for (character_index = 0; character_index < MESSAGE_LEN; character_index = character_index + 1) begin
            @(negedge uart_tx);
            repeat (CLKS_PER_BIT / 2) @(posedge clk);
            if (uart_tx !== 1'b0) begin
                $fatal(1, "PL_TB_FAIL bad start bit at character %0d", character_index);
            end

            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                repeat (CLKS_PER_BIT) @(posedge clk);
                received[bit_index] = uart_tx;
            end

            repeat (CLKS_PER_BIT) @(posedge clk);
            if (uart_tx !== 1'b1) begin
                $fatal(1, "PL_TB_FAIL bad stop bit at character %0d", character_index);
            end
            if (received !== expected_byte(character_index)) begin
                $fatal(1,
                    "PL_TB_FAIL character=%0d expected=0x%02x actual=0x%02x",
                    character_index, expected_byte(character_index), received);
            end
        end

        $display("PL_TB_PASS message=PL_SMOKE_PASS\\n");
        $finish;
    end

    initial begin
        #2ms;
        $fatal(1, "PL_TB_FAIL timeout");
    end
endmodule
