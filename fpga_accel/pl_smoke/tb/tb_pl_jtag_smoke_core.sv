`timescale 1ns/1ps

module tb_pl_jtag_smoke_core;
    logic clk = 1'b0;
    logic [31:0] challenge;
    logic [31:0] response;
    logic [31:0] heartbeat;
    logic [31:0] heartbeat_before;

    always #10 clk = ~clk;

    pl_jtag_smoke_core dut (
        .clk(clk),
        .challenge(challenge),
        .response(response),
        .heartbeat(heartbeat)
    );

    initial begin
        challenge = 32'h1122_3344;
        #1;
        if (response !== 32'hB4E1_693A) begin
            $fatal(1, "PL_JTAG_TB_FAIL challenge 1 response=%08x", response);
        end

        challenge = 32'hC001_D00D;
        #1;
        if (response !== 32'h65C2_8A73) begin
            $fatal(1, "PL_JTAG_TB_FAIL challenge 2 response=%08x", response);
        end

        @(posedge clk);
        heartbeat_before = heartbeat;
        repeat (16) @(posedge clk);
        if (heartbeat == heartbeat_before) begin
            $fatal(1, "PL_JTAG_TB_FAIL heartbeat did not advance");
        end

        $display("PL_JTAG_TB_PASS response=%08x heartbeat_before=%08x heartbeat_after=%08x",
            response, heartbeat_before, heartbeat);
        $finish;
    end

    initial begin
        #10us;
        $fatal(1, "PL_JTAG_TB_FAIL timeout");
    end
endmodule
