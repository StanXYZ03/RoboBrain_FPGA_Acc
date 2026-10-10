`timescale 1ns/1ps

module pl_jtag_smoke_core (
    input  logic        clk,
    input  logic [31:0] challenge,
    output logic [31:0] response,
    output logic [31:0] heartbeat
);
    initial heartbeat = 32'h0000_0000;

    // A configuration-time initial value plus a clocked update is supported by
    // UltraScale+ fabric. Use a plain always block so simulation accepts both.
    always @(posedge clk) begin
        heartbeat <= heartbeat + 1'b1;
    end

    always_comb begin
        response = challenge ^ 32'hA5C3_5A7E;
    end
endmodule
