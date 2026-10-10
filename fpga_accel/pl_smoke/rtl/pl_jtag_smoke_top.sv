`timescale 1ns/1ps

module pl_jtag_smoke_top;
    wire cfgmclk_raw;
    wire cfgmclk;
    wire [31:0] challenge;
    wire [31:0] response;
    wire [31:0] heartbeat;

    STARTUPE3 #(
        .PROG_USR("FALSE"),
        .SIM_CCLK_FREQ(0.0)
    ) startup_clock (
        .CFGCLK(),
        .CFGMCLK(cfgmclk_raw),
        .DI(),
        .EOS(),
        .PREQ(),
        .DO(4'b0000),
        .DTS(4'b1111),
        .FCSBO(1'b0),
        .FCSBTS(1'b1),
        .GSR(1'b0),
        .GTS(1'b0),
        .KEYCLEARB(1'b1),
        .PACK(1'b0),
        .USRCCLKO(1'b0),
        .USRCCLKTS(1'b1),
        .USRDONEO(1'b1),
        .USRDONETS(1'b1)
    );

    BUFG clock_buffer (
        .I(cfgmclk_raw),
        .O(cfgmclk)
    );

    pl_jtag_smoke_core smoke_core (
        .clk(cfgmclk),
        .challenge(challenge),
        .response(response),
        .heartbeat(heartbeat)
    );

    vio_0 jtag_vio (
        .clk(cfgmclk),
        .probe_in0(response),
        .probe_in1(heartbeat),
        .probe_out0(challenge)
    );
endmodule
