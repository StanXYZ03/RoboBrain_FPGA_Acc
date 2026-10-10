`timescale 1ns/1ps

module tb_store_output_rtl;
    reg ap_clk = 1'b0;
    reg ap_rst = 1'b1;
    reg ap_start = 1'b0;
    wire ap_done;
    wire ap_idle;
    wire ap_ready;

    wire awvalid;
    wire [63:0] awaddr;
    wire [31:0] awlen;
    wire [2:0] awsize;
    wire [1:0] awburst;
    wire wvalid;
    wire [127:0] wdata;
    wire [15:0] wstrb;
    wire wlast;
    wire bready;

    wire [6:0] accum_0_address0;
    wire [6:0] accum_1_address0;
    wire [6:0] accum_2_address0;
    wire [6:0] accum_3_address0;
    wire [6:0] accum_4_address0;
    wire [6:0] accum_5_address0;
    wire [6:0] accum_6_address0;
    wire [6:0] accum_7_address0;
    wire accum_0_ce0, accum_1_ce0, accum_2_ce0, accum_3_ce0;
    wire accum_4_ce0, accum_5_ce0, accum_6_ce0, accum_7_ce0;
    reg [31:0] accum_0_q0 = 0, accum_1_q0 = 0;
    reg [31:0] accum_2_q0 = 0, accum_3_q0 = 0;
    reg [31:0] accum_4_q0 = 0, accum_5_q0 = 0;
    reg [31:0] accum_6_q0 = 0, accum_7_q0 = 0;
    reg [31:0] bank0 [0:127];
    reg [31:0] bank1 [0:127];
    reg [31:0] bank2 [0:127];
    reg [31:0] bank3 [0:127];
    reg [31:0] bank4 [0:127];
    reg [31:0] bank5 [0:127];
    reg [31:0] bank6 [0:127];
    reg [31:0] bank7 [0:127];

    integer pending_b = 0;
    integer aw_count = 0;
    integer w_count = 0;
    integer cycles = 0;
    integer i;
    reg [63:0] observed_awaddr [0:7];
    reg [31:0] observed_awlen [0:7];
    reg [127:0] observed_wdata [0:7];
    reg observed_wlast [0:7];
    wire bvalid = (pending_b != 0);

    always #2.5 ap_clk = ~ap_clk;

    q8q8_gemm_p_anonymous_namespace_store_output_tile dut (
        .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start),
        .ap_done(ap_done), .ap_idle(ap_idle), .ap_ready(ap_ready),
        .m_axi_gmem_y_AWVALID(awvalid),
        .m_axi_gmem_y_AWREADY(1'b1),
        .m_axi_gmem_y_AWADDR(awaddr),
        .m_axi_gmem_y_AWLEN(awlen),
        .m_axi_gmem_y_AWSIZE(awsize),
        .m_axi_gmem_y_AWBURST(awburst),
        .m_axi_gmem_y_WVALID(wvalid),
        .m_axi_gmem_y_WREADY(1'b1),
        .m_axi_gmem_y_WDATA(wdata),
        .m_axi_gmem_y_WSTRB(wstrb),
        .m_axi_gmem_y_WLAST(wlast),
        .m_axi_gmem_y_ARREADY(1'b1),
        .m_axi_gmem_y_RVALID(1'b0),
        .m_axi_gmem_y_RDATA(128'b0),
        .m_axi_gmem_y_RLAST(1'b0),
        .m_axi_gmem_y_RID(1'b0),
        .m_axi_gmem_y_RFIFONUM(9'b0),
        .m_axi_gmem_y_RUSER(1'b0),
        .m_axi_gmem_y_RRESP(2'b00),
        .m_axi_gmem_y_BVALID(bvalid),
        .m_axi_gmem_y_BREADY(bready),
        .m_axi_gmem_y_BRESP(2'b00),
        .m_axi_gmem_y_BID(1'b0),
        .m_axi_gmem_y_BUSER(1'b0),
        .output_r(64'h0000_0000_0000_1000),
        .accum_0_address0(accum_0_address0),
        .accum_0_ce0(accum_0_ce0), .accum_0_q0(accum_0_q0),
        .accum_1_address0(accum_1_address0),
        .accum_1_ce0(accum_1_ce0), .accum_1_q0(accum_1_q0),
        .accum_2_address0(accum_2_address0),
        .accum_2_ce0(accum_2_ce0), .accum_2_q0(accum_2_q0),
        .accum_3_address0(accum_3_address0),
        .accum_3_ce0(accum_3_ce0), .accum_3_q0(accum_3_q0),
        .accum_4_address0(accum_4_address0),
        .accum_4_ce0(accum_4_ce0), .accum_4_q0(accum_4_q0),
        .accum_5_address0(accum_5_address0),
        .accum_5_ce0(accum_5_ce0), .accum_5_q0(accum_5_q0),
        .accum_6_address0(accum_6_address0),
        .accum_6_ce0(accum_6_ce0), .accum_6_q0(accum_6_q0),
        .accum_7_address0(accum_7_address0),
        .accum_7_ce0(accum_7_ce0), .accum_7_q0(accum_7_q0),
        .m0(32'd0), .n0(32'd0), .m(32'd3), .n(32'd4)
    );

    always @(posedge ap_clk) begin
        if (accum_0_ce0) accum_0_q0 <= bank0[accum_0_address0];
        if (accum_1_ce0) accum_1_q0 <= bank1[accum_1_address0];
        if (accum_2_ce0) accum_2_q0 <= bank2[accum_2_address0];
        if (accum_3_ce0) accum_3_q0 <= bank3[accum_3_address0];
        if (accum_4_ce0) accum_4_q0 <= bank4[accum_4_address0];
        if (accum_5_ce0) accum_5_q0 <= bank5[accum_5_address0];
        if (accum_6_ce0) accum_6_q0 <= bank6[accum_6_address0];
        if (accum_7_ce0) accum_7_q0 <= bank7[accum_7_address0];

        if (awvalid) begin
            observed_awaddr[aw_count] <= awaddr;
            observed_awlen[aw_count] <= awlen;
            aw_count <= aw_count + 1;
        end
        if (wvalid) begin
            observed_wdata[w_count] <= wdata;
            observed_wlast[w_count] <= wlast;
            if (wstrb !== 16'hffff) $fatal(1, "WSTRB mismatch: %h", wstrb);
            w_count <= w_count + 1;
        end
        // This HLS submodule uses per-word abstract write requests.  The
        // generated gmem_y AXI adapter below it creates WLAST and bursts.
        case ({wvalid, bvalid && bready})
            2'b10: pending_b <= pending_b + 1;
            2'b01: pending_b <= pending_b - 1;
            default: pending_b <= pending_b;
        endcase
        cycles <= cycles + 1;
        if (cycles > 2000) $fatal(1,
            "timeout done=%b idle=%b ready=%b awv=%b wv=%b wl=%b br=%b bv=%b pending=%0d aw=%0d w=%0d state=%b",
            ap_done, ap_idle, ap_ready, awvalid, wvalid, wlast, bready,
            bvalid, pending_b, aw_count, w_count, dut.ap_CS_fsm);
    end

    initial begin
        for (i = 0; i < 128; i = i + 1) begin
            bank0[i] = 0; bank1[i] = 0; bank2[i] = 0; bank3[i] = 0;
            bank4[i] = 0; bank5[i] = 0; bank6[i] = 0; bank7[i] = 0;
        end
        bank0[0]  = 32'h3f800000; bank1[0]  = 32'h40000000;
        bank2[0]  = 32'h40400000; bank3[0]  = 32'h40800000;
        bank0[8]  = 32'hbf800000; bank1[8]  = 32'hc0000000;
        bank2[8]  = 32'hc0400000; bank3[8]  = 32'hc0800000;
        bank0[16] = 32'h3f000000; bank1[16] = 32'h3e800000;
        bank2[16] = 32'h3e000000; bank3[16] = 32'h3d800000;

        repeat (8) @(posedge ap_clk);
        ap_rst <= 1'b0;
        repeat (3) @(posedge ap_clk);
        ap_start <= 1'b1;
        @(posedge ap_clk);
        ap_start <= 1'b0;
        wait (ap_done === 1'b1);
        repeat (8) @(posedge ap_clk);

        if (aw_count !== 3) $fatal(1, "AW count %0d, expected 3", aw_count);
        if (w_count !== 3) $fatal(1, "W count %0d, expected 3", w_count);
        // Internal address requests are expressed in 128-bit words.  The
        // generated AXI adapter restores byte addresses (<< 4).
        if (observed_awaddr[0] !== 64'h100 ||
            observed_awaddr[1] !== 64'h101 ||
            observed_awaddr[2] !== 64'h102)
            $fatal(1, "AW addresses mismatch: %h %h %h",
                   observed_awaddr[0], observed_awaddr[1], observed_awaddr[2]);
        if (observed_awlen[0] !== 1 || observed_awlen[1] !== 1 ||
            observed_awlen[2] !== 1 || awsize !== 3'd0 || awburst !== 2'b00)
            $fatal(1, "abstract write metadata mismatch: len=%h/%h/%h size=%h burst=%h",
                   observed_awlen[0], observed_awlen[1], observed_awlen[2],
                   awsize, awburst);
        if (observed_wdata[0] !== 128'h4080000040400000400000003f800000)
            $fatal(1, "row0 packed data mismatch: %h", observed_wdata[0]);
        if (observed_wdata[1] !== 128'hc0800000c0400000c0000000bf800000)
            $fatal(1, "row1 packed data mismatch: %h", observed_wdata[1]);
        if (observed_wdata[2] !== 128'h3d8000003e0000003e8000003f000000)
            $fatal(1, "row2 packed data mismatch: %h", observed_wdata[2]);
        if (observed_wlast[0] || observed_wlast[1] || observed_wlast[2])
            $fatal(1, "abstract request unexpectedly asserted WLAST");
        if (pending_b !== 0) $fatal(1, "pending responses: %0d", pending_b);

        $display("TB_RESULT=PASS test=store_output_rtl rows=3 words=3 width=128 cycles=%0d", cycles);
        $finish;
    end
endmodule
