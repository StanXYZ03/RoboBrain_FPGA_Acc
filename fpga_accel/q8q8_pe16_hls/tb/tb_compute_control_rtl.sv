`timescale 1ns/1ps

`define WEIGHT_PORTS(I) \
    .weight_q_cache_``I``_address0(weight_q_address[I]), \
    .weight_q_cache_``I``_ce0(weight_q_ce[I]), \
    .weight_q_cache_``I``_q0(weight_q_data[I]), \
    .weight_scale_cache_``I``_address0(weight_scale_address[I]), \
    .weight_scale_cache_``I``_ce0(weight_scale_ce[I]), \
    .weight_scale_cache_``I``_q0(weight_scale_data[I])

`define ACCUM_PORTS(I) \
    .accum_``I``_address0(accum_address0[I]), \
    .accum_``I``_ce0(accum_ce0[I]), \
    .accum_``I``_we0(accum_we0[I]), \
    .accum_``I``_d0(accum_d0[I]), \
    .accum_``I``_address1(accum_address1[I]), \
    .accum_``I``_ce1(accum_ce1[I]), \
    .accum_``I``_q1(accum_q1[I])

module tb_compute_control_rtl;
    reg ap_clk = 1'b0;
    reg ap_rst = 1'b1;
    reg ap_start = 1'b0;
    wire ap_done;
    wire ap_idle;
    wire ap_ready;

    wire [2:0] weight_q_address [0:15];
    wire weight_q_ce [0:15];
    reg [255:0] weight_q_data [0:15];
    wire [2:0] weight_scale_address [0:15];
    wire weight_scale_ce [0:15];
    reg [31:0] weight_scale_data [0:15];

    wire [4:0] activation_q_address;
    wire activation_q_ce;
    reg [255:0] activation_q_data;
    wire [4:0] activation_scale_address;
    wire activation_scale_ce;
    reg [31:0] activation_scale_data;

    wire [5:0] accum_address0 [0:15];
    wire accum_ce0 [0:15];
    wire accum_we0 [0:15];
    wire [31:0] accum_d0 [0:15];
    wire [5:0] accum_address1 [0:15];
    wire accum_ce1 [0:15];
    reg [31:0] accum_q1 [0:15];
    reg [31:0] accum_mem [0:15][0:63];
    integer write_count [0:15];
    integer address_count [0:15][0:63];

    integer bank;
    integer address;
    integer expected;
    integer failures;

    always #2.5 ap_clk = ~ap_clk;

    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : memory_banks
            integer reset_address;
            always @(posedge ap_clk) begin
                if (ap_rst) begin
                    weight_q_data[g] <= 256'b0;
                    weight_scale_data[g] <= 32'b0;
                    accum_q1[g] <= 32'b0;
                    write_count[g] <= 0;
                    for (reset_address = 0; reset_address < 64;
                         reset_address = reset_address + 1) begin
                        accum_mem[g][reset_address] <= 32'b0;
                        address_count[g][reset_address] <= 0;
                    end
                end else begin
                    if (weight_q_ce[g])
                        weight_q_data[g] <= 256'b0;
                    if (weight_scale_ce[g])
                        weight_scale_data[g] <= 32'b0;
                    if (accum_ce1[g])
                        accum_q1[g] <= accum_mem[g][accum_address1[g]];
                    if (accum_ce0[g] && accum_we0[g]) begin
                        accum_mem[g][accum_address0[g]] <= accum_d0[g];
                        write_count[g] <= write_count[g] + 1;
                        address_count[g][accum_address0[g]] <=
                            address_count[g][accum_address0[g]] + 1;
                    end
                end
            end
        end
    endgenerate

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            activation_q_data <= 256'b0;
            activation_scale_data <= 32'b0;
        end else begin
            if (activation_q_ce)
                activation_q_data <= 256'b0;
            if (activation_scale_ce)
                activation_scale_data <= 32'b0;
        end
    end

    q8q8_gemm_p_anonymous_namespace_compute_cached_tile dut (
        .ap_clk(ap_clk),
        .ap_rst(ap_rst),
        .ap_start(ap_start),
        .ap_done(ap_done),
        .ap_idle(ap_idle),
        .ap_ready(ap_ready),
        `WEIGHT_PORTS(0),
        `WEIGHT_PORTS(1),
        `WEIGHT_PORTS(2),
        `WEIGHT_PORTS(3),
        `WEIGHT_PORTS(4),
        `WEIGHT_PORTS(5),
        `WEIGHT_PORTS(6),
        `WEIGHT_PORTS(7),
        `WEIGHT_PORTS(8),
        `WEIGHT_PORTS(9),
        `WEIGHT_PORTS(10),
        `WEIGHT_PORTS(11),
        `WEIGHT_PORTS(12),
        `WEIGHT_PORTS(13),
        `WEIGHT_PORTS(14),
        `WEIGHT_PORTS(15),
        .activation_q_cache_address0(activation_q_address),
        .activation_q_cache_ce0(activation_q_ce),
        .activation_q_cache_q0(activation_q_data),
        .activation_scale_cache_address0(activation_scale_address),
        .activation_scale_cache_ce0(activation_scale_ce),
        .activation_scale_cache_q0(activation_scale_data),
        `ACCUM_PORTS(0),
        `ACCUM_PORTS(1),
        `ACCUM_PORTS(2),
        `ACCUM_PORTS(3),
        `ACCUM_PORTS(4),
        `ACCUM_PORTS(5),
        `ACCUM_PORTS(6),
        `ACCUM_PORTS(7),
        `ACCUM_PORTS(8),
        `ACCUM_PORTS(9),
        `ACCUM_PORTS(10),
        `ACCUM_PORTS(11),
        `ACCUM_PORTS(12),
        `ACCUM_PORTS(13),
        `ACCUM_PORTS(14),
        `ACCUM_PORTS(15),
        .m0(32'd0),
        .m(32'd3),
        .k_blocks(2'd2)
    );

    initial begin : timeout
        repeat (2000) @(posedge ap_clk);
        $display("RTL_CONTROL_RESULT=FAIL reason=timeout");
        $fatal(1);
    end

    initial begin : test
        failures = 0;
        repeat (5) @(posedge ap_clk);
        ap_rst <= 1'b0;
        @(posedge ap_clk);
        ap_start <= 1'b1;
        wait (ap_ready === 1'b1);
        @(posedge ap_clk);
        ap_start <= 1'b0;
        wait (ap_done === 1'b1);
        @(posedge ap_clk);

        for (bank = 0; bank < 16; bank = bank + 1) begin
            if (write_count[bank] != 24) begin
                $display("RTL_CONTROL_MISMATCH bank=%0d writes=%0d expected=24",
                         bank, write_count[bank]);
                failures = failures + 1;
            end
            for (address = 0; address < 64; address = address + 1) begin
                expected = ((address / 4) < 3) ? 2 : 0;
                if (address_count[bank][address] != expected) begin
                    $display("RTL_CONTROL_MISMATCH bank=%0d address=%0d count=%0d expected=%0d",
                             bank, address, address_count[bank][address], expected);
                    failures = failures + 1;
                end
            end
        end

        if (failures == 0) begin
            $display("RTL_CONTROL_RESULT=PASS banks=16 writes_per_bank=24 valid_addresses_per_bank=12");
            $finish;
        end
        $display("RTL_CONTROL_RESULT=FAIL mismatches=%0d", failures);
        $fatal(1);
    end
endmodule

`undef WEIGHT_PORTS
`undef ACCUM_PORTS
