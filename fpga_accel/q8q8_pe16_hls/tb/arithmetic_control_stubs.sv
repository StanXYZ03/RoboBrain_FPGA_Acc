`timescale 1ns/1ps

// These stubs isolate the HLS-generated compute controller and memory address
// schedule. Numerical arithmetic is covered by the C testbench.
module q8q8_gemm_fadd_32ns_32ns_32_8_full_dsp_1 #(
    parameter ID = 1,
    parameter NUM_STAGE = 8,
    parameter din0_WIDTH = 32,
    parameter din1_WIDTH = 32,
    parameter dout_WIDTH = 32
) (
    input wire clk,
    input wire reset,
    input wire [din0_WIDTH-1:0] din0,
    input wire [din1_WIDTH-1:0] din1,
    input wire ce,
    output wire [dout_WIDTH-1:0] dout
);
assign dout = {dout_WIDTH{1'b0}};
endmodule

module q8q8_gemm_fmul_32ns_32ns_32_5_max_dsp_1 #(
    parameter ID = 1,
    parameter NUM_STAGE = 5,
    parameter din0_WIDTH = 32,
    parameter din1_WIDTH = 32,
    parameter dout_WIDTH = 32
) (
    input wire clk,
    input wire reset,
    input wire [din0_WIDTH-1:0] din0,
    input wire [din1_WIDTH-1:0] din1,
    input wire ce,
    output wire [dout_WIDTH-1:0] dout
);
assign dout = {dout_WIDTH{1'b0}};
endmodule

module q8q8_gemm_sitofp_32s_32_6_no_dsp_1 #(
    parameter ID = 1,
    parameter NUM_STAGE = 6,
    parameter din0_WIDTH = 32,
    parameter dout_WIDTH = 32
) (
    input wire clk,
    input wire reset,
    input wire [din0_WIDTH-1:0] din0,
    input wire ce,
    output wire [dout_WIDTH-1:0] dout
);
assign dout = {dout_WIDTH{1'b0}};
endmodule

module q8q8_gemm_mul_8s_8s_16_1_1 #(
    parameter ID = 1,
    parameter NUM_STAGE = 1,
    parameter din0_WIDTH = 8,
    parameter din1_WIDTH = 8,
    parameter dout_WIDTH = 16
) (
    input wire [din0_WIDTH-1:0] din0,
    input wire [din1_WIDTH-1:0] din1,
    output wire [dout_WIDTH-1:0] dout
);
assign dout = {dout_WIDTH{1'b0}};
endmodule

module q8q8_gemm_mac_muladd_8s_8s_16s_17_4_1 #(
    parameter ID = 1,
    parameter NUM_STAGE = 4,
    parameter din0_WIDTH = 8,
    parameter din1_WIDTH = 8,
    parameter din2_WIDTH = 16,
    parameter dout_WIDTH = 17
) (
    input wire clk,
    input wire reset,
    input wire [din0_WIDTH-1:0] din0,
    input wire [din1_WIDTH-1:0] din1,
    input wire [din2_WIDTH-1:0] din2,
    input wire ce,
    output wire [dout_WIDTH-1:0] dout
);
assign dout = {dout_WIDTH{1'b0}};
endmodule
