`timescale 1ns/1ps

// Behavioral equivalents of the two small HLS address MACs.  The data-path
// arithmetic remains stubbed separately; these address operators must stay
// functional so the generated bank/address schedule can be checked.
`define ADDRESS_MAC_STUB(NAME) \
module NAME #( \
    parameter ID = 1, parameter NUM_STAGE = 4, \
    parameter din0_WIDTH = 2, parameter din1_WIDTH = 9, \
    parameter din2_WIDTH = 9, parameter dout_WIDTH = 11 \
)( \
    input wire clk, input wire reset, input wire ce, \
    input wire [din0_WIDTH-1:0] din0, \
    input wire [din1_WIDTH-1:0] din1, \
    input wire [din2_WIDTH-1:0] din2, \
    output wire [dout_WIDTH-1:0] dout \
); \
    reg [din0_WIDTH-1:0] a_reg; \
    reg [din1_WIDTH-1:0] b_reg; \
    reg [din0_WIDTH+din1_WIDTH-1:0] product_reg; \
    reg [dout_WIDTH-1:0] result_reg; \
    always @(posedge clk) begin \
        if (ce) begin \
            a_reg <= din0; \
            b_reg <= din1; \
            product_reg <= a_reg * b_reg; \
            result_reg <= product_reg + din2; \
        end \
    end \
    assign dout = result_reg; \
endmodule

`ADDRESS_MAC_STUB(q8q8_gemm_mac_muladd_2ns_9ns_9ns_11_4_1)
`ADDRESS_MAC_STUB(q8q8_gemm_mac_muladd_4ns_9ns_9ns_13_4_1)

`undef ADDRESS_MAC_STUB
