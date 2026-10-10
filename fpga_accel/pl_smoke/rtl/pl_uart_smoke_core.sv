`timescale 1ns/1ps

module pl_uart_smoke_core #(
    parameter integer CLK_HZ     = 300_000_000,
    parameter integer BAUD       = 115_200,
    parameter integer GAP_CYCLES = 3_000_000
) (
    input  logic clk,
    input  logic rst,
    output logic uart_tx
);
    localparam integer CLKS_PER_BIT = (CLK_HZ + BAUD / 2) / BAUD;
    localparam integer MESSAGE_LEN  = 14;
    localparam integer BAUD_W       = $clog2(CLKS_PER_BIT + 1);
    localparam integer GAP_W        = $clog2(GAP_CYCLES + 1);

    typedef enum logic [1:0] {
        ST_GAP,
        ST_START,
        ST_DATA,
        ST_STOP
    } state_t;

    state_t state;
    logic [BAUD_W-1:0] baud_count;
    logic [GAP_W-1:0] gap_count;
    logic [3:0] message_index;
    logic [2:0] bit_index;
    logic [7:0] tx_byte;

    function automatic logic [7:0] message_byte(input logic [3:0] index);
        case (index)
            4'd0:  message_byte = "P";
            4'd1:  message_byte = "L";
            4'd2:  message_byte = "_";
            4'd3:  message_byte = "S";
            4'd4:  message_byte = "M";
            4'd5:  message_byte = "O";
            4'd6:  message_byte = "K";
            4'd7:  message_byte = "E";
            4'd8:  message_byte = "_";
            4'd9:  message_byte = "P";
            4'd10: message_byte = "A";
            4'd11: message_byte = "S";
            4'd12: message_byte = "S";
            4'd13: message_byte = 8'h0A;
            default: message_byte = 8'h3F;
        endcase
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            state         <= ST_GAP;
            baud_count    <= '0;
            gap_count     <= '0;
            message_index <= '0;
            bit_index     <= '0;
            tx_byte       <= message_byte(0);
            uart_tx       <= 1'b1;
        end else begin
            case (state)
                ST_GAP: begin
                    uart_tx <= 1'b1;
                    if (gap_count == GAP_CYCLES - 1) begin
                        gap_count  <= '0;
                        baud_count <= '0;
                        tx_byte    <= message_byte(message_index);
                        uart_tx    <= 1'b0;
                        state      <= ST_START;
                    end else begin
                        gap_count <= gap_count + 1'b1;
                    end
                end

                ST_START: begin
                    if (baud_count == CLKS_PER_BIT - 1) begin
                        baud_count <= '0;
                        bit_index  <= '0;
                        uart_tx    <= tx_byte[0];
                        state      <= ST_DATA;
                    end else begin
                        baud_count <= baud_count + 1'b1;
                    end
                end

                ST_DATA: begin
                    if (baud_count == CLKS_PER_BIT - 1) begin
                        baud_count <= '0;
                        if (bit_index == 3'd7) begin
                            uart_tx <= 1'b1;
                            state   <= ST_STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                            uart_tx   <= tx_byte[bit_index + 1'b1];
                        end
                    end else begin
                        baud_count <= baud_count + 1'b1;
                    end
                end

                ST_STOP: begin
                    if (baud_count == CLKS_PER_BIT - 1) begin
                        baud_count <= '0;
                        if (message_index == MESSAGE_LEN - 1) begin
                            message_index <= '0;
                            gap_count     <= '0;
                            state         <= ST_GAP;
                        end else begin
                            message_index <= message_index + 1'b1;
                            tx_byte       <= message_byte(message_index + 1'b1);
                            uart_tx       <= 1'b0;
                            state         <= ST_START;
                        end
                    end else begin
                        baud_count <= baud_count + 1'b1;
                    end
                end

                default: begin
                    state   <= ST_GAP;
                    uart_tx <= 1'b1;
                end
            endcase
        end
    end
endmodule
