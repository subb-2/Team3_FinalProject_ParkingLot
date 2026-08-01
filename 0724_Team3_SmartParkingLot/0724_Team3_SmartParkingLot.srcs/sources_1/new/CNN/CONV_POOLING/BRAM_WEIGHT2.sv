`timescale 1ns / 1ps

module BRAM_WEIGHT2 #(
    parameter DEPTH    = 400,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 9,
    parameter OUT_DATA = 8,
    parameter IN_RADDR = 9
) (
    input  logic                clk,
    input  logic                i_we,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    output logic [OUT_DATA-1:0] o_data,
    input  logic [IN_RADDR-1:0] i_raddr
);

    reg [IN_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_we) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule