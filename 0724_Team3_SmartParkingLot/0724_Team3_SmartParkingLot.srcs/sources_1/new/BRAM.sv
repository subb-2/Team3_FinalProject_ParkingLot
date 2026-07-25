`timescale 1ns / 1ps

module BRAM #(
    parameter DEPTH = 150
) (
    input  logic       clk,
    input  logic [7:0] i_data,
    input  logic [7:0] i_waddr,
    output logic [7:0] o_data,
    output logic [7:0] o_rdddr
);
endmodule

reg [0:7] ram[0:DEPTH-1];

