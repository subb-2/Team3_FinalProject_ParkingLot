`timescale 1ns / 1ps

module BRAM_CONV1_1 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule


module BRAM_CONV1_2 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV1_3 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV1_4 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV1_5 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV1_6 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule


module BRAM_CONV2 #(
    parameter DEPTH    = 50,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 9,
    parameter IN_RADDR = 6,
    parameter OUT_DATA = 64
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    logic [5:0] room_num;
    logic [2:0] seat_num;

    assign room_num = i_waddr[8:3];
    assign seat_num = i_waddr[2:0];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            case (seat_num)
                3'd0:    ram[room_num][7:0] <= i_data;
                3'd1:    ram[room_num][15:8] <= i_data;
                3'd2:    ram[room_num][23:16] <= i_data;
                3'd3:    ram[room_num][31:24] <= i_data;
                3'd4:    ram[room_num][39:32] <= i_data;
                3'd5:    ram[room_num][47:40] <= i_data;
                3'd6:    ram[room_num][55:48] <= i_data;
                3'd7:    ram[room_num][63:56] <= i_data;
                default: ;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

