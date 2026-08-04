`timescale 1ns / 1ps

module CONV_POOL_BRAM_TOP (
    input  logic        clk,
    input  logic        reset,
    //AXI -> BRAM_WEIGHT1 side
    input  logic        i_we_c1,
    input  logic [ 7:0] i_data_c1,
    input  logic [ 7:0] i_waddr_c1,
    //AXI -> BRAM_WEIGHT2 side
    input  logic        i_we_c2,
    input  logic [ 7:0] i_data_c2,
    input  logic [11:0] i_waddr_c2,
    //VGA side
    input  logic [31:0] i_pxl_data,
    output logic [7:0] o_raddr,
    //CNN Control <-> CONV1_POOl1 side
    input  logic        i_start_c1,
    //CNN Control <-> CONV2_POOl2 side
    output logic        o_done_c2,
    //BRAM_CONV2 <-> CONV3 side
    output logic [63:0] o_data,
    input  logic [ 5:0] i_raddr
);

    
    logic [7:0] c2_o_raddr;
    logic [7:0] c1_o_data_ch1, c1_o_data_ch2, c1_o_data_ch3, c1_o_data_ch4, c1_o_data_ch5, c1_o_data_ch6;
    logic o_done_c1;
   

    CONV1_POOL1_TOP U_CONV1_POOL1 (
        .clk       (clk),
        .reset     (reset),
        .i_we      (i_we_c1),
        .i_data    (i_data_c1),
        .i_waddr   (i_waddr_c1),
        .i_pxl_data(i_pxl_data),
        .o_raddr   (o_raddr),
        .i_raddr   (c2_o_raddr),
        .o_data_ch1 (c1_o_data_ch1),
        .o_data_ch2 (c1_o_data_ch2),
        .o_data_ch3 (c1_o_data_ch3),
        .o_data_ch4 (c1_o_data_ch4),
        .o_data_ch5 (c1_o_data_ch5),
        .o_data_ch6 (c1_o_data_ch6),
        .i_start   (i_start_c1),
        .o_done    (o_done_c1)
    );

    CONV2_POOL2_TOP U_CONV2_POOL2 (
        .clk       (clk),
        .reset     (reset),
        .i_we      (i_we_c2),
        .i_data    (i_data_c2),
        .i_waddr   (i_waddr_c2),
        .i_pxl_data_0(c1_o_data_ch1 ),
        .i_pxl_data_1(c1_o_data_ch2 ),
        .i_pxl_data_2(c1_o_data_ch3 ),
        .i_pxl_data_3(c1_o_data_ch4 ),
        .i_pxl_data_4(c1_o_data_ch5 ),
        .i_pxl_data_5(c1_o_data_ch6 ), 
        .o_raddr   (c2_o_raddr),
        .o_data    (o_data),
        .i_raddr   (i_raddr),
        .i_start   (o_done_c1),
        .o_done    (o_done_c2)
    );



endmodule
