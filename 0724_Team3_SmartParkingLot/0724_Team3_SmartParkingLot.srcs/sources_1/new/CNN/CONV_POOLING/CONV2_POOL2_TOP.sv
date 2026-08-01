`timescale 1ns / 1ps

module CONV2_POOL2_TOP (
    input  logic        clk,
    input  logic        reset,
    //AXI -> BRAM_WEIGHT2 side
    input  logic        i_we,
    input  logic [ 7:0] i_data,
    input  logic [11:0] i_waddr,
    //BRAM_CONV1 <-> CONV2 side
    input  logic [ 7:0] i_pxl_data_0,
    input  logic [ 7:0] i_pxl_data_1,
    input  logic [ 7:0] i_pxl_data_2,
    input  logic [ 7:0] i_pxl_data_3,
    input  logic [ 7:0] i_pxl_data_4,
    input  logic [ 7:0] i_pxl_data_5,
    output logic [7:0] o_raddr,
    //BRAM_CONV2 <-> CONV3 side
    output logic [63:0] o_data,
    input  logic [ 5:0] i_raddr,
    //CNN Control side
    input  logic        i_start,
    output logic        o_done
);

    //BRAM_WEIGHT2 <-> CONV2 side
    logic [ 7:0] c2_i_weight;
    logic [8:0] c2_o_w_raddr;
    //채널 1~16 카운트
    logic [ 3:0] ch_cnt;

    logic        c_done;
    logic [ 7:0] conv_o_data;

    logic [ 7:0] pool_o_data;
    logic [10:0] p_waddr;
    logic        p_valid;

    logic        o_ch_done;

    logic [7:0] o_data_ch1,o_data_ch2,o_data_ch3,o_data_ch4,o_data_ch5,o_data_ch6;

    assign o_done = o_ch_done && (ch_cnt == 15);


    BRAM_WE2_TOP U_BRAM_WE2 (
    .clk            (clk            ),
    .i_we           (i_we           ),
    .i_data         (i_data         ),
    .i_waddr        (i_waddr        ),
    .o_data_ch1     (o_data_ch1     ),
    .o_data_ch2     (o_data_ch2     ),
    .o_data_ch3     (o_data_ch3     ),
    .o_data_ch4     (o_data_ch4     ),
    .o_data_ch5     (o_data_ch5     ),
    .o_data_ch6     (o_data_ch6     ),
    .i_raddr        (c2_o_w_raddr        )
);

    conv2 U_CONV2(

    .clk        (clk     ),
    .reset      (reset   ),
    .i_start    (i_start ),
    .i_ch_cnt   (ch_cnt),
    .i_pxl_data_0(i_pxl_data_0),
    .i_pxl_data_1(i_pxl_data_1),
    .i_pxl_data_2(i_pxl_data_2),
    .i_pxl_data_3(i_pxl_data_3),
    .i_pxl_data_4(i_pxl_data_4),
    .i_pxl_data_5(i_pxl_data_5),
    .i_weight_0(o_data_ch1 ),
    .i_weight_1(o_data_ch2 ),
    .i_weight_2(o_data_ch3 ),
    .i_weight_3(o_data_ch4 ),
    .i_weight_4(o_data_ch5 ),
    .i_weight_5(o_data_ch6 ),
    .o_raddr(o_raddr),
    .o_w_raddr(c2_o_w_raddr),
    .o_pxl_data(conv_o_data),
    .o_done(c_done)
);

    pooling #(
        .DATA_DEPTH  (10),
        .KERNEL_SIZE (2),
        .POOLING_SIZE(5)
    ) U_POOL2 (
        .clk       (clk),
        .reset     (reset),
        .i_pxl_data(conv_o_data),
        .i_start   (c_done),
        .o_pxl_data(pool_o_data),
        .o_waddr   (p_waddr),
        .o_valid   (p_valid),
        .o_done    (o_ch_done)
    );

    BRAM_CONV2 U_BRAM_CONV2 (
        .clk    (clk),
        .i_valid(p_valid),
        .i_data (pool_o_data),
        .i_waddr(p_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            ch_cnt <= 3'd0;
        end else if (o_ch_done) begin
            ch_cnt <= ch_cnt + 1'b1;
        end
    end

endmodule
