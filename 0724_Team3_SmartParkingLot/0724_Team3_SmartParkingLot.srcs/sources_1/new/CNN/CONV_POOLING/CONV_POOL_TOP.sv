`timescale 1ns / 1ps

module CONV1_POOL1_TOP (
    input  logic        clk,
    input  logic        reset,
    //AXI -> BRAM_WEIGHT1 side
    input  logic        i_we,
    input  logic [ 7:0] i_data,
    input  logic [ 7:0] i_waddr,
    //VGA side
    input  logic [31:0] i_pxl_data,
    output logic [7:0] o_raddr,
    //BRAM_CONV1 Output
    input  logic [7:0] i_raddr,
    output logic [ 7:0] o_data_ch1,
    output logic [ 7:0] o_data_ch2,
    output logic [ 7:0] o_data_ch3,
    output logic [ 7:0] o_data_ch4,
    output logic [ 7:0] o_data_ch5,
    output logic [ 7:0] o_data_ch6,
    //CNN control side
    input  logic        i_start,
    output logic        o_done
);

    logic c_done, p_valid;
    logic [10:0] p_waddr;
    logic [7:0] conv_o_data, pool_o_data;
    //채널 1~6 카운트
    logic [ 2:0] ch_cnt;
    //BRAM_Weight1 <-> CONV1 side
    logic [ 7:0] c1_i_weight;
    logic [ 8:0] c1_o_w_raddr;

    //BRAM_CONV1 Channel Split
    logic valid_ch1, valid_ch2, valid_ch3, valid_ch4, valid_ch5, valid_ch6;
    logic [7:0] b_waddr;
    assign b_waddr = p_waddr - (ch_cnt * 196);

    //BRAM Output Data
    logic o_ch_done;

    assign o_done = o_ch_done && (ch_cnt == 5);

    BRAM_WEIGHT1 U_BRAM_WEIGHT1 (
        .clk    (clk),
        .i_we   (i_we),
        .i_data (i_data),
        .i_waddr(i_waddr),
        .o_data (c1_i_weight),
        .i_raddr(c1_o_w_raddr)
    );

    cnn_conv1 #(
        .IMG_WIDTH(32)
    ) U_CONV1 (
        .clk       (clk),
        .reset     (reset),
        .i_weight  (c1_i_weight),
        .i_pxl_data(i_pxl_data),
        .i_start   (i_start),
        .ch_cnt    (ch_cnt),
        .o_raddr   (o_raddr),
        .o_pxl_data(conv_o_data),
        .o_w_raddr (c1_o_w_raddr),
        .o_done    (c_done)
    );

    pooling #(
        .DATA_DEPTH  (28),
        .KERNEL_SIZE (2),
        .POOLING_SIZE(14)
    ) U_POOL1 (
        .clk       (clk),
        .reset     (reset),
        .i_pxl_data(conv_o_data),
        .i_start   (c_done),
        .o_pxl_data(pool_o_data),
        .o_waddr   (p_waddr),
        .o_valid   (p_valid),
        .o_done    (o_ch_done)
    );

    BRAM_CONV1_1 U_BRAM_CONV1_1 (
        .clk    (clk),
        .i_valid(valid_ch1),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch1)
    );

    BRAM_CONV1_2 U_BRAM_CONV1_2 (
        .clk    (clk),
        .i_valid(valid_ch2),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch2)
    );

    BRAM_CONV1_3 U_BRAM_CONV1_3 (
        .clk    (clk),
        .i_valid(valid_ch3),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch3)
    );

    BRAM_CONV1_4 U_BRAM_CONV1_4 (
        .clk    (clk),
        .i_valid(valid_ch4),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch4)
    );

    BRAM_CONV1_5 U_BRAM_CONV1_5 (
        .clk    (clk),
        .i_valid(valid_ch5),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch5)
    );

    BRAM_CONV1_6 U_BRAM_CONV1_6 (
        .clk    (clk),
        .i_valid(valid_ch6),
        .i_data (pool_o_data),
        .i_waddr(b_waddr),
        .i_raddr(i_raddr),
        .o_data (o_data_ch6)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            ch_cnt <= 3'd0;
            valid_ch1 <= 0;
            valid_ch2 <= 0;
            valid_ch3 <= 0;
            valid_ch4 <= 0;
            valid_ch5 <= 0;
            valid_ch6 <= 0;
        end else begin
            // Default assignments to ensure one-clock-cycle pulses
            valid_ch1 <= 0;
            valid_ch2 <= 0;
            valid_ch3 <= 0;
            valid_ch4 <= 0;
            valid_ch5 <= 0;
            valid_ch6 <= 0;

            if (o_ch_done) begin
                if (ch_cnt < 6) begin
                    ch_cnt <= ch_cnt + 1'b1;
                end else begin
                    ch_cnt <= 3'd0;
                end
            end
            if (p_valid) begin
                case (ch_cnt)
                    3'd0: valid_ch1 <= 1'b1;
                    3'd1: valid_ch2 <= 1'b1;
                    3'd2: valid_ch3 <= 1'b1;
                    3'd3: valid_ch4 <= 1'b1;
                    3'd4: valid_ch5 <= 1'b1;
                    3'd5: valid_ch6 <= 1'b1;
                    default: ;
                endcase
            end
        end
    end

endmodule
