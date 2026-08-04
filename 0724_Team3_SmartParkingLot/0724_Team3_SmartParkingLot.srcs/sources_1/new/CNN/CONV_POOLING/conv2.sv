`timescale 1ns / 1ps

module conv2 (

    input logic       clk,
    input logic       reset,
    input logic       i_start,
    input logic [3:0] i_ch_cnt,

    input logic [7:0] i_pxl_data_0,
    input logic [7:0] i_pxl_data_1,
    input logic [7:0] i_pxl_data_2,
    input logic [7:0] i_pxl_data_3,
    input logic [7:0] i_pxl_data_4,
    input logic [7:0] i_pxl_data_5,

    input logic [7:0] i_weight_0,
    input logic [7:0] i_weight_1,
    input logic [7:0] i_weight_2,
    input logic [7:0] i_weight_3,
    input logic [7:0] i_weight_4,
    input logic [7:0] i_weight_5,

    output logic [7:0] o_raddr,
    output logic [8:0] o_w_raddr,


    output logic [7:0] o_pxl_data,
    output logic       o_done
);

    localparam logic [7:0] BIAS_PARAM[0:15] = '{
        8'hF9,
        8'h0D,
        8'h0B,
        8'hF0,
        8'hE9,
        8'hFF,
        8'h0B,
        8'h09,
        8'h04,
        8'h08,
        8'hE8,
        8'h06,
        8'h03,
        8'h05,
        8'h09,
        8'h04
    };
        

  logic [ 7:0] bias;
  logic [20:0] w_pxl_data_0;
  logic [20:0] w_pxl_data_1;
  logic [20:0] w_pxl_data_2;
  logic [20:0] w_pxl_data_3;
  logic [20:0] w_pxl_data_4;
  logic [20:0] w_pxl_data_5;
  logic        conv2_done;
  logic signed [22:0] conv2_sum;

  logic [10:0] w_raddr;
  logic [11:0] w_wraddr;

  assign o_raddr = w_raddr[7:0];
  assign o_w_raddr = w_wraddr[8:0];


  logic        conv2_done_dly0;
  logic        conv2_done_dly1; 

  assign bias = BIAS_PARAM[i_ch_cnt];

    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            conv2_sum <= 0;
        end
        else if(conv2_done) begin
            conv2_sum <= $signed(w_pxl_data_0) + 
                         $signed(w_pxl_data_1) +
                         $signed(w_pxl_data_2) + 
                         $signed(w_pxl_data_3) +
                         $signed(w_pxl_data_4) +
                         $signed(w_pxl_data_5) +
                         ($signed({{13{bias[7]}}, bias}) <<< 6);
                          ;
        end
        else begin
            conv2_sum <= conv2_sum;
        end
    end

    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            conv2_done_dly0 <= 0;
            conv2_done_dly1 <= 0;
        end
        else begin
            conv2_done_dly0 <= conv2_done;
            conv2_done_dly1 <= conv2_done_dly0;
        end
    end

    assign o_done = conv2_done_dly1;


    always @ (posedge clk or posedge reset) begin
          if(reset) begin
            o_pxl_data <= 8'd0;
          end
          else if(conv2_done_dly0) begin
                if ($signed(conv2_sum) < 0) begin
                  o_pxl_data <= 8'd0;
                end else if (($signed(conv2_sum) >>> 8) > 21'd255) begin
                  o_pxl_data <= 8'd255;
                end else begin
                  o_pxl_data <= $signed(conv2_sum[15:8]);
                end
          end
          else begin
            o_pxl_data <= o_pxl_data;
          end
    end


  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_0 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_0),
      .i_pxl_data(i_pxl_data_0),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (w_raddr),
      .o_pxl_data(w_pxl_data_0),
      .o_w_raddr (w_wraddr),
      .o_done    (conv2_done)
  );

  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_1 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_1),
      .i_pxl_data(i_pxl_data_1),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (),
      .o_pxl_data(w_pxl_data_1),
      .o_w_raddr (),
      .o_done    ()
  );

  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_2 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_2),
      .i_pxl_data(i_pxl_data_2),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (),
      .o_pxl_data(w_pxl_data_2),
      .o_w_raddr (),
      .o_done    ()
  );

  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_3 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_3),
      .i_pxl_data(i_pxl_data_3),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (),
      .o_pxl_data(w_pxl_data_3),
      .o_w_raddr (),
      .o_done    ()
  );
  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_4 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_4),
      .i_pxl_data(i_pxl_data_4),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (),
      .o_pxl_data(w_pxl_data_4),
      .o_w_raddr (),
      .o_done    ()
  );

  cnn_conv2 #(
      .IMG_WIDTH(14)
  ) U_CNN_CONV2_5 (
      .clk       (clk),
      .reset     (reset),
      .i_weight  (i_weight_5),
      .i_pxl_data(i_pxl_data_5),
      .i_start   (i_start),
      .ch_cnt    (i_ch_cnt),
      .o_raddr   (),
      .o_pxl_data(w_pxl_data_5),
      .o_w_raddr (),
      .o_done    ()
  );





endmodule
