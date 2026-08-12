`timescale 1ns / 1ps

module cnn_conv1 #(
    parameter IMG_WIDTH = 32
) (
    input logic        clk,
    input logic        reset,
    input logic [ 7:0] i_weight,
    input logic [31:0] i_pxl_data,
    input logic        i_start,
    input logic [ 2:0] ch_cnt,

    output logic [7:0] o_raddr,
    output logic [7:0] o_pxl_data,
    output logic [8:0] o_w_raddr,
    output logic       o_done
);

  typedef enum logic [2:0] {
    IDLE,
    LOAD,
    CALC,
    DONE,
    WAIT_CHANNEL
  } state_t;

  state_t c_state;
  state_t n_state;

  localparam logic [7:0] BIAS_PARAM[0:5] = '{8'hF7, 8'h1E, 8'h04, 8'h02, 8'h02, 8'h09};

  logic        [ 4:0] cnt;
  logic        [ 2:0] calc_cnt;

  logic        [ 4:0] pxl_x;
  logic        [ 4:0] pxl_y;

  logic        [ 2:0] ch_cnt_d;
  logic               initial_load;

  logic        [31:0] row_buffer       [ 0:4];

  logic signed [ 7:0] pxl_reg          [0:24];
  logic signed [ 7:0] weight_reg       [0:24];

  logic signed [20:0] conv_result;
  logic signed [20:0] conv_bias_result;

  (* use_dsp = "yes" *)
  logic signed [15:0] step0            [0:24];
  logic signed [16:0] step1            [0:12];
  logic signed [17:0] step2            [ 0:6];
  logic signed [18:0] step3            [ 0:3];
  logic signed [19:0] step4            [ 0:1];

  logic [7:0] r_raddr;
  logic [1:0] buf_cnt;

  assign o_raddr = {1'b0, buf_cnt, r_raddr[4:0]};

  // 현재 출력 위치의 5x5 픽셀 선택
  always_comb begin
    for (int row_idx = 0; row_idx < 5; row_idx++) begin
      for (int col_idx = 0; col_idx < 5; col_idx++) begin
        pxl_reg[(row_idx*5)+col_idx] = $signed({7'b0000000, row_buffer[row_idx][pxl_x+col_idx]});
      end
    end
  end

  // Bias 적용
  assign conv_bias_result = conv_result + $signed(
      {{13{BIAS_PARAM[ch_cnt_d][7]}}, BIAS_PARAM[ch_cnt_d]}
  );

  // Weight 메모리 주소
  always_comb begin
    if (cnt < 5'd25) begin
      o_w_raddr = ({6'd0, ch_cnt_d} * 9'd25) + {4'd0, cnt};
    end else begin
      o_w_raddr = ({6'd0, ch_cnt_d} * 9'd25) + 9'd24;
    end
  end

  // 입력 픽셀 메모리 주소
  always_comb begin
    r_raddr = 8'd0;

    if (c_state == LOAD) begin
      if (initial_load) begin
        if (cnt < 5'd5) begin
          r_raddr = {3'd0, cnt};
        end else begin
          r_raddr = 8'd4;
        end
      end else begin
        r_raddr = {3'd0, pxl_y} + 8'd4;
      end
    end
  end

  // FSM State Register
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      c_state <= IDLE;
    end else begin
      c_state <= n_state;
    end
  end

  // FSM Next State Logic
  always_comb begin
    n_state = c_state;

    case (c_state)
      IDLE: begin
        if (i_start) begin
          n_state = LOAD;
        end
      end

      LOAD: begin
        if (initial_load) begin
          if (cnt == 5'd25) begin
            n_state = CALC;
          end
        end else begin
          if (cnt == 5'd1) begin
            n_state = CALC;
          end
        end
      end

      CALC: begin
        if (calc_cnt == 3'd5) begin
          n_state = DONE;
        end
      end

      DONE: begin
        if (pxl_x < 5'd27) begin
          n_state = CALC;
        end else if (pxl_y < 5'd27) begin
          n_state = LOAD;
        end else if (ch_cnt_d == 3'd5) begin
          n_state = IDLE;
        end else begin
          n_state = WAIT_CHANNEL;
        end
      end

      WAIT_CHANNEL: begin
        if (ch_cnt != ch_cnt_d) begin
          n_state = LOAD;
        end
      end

      default: begin
        n_state = IDLE;
      end
    endcase
  end

  // Datapath & Control
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      cnt          <= 5'd0;
      calc_cnt     <= 3'd0;

      pxl_x        <= 5'd0;
      pxl_y        <= 5'd0;

      ch_cnt_d     <= 3'd0;
      initial_load <= 1'b1;

      conv_result  <= 21'sd0;

      o_pxl_data   <= 8'd0;
      o_done       <= 1'b0;

      buf_cnt <= 2'd0;
    end else begin
      o_done <= 1'b0;

      case (c_state)
        IDLE: begin
          cnt          <= 5'd0;
          calc_cnt     <= 3'd0;

          pxl_x        <= 5'd0;
          pxl_y        <= 5'd0;

          initial_load <= 1'b1;

          if (i_start) begin
            ch_cnt_d <= ch_cnt;
          end
        end

        LOAD: begin
          calc_cnt <= 3'd0;

          if (initial_load) begin
            if (cnt < 5'd25) begin
              cnt <= cnt + 1'b1;
            end

            if ((cnt > 5'd0) && (cnt <= 5'd25)) begin

              weight_reg[cnt-1'b1] <= $signed(i_weight);
            end

            if ((cnt > 5'd0) && (cnt <= 5'd5)) begin

              row_buffer[cnt-1'b1] <= i_pxl_data;
            end
          end else begin
            if (cnt < 5'd1) begin
              cnt <= cnt + 1'b1;
            end

            if (cnt == 5'd1) begin
              row_buffer[4] <= i_pxl_data;
            end
          end
        end

        CALC: begin
          cnt <= 5'd0;

          if (calc_cnt < 3'd5) begin
            calc_cnt <= calc_cnt + 1'b1;
          end

          if (calc_cnt == 3'd0) begin
            for (int i = 0; i < 25; i++) begin
              step0[i] <= pxl_reg[i] * weight_reg[i];
            end
          end else if (calc_cnt == 3'd1) begin
            for (int i = 0; i < 12; i++) begin
              step1[i] <= step0[2*i] + step0[(2*i)+1];
            end

            step1[12] <= step0[24];
          end else if (calc_cnt == 3'd2) begin
            for (int i = 0; i < 6; i++) begin
              step2[i] <= step1[2*i] + step1[(2*i)+1];
            end

            step2[6] <= step1[12];
          end else if (calc_cnt == 3'd3) begin
            for (int i = 0; i < 3; i++) begin
              step3[i] <= step2[2*i] + step2[(2*i)+1];
            end

            step3[3] <= step2[6];
          end else if (calc_cnt == 3'd4) begin
            step4[0] <= step3[0] + step3[1];

            step4[1] <= step3[2] + step3[3];
          end else if (calc_cnt == 3'd5) begin
            conv_result <= step4[0] + step4[1];
          end
        end

        DONE: begin
          o_done   <= 1'b1;
          calc_cnt <= 3'd0;
          cnt      <= 5'd0;

          if (conv_bias_result < 0) begin
            o_pxl_data <= 8'd0;
          end else if ((conv_bias_result >>> 1) > 21'd255) begin
            o_pxl_data <= 8'd255;
          end else begin
            o_pxl_data <= conv_bias_result[8:1];
          end

          if (pxl_x < 5'd27) begin
            pxl_x <= pxl_x + 1'b1;
          end else if (pxl_y < 5'd27) begin
            pxl_x <= 5'd0;
            pxl_y <= pxl_y + 1'b1;

            row_buffer[0] <= row_buffer[1];
            row_buffer[1] <= row_buffer[2];
            row_buffer[2] <= row_buffer[3];
            row_buffer[3] <= row_buffer[4];

            initial_load <= 1'b0;
          end else begin
            pxl_x        <= 5'd0;
            pxl_y        <= 5'd0;
            initial_load <= 1'b1;
            if(ch_cnt_d == 3'd5) buf_cnt <= buf_cnt + 1; 
          end
           
        end

        WAIT_CHANNEL: begin
          cnt          <= 5'd0;
          calc_cnt     <= 3'd0;
          initial_load <= 1'b1;

          if (ch_cnt != ch_cnt_d) begin
            ch_cnt_d <= ch_cnt;
          end
        end

        default: begin
          cnt          <= 5'd0;
          calc_cnt     <= 3'd0;
          initial_load <= 1'b1;
        end
      endcase
    end
  end

endmodule

module cnn_conv2 #(
    parameter IMG_WIDTH = 14
) (
    input logic       clk,
    input logic       reset,
    input logic [7:0] i_weight,
    input logic [7:0] i_pxl_data,
    input logic       i_start,
    input logic [3:0] ch_cnt,

    output logic [10:0] o_raddr,
    output logic [20:0] o_pxl_data,
    output logic [11:0] o_w_raddr,
    output logic        o_done
);

  typedef enum logic [2:0] {
    IDLE,
    LOAD,
    CALC,
    DONE,
    WAIT_CHANNEL
  } state_t;

  state_t             c_state;
  state_t             n_state;

  logic        [ 4:0] cnt;
  logic        [ 4:0] load_count;
  logic        [ 2:0] calc_cnt;

  logic        [ 2:0] con_x;
  logic        [ 2:0] con_y;
  logic        [ 3:0] pxl_x;
  logic        [ 3:0] pxl_y;

  logic        [10:0] base_addr;
  logic        [10:0] pxl_addr;

  logic        [ 3:0] ch_cnt_d;
  logic               weight_loaded;

  logic        [ 7:0] pxl_reg       [0:24];
  logic signed [ 7:0] weight_reg    [0:24];

  logic signed [20:0] conv_result;

  (* use_dsp = "yes" *)
  logic signed [16:0] step0         [0:24];
  logic signed [17:0] step1         [0:12];
  logic signed [18:0] step2         [ 0:6];
  logic signed [19:0] step3         [ 0:3];
  logic signed [20:0] step4         [ 0:1];

  assign pxl_addr = base_addr + (con_y * IMG_WIDTH) + con_x;
  assign o_raddr = pxl_addr;
  assign o_pxl_data = conv_result;

  always_comb begin
    if (pxl_x == 4'd0) begin
      load_count = 5'd25;
    end else begin
      load_count = 5'd5;
    end
  end

  always_comb begin
    if (cnt < 5'd25) begin
      o_w_raddr = ({8'd0, ch_cnt_d} * 12'd25) + {7'd0, cnt};
    end else begin
      o_w_raddr = ({8'd0, ch_cnt_d} * 12'd25) + 12'd24;
    end
  end

  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      c_state <= IDLE;
    end else begin
      c_state <= n_state;
    end
  end

  always_comb begin
    n_state = c_state;

    case (c_state)
      IDLE: begin
        if (i_start) begin
          n_state = LOAD;
        end
      end

      LOAD: begin
        if (cnt == load_count) begin
          n_state = CALC;
        end
      end

      CALC: begin
        if (calc_cnt == 3'd5) begin
          n_state = DONE;
        end
      end

      DONE: begin
        if ((pxl_x == 4'd9) && (pxl_y == 4'd9)) begin

          if (ch_cnt_d == 4'd15) begin
            n_state = IDLE;
          end else begin
            n_state = WAIT_CHANNEL;
          end
        end else begin
          n_state = LOAD;
        end
      end

      WAIT_CHANNEL: begin
        if (ch_cnt != ch_cnt_d) begin
          n_state = LOAD;
        end
      end

      default: begin
        n_state = IDLE;
      end
    endcase
  end

  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      cnt           <= 5'd0;
      calc_cnt      <= 3'd0;

      con_x         <= 3'd0;
      con_y         <= 3'd0;
      pxl_x         <= 4'd0;
      pxl_y         <= 4'd0;

      base_addr     <= 11'd0;
      ch_cnt_d      <= 4'd0;
      weight_loaded <= 1'b0;

      conv_result   <= 21'sd0;
      o_done        <= 1'b0;
    end else begin
      o_done <= 1'b0;

      case (c_state)
        IDLE: begin
          cnt           <= 5'd0;
          calc_cnt      <= 3'd0;

          con_x         <= 3'd0;
          con_y         <= 3'd0;
          pxl_x         <= 4'd0;
          pxl_y         <= 4'd0;

          base_addr     <= 11'd0;
          weight_loaded <= 1'b0;

          if (i_start) begin
            ch_cnt_d <= ch_cnt;
          end
        end

        LOAD: begin
          calc_cnt <= 3'd0;

          if (cnt < load_count) begin
            cnt <= cnt + 1'b1;

            if (pxl_x == 4'd0) begin
              if (con_x == 3'd4) begin
                con_x <= 3'd0;
                con_y <= con_y + 1'b1;
              end else begin
                con_x <= con_x + 1'b1;
              end
            end else begin
              con_x <= 3'd4;

              if (con_y < 3'd4) begin
                con_y <= con_y + 1'b1;
              end
            end
          end

          if ((cnt > 5'd0) && (cnt <= load_count)) begin

            if (pxl_x == 4'd0) begin
              pxl_reg[cnt-1'b1] <= i_pxl_data;
            end else begin
              pxl_reg[((cnt-1'b1)*5)+4] <= i_pxl_data;
            end

            if (!weight_loaded) begin
              weight_reg[cnt-1'b1] <= $signed(i_weight);
            end
          end

          if ((!weight_loaded) && (cnt == 5'd25)) begin

            weight_loaded <= 1'b1;
          end
        end

        CALC: begin
          cnt <= 5'd0;

          if (calc_cnt < 3'd5) begin
            calc_cnt <= calc_cnt + 1'b1;
          end

          if (calc_cnt == 3'd0) begin
            for (int i = 0; i < 25; i++) begin
              step0[i] <= $signed({1'b0, pxl_reg[i]}) * $signed(weight_reg[i]);
            end
          end else if (calc_cnt == 3'd1) begin
            for (int i = 0; i < 12; i++) begin
              step1[i] <= step0[2*i] + step0[(2*i)+1];
            end

            step1[12] <= step0[24];
          end else if (calc_cnt == 3'd2) begin
            for (int i = 0; i < 6; i++) begin
              step2[i] <= step1[2*i] + step1[(2*i)+1];
            end

            step2[6] <= step1[12];
          end else if (calc_cnt == 3'd3) begin
            for (int i = 0; i < 3; i++) begin
              step3[i] <= step2[2*i] + step2[(2*i)+1];
            end

            step3[3] <= step2[6];
          end else if (calc_cnt == 3'd4) begin
            step4[0] <= step3[0] + step3[1];
            step4[1] <= step3[2] + step3[3];
          end else if (calc_cnt == 3'd5) begin
            conv_result <= step4[0] + step4[1];
          end
        end

        DONE: begin
            o_done   <= 1'b1;
            cnt      <= 5'd0;
            calc_cnt <= 3'd0;
          if (pxl_x < 4'd9) begin
            for (int row_idx = 0; row_idx < 5; row_idx++) begin
              pxl_reg[(row_idx*5)+0] <= pxl_reg[(row_idx*5)+1];
              pxl_reg[(row_idx*5)+1] <= pxl_reg[(row_idx*5)+2];
              pxl_reg[(row_idx*5)+2] <= pxl_reg[(row_idx*5)+3];
              pxl_reg[(row_idx*5)+3] <= pxl_reg[(row_idx*5)+4];
            end

            pxl_x     <= pxl_x + 1'b1;
            base_addr <= base_addr + 1'b1;
            con_x     <= 3'd4;
            con_y     <= 3'd0;
          end else if (pxl_y < 4'd9) begin
            pxl_x     <= 4'd0;
            pxl_y     <= pxl_y + 1'b1;
            base_addr <= base_addr + 11'd5;
            con_x     <= 3'd0;
            con_y     <= 3'd0;
          end else begin
            pxl_x     <= 4'd0;
            pxl_y     <= 4'd0;
            base_addr <= 11'd0;
            con_x     <= 3'd0;
            con_y     <= 3'd0;
          end
        end

        WAIT_CHANNEL: begin
          cnt           <= 5'd0;
          calc_cnt      <= 3'd0;

          con_x         <= 3'd0;
          con_y         <= 3'd0;
          pxl_x         <= 4'd0;
          pxl_y         <= 4'd0;

          base_addr     <= 11'd0;
          weight_loaded <= 1'b0;

          if (ch_cnt != ch_cnt_d) begin
            ch_cnt_d <= ch_cnt;
          end
        end

        default: begin
          cnt      <= 5'd0;
          calc_cnt <= 3'd0;
        end
      endcase
    end
  end

endmodule


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
        8'h08,
        8'h0E,
        8'h01,
        8'h03,
        8'hFF,
        8'h05,
        8'h08,
        8'hFD,
        8'h01,
        8'hFA,
        8'hFF,
        8'h00,
        8'h05,
        8'h05,
        8'h08,
        8'hFF
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
        if (reset) begin
            o_pxl_data <= 8'd0;
        end
        else if (conv2_done_dly0) begin
            if ($signed(conv2_sum) < 0) begin
                o_pxl_data <= 8'd0;                          // ReLU
            end else if (($signed(conv2_sum) >>> 9) > 21'sd255) begin
                o_pxl_data <= 8'd255;                        // Saturation
            end else begin
                o_pxl_data <= conv2_sum[16:9];               // Valid Range (0~255)
            end
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
