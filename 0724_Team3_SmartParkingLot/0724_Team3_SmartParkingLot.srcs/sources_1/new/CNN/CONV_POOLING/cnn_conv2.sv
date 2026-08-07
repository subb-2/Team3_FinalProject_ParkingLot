`timescale 1ns / 1ps

module cnn_conv2 #(
    parameter IMG_WIDTH = 14
) (
    input  logic        clk,
    input  logic        reset,
    input  logic [ 7:0] i_weight,
    input  logic [ 7:0] i_pxl_data,
    input  logic        i_start,
    input  logic [ 3:0] ch_cnt,
    output logic [10:0] o_raddr,
    output logic [20:0] o_pxl_data,
    output logic [11:0] o_w_raddr,
    output logic        o_done
);

  typedef enum logic [1:0] {
    IDLE,
    LOAD,
    CALC,
    DONE
  } state_t;

  state_t c_state, n_state;

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

  logic               pxl_start;
  logic        [ 4:0] cnt;
  logic        [ 2:0] con_x;
  logic        [ 2:0] con_y;
  logic        [ 3:0] pxl_x;
  logic        [ 3:0] pxl_y;

  logic        [10:0] base_addr;
  logic        [10:0] pxl_addr;
  logic        [ 2:0] calc_cnt;

  logic        [ 7:0] selected_pxl;
  logic        [ 7:0] bias;

  // Pool1 출력은 unsigned 8-bit
  logic        [ 7:0] pxl_reg          [0:24];

  // Weight는 signed 8-bit
  logic signed [ 7:0] weight_reg       [0:24];

  logic signed [20:0] conv_result;
  logic signed [20:0] conv_bias_result;
  logic        [ 3:0] ch_cnt_d;
  logic        [ 7:0] w_raddr_cnt;
  logic               wait_ch_change;

  // unsigned pixel을 9-bit signed 양수로 확장하여 곱하므로
  // 곱셈 결과는 17-bit로 설정
  logic signed [16:0] step0            [0:24];
  logic signed [17:0] step1            [0:12];
  logic signed [18:0] step2            [ 0:6];
  logic signed [19:0] step3            [ 0:3];
  logic signed [20:0] step4            [ 0:1];

  assign pxl_addr = base_addr + (con_y * IMG_WIDTH) + con_x;
  assign bias = BIAS_PARAM[ch_cnt];
  assign o_raddr = pxl_addr;

  assign o_pxl_data = conv_result;

  assign o_w_raddr = {4'd0, w_raddr_cnt} + ({8'd0, ch_cnt} * 12'd25);

  assign selected_pxl = i_pxl_data;

  always_ff @(posedge clk, posedge reset) begin
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
        if (i_start || (pxl_start && (!wait_ch_change || (ch_cnt != ch_cnt_d)))) n_state = LOAD;
      end

      LOAD: begin
        if (cnt == 5'd25) n_state = CALC;
      end

      CALC: begin
        if (calc_cnt == 3'd5) n_state = DONE;
      end

      DONE: begin
        n_state = IDLE;
      end

      default: begin
        n_state = IDLE;
      end
    endcase
  end

  always_ff @(posedge clk, posedge reset) begin
    if (reset) begin
      cnt         <= 5'd0;
      calc_cnt    <= 3'd0;
      con_x       <= 3'd0;
      con_y       <= 3'd0;
      pxl_x       <= 4'd0;
      pxl_y       <= 4'd0;
      base_addr   <= 11'd0;
      o_done      <= 1'b0;
      o_pxl_data  <= 8'd0;
      w_raddr_cnt <= 8'd0;
      conv_result <= 21'sd0;
      ch_cnt_d    <= 4'd0;
      pxl_start   <= 1'b0;
      wait_ch_change <= 1'b0;
    end else begin
      ch_cnt_d <= ch_cnt;
      o_done   <= 1'b0;

      if ((c_state == DONE) && (pxl_x == 4'd9) && (pxl_y == 4'd9)) begin
        wait_ch_change <= 1'b1;
      end else if (ch_cnt != ch_cnt_d) begin
        wait_ch_change <= 1'b0;
      end

      if (ch_cnt != ch_cnt_d) begin
        base_addr <= 11'd0;
        pxl_x     <= 4'd0;
        pxl_y     <= 4'd0;
      end

      case (c_state)
        IDLE: begin
          cnt         <= 5'd0;
          calc_cnt    <= 3'd0;
          w_raddr_cnt <= 8'd0;
          con_x       <= 3'd0;
          con_y       <= 3'd0;

          if (i_start) begin
            pxl_start <= 1'b1;
          end
        end

        LOAD: begin
          if (cnt < 5'd25) begin
            cnt         <= cnt + 1'b1;
            w_raddr_cnt <= w_raddr_cnt + 1'b1;

            if (con_x == 3'd4) begin
              con_x <= 3'd0;
              con_y <= con_y + 1'b1;
            end else begin
              con_x <= con_x + 1'b1;
            end
          end

          if ((cnt > 5'd0) && (cnt <= 5'd25)) begin
            pxl_reg[cnt-1]    <= selected_pxl;
            weight_reg[cnt-1] <= i_weight;
          end
        end

        CALC: begin
          if (calc_cnt < 3'd5) begin
            calc_cnt <= calc_cnt + 1'b1;
          end

          if (calc_cnt == 3'd0) begin
            for (int i = 0; i < 25; i++) begin
              step0[i] <= $signed({1'b0, pxl_reg[i]}) * $signed(weight_reg[i]);
            end
          end else if (calc_cnt == 3'd1) begin
            for (int i = 0; i < 12; i++) begin
              step1[i] <= step0[2*i] + step0[2*i+1];
            end

            step1[12] <= step0[24];
          end else if (calc_cnt == 3'd2) begin
            for (int i = 0; i < 6; i++) begin
              step2[i] <= step1[2*i] + step1[2*i+1];
            end

            step2[6] <= step1[12];
          end else if (calc_cnt == 3'd3) begin
            for (int i = 0; i < 3; i++) begin
              step3[i] <= step2[2*i] + step2[2*i+1];
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
          o_done <= 1'b1;

          if (ch_cnt == ch_cnt_d) begin
            if (pxl_x == 4'd9) begin
              pxl_x <= 4'd0;

              if (pxl_y == 4'd9) begin
                pxl_y     <= 4'd0;
                base_addr <= 11'd0;

                if (ch_cnt == 4'd15) begin
                  pxl_start <= 1'b0;
                end
              end else begin
                pxl_y     <= pxl_y + 1'b1;
                base_addr <= base_addr + 11'd5;
              end
            end else begin
              pxl_x     <= pxl_x + 1'b1;
              base_addr <= base_addr + 11'd1;
            end
          end
        end
      endcase
    end
  end

endmodule
