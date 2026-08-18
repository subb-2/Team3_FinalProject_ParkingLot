`timescale 1ns / 1ps

// =================================================================
// Conv3 Top Module (LeNet-5 C5 Layer)
// - 입력: 5x5x16 (400개 픽셀) -> 8개씩 묶어 50개 주소 (o_raddr: 0 ~ 49)
// - 출력: 1x1x120 (120개 픽셀, Signed 8-bit, ReLU) (o_waddr: 0 ~ 119)
// - 가중치: 6000개 주소 (o_w_raddr: 0 ~ 5999)
// =================================================================
module Conv3 (
    input logic clk,
    input logic reset,

    // Control Unit
    input  logic i_start,  // 1클록 Pulse 신호
    output logic o_done,   // 연산 완료 신호

    // Feature Map BRAM Read Address
    input  logic [63:0] i_pxl_data,  // 8개 픽셀 (64-bit)
    output logic [ 5:0] o_raddr,     // 0 ~ 49 (6-bit)

    // BRAM WEIGHT_Conv3 (주소 13-bit)
    input  logic [63:0] i_weight,  // 8개 가중치 (64-bit)
    output logic [12:0] o_w_raddr, // 0 ~ 5999 (13-bit)

    // BRAM Conv3 Output
    output logic signed [7:0] o_pxl_data,  // 최종 결과 (ReLU 적용)
    output logic        [6:0] o_waddr,     // 0 ~ 119 (7-bit)
    output logic              o_we
);

    // -------------------------------------------------------------
    // FSM & 카운터 정의
    // -------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        RUN,
        DONE
    } conv_state;

    conv_state state, next_state;

    logic [5:0] cnt_step;  // 0 ~ 49 (400개 / 8 = 50번 읽기)
    logic [6:0] cnt_out;  // 0 ~ 119 (총 120개 출력 픽셀)

    // 파이프라인 지연 레지스터
    logic valid_d1, valid_d2, valid_d3, valid_d4;
    logic last_step_d1, last_step_d2, last_step_d3, last_step_d4;
    logic first_step_d1, first_step_d2;
    logic [6:0] cnt_out_d1, cnt_out_d2, cnt_out_d3, cnt_out_d4;

    // -------------------------------------------------------------
    // 1. FSM
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (i_start) next_state = RUN;
            RUN: if (cnt_out == 7'd119 && cnt_step == 6'd49) next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // 카운터 제어 (120개 출력 x 50 step)
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            cnt_step <= 6'd0;
            cnt_out  <= 7'd0;
        end else if (state == RUN) begin
            if (cnt_step == 6'd49) begin
                cnt_step <= 6'd0;
                cnt_out  <= (cnt_out == 7'd119) ? 7'd0 : cnt_out + 1'b1;
            end else begin
                cnt_step <= cnt_step + 1'b1;
            end
        end else if (state == IDLE) begin
            cnt_step <= 6'd0;
            cnt_out  <= 7'd0;
        end
    end

    // -------------------------------------------------------------
    // 2. 주소 생성 및 파이프라인 지연
    // -------------------------------------------------------------
    // 입력 피처맵 주소: 0 ~ 49 사이를 120번 반복 연산
    assign o_raddr   = cnt_step;

    // 가중치 주소: 120개 출력 각각 50 step씩 참조 (0 ~ 5999)
    assign o_w_raddr = (cnt_out * 50) + cnt_step;

    assign o_done    = (state == DONE);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            valid_d1      <= 1'b0;
            valid_d2      <= 1'b0;
            valid_d3      <= 1'b0;
            valid_d4      <= 1'b0;
            first_step_d1 <= 1'b0;
            first_step_d2 <= 1'b0;
            last_step_d1  <= 1'b0;
            last_step_d2  <= 1'b0;
            last_step_d3  <= 1'b0;
            last_step_d4  <= 1'b0;
            cnt_out_d1    <= 7'd0;
            cnt_out_d2    <= 7'd0;
            cnt_out_d3    <= 7'd0;
            cnt_out_d4    <= 7'd0;
        end else begin
            valid_d1      <= (state == RUN);
            valid_d2      <= valid_d1;
            valid_d3      <= valid_d2;
            valid_d4      <= valid_d3;

            first_step_d1 <= (state == RUN) && (cnt_step == 6'd0);
            first_step_d2 <= first_step_d1;

            last_step_d1  <= (state == RUN) && (cnt_step == 6'd49);
            last_step_d2  <= last_step_d1;
            last_step_d3  <= last_step_d2;
            last_step_d4  <= last_step_d3;

            cnt_out_d1    <= cnt_out;
            cnt_out_d2    <= cnt_out_d1;
            cnt_out_d3    <= cnt_out_d2;
            cnt_out_d4    <= cnt_out_d3;
        end
    end

    // -------------------------------------------------------------
    // 3. 하위 연산 모듈 인스턴스화
    // -------------------------------------------------------------
    Mac_Adder_Tree_Conv3 u_mac_adder_tree (
        .clk         (clk),
        .reset       (reset),
        .i_valid     (valid_d1),
        .i_clear     (first_step_d2),
        .i_last      (last_step_d3),
        .i_cnt_filter(cnt_out_d3),     // 7-bit [6:0] 전체 전달
        .i_pxl_data  (i_pxl_data),
        .i_weight    (i_weight),
        .o_pxl_data  (o_pxl_data)
    );

    // -------------------------------------------------------------
    // 4. BRAM 쓰기 제어
    // -------------------------------------------------------------
    assign o_we    = valid_d4 && last_step_d4;
    assign o_waddr = cnt_out_d4;

endmodule

`timescale 1ns / 1ps

// =================================================================
// Mac_Adder_Tree_Conv3 Module (LeNet-5 C5 Layer용)
// - 120개 출력 채널에 맞춘 7-bit Filter Index (0~119)
// - 120개 Bias LUT 적용 및 Adder Tree, Quantization(ReLu)
// =================================================================
module Mac_Adder_Tree_Conv3 (
    input logic clk,
    input logic reset,

    input logic       i_valid,
    input logic       i_clear,
    input logic       i_last,
    input logic [6:0] i_cnt_filter, // 7-bit (0~119 커버)

    input logic [63:0] i_pxl_data,
    input logic [63:0] i_weight,

    output logic [7:0] o_pxl_data
);

    // 120개 출력 픽셀에 대응하는 120개 Bias 테이블 (0~119)
    localparam logic signed [7:0] BIAS_LUT_Conv3[0:119] = '{
        8'hf0,
        8'h00,
        8'h0b,
        8'h11,
        8'hfe,
        8'hfe,
        8'h15,
        8'hfa,
        8'hfd,
        8'hfc,
        8'hf3,
        8'hff,
        8'h0c,
        8'h06,
        8'h07,
        8'h0c,
        8'hfe,
        8'h0b,
        8'h02,
        8'h06,
        8'h07,
        8'hf5,
        8'hff,
        8'hfd,
        8'hfb,
        8'h02,
        8'hfd,
        8'h00,
        8'h00,
        8'h08,
        8'hfe,
        8'h09,
        8'hff,
        8'hff,
        8'hf1,
        8'h03,
        8'hf6,
        8'hff,
        8'h07,
        8'h08,
        8'hee,
        8'h01,
        8'hfc,
        8'h03,
        8'hfc,
        8'hf8,
        8'h05,
        8'h08,
        8'h02,
        8'hf4,
        8'hfe,
        8'hf7,
        8'hfb,
        8'hef,
        8'h07,
        8'h0b,
        8'h08,
        8'h09,
        8'hfb,
        8'hfa,
        8'hf6,
        8'hfc,
        8'h02,
        8'h11,
        8'h06,
        8'h01,
        8'h04,
        8'hfc,
        8'h01,
        8'h04,
        8'h02,
        8'hfe,
        8'h05,
        8'hfe,
        8'h09,
        8'hfb,
        8'h13,
        8'h06,
        8'hfc,
        8'hfc,
        8'hff,
        8'h04,
        8'h07,
        8'h04,
        8'hf4,
        8'h00,
        8'h00,
        8'hff,
        8'h0d,
        8'hed,
        8'hff,
        8'hf9,
        8'h02,
        8'h00,
        8'h07,
        8'h00,
        8'h04,
        8'hfe,
        8'h0a,
        8'h08,
        8'hfb,
        8'h03,
        8'h0e,
        8'h0a,
        8'hff,
        8'h00,
        8'h01,
        8'hfa,
        8'hf1,
        8'h00,
        8'hff,
        8'hf7,
        8'hff,
        8'hfd,
        8'h04,
        8'h00,
        8'h02,
        8'h00,
        8'h00,
        8'h08
    };

    (* use_dsp = "yes" *)
    logic signed [15:0] mult[0:7];
    logic valid_stage2;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            valid_stage2 <= 1'b0;
        end else begin
            valid_stage2 <= i_valid;
        end
    end

    // 8비트 곱셈기 8개
    always_ff @(posedge clk) begin
        if (i_valid) begin
            mult[0] <= $signed(
                {1'b0, i_pxl_data[7:0]}
            ) * $signed(
                i_weight[7:0]
            );
            mult[1] <= $signed(
                {1'b0, i_pxl_data[15:8]}
            ) * $signed(
                i_weight[15:8]
            );
            mult[2] <= $signed(
                {1'b0, i_pxl_data[23:16]}
            ) * $signed(
                i_weight[23:16]
            );
            mult[3] <= $signed(
                {1'b0, i_pxl_data[31:24]}
            ) * $signed(
                i_weight[31:24]
            );
            mult[4] <= $signed(
                {1'b0, i_pxl_data[39:32]}
            ) * $signed(
                i_weight[39:32]
            );
            mult[5] <= $signed(
                {1'b0, i_pxl_data[47:40]}
            ) * $signed(
                i_weight[47:40]
            );
            mult[6] <= $signed(
                {1'b0, i_pxl_data[55:48]}
            ) * $signed(
                i_weight[55:48]
            );
            mult[7] <= $signed(
                {1'b0, i_pxl_data[63:56]}
            ) * $signed(
                i_weight[63:56]
            );
        end
    end

    // Adder Tree 덧셈 로직
    logic signed [16:0] sum_st1  [0:3];
    logic signed [17:0] sum_st2  [0:1];
    logic signed [18:0] tree_sum;

    always_comb begin
        sum_st1[0] = mult[0] + mult[1];
        sum_st1[1] = mult[2] + mult[3];
        sum_st1[2] = mult[4] + mult[5];
        sum_st1[3] = mult[6] + mult[7];

        sum_st2[0] = sum_st1[0] + sum_st1[1];
        sum_st2[1] = sum_st1[2] + sum_st1[3];

        tree_sum   = sum_st2[0] + sum_st2[1];
    end

    // 누적기 (Accumulator)
    logic signed [31:0] accum;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            accum <= 32'sd0;
        end else if (valid_stage2) begin
            if (i_clear) accum <= $signed(tree_sum);
            else accum <= accum + $signed(tree_sum);
        end
    end

    // 3. Bias Left Shift (<<< 3 또는 << 3) 적용
    logic signed [31:0] bias_expanded;
    logic signed [31:0] sum_with_bias;
    logic signed [31:0] scaled_val;

    // Bias 8-bit -> 32-bit Sign Extension 후 <<< 3 (Left Shift)
    assign bias_expanded = $signed(
        {{24{BIAS_LUT_Conv3[i_cnt_filter][7]}}, BIAS_LUT_Conv3[i_cnt_filter]}
    ) <<< 3;
    assign sum_with_bias = accum + bias_expanded;

    // 4. Right Shift (>>> 8) 및 Saturation (0 ~ 255) 적용
    assign scaled_val = sum_with_bias >>> 8;  // SHIFT3 = 8

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            o_pxl_data <= 8'd0;
        end else if (i_last) begin
            if (scaled_val < 32'sd0) o_pxl_data <= 8'd0;  // ReLU
            else if (scaled_val > 32'sd255)
                o_pxl_data <= 8'd255;  // u8 Saturation (255)
            else o_pxl_data <= scaled_val[7:0];
        end
    end

endmodule
