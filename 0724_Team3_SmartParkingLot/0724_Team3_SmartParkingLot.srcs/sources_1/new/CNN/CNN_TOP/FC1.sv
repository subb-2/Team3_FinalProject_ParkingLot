`timescale 1ns / 1ps

// =================================================================
// 1. FC1 Top Module (Bias 내부 하드코딩 버전)
// - 입력: 120개 (64-bit 버스, 8개씩 15번 읽음)
// - 출력: 84개 노드 (Signed 8-bit, ReLU 적용)
// - Weight 주소: 120 x 84 / 8 = 1260개 (0 ~ 1259)
// =================================================================
module FC1 (
    input logic clk,
    input logic reset,

    // Control Unit
    input  logic i_start,  // 1클록 Pulse 신호
    output logic o_done,   // 연산 완료 신호

    // BRAM Feature Map Input (120개 -> 8개씩 15번 읽음)
    input  logic [63:0] i_pxl_data,  // 8개 픽셀 (64-bit)
    output logic [ 3:0] o_raddr,     // 0 ~ 14

    // BRAM WEIGHT_FC1 (1260개 주소)
    input  logic [63:0] i_weight,  // 8개 가중치 (64-bit)
    output logic [10:0] o_w_raddr, // 0 ~ 1259

    // BRAM FC1 Output (FC2의 입력으로 전달될 84개 결과)
    output logic [7:0] o_fc_data,  // 최종 84개 결과 (ReLU 적용)
    output logic [6:0] o_waddr,    // 0 ~ 83
    output logic       o_we
);

    // -------------------------------------------------------------
    // FSM & 카운터 정의
    // -------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        RUN,
        DONE
    } fc1_state;

    fc1_state state, next_state;

    logic [3:0] cnt_step;  // 0 ~ 14 (120 / 8 = 15번 읽기)
    logic [6:0] cnt_node;  // 0 ~ 83 (84개 노드)

    // 파이프라인 지연 레지스터
    logic valid_d1, valid_d2, valid_d3, valid_d4;
    logic last_step_d1, last_step_d2, last_step_d3, last_step_d4;
    logic first_step_d1, first_step_d2;
    logic [6:0] cnt_node_d1, cnt_node_d2, cnt_node_d3, cnt_node_d4;

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
            RUN: if (cnt_node == 7'd83 && cnt_step == 4'd14) next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // 카운터 제어
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            cnt_step <= 4'd0;
            cnt_node <= 7'd0;
        end else if (state == RUN) begin
            if (cnt_step == 4'd14) begin
                cnt_step <= 4'd0;
                cnt_node <= (cnt_node == 7'd83) ? 7'd0 : cnt_node + 1'b1;
            end else begin
                cnt_step <= cnt_step + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------
    // 2. 주소 생성 및 지연 파이프라인
    // -------------------------------------------------------------
    assign o_raddr   = cnt_step;  // 0 ~ 14
    assign o_w_raddr = (cnt_node * 15) + cnt_step;  // 0 ~ 1259
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
            cnt_node_d1   <= 7'd0;
            cnt_node_d2   <= 7'd0;
            cnt_node_d3   <= 7'd0;
            cnt_node_d4   <= 7'd0;
        end else begin
            valid_d1      <= (state == RUN);
            valid_d2      <= valid_d1;
            valid_d3      <= valid_d2;
            valid_d4      <= valid_d3;

            first_step_d1 <= (cnt_step == 4'd0);
            first_step_d2 <= first_step_d1;

            last_step_d1  <= (cnt_step == 4'd14);
            last_step_d2  <= last_step_d1;
            last_step_d3  <= last_step_d2;
            last_step_d4  <= last_step_d3;

            cnt_node_d1   <= cnt_node;
            cnt_node_d2   <= cnt_node_d1;
            cnt_node_d3   <= cnt_node_d2;
            cnt_node_d4   <= cnt_node_d3;
        end
    end

    // -------------------------------------------------------------
    // 3. 하위 연산 모듈 인스턴스화
    // -------------------------------------------------------------
    Mac_Adder_Tree_FC1 u_mac_adder_tree (
        .clk       (clk),
        .reset     (reset),
        .i_valid   (valid_d1),
        .i_clear   (first_step_d2),
        .i_last    (last_step_d3),
        .i_cnt_node(cnt_node_d3),    // 0 ~ 83 노드 인덱스 전달
        .i_pxl_data(i_pxl_data),
        .i_weight  (i_weight),
        .o_fc_data (o_fc_data)
    );

    // -------------------------------------------------------------
    // 4. BRAM 쓰기 제어
    // -------------------------------------------------------------
    assign o_we    = valid_d4 && last_step_d4;
    assign o_waddr = cnt_node_d4;

endmodule


// =================================================================
// 2. Mac_Adder_Tree_FC1 Module (Bias LUT 내장)
// =================================================================
module Mac_Adder_Tree_FC1 (
    input logic clk,
    input logic reset,

    // 제어 신호
    input logic       i_valid,
    input logic       i_clear,
    input logic       i_last,
    input logic [6:0] i_cnt_node, // 노드 인덱스 (0 ~ 83)

    // 입력 데이터 & 가중치
    input logic [63:0] i_pxl_data,
    input logic [63:0] i_weight,

    // 최종 결과 (Signed 8-bit / ReLU 적용)
    output logic [7:0] o_fc_data
);

    // -------------------------------------------------------------
    // Bias 하드코딩 (84개 노드 전용 LUT)
    // -------------------------------------------------------------
    localparam logic signed [7:0] BIAS_LUT_FC1[0:83] = '{
        8'h03,
        8'hF7,
        8'h03,
        8'hF9,
        8'h05,
        8'h09,
        8'hF6,
        8'hFA,
        8'hF8,
        8'hFE,
        8'h06,
        8'hF9,
        8'h05,
        8'h08,
        8'hFB,
        8'hF9,
        8'h05,
        8'hFF,
        8'h0B,
        8'hFE,
        8'hFB,
        8'h09,
        8'hF8,
        8'h05,
        8'h04,
        8'h04,
        8'hFF,
        8'hFC,
        8'hFE,
        8'hF7,
        8'h01,
        8'hF7,
        8'hFA,
        8'h01,
        8'h05,
        8'hFC,
        8'h04,
        8'hF8,
        8'h05,
        8'hFE,
        8'hF6,
        8'h03,
        8'hF4,
        8'hF3,
        8'hFC,
        8'h04,
        8'h06,
        8'hFA,
        8'hFF,
        8'h09,
        8'h0B,
        8'h0D,
        8'hFB,
        8'h0D,
        8'h09,
        8'hF4,
        8'h05,
        8'h07,
        8'hF8,
        8'h06,
        8'hF3,
        8'h08,
        8'hF7,
        8'hFC,
        8'h09,
        8'h05,
        8'h0A,
        8'hFA,
        8'h08,
        8'h0B,
        8'hFE,
        8'h01,
        8'hF9,
        8'h03,
        8'h09,
        8'h04,
        8'h05,
        8'h02,
        8'hF9,
        8'h0C,
        8'hFB,
        8'hF4,
        8'hFA,
        8'h04
    };

    // -------------------------------------------------------------
    // 1. Stage 1: 8병렬 Signed 곱셈
    // -------------------------------------------------------------
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

    // -------------------------------------------------------------
    // 2. Stage 2: Adder Tree (8개 곱셈 결과 덧셈)
    // -------------------------------------------------------------
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

    // -------------------------------------------------------------
    // 3. Accumulator (15회 x 8개 = 120개 입력 누적)
    // -------------------------------------------------------------
    logic signed [31:0] accum;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            accum <= 32'sd0;
        end else if (valid_stage2) begin
            if (i_clear) accum <= $signed(tree_sum);
            else accum <= accum + $signed(tree_sum);
        end
    end

    // -------------------------------------------------------------
    // 4. Bias 가산(Sign Extension) + Shift 8 + ReLU + Saturation
    // -------------------------------------------------------------
    logic signed [31:0] bias_expanded;
    logic signed [31:0] sum_with_bias;
    logic signed [31:0] scaled_val;

    // 32비트 연산 공간으로 먼저 확정한 후 <<< 2 시프트 연산 수행 (비트 손실 방지)
    assign bias_expanded = $signed(
        {{24{BIAS_LUT_FC1[i_cnt_node][7]}}, BIAS_LUT_FC1[i_cnt_node]}
    ) <<< 2;
    assign sum_with_bias = accum + bias_expanded;
    assign scaled_val = sum_with_bias >>> 7;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            o_fc_data <= 8'd0;
        end else if (i_last) begin
            // ReLU & Unsigned 8-bit Saturation (0 ~ 255)
            if (scaled_val < 32'sd0) o_fc_data <= 8'd0;
            else if (scaled_val > 32'sd255) o_fc_data <= 8'd255;
            else o_fc_data <= scaled_val[7:0];
        end
    end

endmodule
