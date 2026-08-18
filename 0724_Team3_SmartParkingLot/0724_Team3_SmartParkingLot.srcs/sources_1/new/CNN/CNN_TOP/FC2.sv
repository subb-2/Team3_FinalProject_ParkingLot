`timescale 1ns / 1ps

// =================================================================
// 1. FC2 Top Controller Module
// =================================================================
module FC2 (
    input  logic        clk,
    input  logic        reset,

    // Control Unit
    input  logic        i_start,
    output logic        o_done,

    // BRAM FC1 Output (FC2 Input: 84개 -> 8개씩 11번 읽음)
    input  logic [63:0] i_pxl_data,   // 8개씩 읽음 (64-bit)
    output logic [ 3:0] o_raddr,      // 0 ~ 10 -> 11번

    // BRAM WEIGHT_FC2 (84 x 10 / 8 = 110개 주소)
    input  logic [63:0] i_weight,     // 8개씩 읽음 (64-bit)
    output logic [ 6:0] o_w_raddr,    // 0 ~ 109 -> 110번

    // BRAM FC2 Output (최종 10개 클래스 Score/Logit)
    output logic signed [7:0] o_fc_data,   // 최종 10개 결과
    output logic        [3:0] o_waddr,     // 0 ~ 9
    output logic              o_we
);

    // -------------------------------------------------------------
    // FSM & 카운터 정의
    // -------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        RUN,
        DONE
    } fc2_state;

    fc2_state state, next_state;

    logic [3:0] cnt_step;  // 0 ~ 10 (84 / 8 = 10.5 -> 11번 읽기)
    logic [3:0] cnt_node;  // 0 ~ 9  (10개 클래스)
    logic       done_reading;

    // 파이프라인 지연 레지스터
    logic valid_d1, valid_d2, valid_d3, valid_d4;
    logic last_step_d1, last_step_d2, last_step_d3, last_step_d4;
    logic first_step_d1, first_step_d2, first_step_d3;
    logic [3:0] cnt_node_d1, cnt_node_d2, cnt_node_d3, cnt_node_d4;

    // -------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (i_start) next_state = RUN;
            
            // 🟢 [핵심 수정] 마지막 Node 9 / Step 10의 결과가 BRAM에 완벽히 써진 시점(d4)에 DONE으로 전환
            RUN:  if (valid_d4 && last_step_d4 && (cnt_node_d4 == 4'd9)) next_state = DONE;
            
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------
    // 카운터 및 읽기 종료 상태 제어
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            cnt_step     <= 4'd0;
            cnt_node     <= 4'd0;
            done_reading <= 1'b0;
        end else if (state == RUN) begin
            if (cnt_node == 4'd9 && cnt_step == 4'd10) begin
                // 마지막 데이터 읽기 요청 완료 후 카운터 동결 및 완료 플래그 세팅
                cnt_step     <= cnt_step;
                cnt_node     <= cnt_node;
                done_reading <= 1'b1;
            end else if (cnt_step == 4'd10) begin
                cnt_step     <= 4'd0;
                cnt_node     <= cnt_node + 1'b1;
                done_reading <= 1'b0;
            end else begin
                cnt_step     <= cnt_step + 1'b1;
                done_reading <= 1'b0;
            end
        end else begin
            cnt_step     <= 4'd0;
            cnt_node     <= 4'd0;
            done_reading <= 1'b0;
        end
    end

    // -------------------------------------------------------------
    // 주소 생성 및 지연 파이프라인
    // -------------------------------------------------------------
    assign o_raddr   = cnt_step;  // 0 ~ 10
    assign o_w_raddr = (cnt_node * 11) + cnt_step;
    assign o_done    = (state == DONE);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            valid_d1      <= 1'b0;
            valid_d2      <= 1'b0;
            valid_d3      <= 1'b0;
            valid_d4      <= 1'b0;
            first_step_d1 <= 1'b0;
            first_step_d2 <= 1'b0;
            first_step_d3 <= 1'b0;
            last_step_d1  <= 1'b0;
            last_step_d2  <= 1'b0;
            last_step_d3  <= 1'b0;
            last_step_d4  <= 1'b0;
            cnt_node_d1   <= 4'd0;
            cnt_node_d2   <= 4'd0;
            cnt_node_d3   <= 4'd0;
            cnt_node_d4   <= 4'd0;
        end else begin
            // 읽기 동작이 완료되면 추가 valid 생성 중단
            valid_d1      <= (state == RUN) && !done_reading;
            valid_d2      <= valid_d1;
            valid_d3      <= valid_d2;
            valid_d4      <= valid_d3;

            first_step_d1 <= (cnt_step == 4'd0);
            first_step_d2 <= first_step_d1;
            first_step_d3 <= first_step_d2;

            last_step_d1  <= (cnt_step == 4'd10);
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
    // 하위 연산 모듈 인스턴스화
    // -------------------------------------------------------------
    Mac_Adder_Tree_FC2 u_mac_adder_tree (
        .clk        (clk),
        .reset      (reset),
        .i_valid    (valid_d1),
        .i_clear    (first_step_d2),
        .i_last     (last_step_d1),
        .i_cnt_node (cnt_node_d3),
        .i_pxl_data (i_pxl_data),
        .i_weight   (i_weight),
        .o_fc_data  (o_fc_data)
    );

    // -------------------------------------------------------------
    // BRAM 쓰기 제어 (Stage 4)
    // -------------------------------------------------------------
    assign o_we    = valid_d4 && last_step_d4;
    assign o_waddr = cnt_node_d4;

endmodule


// =================================================================
// 2. FC2 전용 연산 모듈 (Bias LUT 내장, ReLU 제거 & Signed Saturation)
// =================================================================
module Mac_Adder_Tree_FC2 (
    input  logic        clk,
    input  logic        reset,

    // 제어 신호
    input  logic        i_valid,
    input  logic        i_clear,
    input  logic        i_last,
    input  logic [3:0]  i_cnt_node,   // 노드 인덱스 (0 ~ 9)

    // 입력 데이터 & 가중치
    input  logic [63:0] i_pxl_data,
    input  logic [63:0] i_weight,

    // 최종 결과 (Signed 8-bit)
    output logic signed [7:0] o_fc_data
);

    // -------------------------------------------------------------
    // Bias 하드코딩 (10개 노드)
    // -------------------------------------------------------------
    localparam logic signed [31:0] BIAS_LUT[0:9] = '{
        32'hffff_fff8, // f8 
        32'hffff_fffa, // fa 
        32'hffff_fff7, // f7 
        32'h0000_0005, // 05
        32'hffff_fffe, // fe
        32'h0000_000e, // 0e 
        32'hffff_fff7, // f7 
        32'h0000_0009, // 09 
        32'hffff_ffff, // ff
        32'h0000_0004  // 04
    };

    // -------------------------------------------------------------
    // 파이프라인 내부 지연 신호
    // -------------------------------------------------------------
    logic last_stage2, last_stage3;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            last_stage2 <= 1'b0;
            last_stage3 <= 1'b0;
        end else begin
            last_stage2 <= i_last;       // 1클록 지연 (Stage 2용)
            last_stage3 <= last_stage2;  // 2클록 지연 (Stage 3 출력용)
        end
    end

    // -------------------------------------------------------------
    // 1. Stage 1: 병렬 곱셈 (11번째 상위 4바이트 마스킹)
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
            // 하위 4개 바이트 (0~3번): 항상 유효 데이터
            mult[0] <= $signed({1'b0, i_pxl_data[ 7: 0]}) * $signed(i_weight[ 7: 0]);
            mult[1] <= $signed({1'b0, i_pxl_data[15: 8]}) * $signed(i_weight[15: 8]);
            mult[2] <= $signed({1'b0, i_pxl_data[23:16]}) * $signed(i_weight[23:16]);
            mult[3] <= $signed({1'b0, i_pxl_data[31:24]}) * $signed(i_weight[31:24]);

            // 마지막 11번째 읽기(i_last = 1)일 때는 상위 4개 바이트 곱셈 결과를 0으로 마스킹
            if (i_last) begin
                mult[4] <= 16'sd0;
                mult[5] <= 16'sd0;
                mult[6] <= 16'sd0;
                mult[7] <= 16'sd0;
            end else begin
                mult[4] <= $signed({1'b0, i_pxl_data[39:32]}) * $signed(i_weight[39:32]);
                mult[5] <= $signed({1'b0, i_pxl_data[47:40]}) * $signed(i_weight[47:40]);
                mult[6] <= $signed({1'b0, i_pxl_data[55:48]}) * $signed(i_weight[55:48]);
                mult[7] <= $signed({1'b0, i_pxl_data[63:56]}) * $signed(i_weight[63:56]);
            end
        end
    end

    // -------------------------------------------------------------
    // 2. Stage 2: Adder Tree
    // -------------------------------------------------------------
    logic signed [16:0] sum_st1 [0:3];
    logic signed [17:0] sum_st2 [0:1];
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
    // 3. Accumulator
    // -------------------------------------------------------------
    logic signed [31:0] accum;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            accum <= 32'sd0;
        end else if (valid_stage2) begin
            if (i_clear) accum <= $signed(tree_sum);
            else         accum <= accum + $signed(tree_sum);
        end
    end

    // -------------------------------------------------------------
    // 4. Bias 가산 + Shift Right 1 (Signed Saturation)
    // -------------------------------------------------------------
    logic signed [31:0] sum_with_bias;
    logic signed [31:0] scaled_val;

    assign sum_with_bias = accum + (BIAS_LUT[i_cnt_node] <<< 2); // Bias Shift = 2
    
    // 2. Python 레퍼런스의 FC2 Output Right Shift 값 적용 (예: >>> 10)
    assign scaled_val    = sum_with_bias >>> 10; 

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            o_fc_data <= 8'sd0;
        end else if (last_stage3) begin
            // 3. scaled_val 기준으로 8-bit Signed 범위 (-128 ~ 127) Saturation 처리
            if (scaled_val > 32'sd127)        o_fc_data <= 8'sd127;
            else if (scaled_val < -32'sd128)  o_fc_data <= -8'sd128;
            else                              o_fc_data <= scaled_val[7:0];
        end
    end

endmodule
