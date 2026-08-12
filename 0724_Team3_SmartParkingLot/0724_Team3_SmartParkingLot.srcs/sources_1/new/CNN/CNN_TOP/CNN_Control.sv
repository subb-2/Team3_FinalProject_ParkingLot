`timescale 1ns / 1ps

module CNN_Control (
    input  logic clk,
    input  logic reset,         // Active-Low Reset (!reset)

    // from VGA (TOP 모듈에서 Edge Detect 처리된 1-cycle pulse)
    input  logic i_vga_done,    // VGA 프레임 준비 완료 신호 (1회)

    // to Conv1 (전체 CNN 시작 신호)
    output logic o_cnn_start,   // 각 숫자 추론 시작 신호 (각 숫자마다 1클럭 펄스, 총 4회)

    // from FC2 / INF_REG (7-cycle pulse 신호)
    input  logic i_inf_done,    // 각 숫자 추론 완료 신호 (총 4회 입력)

    // to UART
    output logic o_inf_done     // 번호판 4자리 최종 추론 완료 신호 (1회)
);

    // -------------------------------------------------------------
    // 0. Active-High Reset 변환 및 INF Edge Detector 인스턴스화
    // -------------------------------------------------------------
    logic rst_act_high;
    assign rst_act_high = ~reset;

    logic w_inf_done_pulse;

    // INF_REG의 7-cycle pulse -> 1-cycle 상승 엣지 펄스 추출
    async_edge_detect u_edge_inf (
        .clk       (clk),
        .reset     (rst_act_high),
        .async_in  (i_inf_done),
        .pos_edge  (w_inf_done_pulse),
        .neg_edge  (),
        .any_edge  ()
    );

    // -------------------------------------------------------------
    // FSM State & Counter
    // -------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE      = 2'd0,
        CNN_START = 2'd1,
        CNN_WAIT  = 2'd2,
        DONE      = 2'd3
    } cnn_state;

    cnn_state state, next_state;

    // 숫자 4개 카운팅용 레지스터 (0 ~ 3)
    logic [1:0] digit_cnt;

    // -------------------------------------------------------------
    // 1. FSM State Register & Digit Counter (Active-Low Reset)
    // -------------------------------------------------------------
    always_ff @(posedge clk or negedge reset) begin
        if (!reset) begin
            state     <= IDLE;
            digit_cnt <= 2'd0;
        end else begin
            state <= next_state;

            // 숫자 카운터 제어 (내부 Edge Detector로 정제된 1-cycle 펄스 사용)
            if (state == IDLE) begin
                digit_cnt <= 2'd0;
            end else if (state == CNN_WAIT && w_inf_done_pulse) begin
                if (digit_cnt < 2'd3) begin
                    digit_cnt <= digit_cnt + 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------
    // 2. Next State Combinational Logic
    // -------------------------------------------------------------
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                // TOP 모듈에서 정제되어 들어온 VGA 완료 펄스 감지 시 첫 번째 숫자 추론 시작
                if (i_vga_done) next_state = CNN_START;
            end

            CNN_START: begin
                next_state = CNN_WAIT;
            end

            CNN_WAIT: begin
                // INF 완료 엣지 감지 시 동작
                if (w_inf_done_pulse) begin
                    if (digit_cnt == 2'd3) begin
                        next_state = DONE;
                    end else begin
                        next_state = CNN_START;
                    end
                end
            end

            DONE: begin
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------
    // 3. Registered Output Signals (Glitch-Free Register Output)
    // -------------------------------------------------------------
    always_ff @(posedge clk or negedge reset) begin
        if (!reset) begin
            o_cnn_start <= 1'b0;
            o_inf_done  <= 1'b0;
        end else begin
            o_cnn_start <= (next_state == CNN_START);
            o_inf_done  <= (next_state == DONE);
        end
    end

endmodule