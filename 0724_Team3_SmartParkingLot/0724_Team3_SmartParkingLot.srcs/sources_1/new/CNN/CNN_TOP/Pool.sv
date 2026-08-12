`timescale 1ns / 1ps

module pooling #(
    parameter DATA_DEPTH   = 28,
    parameter KERNEL_SIZE  = 2,
    parameter POOLING_SIZE = 14
) (
    input  logic        clk,
    input  logic        reset,
    input  logic [ 7:0] i_pxl_data,
    input  logic        i_start,
    output logic [ 7:0] o_pxl_data,
    output logic [10:0] o_waddr,
    output logic        o_valid,
    output logic        o_done,

    input logic i_done
);

    typedef enum logic [1:0] {
        IDLE   = 2'b00,
        STREAM,
        DONE
    } p_state;

    p_state c_state, n_state;

    // 1. 제곱 연산(**)을 컴파일 타임 상수로 변환 (하드웨어 자원 소비 0)
    localparam TOTAL_OUT = POOLING_SIZE * POOLING_SIZE;

    // 한 줄 저장
    reg   [                   7:0] temp          [0:DATA_DEPTH-1];

    // 행/열 카운트 및 레지스터
    logic [$clog2(DATA_DEPTH)-1:0] col_cnt;
    logic                          row_parity;
    logic [                   7:0] reg_max;
    logic [                  10:0] waddr_cnt;
    logic [                   7:0] vmax;

    // 2. % 연산 제거를 위한 출력 카운터 및 완료 레지스터 플래그
    logic [ $clog2(TOTAL_OUT)-1:0] ch_out_cnt;
    logic                          last_out_done;

    // col_cnt 번째 열 max값 판단
    assign vmax = (temp[col_cnt] > i_pxl_data) ? temp[col_cnt] : i_pxl_data;

    // State Register
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            c_state <= IDLE;
        end else begin
            c_state <= n_state;
        end
    end

    // Next State CL (조합 논리 연산 제거로 Critical Path 해결)
    always_comb begin
        o_done  = 1'b0;
        n_state = c_state;
        case (c_state)
            IDLE: begin
                if (i_start) begin
                    n_state = STREAM;
                end
            end
            STREAM: begin
                // 나누기/나머지 연산 없이 레지스터 플래그 하나만으로 상태 전환
                if (last_out_done) begin
                    n_state = DONE;
                end
            end
            DONE: begin
                o_done  = 1'b1;
                n_state = IDLE;
            end
            default: n_state = IDLE;
        endcase
    end

    // Data Logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            col_cnt       <= '0;
            row_parity    <= 1'b0;
            reg_max       <= '0;
            waddr_cnt     <= '0;
            o_waddr       <= '0;
            o_valid       <= 1'b0;
            ch_out_cnt    <= '0;
            last_out_done <= 1'b0;
        end else if (i_done) begin
            col_cnt       <= '0;
            row_parity    <= 1'b0;
            reg_max       <= '0;
            waddr_cnt     <= '0;
            o_waddr       <= '0;
            o_valid       <= 1'b0;
            ch_out_cnt    <= '0;
            last_out_done <= 1'b0;
        end else begin
            o_valid       <= 1'b0;
            last_out_done <= 1'b0;

            if (c_state == IDLE) begin
                ch_out_cnt <= '0;
            end

            if (i_start) begin
                if (row_parity == 1'b0) begin
                    temp[col_cnt] <= i_pxl_data;
                end else begin  // 짝수 행일 때
                    // 3. % 2 연산 대신 비트[0] 확인으로 대체
                    if (col_cnt[0] == 1'b0) begin
                        reg_max <= vmax;
                    end else begin  // 홀수 열
                        o_pxl_data <= (reg_max > vmax) ? reg_max : vmax;
                        o_waddr    <= waddr_cnt;
                        o_valid    <= 1'b1;
                        waddr_cnt  <= waddr_cnt + 1'b1;

                        // 채널별 마지막 출력 인덱스(TOTAL_OUT - 1) 도달 확인
                        if (ch_out_cnt == TOTAL_OUT - 1) begin
                            ch_out_cnt <= '0;
                            last_out_done <= 1'b1; // 플래그를 정류하여 다음 클록에 STREAM -> DONE 전환
                        end else begin
                            ch_out_cnt <= ch_out_cnt + 1'b1;
                        end
                    end
                end

                if (col_cnt == DATA_DEPTH - 1) begin
                    row_parity <= ~row_parity;
                    col_cnt    <= '0;
                end else begin
                    col_cnt <= col_cnt + 1'b1;
                end
            end
        end
    end

endmodule
