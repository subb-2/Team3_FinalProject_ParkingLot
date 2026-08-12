`timescale 1ns / 1ps

module async_edge_detect (
    input  logic clk,
    input  logic reset,       // Active-High Reset (1일 때 리셋)
    input  logic async_in,    // 외부 비동기 입력 신호
    
    output logic pos_edge,    // 상승 엣지 펄스 (0 -> 1)
    output logic neg_edge,    // 하강 엣지 펄스 (1 -> 0)
    output logic any_edge     // 모든 엣지 펄스 (변화 감지)
);

    // Vivado 등 FPGA 툴에 동기화용 레지스터임을 알림
    (* ASYNC_REG = "TRUE" *) logic sync_ff0;
    (* ASYNC_REG = "TRUE" *) logic sync_ff1;
    
    // 엣지 비교용 지연 레지스터
    logic sync_ff2;

    // -------------------------------------------------------------
    // 1. 2-Stage Flip-Flop Synchronizer (Active-High Reset)
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            sync_ff0 <= 1'b0;
            sync_ff1 <= 1'b0;
        end else begin
            sync_ff0 <= async_in;   // 1차 샘플링
            sync_ff1 <= sync_ff0;   // 2차 샘플링 (클록 시스템 동기화)
        end
    end

    // -------------------------------------------------------------
    // 2. Edge Detection을 위한 1클록 지연 (Active-High Reset)
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            sync_ff2 <= 1'b0;
        end else begin
            sync_ff2 <= sync_ff1;   // 1클록 지연
        end
    end

    // -------------------------------------------------------------
    // 3. Combinational Logic (엣지 출력)
    // -------------------------------------------------------------
    assign pos_edge =  sync_ff1 & ~sync_ff2; // 상승 엣지 (0 -> 1)
    assign neg_edge = ~sync_ff1 &  sync_ff2; // 하강 엣지 (1 -> 0)
    assign any_edge =  sync_ff1 ^  sync_ff2; // 아무 변화나 감지 (XOR)

endmodule