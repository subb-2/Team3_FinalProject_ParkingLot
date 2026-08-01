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
    //BRAM으로 8bit 보낼 때마다 발생하는 신호
    output logic        o_valid,
    output logic        o_done
);

    typedef enum logic [1:0] {
        IDLE   = 2'b00,
        STREAM,
        DONE
    } p_state;

    p_state c_state, n_state;

    //한 줄 저장
    reg   [                        7:0] temp       [0:DATA_DEPTH-1];
    //행 카운트
    logic [     $clog2(DATA_DEPTH)-1:0] col_cnt;
    logic                               row_parity;
    //홀수 열 max 값 register
    logic [                        7:0] reg_max;
    //BRAM에 보낼 때마다 count. o_done 신호 생성 위해
    logic [10:0] waddr_cnt;
    logic [                        7:0] vmax;

    //col_cnt 번째 열 max값 판단
    assign vmax = (temp[col_cnt] > i_pxl_data) ? temp[col_cnt] : i_pxl_data;


    //state register
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            c_state <= IDLE;
        end else begin
            c_state <= n_state;
        end
    end

    //Next State CL
    always_comb begin
        o_done  = 0;
        n_state = c_state;
        case (c_state)
            IDLE: begin
                if (i_start) begin
                    n_state = STREAM;
                end
            end
            STREAM: begin
                //Pooling 다 끝나면 넘어가 (다채널 대응을 위해 modulo 연산 사용)
                
                if (((o_waddr + 1) % (POOLING_SIZE**2) == 0) && o_valid) begin
                    n_state = DONE;
                end
            end
            DONE: begin
                //done 신호 내보내고 넘어가
                o_done  = 1;
                n_state = IDLE;
            end
            default: n_state = IDLE;
        endcase
    end

    //Data Logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            col_cnt <= 0;
            row_parity <= 0;
            reg_max <= 0;
            waddr_cnt <= 0;
            o_waddr <= 0;
            o_valid <= 0;
        end else begin
            o_valid <= 0;
            if (i_start) begin
                if (row_parity == 0) begin
                    temp[col_cnt] <= i_pxl_data;
                end else begin  //한 줄 다 받았을 때
                    if (col_cnt % 2 == 0) begin //좌측 열 상하 2개 max 레지스터에 저장
                        reg_max <= vmax;
                    end else begin  //우측 열 max와 좌측 열 max 비교해서 send
                        o_pxl_data <= (reg_max > vmax) ? reg_max : vmax;
                        o_waddr <= waddr_cnt;
                        o_valid <= 1;
                        waddr_cnt <= waddr_cnt + 1;
                    end
                end
                if (col_cnt == DATA_DEPTH - 1) begin
                    row_parity <= ~row_parity;
                    col_cnt <= 0;
                end else begin
                    col_cnt <= col_cnt + 1;
                end
            end
        end
    end


endmodule
