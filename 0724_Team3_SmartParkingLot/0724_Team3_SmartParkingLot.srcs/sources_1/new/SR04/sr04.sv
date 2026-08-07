`timescale 1ns / 1ps

module sr04 (
    input  logic clk,
    input  logic pclk,
    input  logic reset,
    input  logic o_echo,
    input  logic i_cnn_done,
    
    output logic o_trigger,
    output logic o_capture,
    output logic o_close
);

    wire [11:0] w_distance; // 센서에서 측정된 거리를 FSM으로 전달하는 선

    logic s_cnn_done;

    SYNC_2FF #(
        .WIDTH(1)
    ) U_SYNC_cnndone (
        .clk(clk),
        .reset(reset),
        .async_in(i_cnn_done),
        .sync_out(s_cnn_done)
    );
    // 초음파 센서 구동부 인스턴스
    SR04_Controller U_SR04_CONTROLLER(
        .clk(clk),
        .reset(reset),
        .echo(o_echo),
        .trigger(o_trigger),
        .distance(w_distance)
    );
    // 카메라 및 문 제어 FSM 인스턴스
    sr04_fsm U_FSM_CONTROL (
        .clk(pclk),
        .reset(reset),
        .i_distance(w_distance),
        .i_cnn_done(s_cnn_done),
        .o_capture(o_capture),
        .o_close(o_close)
    );

endmodule

module sr04_fsm (
    input  logic        clk,
    input  logic        reset,
    input  logic [11:0] i_distance,
    input  logic        i_cnn_done,
    output logic        o_capture,
    output logic        o_close
);

    typedef enum logic [2:0] {
        IDLE,
        WAIT,
        WAIT_CNN,
        OPEN,
        WAIT_DELAY
    } state_e;

    // Variable
    localparam DISTANCE = 3;
    localparam DELAY    = 742_500_000;

    state_e state;
    logic [31:0] delay_cnt;

    logic [11:0] s_distance;

    SYNC_2FF #(
        .WIDTH(12)
    ) U_SYNC_SR04(
        .clk(pclk),
        .reset(reset),
        .async_in(i_distance),
        .sync_out(s_distance)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            delay_cnt <= 32'd0;
        end else begin
            case (state)
                IDLE: begin
                    delay_cnt <= 32'd0;
                    if (s_distance > 12'd0 && s_distance <= DISTANCE) begin
                        state <= WAIT;
                    end
                end
                WAIT: begin
                    if (delay_cnt >= DELAY) begin
                        state <= WAIT_CNN;
                        delay_cnt <= 32'd0;
                    end else begin
                        delay_cnt <= delay_cnt + 1'b1;
                    end
                end
                WAIT_CNN: begin
                    if (i_cnn_done) begin
                        state <= OPEN;
                    end
                end
                OPEN: begin
                    delay_cnt <= 32'd0;
                    state <= WAIT_DELAY;
                end
                WAIT_DELAY: begin
                    if (delay_cnt >= DELAY) begin
                        state <= IDLE;
                    end else begin
                        delay_cnt <= delay_cnt + 1'b1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            o_capture <= 1'b0;
            o_close <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    o_close   <= 1'b1;
                    o_capture <= 1'b0;
                end
                WAIT: begin
                    o_close <= 1'b1;
                    if (delay_cnt == DELAY - 1) begin
                        o_capture <= 1'b1;
                    end
                end
                WAIT_CNN: begin
                    o_close   <= 1'b1;
                    o_capture <= 1'b1;
                end
                OPEN: begin
                    o_close   <= 1'b0;
                    o_capture <= 1'b0;
                end
                WAIT_DELAY: begin
                    o_close   <= 1'b0;
                    o_capture <= 1'b0;
                end
                default: begin
                    o_close   <= 1'b1;
                    o_capture <= 1'b0;
                end
            endcase
        end
    end

endmodule
