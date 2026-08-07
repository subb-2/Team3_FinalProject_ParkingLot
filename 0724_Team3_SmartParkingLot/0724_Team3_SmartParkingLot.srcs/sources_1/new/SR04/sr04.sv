`timescale 1ns / 1ps

module sr04 (
    input  logic clk,
    input  logic pclk,
    input  logic reset,
    input  logic o_echo,
    input  logic i_cnn_done,
    input  logic i_vga_done,
    
    output logic o_trigger,
    output logic o_capture,
    output logic o_close
);

    logic [11:0] w_distance, s_distance; // 센서에서 측정된 거리를 FSM으로 전달하는 선

    logic s_cnn_done;
    logic w_distance_val, s_distance_val;
    
    always_ff @(posedge pclk or posedge reset) begin
        if (reset) begin
            s_distance <= 12'd0;
        end else if (s_distance_val) begin
            s_distance <= w_distance;
        end
    end

    PulseSync_100M_to_25M U_SYNC_cnndone(
        .clk_100m(clk),
        .clk_25m(pclk),
        .reset(reset),
        .pulse_in(i_cnn_done),
        .pulse_out(s_cnn_done)
    );
    // 초음파 센서 구동부 인스턴스
    SR04_Controller U_SR04_CONTROLLER(
        .clk(clk),
        .reset(reset),
        .echo(o_echo),
        .trigger(o_trigger),
        .distance(w_distance),
        .distance_val(w_distance_val)
    );
    PulseSync_100M_to_25M U_SYNC_distanceval(
        .clk_100m(clk),
        .clk_25m(pclk),
        .reset(reset),
        .pulse_in(w_distance_val),
        .pulse_out(s_distance_val)
    );
    // 카메라 및 문 제어 FSM 인스턴스
    sr04_fsm U_FSM_CONTROL (
        .clk(pclk),
        .reset(reset),
        .i_distance_val(s_distance_val),
        .i_distance(s_distance),
        .i_cnn_done(s_cnn_done),
        .i_vga_done(i_vga_done),
        .o_capture(o_capture),
        .o_close(o_close)
    );

endmodule

module sr04_fsm (
    input  logic        clk,
    input  logic        reset,
    input  logic        i_distance_val,
    input  logic [11:0] i_distance,
    input  logic        i_cnn_done,
    input  logic        i_vga_done,
    output logic        o_capture,
    output logic        o_close
);

    typedef enum logic [2:0] {
        IDLE,
        WAIT,
        WAIT_VGA,
        WAIT_CNN,
        OPEN,
        WAIT_GONE,
        WAIT_DELAY
    } state_e;

    // Variable
    localparam DISTANCE = 3;
    localparam DELAY    = 50_000_000;

    state_e state;
    logic [31:0] delay_cnt;
    logic [2:0]  gone_cnt;


    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state <= IDLE;
            delay_cnt <= 32'd0;
            gone_cnt  <= 3'b000;
        end else begin
            case (state)
                IDLE: begin
                    delay_cnt <= 32'd0;
                    if (i_distance > 12'd0 && i_distance <= DISTANCE) begin
                        state <= WAIT;
                    end
                end
                WAIT: begin
                    if (delay_cnt >= DELAY) begin
                        state <= WAIT_VGA;
                        delay_cnt <= 32'd0;
                    end else begin
                        delay_cnt <= delay_cnt + 1'b1;
                    end
                end
                WAIT_VGA: begin
                    if(i_vga_done) begin
                        state <= WAIT_CNN;
                    end
                end
                WAIT_CNN: begin
                    if (i_cnn_done) begin
                        state <= OPEN;
                    end
                end
                OPEN: begin
                    delay_cnt <= 32'd0;
                    gone_cnt  <= 3'b000;
                    state <= WAIT_GONE;
                end
                WAIT_GONE: begin
                    if(i_distance_val) begin
                        if(i_distance > 12'd5 && i_distance < 12'd100) begin
                            if(gone_cnt >= 3'b111) begin
                                state    <= WAIT_DELAY;
                                gone_cnt <= 3'b000;
                            end else begin
                                gone_cnt <= gone_cnt + 1'b1;
                            end
                        end else begin
                            gone_cnt <= 3'b000;
                        end
                    end
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
                    end else begin
                        o_capture <= 1'b0;
                    end
                end
                WAIT_VGA: begin
                    o_close   <= 1'b1;
                    o_capture <= 1'b1;
                end
                WAIT_CNN: begin
                    o_close   <= 1'b1;
                    o_capture <= 1'b0;
                end
                OPEN: begin
                    o_close   <= 1'b0;
                    o_capture <= 1'b0;
                end
                WAIT_GONE: begin
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
