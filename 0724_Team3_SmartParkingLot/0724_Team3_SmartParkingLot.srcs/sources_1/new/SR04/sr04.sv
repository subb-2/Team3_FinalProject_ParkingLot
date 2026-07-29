`timescale 1ns / 1ps

module sr04 (
    input  logic clk,
    input  logic reset,
    input  logic echo,
    input  logic i_cnn_done,
    
    // output logic trigger,
    output logic o_capture,
    output logic o_open,
    output logic o_close
);

    wire [8:0] w_distance; // 센서에서 측정된 거리를 FSM으로 전달하는 선

    // 초음파 센서 구동부 인스턴스
    sr04_control_top U_SENSOR_TOP (
        .clk(clk),
        .reset(reset),
        .echo(echo),
        .trigger(trigger),
        .distance(w_distance)
    );

    // 카메라 및 문 제어 FSM 인스턴스
    sr04_fsm U_FSM_CONTROL (
        .clk(clk),
        .reset(reset),
        .i_distance(w_distance),
        .i_cnn_done(i_cnn_done),
        .o_capture(o_capture),
        .o_open(o_open),
        .o_close(o_close)
    );

endmodule

module sr04_fsm (
    input  logic       clk,
    input  logic       reset,
    input  logic [8:0] i_distance,
    input  logic       i_cnn_done,
    output logic       o_capture,
    output logic       o_open,
    output logic       o_close
);

    typedef enum logic [2:0] {
        IDLE,
        WAIT,
        WAIT_CNN,
        OPEN,
        WAIT_DELAY
    } state_e;

    state_e state;
    logic [31:0] delay_cnt;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            delay_cnt <= 32'd0;
        end else begin
            case (state)
                IDLE: begin
                    delay_cnt <= 32'd0;
                    if (i_distance > 9'd0 && i_distance <= 9'd3) begin
                        state <= WAIT;
                    end
                end
                WAIT: begin
                    if (delay_cnt >= 200_000_000) begin
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
                    if (i_distance >= 9'd3 || i_distance == 9'd0) begin

                    end
                end
                WAIT_DELAY: begin
                    if (delay_cnt >= 200_000_000) begin
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
            o_open <= 1'b0;
            o_close <= 1'b0;
        end else begin
            o_capture <= 1'b0;
            o_close   <= 1'b0;

            case (state)
                IDLE: begin
                    o_open <= 1'b0;
                end
                WAIT: begin
                    if (delay_cnt == 200_000_000 - 1) begin
                        o_capture <= 1'b1;
                    end
                end
                WAIT_CNN: begin

                end
                OPEN: begin
                    o_open <= 1'b1;
                end
                WAIT_DELAY: begin
                    o_open <= 1'b1;
                    if (delay_cnt == 200_000_000 - 1) begin
                        o_open  <= 1'b0;
                        o_close <= 1'b1;
                    end
                end
                default: begin
                    o_capture <= 1'b0;
                    o_open <= 1'b0;
                    o_close <= 1'b0;
                end
            endcase
        end
    end

endmodule

module sr04_control_top (
    input        clk,
    input        reset,
    input        echo,
    output       trigger,
    output [8:0] distance
);

    wire         w_tick_1us;
    wire         sr04_start;

    // 60ms 마다 자동 start 
    logic [22:0] auto_start_cnt;
    logic        r_sr04_start;

    assign sr04_start = r_sr04_start;

    localparam CNT_60MS = 23'd6_000_000;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            auto_start_cnt <= 23'd0;
            r_sr04_start   <= 1'b0;
        end else begin
            if (auto_start_cnt >= CNT_60MS - 1) begin
                auto_start_cnt <= 23'd0;
                r_sr04_start   <= 1'b1;
            end else begin
                auto_start_cnt <= auto_start_cnt + 1'b1;
                r_sr04_start   <= 1'b0;
            end
        end
    end

    tick_gen_1us U_TICK_1us (
        .clk    (clk),
        .reset  (reset),
        .clk_1us(w_tick_1us)
    );

    sr04_controller U_SR04 (
        .clk     (clk),
        .reset   (reset),
        .tick_1  (w_tick_1us),
        .start   (sr04_start),
        .echo    (echo),
        .trigger (trigger),
        .distance(distance)
    );

endmodule

module sr04_controller (
    input            clk,
    input            reset,
    input            tick_1,
    input            start,
    input            echo,
    output reg       trigger,
    output reg [8:0] distance
);

    //============================================================
    // FSM States
    // IDLE_S   : Wait for start request
    // trigger_S   : Drive trigger high for 10us
    // WAIT_S   : Wait for ECHO rising edge (start of echo pulse)
    // CALC_S   : Count ECHO high width in microseconds
    // CALC_S2  : Convert echo_cnt(us) to distance(cm) by /58
    //============================================================
    localparam IDLE_S = 3'd0, trigger_S = 3'd1, WAIT_S = 3'd2, CALC_S = 3'd3, CALC_S2 = 3'd4;

    parameter TIMEOUT_WAIT = 25000; // (25ms) maximum waiting time for echo_rise
    parameter TIMEOUT_CALC = 25000; // (25ms) maximum time allowed while measuring echo high width

    reg [ 2:0] c_state;
    reg [ 3:0] trigger_cnt;
    reg [14:0] echo_cnt;
    reg [14:0] timeout_cnt;
    reg [18:0] distance_x10;
    reg [18:0] distance_div;

    reg echo_n, echo_f;
    reg edge_reg, echo_rise, echo_fall;

    wire echo_sync;

    assign echo_sync = echo_n;

    // Synchronizer for async ECHO
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            echo_f <= 1'b0;
            echo_n <= 1'b0;
        end else begin
            echo_f <= echo;
            echo_n <= echo_f;
        end
    end

    // Edge detection
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            edge_reg  <= 1'b0;
            echo_rise <= 1'b0;
            echo_fall <= 1'b0;
        end else begin
            echo_rise <= (~edge_reg) & echo_sync;
            echo_fall <= edge_reg & (~echo_sync);
            edge_reg  <= echo_sync;
        end
    end

    // Main FSM
    always @(posedge clk, posedge reset) begin
        if (reset) begin
            c_state      <= IDLE_S;
            trigger      <= 1'b0;
            distance     <= 13'd0;
            trigger_cnt  <= 4'd0;
            echo_cnt     <= 15'd0;
            timeout_cnt  <= 15'd0;
            distance_x10 <= 19'd0;
            distance_div <= 19'd0;
        end else begin
            case (c_state)

                // initialize counters and wait for start
                IDLE_S: begin
                    trigger      <= 1'b0;
                    trigger_cnt  <= 4'd0;
                    echo_cnt     <= 15'd0;
                    timeout_cnt  <= 15'd0;
                    distance_x10 <= 19'd0;
                    distance_div <= 19'd0;
                    if (start) begin
                        trigger <= 1'b1;
                        c_state <= trigger_S;
                    end
                end

                // keep trigger high for 10us
                trigger_S: begin
                    trigger <= 1'b1;
                    if (tick_1) begin
                        if (trigger_cnt == 4'd10) begin
                            trigger     <= 1'b0;
                            trigger_cnt <= 4'd0;
                            timeout_cnt <= 15'd0;
                            c_state     <= WAIT_S;
                        end else begin
                            trigger_cnt <= trigger_cnt + 1;
                        end
                    end
                end

                // Wait for ECHO rising edge, If TIMEOUT_WAIT, return IDLE
                WAIT_S: begin
                    if (echo_rise) begin
                        echo_cnt    <= 15'd0;
                        timeout_cnt <= 15'd0;
                        c_state     <= CALC_S;
                    end else begin
                        if (tick_1) begin
                            if (timeout_cnt >= TIMEOUT_WAIT - 1) begin
                                distance <= 13'd0;
                                c_state  <= IDLE_S;
                            end else begin
                                timeout_cnt <= timeout_cnt + 1;
                            end
                        end
                    end
                end

                // Measure echo high duration(ms)
                CALC_S: begin
                    if (echo_fall) begin
                        distance_x10 <= echo_cnt * 10; // for fixed-point scaling XXX.X cm
                        distance_div <= 19'd0;
                        c_state <= CALC_S2;
                    end else if (tick_1) begin
                        echo_cnt <= echo_cnt + 1;
                        if (timeout_cnt >= TIMEOUT_CALC - 1) begin
                            distance <= 13'd0;
                            c_state  <= IDLE_S;
                        end else begin
                            timeout_cnt <= timeout_cnt + 1;
                        end
                    end
                end

                // Convert time(us) to distance(cm)
                // distance(cm) = echo_time(us) / 58
                // Replaced direct division with iterative subtraction
                // to avoid 'NEGATIVE SLACK'
                CALC_S2: begin
                    if (distance_x10 >= 19'd58) begin
                        distance_x10 <= distance_x10 - 19'd58;  // subtract 58
                        distance_div <= distance_div + 1'b1;  // quotient++
                    end else begin
                        distance <= distance_div;
                        c_state  <= IDLE_S;
                    end
                end
                default: c_state <= IDLE_S;
            endcase
        end
    end

endmodule

module tick_gen_1us (
    input      clk,
    input      reset,
    output reg clk_1us
);
    reg [6:0] count_reg;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count_reg <= 0;
            clk_1us   <= 1'b0;
        end else if (count_reg == 99) begin
            count_reg <= 0;
            clk_1us   <= 1'b1;
        end else begin
            count_reg <= count_reg + 1;
            clk_1us   <= 1'b0;
        end
    end
endmodule
