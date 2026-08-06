module SR04_Controller (
    input         clk,
    input         reset,
    input         echo,
    output        trigger,
    output [11:0] distance
);

    // tick generator
    wire w_tick_1MHz;

    SR04_controller U_SR04_CTRL (
        .clk(clk),
        .reset(reset),
        .tick_1MHz(w_tick_1MHz),
        .echo(echo),
        .trigger(trigger),
        .distance(distance)
    );

    tick_gen_1MHz U_TICK_GEN (
        .clk(clk),
        .reset(reset),
        .tick_us(w_tick_1MHz)
    );
endmodule



module SR04_controller (
    input  logic        clk,
    input  logic        reset,
    input  logic        tick_1MHz,
    input  logic        echo,
    output logic        trigger,
    output logic [11:0] distance
);

    localparam [1:0] IDLE = 2'b00, TRIGGER = 2'b01, ECHO = 2'b10;

    // FSM variable
    logic [1:0] current_st, next_st;
    // managing output variable
    logic [11:0]distance_reg;
    logic trigger_reg;
    // trigger counter
    logic [3:0] trigger_cnt_next, trigger_cnt_reg;
    // echo counter
    logic [15:0] echo_cnt_next, echo_cnt_reg;
    // echo signal synchronizer
    logic echo_reg1, echo_reg2;
    // tick resigter for detect rising edge
    logic tick_reg;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            current_st      <= 0;
            tick_reg        <= 0;
            trigger_cnt_reg <= 0;
            echo_cnt_reg    <= 0;
            echo_reg1       <= 0;
            echo_reg2       <= 0;
            distance_reg    <= 11'b0;
            trigger_reg     <= 0;
        end else begin
            current_st <= next_st;
            tick_reg <= tick_1MHz;   //register 값 변경은 clk에 동기해서!
            trigger_cnt_reg <= trigger_cnt_next;
            echo_cnt_reg <= echo_cnt_next;
            distance_reg <= distance;
            trigger_reg <= trigger;
            // 2-stage synchronizer
            echo_reg1 <= echo;
            echo_reg2 <= echo_reg1;
        end
    end

    always @(*) begin
        next_st          = current_st;
        distance         = distance_reg;
        trigger_cnt_next = trigger_cnt_reg;
        echo_cnt_next    = echo_cnt_reg;
        trigger          = trigger_reg;

        case (current_st)
            IDLE: begin
                // initialize counter
                trigger_cnt_next = 0;
                echo_cnt_next    = 0;
                next_st = TRIGGER;
            end
            TRIGGER: begin
                if (tick_1MHz == 1'b1 && tick_reg == 1'b0) begin // rising edge detect
                    if (trigger_cnt_reg < 11) begin
                        trigger = 1'b1;
                        trigger_cnt_next = trigger_cnt_next + 1;
                    end else begin
                        trigger = 1'b0;
                        next_st = ECHO;
                    end
                end
            end
            ECHO: begin
                if (tick_1MHz == 1'b1 && tick_reg == 1'b0) begin // rising edge detect
                    if (echo_reg2 == 1) begin
                        echo_cnt_next = echo_cnt_next + 1;
                    end else if (echo_cnt_reg > 0 && echo_reg2 == 0) begin
                        distance = (echo_cnt_reg * 25'd1130) >> 16;
                        next_st  = IDLE;
                    end
                end
            end
        endcase
    end

endmodule

module tick_gen_1MHz (
    input  logic clk,
    input  logic reset,
    output logic tick_us
);

    // 1usec = 10^-6 sec -> 1_000_000 Hz = 1MHz
    // 100Mhz / 100
    //parameter F_COUNT = 100_000_000 / 100;
    parameter F_COUNT = 100;

    logic [$clog2(F_COUNT)-1:0] r_counter;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            r_counter <= 0;
            tick_us   <= 1'b0;
        end else begin
            r_counter <= r_counter + 1;
            if (r_counter == (F_COUNT - 1)) begin
                r_counter <= 0;
                tick_us   <= 1'b1;
            end else begin
                tick_us <= 1'b0;
            end
        end
    end

endmodule