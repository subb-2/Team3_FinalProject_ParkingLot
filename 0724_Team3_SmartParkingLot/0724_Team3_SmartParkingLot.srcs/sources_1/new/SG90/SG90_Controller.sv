module SG90_Controller(
    input  logic pclk,
    input  logic reset,
    input  logic i_open,
    input  logic i_close,
    output logic o_pwm
);

    localparam OPEN  = 0,
               CLOSE = 1;
    logic state, n_state;

    localparam UNIT = 74_250;
    localparam PWM = 40;
    logic [$clog2(UNIT)-1:0] unit_cnt; // 0.5ms
    logic [$clog2(PWM)-1:0]  pwm_cnt;  // 20ms

    /********* state update *********/
    always_ff @(posedge pclk, posedge reset) begin
        if(reset) begin
            state <= CLOSE;
        end else begin
            state <= n_state;
        end
    end

    /******* next state logic *******/
    always_comb begin
        n_state = state;
        case(state)
            CLOSE: if(i_open)  n_state = OPEN;
            OPEN:  if(i_close) n_state = CLOSE;
        endcase
    end

    /********* output logic *********/
    always_comb begin
        case(state)
            OPEN: begin
            // 90 degree: 1.5ms HIGH - 18.5ms LOW
                if(pwm_cnt < 3) o_pwm = 1'b1;
                else            o_pwm = 1'b0;
            end
            CLOSE: begin
            // 0 degree: 0.5ms HIGH - 19.5ms LOW
                if(pwm_cnt < 1) o_pwm = 1'b1;
                else            o_pwm = 1'b0;
            end
            default: o_pwm = 1'b0;
        endcase
    end

    /********** pwm counter *********/
    always_ff @(posedge pclk, posedge reset) begin
        if(reset) begin
            unit_cnt <= 0;
            pwm_cnt  <= 0;
        end else begin
            if(unit_cnt == UNIT-1) begin
                if(pwm_cnt == PWM-1) pwm_cnt <= 0;
                else                 pwm_cnt <= pwm_cnt + 1;
                unit_cnt <= 0;
            end else begin
                unit_cnt <= unit_cnt + 1;
            end
        end
    end
endmodule
