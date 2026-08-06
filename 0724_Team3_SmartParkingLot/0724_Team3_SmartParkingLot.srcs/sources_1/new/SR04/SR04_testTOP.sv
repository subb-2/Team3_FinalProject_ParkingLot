module SR04_testTOP(
    input  logic clk,
    input  logic reset,
    
    input  logic echo,
    output logic trigger,

    output logic pwm,

    output logic led
);

    // SR04 -> SG90
    logic w_close, w_gate;

    always_comb begin
        if(w_close) w_gate = 1'b0;  // close
        else        w_gate = 1'b1;  // open
    end

    sr04 U_SR04(
        .clk(clk),
        .reset(reset),
        .echo(echo),
        .i_cnn_done(1'b1),
        .trigger(trigger),
        .o_capture(led),
        .o_close(w_close)
    );
    SG90_Controller U_SG90(
        .clk(clk),
        .reset(reset),
        .i_open(w_gate),
        .i_close(~w_gate),
        .o_pwm(pwm)
    );
endmodule