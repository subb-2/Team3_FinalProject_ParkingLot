module SG90_test(
    input  logic clk,
    input  logic reset,
    input  logic sw,
    output logic pwm
);
    
    SG90_Controller U_SG90Ctrl(
        .clk(clk),
        .reset(reset),
        .i_open(sw),
        .i_close(~sw),
        .o_pwm(pwm)
    );
endmodule