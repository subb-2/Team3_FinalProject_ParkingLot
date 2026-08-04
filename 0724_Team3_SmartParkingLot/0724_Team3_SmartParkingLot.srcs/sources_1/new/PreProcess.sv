module PreProcess(
    input  logic clk,
    input  logic reset,

    // VGA from PS
    input  logic        pclk,
    input  logic [23:0] RGB,
    input  logic [10:0] x_pixel,
    input  logic [10:0] y_pixel,

    // HW
    input  logic echo,      // from SR04
    output logic trigger,   // to SR04
    output logic pwm,       // to SG90

    // CNN
    input  logic        cnn_done,
    input  logic [7:0]  pixel_addr,
    output logic [31:0] pixel_data
);
    
    // SR04 -> SG90
    logic w_close, w_gate;
    // SR04 -> VGA
    logic w_capture;
    logic w_vga_done;

    /********* Sensor *********/
    sr04 U_SR04(
        .clk(clk),
        .reset(reset),
        .o_echo(echo),
        .i_cnn_done(cnn_done),
        .o_trigger(trigger),
        .o_capture(w_capture),
        .o_close(w_close)
    );
    SG90_Controller U_SG90(
        .clk(clk),
        .reset(reset),
        .i_open(~w_close),
        .i_close(w_close),
        .o_pwm(pwm)
    );

    /********* VGA *********/
    VGA_top U_VGA_TOP(
        .i_pixel_clk(pclk),
        .reset(reset),
        .i_RGB(RGB),
        .i_x_pixel(x_pixel),
        .i_y_pixel(y_pixel),
        .i_pixel_addr(pixel_addr),
        .o_pixel_data(pixel_data),
        .i_capture(w_capture),
        .o_vga_done(w_vga_done)
    );

endmodule