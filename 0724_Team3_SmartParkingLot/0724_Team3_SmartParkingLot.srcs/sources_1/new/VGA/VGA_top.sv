module VGA_top(
    input  logic i_pixel_clk,
    input  logic reset,

    // from Pcam(PS)
    input  logic [23:0] i_RGB,
    input  logic [10:0] i_x_pixel,
    input  logic [10:0] i_y_pixel,

    // from CNN
    input  logic [7:0]  i_pixel_addr,
    output logic [31:0] o_pixel_data,

    // from SR04
    input  logic i_capture,

    // to CNN
    output logic o_vga_done
);

    logic frame_we;
    logic [31:0] frame_line;
    logic [7:0]  frame_waddr;

    FrameController U_FrameController(
        .i_pixel_clk(i_pixel_clk),
        .reset(reset),
        .i_RGB(i_RGB),
        .i_x_pixel(i_x_pixel),
        .i_y_pixel(i_y_pixel),
        .o_we(frame_we),
        .o_data(frame_line),
        .o_addr(frame_waddr),
        .o_vga_done(o_vga_done)
    );
    PixelBuffer U_PixelBuffer(
        .i_wclk(i_pixel_clk),
        .i_we(frame_we && i_capture),
        .i_waddr(frame_waddr),
        .i_data(frame_line),
        .i_raddr(i_pixel_addr),
        .o_data(o_pixel_data)
    );
endmodule