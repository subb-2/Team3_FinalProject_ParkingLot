module PixelBuffer(
    // write
    input  logic        i_wclk,
    input  logic        i_we,
    input  logic [7:0]  i_waddr,
    input  logic [31:0] i_data,

    // read
    // input  logic        i_rclk,
    input  logic [7:0]  i_raddr,
    output logic [31:0] o_data
);

    logic [31:0] PixelRAM[0:127];

    // sync write
    always_ff @(posedge i_wclk) begin
        if(i_we) PixelRAM[i_waddr] <= i_data;
    end

    // async read
    always_comb begin
        case(i_raddr)
            // initially padding
            8'd0,  8'd1,  8'd30,  8'd31,
            8'd32, 8'd33, 8'd62,  8'd63,
            8'd64, 8'd65, 8'd94,  8'd95,
            8'd96, 8'd97, 8'd126, 8'd127: begin
                o_data = 32'hffff_ffff;
            end
            default: o_data = PixelRAM[i_raddr];
        endcase
    end
endmodule