module FrameController(
    input  logic i_pixel_clk,
    input  logic reset,

    // from SR04
    input  logic i_capture,

    // from PS(Pcam)
    input  logic [23:0] i_RGB,
    input  logic [10:0] i_x_pixel,
    input  logic [10:0] i_y_pixel,

    // to FrameBuffer
    output logic        o_we,
    output logic [31:0] o_data,
    output logic [7:0]  o_addr,

    // to CNN_top
    output logic o_vga_done
);

    // position number
    logic [3:0]  position;
    // Pixel data
    logic [23:0] PXLcrop;
    logic        PXLmono;

    FrameCrop U_FrameCrop(
        .i_pdata(i_RGB),
        .i_x_pixel(i_x_pixel),
        .i_y_pixel(i_y_pixel),
        .o_pdata(PXLcrop),
        .o_pos(position)
    );
    FrameMono U_FrameMono(
        .i_pdata(PXLcrop),
        .o_pdata(PXLmono)
    );
    FrameRegister U_FrameRegister(
        .i_pixel_clk(i_pixel_clk),
        .reset(reset),
        .i_pos(position),
        .i_pdata(PXLmono),
        .o_we(o_we),
        .o_data(o_data),
        .o_addr(o_addr)
    );
endmodule


module FrameCrop(
    input  logic [23:0] i_pdata,
    input  logic [10:0] i_x_pixel,
    input  logic [10:0] i_y_pixel,
    output logic [23:0] o_pdata,
    output logic [3:0]  o_pos
);

    logic [10:0] x_en, y_en;
    assign y_en = ((i_y_pixel >= 456) && (i_y_pixel < 624)) ? 1'b1 : 1'b0;

    // position
    always_comb begin
             if((i_x_pixel >=  624) && (i_x_pixel <  792)) o_pos = 4'b0001;
        else if((i_x_pixel >=  792) && (i_x_pixel <  960)) o_pos = 4'b0010;
        else if((i_x_pixel >=  960) && (i_x_pixel < 1128)) o_pos = 4'b0100;
        else if((i_x_pixel >= 1128) && (i_x_pixel < 1296)) o_pos = 4'b1000;
        else o_pos = 4'b0000;
    end

    // pixel data pass
    always_comb begin
        case(o_pos)
            4'b0001: x_en = (i_x_pixel -  624) % 6;
            4'b0010: x_en = (i_x_pixel -  792) % 6;
            4'b0100: x_en = (i_x_pixel -  960) % 6;
            4'b1000: x_en = (i_x_pixel - 1128) % 6;
        endcase
    end

    // pdata out
    always_comb begin
        if(x_en && y_en) o_pdata = i_pdata;
        else             o_pdata = 24'd0;
    end
endmodule


module FrameMono(
    input  logic [23:0] i_pdata,
    output logic        o_pdata
);

    logic [7:0] gray;
    assign gray = (i_pdata[23:16] >> 2) + (i_pdata[15:8] >> 1) + (i_pdata[15:8] >> 3) + (i_pdata[7:0] >> 3);

    // binary filter
    always_comb begin
        if(gray >= 8'b0111_1111) o_pdata = 1'b1;
        else                     o_pdata = 1'b0;
    end
endmodule


module FrameRegister(
    input  logic        i_pixel_clk,
    input  logic        reset,
    input  logic [3:0]  i_pos,
    input  logic        i_pdata,
    output logic        o_we,
    output logic [31:0] o_data,
    output logic [7:0]  o_addr
);
    logic [27:0] FrameReg;
    logic [4:0]  bitCnt, lineCnt;   // 0 ~ 28

    always_ff @(posedge i_pixel_clk, posedge reset) begin
        if(reset) begin
            o_we    <= 1'b0;
            o_data  <= 32'd0;
            bitCnt  <= 5'd0;
            lineCnt <= 5'd0;
        end else begin
            if(i_pos && bitCnt < 27) begin
                FrameReg <= {FrameReg[26:0], i_pdata};
                bitCnt   <= bitCnt + 1;
                o_we     <= 1'b0;
                o_data   <= 32'd0;
            end else if(bitCnt == 27) begin
                o_data   <= {2'b11, FrameReg[26:0], i_pdata, 2'b11}; // white padding 28*28 -> 32*32
                o_we     <= 1'b1;
                bitCnt   <= 0;
                if(lineCnt == 27) lineCnt <= 0;
                else              lineCnt <= lineCnt + 1;
            end
        end
    end

    always_comb begin
        case(i_pos)
            4'b0001: o_addr = lineCnt + 2;
            4'b0010: o_addr = lineCnt + 34;
            4'b0100: o_addr = lineCnt + 66;
            4'b1000: o_addr = lineCnt + 98;
            default: o_addr = 0;
        endcase
    end
endmodule