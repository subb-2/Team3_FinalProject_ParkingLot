module FrameController(
    input  logic i_pixel_clk,
    input  logic reset,

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
    logic        valid;
    // Pixel data
    logic [23:0] PXLcrop;
    logic        PXLmono;

    FrameCrop U_FrameCrop(
        .i_pixel_clk(i_pixel_clk),
        .reset(reset),
        .i_pdata(i_RGB),
        .i_x_pixel(i_x_pixel),
        .i_y_pixel(i_y_pixel),
        .o_pdata(PXLcrop),
        .o_valid(valid)
    );
    FrameMono U_FrameMono(
        .i_pdata(PXLcrop),
        .o_pdata(PXLmono)
    );
    FrameRegister U_FrameRegister(
        .i_pixel_clk(i_pixel_clk),
        .reset(reset),
        .i_valid(valid),
        .i_pdata(PXLmono),
        .o_we(o_we),
        .o_data(o_data),
        .o_addr(o_addr),
        .o_vga_done(o_vga_done)
    );
endmodule


module FrameCrop(
    input  logic        i_pixel_clk,
    input  logic        reset,
    input  logic [23:0] i_pdata,
    input  logic [10:0] i_x_pixel,
    input  logic [10:0] i_y_pixel,
    output logic [23:0] o_pdata,
    output logic        o_valid
);

    logic x_en, y_en;

    // pixel pass
    // assign y_en = ((i_y_pixel >= 456) && (i_y_pixel < 624) && !((i_y_pixel - 456) % 6));
    // always_comb begin
    //          if((i_x_pixel >=  624) && (i_x_pixel <  792)) x_en = (i_x_pixel -  624) % 6 ? 1'b0 : 1'b1;
    //     else if((i_x_pixel >=  792) && (i_x_pixel <  960)) x_en = (i_x_pixel -  792) % 6 ? 1'b0 : 1'b1;
    //     else if((i_x_pixel >=  960) && (i_x_pixel < 1128)) x_en = (i_x_pixel -  960) % 6 ? 1'b0 : 1'b1;
    //     else if((i_x_pixel >= 1128) && (i_x_pixel < 1296)) x_en = (i_x_pixel - 1128) % 6 ? 1'b0 : 1'b1;
    //     else x_en = 1'b0;
    // end

    // // pdata out
    // always_comb begin
    //     if(x_en && y_en) begin
    //         o_pdata = i_pdata;
    //         o_valid = 1'b1;
    //     end else begin
    //         o_pdata = 24'd0;
    //         o_valid = 1'b0;
    //     end
    // end

    assign x_en = ((i_x_pixel >= 208) && (i_x_pixel < 432));
    assign y_en = ((i_y_pixel >= 212) && (i_y_pixel < 268) && (i_y_pixel[0] == 1'b0));

    localparam WAIT = 0,
               ENABLE = 1;
    logic state, n_state;

    logic pxlCnt;

    /********* state update *********/
    always_ff @(posedge i_pixel_clk, posedge reset) begin
        if(reset) begin
            state <= WAIT;
        end else begin
            state <= n_state;
        end
    end

    /******* next state logic *******/
    always_comb begin
        n_state = state;
        case(state)
            WAIT: if(x_en && y_en) n_state = ENABLE;
            ENABLE: if(!(x_en && y_en)) n_state = WAIT;
        endcase
    end

    /******** output logic ********/
    always_ff @(posedge i_pixel_clk, posedge reset) begin
        if(reset) begin
            o_valid <= 0;
            pxlCnt  <= 0;
        end else begin
            case(state)
                WAIT: begin
                    if(x_en && y_en) begin
                        o_valid <= 1'b1;
                        o_pdata <= i_pdata;
                        pxlCnt  <= 1'b0;
                    end else begin
                        o_valid <= 1'b0;
                        o_pdata <= 24'd0;
                        pxlCnt  <= 0;
                    end
                end
                ENABLE: begin
                    if(x_en && y_en) begin
                        if(pxlCnt == 1) begin
                            o_valid <= 1'b1;
                            o_pdata <= i_pdata;
                            pxlCnt  <= 0;
                        end else begin
                            o_valid <= 1'b0;
                            o_pdata <= 24'd0;
                            pxlCnt  <= pxlCnt + 1;
                        end
                    end else begin
                        o_valid <= 1'b0;
                        o_pdata <= 24'd0;
                        pxlCnt  <= 1'b0;
                    end
                end
                default: begin
                    o_valid <= 1'b0;
                    o_pdata <= 24'd0;
                    pxlCnt  <= 1'b0;
                end
            endcase
        end
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
    input  logic        i_valid,
    input  logic        i_pdata,
    output logic        o_we,
    output logic [31:0] o_data,
    output logic [7:0]  o_addr,
    output logic        o_vga_done
);
    logic [27:0] FrameReg;
    logic [4:0]  bitCnt, lineCnt;   // 0 ~ 28
    logic [1:0]  posCnt;            // 0 ~ 3
    logic [1:0]  clkCnt;            // 0 ~ 1

    // always_ff @(posedge i_pixel_clk, posedge reset) begin
    //     if(reset) begin
    //         o_we       <= 1'b0;
    //         o_data     <= 32'd0;
    //         o_vga_done <= 1'b0;
    //         bitCnt     <= 5'd0;
    //         lineCnt    <= 5'd0;
    //         posCnt     <= 2'd0;
    //     end else begin
    //         o_we       <= 1'b0;
    //         o_vga_done <= 1'b0;
    //         if(i_valid && i_pos && bitCnt < 27) begin
    //             FrameReg   <= {FrameReg[26:0], i_pdata};
    //             bitCnt     <= bitCnt + 1;
    //             o_data     <= 32'd0;
    //         end else if(bitCnt == 27) begin
    //             o_data   <= {2'b11, FrameReg[26:0], i_pdata, 2'b11}; // white padding 28*28 -> 32*32
    //             o_we     <= 1'b1;
    //             bitCnt   <= 0;
    //             if(lineCnt == 27) begin
    //                 if(posCnt == 3) begin
    //                     lineCnt    <= 0;
    //                     o_vga_done <= 1'b1;
    //                 end else begin
    //                     posCnt  <= posCnt + 1;
    //                 end
    //             end else begin
    //                 lineCnt <= lineCnt + 1;
    //             end
    //         end
    //     end
    // end

    // always_comb begin
    //     case(i_pos)
    //         4'b0001: o_addr = lineCnt + 2;
    //         4'b0010: o_addr = lineCnt + 34;
    //         4'b0100: o_addr = lineCnt + 66;
    //         4'b1000: o_addr = lineCnt + 98;
    //         default: o_addr = 0;
    //     endcase
    // end

    localparam IDLE = 0,
               PIXEL_REG = 1,
               WRITE = 2,
               VGA_DONE = 3;
    logic [1:0] state, n_state;

    /********* state update *********/
    always_ff @(posedge i_pixel_clk, posedge reset) begin
        if(reset) begin
            state <= IDLE;
        end else begin
            state <= n_state;
        end
    end

    /******* next state logic *******/
    always_comb begin
        n_state = state;
        case(state)
            IDLE:      if(i_valid)                       n_state = PIXEL_REG;
            PIXEL_REG: if(bitCnt == 28)                  n_state = WRITE;
            WRITE:     if(lineCnt == 27 && posCnt == 3)  n_state = VGA_DONE;
                       else                              n_state = PIXEL_REG;
            VGA_DONE:  if(clkCnt == 2)                   n_state = IDLE;
            default:                                     n_state = IDLE;
        endcase
    end

    /******** output logic ********/
    always_ff @(posedge i_pixel_clk, posedge reset) begin
        if(reset) begin
            o_we       <= 1'b0;
            o_data     <= 32'd0;
            o_vga_done <= 1'b0;
            bitCnt     <= 5'd0;
            lineCnt    <= 5'd0;
            posCnt     <= 2'd0;
            clkCnt     <= 1'b0;
        end else begin
            case(state)
            IDLE: begin
                o_we <= 1'b0;
                if(i_valid) begin
                    FrameReg <= {27'd0, i_pdata};
                    bitCnt   <= bitCnt + 1;
                end
            end
            PIXEL_REG: begin
                o_we <= 1'b0;
                if(i_valid && bitCnt < 28) begin
                    FrameReg <= {FrameReg[26:0], i_pdata};
                    bitCnt   <= bitCnt + 1;
                    o_data   <= 32'd0;
                end
            end
            WRITE: begin
                o_we   <= 1'b1;
                o_data <= {2'b11, FrameReg, 2'b11}; // white padding 28*28 -> 32*32
                bitCnt <= 0;
                posCnt <= posCnt + 1;
                case(posCnt)
                    2'd0: o_addr <= lineCnt + 2;
                    2'd1: o_addr <= lineCnt + 34;
                    2'd2: o_addr <= lineCnt + 66;
                    2'd3: o_addr <= lineCnt + 98;
                    default: o_addr <= 0;
                endcase
                if(posCnt == 3) begin
                    lineCnt <= lineCnt + 1;
                end
                if(i_valid) begin
                    FrameReg <= {27'd0, i_pdata};
                    bitCnt   <= 1;
                end
            end
            VGA_DONE: begin
                if(clkCnt == 2) begin
                    o_vga_done <= 1'b0;
                    clkCnt     <= 1'b0;
                    bitCnt     <= 5'd0;
                    lineCnt    <= 5'd0;
                    posCnt     <= 2'd0;
                end else begin
                    o_vga_done <= 1'b1;
                    clkCnt     <= clkCnt + 1;
                end
            end
            default: begin
                o_we       <= 1'b0;
                o_data     <= 32'd0;
                o_vga_done <= 1'b0;
                bitCnt     <= 5'd0;
                lineCnt    <= 5'd0;
                posCnt     <= 2'd0;
                clkCnt     <= 1'b0;
            end
            endcase
        end
    end
endmodule