`timescale 1ns / 1ps

module BRAM_WE2_TOP #(
    parameter DEPTH    = 2400,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 12,
    parameter OUT_DATA = 8,
    parameter IN_RADDR = 9
) (
    input  logic                clk,
    input  logic                i_we,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    output logic [OUT_DATA-1:0] o_data_ch1,
    output logic [OUT_DATA-1:0] o_data_ch2,
    output logic [OUT_DATA-1:0] o_data_ch3,
    output logic [OUT_DATA-1:0] o_data_ch4,
    output logic [OUT_DATA-1:0] o_data_ch5,
    output logic [OUT_DATA-1:0] o_data_ch6,
    input  logic [IN_RADDR-1:0] i_raddr
);

//waddr -> 12bit로 들어오는 걸 6개로 쪼개서 해당 BRAM에 넣어줘야 함
//raddr -> 9bit로 들어옴. 딱히 처리할 필요 x 

    logic we_ch1, we_ch2, we_ch3, we_ch4, we_ch5, we_ch6;
    logic [8:0] waddr_ch1, waddr_ch2, waddr_ch3, waddr_ch4, waddr_ch5, waddr_ch6;
    logic [2:0] ch_num;

    assign ch_num = i_waddr / 400;
    assign waddr_ch1 = i_waddr - ch_num * 400;
    assign waddr_ch2 = i_waddr - ch_num * 400;
    assign waddr_ch3 = i_waddr - ch_num * 400;
    assign waddr_ch4 = i_waddr - ch_num * 400;
    assign waddr_ch5 = i_waddr - ch_num * 400;
    assign waddr_ch6 = i_waddr - ch_num * 400;

    always_comb begin
        //we 초기화
        we_ch1 = 1'b0;
        we_ch2 = 1'b0;
        we_ch3 = 1'b0;
        we_ch4 = 1'b0;
        we_ch5 = 1'b0;
        we_ch6 = 1'b0;
        case(ch_num) 
        3'd0: begin 
            we_ch1 = i_we;
        end
        3'd1: begin 
            we_ch2 = i_we;
        end
        3'd2: begin 
            we_ch3 = i_we;
        end
        3'd3: begin 
            we_ch4 = i_we;
        end
        3'd4: begin 
            we_ch5 = i_we;
        end
        3'd5: begin 
            we_ch6 = i_we;
        end
        endcase
    end

BRAM_WEIGHT2 U_BRAM_WEIGHT2_1 (
    .clk        (clk    ),
    .i_we       (we_ch1 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch1),
    .o_data     (o_data_ch1),
    .i_raddr    (i_raddr)
);

BRAM_WEIGHT2 U_BRAM_WEIGHT2_2 (
    .clk        (clk    ),
    .i_we       (we_ch2 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch2),
    .o_data     (o_data_ch2),
    .i_raddr    (i_raddr)
);

BRAM_WEIGHT2 U_BRAM_WEIGHT2_3 (
    .clk        (clk    ),
    .i_we       (we_ch3 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch3),
    .o_data     (o_data_ch3),
    .i_raddr    (i_raddr)
);

BRAM_WEIGHT2 U_BRAM_WEIGHT2_4 (
    .clk        (clk    ),
    .i_we       (we_ch4 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch4),
    .o_data     (o_data_ch4 ),
    .i_raddr    (i_raddr)
);

BRAM_WEIGHT2 U_BRAM_WEIGHT2_5 (
    .clk        (clk    ),
    .i_we       (we_ch5 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch5),
    .o_data     (o_data_ch5 ),
    .i_raddr    (i_raddr)
);

BRAM_WEIGHT2 U_BRAM_WEIGHT2_6 (
    .clk        (clk    ),
    .i_we       (we_ch6 ),
    .i_data     (i_data ),
    .i_waddr    (waddr_ch6),
    .o_data     (o_data_ch6),
    .i_raddr    (i_raddr)
);
endmodule
