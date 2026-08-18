`timescale 1ns / 1ps

module BRAM_WEIGHT1 #(
    parameter DEPTH    = 150,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter OUT_DATA = 8,
    parameter IN_RADDR = 9
) (
    input  logic                clk,
    input  logic                i_we,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    output logic [OUT_DATA-1:0] o_data,
    input  logic [IN_RADDR-1:0] i_raddr
);

    reg [IN_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_we) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV1 #(
    parameter DEPTH    = 196,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 8,
    parameter IN_RADDR = 8,
    parameter OUT_DATA = 8
) (
    input  logic                clk,
    input  logic                i_valid,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    input  logic [IN_RADDR-1:0] i_raddr,
    output logic [OUT_DATA-1:0] o_data
);

    reg [OUT_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_WEIGHT2 #(
    parameter DEPTH    = 400,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 9,
    parameter OUT_DATA = 8,
    parameter IN_RADDR = 9
) (
    input  logic                clk,
    input  logic                i_we,
    input  logic [ IN_DATA-1:0] i_data,
    input  logic [IN_WADDR-1:0] i_waddr,
    output logic [OUT_DATA-1:0] o_data,
    input  logic [IN_RADDR-1:0] i_raddr
);

    reg [IN_DATA-1:0] ram[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (i_we) begin
            ram[i_waddr] <= i_data;
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule


module BRAM_CONV2 #(
    parameter DEPTH    = 50,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 9,
    parameter IN_RADDR = 6,
    parameter OUT_DATA = 64
) (
    input  logic                 clk,
    input  logic                 i_valid,
    input  logic [ IN_DATA-1:0]  i_data,
    input  logic [IN_WADDR-1:0]  i_waddr,
    input  logic [IN_RADDR-1:0]  i_raddr,
    output logic [OUT_DATA-1:0]  o_data
);

    // 1. Vivado 합성기에게 BRAM 매핑을 강제하는 속성(Attribute) 부여
    (* ram_style = "block" *) logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    // 주소 비트 분할
    logic [IN_RADDR-1:0] room_num;
    logic [2:0]          seat_num;

    assign room_num = i_waddr[IN_WADDR-1:3]; // i_waddr[8:3]
    assign seat_num = i_waddr[2:0];

    // 2. 8비트 Byte Write Enable 신호 생성 (1-Hot Encoding)
    logic [7:0] byte_we;

    always_comb begin
        byte_we = 8'b0;
        if (i_valid) begin
            byte_we[seat_num] = 1'b1;
        end
    end

    // 3. Vivado BRAM Inference 표준 가이드라인 적용 (for-loop 기반 Write)
    integer i;
    always_ff @(posedge clk) begin
        for (i = 0; i < 8; i = i + 1) begin
            if (byte_we[i]) begin
                // Indexed Part-Select (+: 8) 이용해 i번째 바이트에 write
                ram[room_num][(i * 8) +: 8] <= i_data;
            end
        end
    end

    // 4. 동기식 Read Port (BRAM Read latency = 1 cycle)
    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule


module BRAM_WE2_TOP #(
    parameter DEPTH    = 2400,
    parameter IN_DATA  = 8,
    parameter IN_WADDR = 12,
    parameter OUT_DATA = 8,
    parameter IN_RADDR = 9
) (
    input logic                clk,
    input logic                i_we,
    input logic [ IN_DATA-1:0] i_data,
    input logic [IN_WADDR-1:0] i_waddr,

    output logic [OUT_DATA-1:0] o_data_ch1,
    output logic [OUT_DATA-1:0] o_data_ch2,
    output logic [OUT_DATA-1:0] o_data_ch3,
    output logic [OUT_DATA-1:0] o_data_ch4,
    output logic [OUT_DATA-1:0] o_data_ch5,
    output logic [OUT_DATA-1:0] o_data_ch6,

    input logic [IN_RADDR-1:0] i_raddr
);

    /*
     * Stage 1
     * 전체 주소에서 출력 채널 계산
     */
    logic [         3:0] out_ch_s1;
    logic [IN_WADDR-1:0] waddr_s1;
    logic [ IN_DATA-1:0] data_s1;
    logic                we_s1 = 1'b0;

    logic [        19:0] out_ch_mult_comb;

    assign out_ch_mult_comb = i_waddr * 20'd437;

    always_ff @(posedge clk) begin
        out_ch_s1 <= out_ch_mult_comb[19:16];
        waddr_s1  <= i_waddr;
        data_s1   <= i_data;
        we_s1     <= i_we;
    end

    /*
     * Stage 2
     * 150개 단위 블록 안의 상대 주소 계산
     */
    logic [        7:0] rel_addr_s2;
    logic [        3:0] out_ch_s2;
    logic [IN_DATA-1:0] data_s2;
    logic               we_s2 = 1'b0;

    logic [       11:0] out_ch_x150_comb;

    /*
     * 반드시 12비트로 확장한 뒤 시프트해야 합니다.
     * 4비트 out_ch를 바로 시프트하면 상위 비트가 잘릴 수 있습니다.
     */
    assign out_ch_x150_comb =
        ({8'd0, out_ch_s1} << 7) +
        ({8'd0, out_ch_s1} << 4) +
        ({8'd0, out_ch_s1} << 2) +
        ({8'd0, out_ch_s1} << 1);

    always_ff @(posedge clk) begin
        rel_addr_s2 <= waddr_s1 - out_ch_x150_comb;
        out_ch_s2   <= out_ch_s1;
        data_s2     <= data_s1;
        we_s2       <= we_s1;
    end

    /*
     * Stage 3
     * 입력 채널 번호 계산
     */
    logic [        2:0] in_ch_idx_s3;
    logic [        7:0] rel_addr_s3;
    logic [        3:0] out_ch_s3;
    logic [IN_DATA-1:0] data_s3;
    logic               we_s3 = 1'b0;

    logic [       12:0] in_ch_mult_comb;

    assign in_ch_mult_comb = rel_addr_s2 * 13'd41;

    always_ff @(posedge clk) begin
        in_ch_idx_s3 <= in_ch_mult_comb[12:10];
        rel_addr_s3  <= rel_addr_s2;
        out_ch_s3    <= out_ch_s2;
        data_s3      <= data_s2;
        we_s3        <= we_s2;
    end

    /*
     * Stage 4
     * 최종 BRAM 주소와 write enable 생성
     */
    logic [7:0] kernel_idx_comb;
    logic [7:0] in_ch_x25_comb;
    logic [8:0] out_ch_x25_comb;

    logic [8:0] target_waddr_q;
    logic [IN_DATA-1:0] wdata_q;
    logic [5:0] we_ch_q = 6'b000000;

    assign in_ch_x25_comb =
        ({5'd0, in_ch_idx_s3} << 4) +
        ({5'd0, in_ch_idx_s3} << 3) +
         {5'd0, in_ch_idx_s3};

    assign kernel_idx_comb = rel_addr_s3 - in_ch_x25_comb;

    assign out_ch_x25_comb =
        ({5'd0, out_ch_s3} << 4) +
        ({5'd0, out_ch_s3} << 3) +
         {5'd0, out_ch_s3};

    always_ff @(posedge clk) begin
        target_waddr_q <= out_ch_x25_comb + {1'b0, kernel_idx_comb};

        wdata_q <= data_s3;

        we_ch_q <= 6'b000000;

        if (we_s3) begin
            case (in_ch_idx_s3)
                3'd0: we_ch_q[0] <= 1'b1;
                3'd1: we_ch_q[1] <= 1'b1;
                3'd2: we_ch_q[2] <= 1'b1;
                3'd3: we_ch_q[3] <= 1'b1;
                3'd4: we_ch_q[4] <= 1'b1;
                3'd5: we_ch_q[5] <= 1'b1;
                default: we_ch_q <= 6'b000000;
            endcase
        end
    end

    /*
     * BRAM instances
     */
    BRAM_WEIGHT2 U_BRAM_WEIGHT2_1 (
        .clk    (clk),
        .i_we   (we_ch_q[0]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch1),
        .i_raddr(i_raddr)
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_2 (
        .clk    (clk),
        .i_we   (we_ch_q[1]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch2),
        .i_raddr(i_raddr)
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_3 (
        .clk    (clk),
        .i_we   (we_ch_q[2]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch3),
        .i_raddr(i_raddr)
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_4 (
        .clk    (clk),
        .i_we   (we_ch_q[3]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch4),
        .i_raddr(i_raddr)
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_5 (
        .clk    (clk),
        .i_we   (we_ch_q[4]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch5),
        .i_raddr(i_raddr)
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_6 (
        .clk    (clk),
        .i_we   (we_ch_q[5]),
        .i_data (wdata_q),
        .i_waddr(target_waddr_q),
        .o_data (o_data_ch6),
        .i_raddr(i_raddr)
    );

endmodule

module BRAM_WEIGHT3 #(
    parameter DEPTH      = 6000,  // 64-bit 워드 기준 (6000 * 8 = 48,000 Bytes)
    parameter IN_DATA    = 8,
    parameter IN_WADDR   = 16,    // 0 ~ 47999 (Byte Address)
    parameter OUT_DATA   = 64,
    parameter IN_RADDR   = 13     // 0 ~ 5999 (64-bit Word Address)
)(
    input  logic                   clk,
    input  logic                   i_we,
    input  logic [IN_DATA-1:0]     i_data,
    input  logic [IN_WADDR-1:0]    i_waddr,
    input  logic [IN_RADDR-1:0]    i_raddr,
    output logic [OUT_DATA-1:0]    o_data
);

    // Xilinx Block RAM 인퍼런스
    (* ram_style = "block" *)
    logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    // 바이트 주소를 64비트 워드 주소 및 바이트 오프셋으로 분할
    logic [IN_RADDR-1:0] w_word_waddr;
    logic [2:0]          w_byte_sel;
    logic [7:0]          w_wea;

    assign w_word_waddr = i_waddr[IN_WADDR-1:3]; // i_waddr / 8
    assign w_byte_sel  = i_waddr[2:0];          // i_waddr % 8

    // 8-bit Byte Enable 마스크 생성
    always_comb begin
        w_wea = 8'b0;
        if (i_we) begin
            w_wea = 8'b0000_0001 << w_byte_sel;
        end
    end

    // Xilinx Standard Byte-Write Enable
    always_ff @(posedge clk) begin
        for (int i = 0; i < 8; i++) begin
            if (w_wea[i]) begin
                ram[w_word_waddr][(i*8) +: 8] <= i_data;
            end
        end
    end

    // Synchronous Read
    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_CONV3 #(
    parameter DEPTH      = 15,    // 15 x 64-bit (120 Bytes)
    parameter IN_DATA    = 8,
    parameter IN_WADDR   = 7,     // 0 ~ 119
    parameter IN_RADDR   = 4,     // 0 ~ 14
    parameter OUT_DATA   = 64
)(
    input  logic                   clk,
    input  logic                   i_valid,
    input  logic [IN_DATA-1:0]     i_data,
    input  logic [IN_WADDR-1:0]    i_waddr,
    input  logic [IN_RADDR-1:0]    i_raddr,
    output logic [OUT_DATA-1:0]    o_data
);

    // Block RAM 하드웨어 블록을 사용하도록 지정
    (* ram_style = "block" *)
    logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    logic [IN_RADDR-1:0] w_word_waddr;
    logic [2:0]          w_byte_sel;
    logic [7:0]          w_wea;

    assign w_word_waddr = i_waddr[IN_WADDR-1:3];
    assign w_byte_sel  = i_waddr[2:0];

    always_comb begin
        w_wea = 8'b0;
        if (i_valid) begin
            w_wea = 8'b0000_0001 << w_byte_sel;
        end
    end

    always_ff @(posedge clk) begin
        for (int i = 0; i < 8; i++) begin
            if (w_wea[i]) begin
                ram[w_word_waddr][(i*8) +: 8] <= i_data;
            end
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule


module BRAM_WEIGHT4 #(
    parameter DEPTH      = 1260,  // 64-bit 워드 기준 (1260 * 8 = 10,080 Bytes)
    parameter IN_DATA    = 8,
    parameter IN_WADDR   = 14,    // 0 ~ 10079 (Byte Address)
    parameter OUT_DATA   = 64,
    parameter IN_RADDR   = 11     // 0 ~ 1259 (64-bit Word Address)
)(
    input  logic                   clk,
    input  logic                   i_we,
    input  logic [IN_DATA-1:0]     i_data,
    input  logic [IN_WADDR-1:0]    i_waddr,
    input  logic [IN_RADDR-1:0]    i_raddr,
    output logic [OUT_DATA-1:0]    o_data
);

    (* ram_style = "block" *)
    logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    logic [IN_RADDR-1:0] w_word_waddr;
    logic [2:0]          w_byte_sel;
    logic [7:0]          w_wea;

    assign w_word_waddr = i_waddr[IN_WADDR-1:3];
    assign w_byte_sel  = i_waddr[2:0];

    always_comb begin
        w_wea = 8'b0;
        if (i_we) begin
            w_wea = 8'b0000_0001 << w_byte_sel;
        end
    end

    always_ff @(posedge clk) begin
        for (int i = 0; i < 8; i++) begin
            if (w_wea[i]) begin
                ram[w_word_waddr][(i*8) +: 8] <= i_data;
            end
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_FC #(
    parameter DEPTH      = 11,    // 11 x 64-bit (88 Bytes)
    parameter IN_DATA    = 8,
    parameter IN_WADDR   = 7,     // 0 ~ 83
    parameter IN_RADDR   = 4,     // 0 ~ 10
    parameter OUT_DATA   = 64
)(
    input  logic                   clk,
    input  logic                   i_we,
    input  logic [IN_DATA-1:0]     i_data,
    input  logic [IN_WADDR-1:0]    i_waddr,
    input  logic [IN_RADDR-1:0]    i_raddr,
    output logic [OUT_DATA-1:0]    o_data
);

    // Block RAM 하드웨어 블록을 사용하도록 지정
    (* ram_style = "block" *)
    logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    logic [IN_RADDR-1:0] w_word_waddr;
    logic [2:0]          w_byte_sel;
    logic [7:0]          w_wea;

    assign w_word_waddr = i_waddr[IN_WADDR-1:3];
    assign w_byte_sel  = i_waddr[2:0];

    initial begin
        for (int k = 0; k < DEPTH; k++) begin
            ram[k] = '0;
        end
    end

    always_comb begin
        w_wea = 8'b0;
        if (i_we) begin
            w_wea = 8'b0000_0001 << w_byte_sel;
        end
    end

    always_ff @(posedge clk) begin
        for (int i = 0; i < 8; i++) begin
            if (w_wea[i]) begin
                ram[w_word_waddr][(i*8) +: 8] <= i_data;
            end
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule

module BRAM_WEIGHT5 #(
    parameter DEPTH      = 110,   // 64-bit 워드 기준 (110 * 8 = 880 Bytes)
    parameter IN_DATA    = 8,
    parameter IN_WADDR   = 10,    // 0 ~ 879 (Byte Address)
    parameter OUT_DATA   = 64,
    parameter IN_RADDR   = 7      // 0 ~ 109 (64-bit Word Address)
)(
    input  logic                   clk,
    input  logic                   i_we,
    input  logic [IN_DATA-1:0]     i_data,
    input  logic [IN_WADDR-1:0]    i_waddr,
    input  logic [IN_RADDR-1:0]    i_raddr,
    output logic [OUT_DATA-1:0]    o_data
);

    (* ram_style = "block" *)
    logic [OUT_DATA-1:0] ram [0:DEPTH-1];

    logic [IN_RADDR-1:0] w_word_waddr;
    logic [2:0]          w_byte_sel;
    logic [7:0]          w_wea;

    assign w_word_waddr = i_waddr[IN_WADDR-1:3];
    assign w_byte_sel  = i_waddr[2:0];

    always_comb begin
        w_wea = 8'b0;
        if (i_we) begin
            w_wea = 8'b0000_0001 << w_byte_sel;
        end
    end

    always_ff @(posedge clk) begin
        for (int i = 0; i < 8; i++) begin
            if (w_wea[i]) begin
                ram[w_word_waddr][(i*8) +: 8] <= i_data;
            end
        end
    end

    always_ff @(posedge clk) begin
        o_data <= ram[i_raddr];
    end

endmodule
