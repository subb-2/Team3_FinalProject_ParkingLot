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
    parameter DEPTH     = 2400,
    parameter IN_DATA   = 8,
    parameter IN_WADDR  = 12,
    parameter OUT_DATA  = 8,
    parameter IN_RADDR  = 9
) (
    input  logic                 clk,
    input  logic                 i_we,
    input  logic [ IN_DATA-1:0]  i_data,
    input  logic [IN_WADDR-1:0]  i_waddr,
    output logic [OUT_DATA-1:0]  o_data_ch1,
    output logic [OUT_DATA-1:0]  o_data_ch2,
    output logic [OUT_DATA-1:0]  o_data_ch3,
    output logic [OUT_DATA-1:0]  o_data_ch4,
    output logic [OUT_DATA-1:0]  o_data_ch5,
    output logic [OUT_DATA-1:0]  o_data_ch6,
    input  logic [IN_RADDR-1:0]  i_raddr
);

    // =========================================================================
    // 1. 내부 주소 및 인덱스 계산 신호 (조합 회로)
    // =========================================================================
    logic [3:0] out_ch;        // 출력 채널 인덱스 (0 ~ 15)
    logic [7:0] rel_addr;      // 150개 단위 블록 내 상대 주소 (0 ~ 149)
    logic [2:0] in_ch_idx;     // 25개 단위 입력 채널 순서 (0 ~ 5)
    logic [4:0] kernel_idx;    // 5x5 필터 내부 픽셀 인덱스 (0 ~ 24)
    
    logic [8:0] target_waddr;  // 각 BRAM 내부 저장 주소 (0 ~ 399)

    logic we_ch1, we_ch2, we_ch3, we_ch4, we_ch5, we_ch6;

    // =========================================================================
    // 2. 주소 디코딩 (나누기/모듈로 연산자 완전 제거 - 고정소수점 역수 곱셈 방식)
    // =========================================================================
    
    // 1) out_ch = i_waddr / 150
    // 1/150 ≈ 437 / 65536 (2^16) -> 0~2399 범위에서 100% 정밀도 일치
    assign out_ch = (i_waddr * 20'd437) >> 16;

    // 2) rel_addr = i_waddr % 150 = i_waddr - (out_ch * 150)
    assign rel_addr = i_waddr - ((out_ch << 7) + (out_ch << 4) + (out_ch << 2) + (out_ch << 1));

    // 3) in_ch_idx = rel_addr / 25
    // 1/25 ≈ 41 / 1024 (2^10) -> 0~149 범위에서 100% 정밀도 일치
    assign in_ch_idx = (rel_addr * 13'd41) >> 10;

    // 4) kernel_idx = rel_addr % 25 = rel_addr - (in_ch_idx * 25)
    assign kernel_idx = rel_addr - ((in_ch_idx << 4) + (in_ch_idx << 3) + in_ch_idx);

    // 5) target_waddr = (out_ch * 25) + kernel_idx
    assign target_waddr = (out_ch << 4) + (out_ch << 3) + out_ch + kernel_idx;

    // =========================================================================
    // 3. Write Enable 라우팅 (조합 회로)
    // =========================================================================
    always_comb begin
        we_ch1 = 1'b0;
        we_ch2 = 1'b0;
        we_ch3 = 1'b0;
        we_ch4 = 1'b0;
        we_ch5 = 1'b0;
        we_ch6 = 1'b0;

        if (i_we) begin
            case (in_ch_idx)
                3'd0: we_ch1 = 1'b1;
                3'd1: we_ch2 = 1'b1;
                3'd2: we_ch3 = 1'b1;
                3'd3: we_ch4 = 1'b1;
                3'd4: we_ch5 = 1'b1;
                3'd5: we_ch6 = 1'b1;
                default: ;
            endcase
        end
    end

    // =========================================================================
    // 4. BRAM 인스턴스화
    // =========================================================================
    BRAM_WEIGHT2 U_BRAM_WEIGHT2_1 (
        .clk     (clk         ),
        .i_we    (we_ch1      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch1  ),
        .i_raddr (i_raddr     )
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_2 (
        .clk     (clk         ),
        .i_we    (we_ch2      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch2  ),
        .i_raddr (i_raddr     )
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_3 (
        .clk     (clk         ),
        .i_we    (we_ch3      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch3  ),
        .i_raddr (i_raddr     )
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_4 (
        .clk     (clk         ),
        .i_we    (we_ch4      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch4  ),
        .i_raddr (i_raddr     )
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_5 (
        .clk     (clk         ),
        .i_we    (we_ch5      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch5  ),
        .i_raddr (i_raddr     )
    );

    BRAM_WEIGHT2 U_BRAM_WEIGHT2_6 (
        .clk     (clk         ),
        .i_we    (we_ch6      ),
        .i_data  (i_data      ),
        .i_waddr (target_waddr),
        .o_data  (o_data_ch6  ),
        .i_raddr (i_raddr     )
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
