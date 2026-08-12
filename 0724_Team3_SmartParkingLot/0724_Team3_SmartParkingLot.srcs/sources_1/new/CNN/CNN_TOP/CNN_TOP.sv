`timescale 1ns / 1ps

// =================================================================
// CNN_Top Module (CONV1~CONV3 + FC1~FC2 + Control Integrated)
// =================================================================
module CNN_Top (
    input  logic        clk,
    input  logic        reset,              // Active-Low Reset (!reset)

    // VGA Control & Image Memory Interface
    input  logic        i_vga_done,         // VGA 프레임 준비 완료 신호
    input  logic [31:0] i_pxl_data,         // VGA 픽셀 데이터 (Conv1 입력)
    output logic [ 7:0] o_vga_raddr,        // VGA 픽셀 읽기 주소

    // BRAM_WEIGHT1 Interface (Conv1 Weight Load)
    input  logic        i_w1_we,
    input  logic [ 7:0] i_w1_data,
    input  logic [ 7:0] i_w1_waddr,

    // BRAM_WEIGHT2 Interface (Conv2 Weight Load)
    input  logic        i_w2_we,
    input  logic [ 7:0] i_w2_data,
    input  logic [11:0] i_w2_waddr,

    // BRAM_WEIGHT3 Interface (Conv3 Weight Load)
    input  logic        i_w3_we,
    input  logic [ 7:0] i_w3_data,
    input  logic [15:0] i_w3_waddr,

    // BRAM_WEIGHT4 Interface (FC1 Weight Load)
    input  logic        i_w4_we,
    input  logic [ 7:0] i_w4_data,
    input  logic [13:0] i_w4_waddr,

    // BRAM_WEIGHT5 Interface (FC2 Weight Load)
    input  logic        i_w5_we,
    input  logic [ 7:0] i_w5_data,
    input  logic [ 9:0] i_w5_waddr,

    // UART / External Output Interface
    output logic [15:0] o_inf_out,          // Argmax 결과 (0 ~ 9 Class ID)
    output logic        o_inf_done,         // 최종 추론 완료 신호 (1-clock pulse)
    
    // SR04
    output logic        o_cnn_done
);

    // -------------------------------------------------------------
    // 내부 연결 신호 선언
    // -------------------------------------------------------------
    logic        w_vga_done;

    logic        w_cnn_start;               // Conv1 시작 신호

    // Active-High Reset 변환
    logic rst_act_high;
    assign rst_act_high = ~reset;

    // CONV1_POOL1 & CONV2_POOL2 <-> Conv3 연결 신호
    logic        w_conv2_done;              // Conv2+Pool2 완료 -> Conv3 시작 신호
    logic [63:0] w_conv2_rdata;             // BRAM_CONV2 읽기 데이터 (64-bit)
    logic [ 5:0] w_conv3_raddr;             // Conv3에서 BRAM_CONV2로 보내는 Read Addr (0~49)

    // Conv3 <-> BRAM_WEIGHT3
    logic [12:0] w_w3_raddr;
    logic [63:0] w_w3_rdata;

    // Conv3 <-> BRAM_CONV3
    logic signed [7:0] w_conv3_data;
    logic [ 6:0] w_conv3_waddr;             // 0 ~ 119 쓰기 주소
    logic        w_conv3_we;
    logic        w_conv3_done;

    // BRAM_CONV3 <-> FC1
    logic [ 3:0] w_bram_conv3_raddr;        // 0 ~ 14 (15개 Word) 읽기 주소
    logic [63:0] w_bram_conv3_rdata;

    // FC1 <-> BRAM_WEIGHT4
    logic [10:0] w_w4_raddr;
    logic [63:0] w_w4_rdata;

    // FC1 <-> BRAM_FC
    logic signed [7:0] w_fc1_data;
    logic [ 6:0] w_fc1_waddr;
    logic        w_fc1_we;
    logic        w_fc1_done;

    // BRAM_FC <-> FC2
    logic [ 3:0] w_bram_fc_raddr;           // 0 ~ 10 (11개 Word) 읽기 주소
    logic [63:0] w_bram_fc_rdata;

    // FC2 <-> BRAM_WEIGHT5
    logic [ 6:0] w_w5_raddr;
    logic [63:0] w_w5_rdata;

    // FC2 <-> INF_REG
    logic signed [7:0] w_fc2_data;
    logic [ 3:0] w_fc2_waddr;
    logic        w_fc2_we;
    logic        w_fc2_done;

    // -------------------------------------------------------------
    // 0. Edge Detect
    // -------------------------------------------------------------
    async_edge_detect U_EDGE_DETECT(
        .clk       (clk),
        .reset     (rst_act_high),    // Active-High Reset 전달 (~reset)
        .async_in  (i_vga_done),      // 외부 비동기 입력 신호
        
        .pos_edge  (w_vga_done),      // 상승 엣지 펄스 (0 -> 1)
        .neg_edge  (),               // 미사용 포트
        .any_edge  ()                // 미사용 포트
    );

    // -------------------------------------------------------------
    // 1. CNN Control Unit
    // -------------------------------------------------------------
    CNN_Control u_control (
        .clk         (clk),
        .reset       (reset),               // Active-Low Reset
        .i_vga_done  (w_vga_done),
        .o_cnn_start (w_cnn_start),         // 파이프라인 최선두 (Conv1) 트리거
        .i_inf_done  (o_cnn_done), // INF_REG 완료 신호
        .o_inf_done  (o_inf_done)           // 외부 알림
    );

    // -------------------------------------------------------------
    // 2. CONV1 + POOL1 + CONV2 + POOL2 Top Subsystem
    // -------------------------------------------------------------
    CONV_POOL_BRAM_TOP u_conv1_conv2_top (
        .clk        (clk),
        .reset      (rst_act_high),         // Active-High Reset 전달

        // Weight 1 Write Interface
        .i_we_c1    (i_w1_we),
        .i_data_c1  (i_w1_data),
        .i_waddr_c1 (i_w1_waddr),

        // Weight 2 Write Interface
        .i_we_c2    (i_w2_we),
        .i_data_c2  (i_w2_data),
        .i_waddr_c2 (i_w2_waddr),

        // VGA Image Input Interface
        .i_pxl_data (i_pxl_data),
        .o_raddr    (o_vga_raddr),

        // Control Pipeline
        .i_start_c1 (w_cnn_start),          // 전체 CNN 시작 신호
        .o_done_c2  (w_conv2_done),         // Conv2 완료 -> Conv3 시작 신호로 전달

        // Output BRAM_CONV2 Interface (Conv3 연결)
        .o_data     (w_conv2_rdata),        // Conv3로 입력될 64-bit 픽셀 데이터
        .i_raddr    (w_conv3_raddr)         // Conv3에서 요청하는 Read Addr
    );

    // -------------------------------------------------------------
    // 3. Conv3 계층 & BRAM_WEIGHT3
    // -------------------------------------------------------------
    Conv3 u_conv3 (
        .clk        (clk),
        .reset      (rst_act_high),
        .i_start    (w_conv2_done),         // Conv2/Pool2 완료 시 자동 실행
        .o_done     (w_conv3_done),
        .i_pxl_data (w_conv2_rdata),        // BRAM_CONV2 64-bit 읽기 데이터
        .o_raddr    (w_conv3_raddr),        // BRAM_CONV2 6-bit Read Address (0 ~ 49)
        .i_weight   (w_w3_rdata),
        .o_w_raddr  (w_w3_raddr),
        .o_pxl_data (w_conv3_data),
        .o_waddr    (w_conv3_waddr),
        .o_we       (w_conv3_we)
    );

    BRAM_WEIGHT3 #(
        .DEPTH(6000), .IN_DATA(8), .IN_WADDR(16), .OUT_DATA(64), .IN_RADDR(13)
    ) u_bram_weight3 (
        .clk     (clk),
        .i_we    (i_w3_we),
        .i_data  (i_w3_data),
        .i_waddr (i_w3_waddr),
        .i_raddr (w_w3_raddr),
        .o_data  (w_w3_rdata)
    );

    // -------------------------------------------------------------
    // 4. Intermediate BRAM_CONV3 (120 Byte Write -> 15 x 64-bit Read)
    // -------------------------------------------------------------
    BRAM_CONV3 #(
        .DEPTH(15), .IN_DATA(8), .IN_WADDR(7), .IN_RADDR(4), .OUT_DATA(64)
    ) u_bram_conv3 (
        .clk     (clk),
        .i_valid (w_conv3_we),
        .i_data  (w_conv3_data),
        .i_waddr (w_conv3_waddr),           // 7-bit (0 ~ 119)
        .i_raddr (w_bram_conv3_raddr),      // 4-bit (0 ~ 14)
        .o_data  (w_bram_conv3_rdata)
    );

    // -------------------------------------------------------------
    // 5. FC1 계층 & BRAM_WEIGHT4
    // -------------------------------------------------------------
    FC1 u_fc1 (
        .clk        (clk),
        .reset      (rst_act_high),
        .i_start    (w_conv3_done),
        .o_done     (w_fc1_done),
        .i_pxl_data (w_bram_conv3_rdata),
        .o_raddr    (w_bram_conv3_raddr),
        .i_weight   (w_w4_rdata),
        .o_w_raddr  (w_w4_raddr),
        .o_fc_data  (w_fc1_data),
        .o_waddr    (w_fc1_waddr),
        .o_we       (w_fc1_we)
    );

    BRAM_WEIGHT4 #(
        .DEPTH(1260), .IN_DATA(8), .IN_WADDR(14), .OUT_DATA(64), .IN_RADDR(11)
    ) u_bram_weight4 (
        .clk     (clk),
        .i_we    (i_w4_we),
        .i_data  (i_w4_data),
        .i_waddr (i_w4_waddr),
        .i_raddr (w_w4_raddr),
        .o_data  (w_w4_rdata)
    );

    // -------------------------------------------------------------
    // 6. Intermediate BRAM_FC (84 Byte Write -> 11 x 64-bit Read)
    // -------------------------------------------------------------
    BRAM_FC #(
        .DEPTH(11), .IN_DATA(8), .IN_WADDR(7), .IN_RADDR(4), .OUT_DATA(64)
    ) u_bram_fc (
        .clk     (clk),
        .i_we    (w_fc1_we),
        .i_data  (w_fc1_data),
        .i_waddr (w_fc1_waddr),
        .i_raddr (w_bram_fc_raddr),
        .o_data  (w_bram_fc_rdata)
    );

    // -------------------------------------------------------------
    // 7. FC2 계층 & BRAM_WEIGHT5
    // -------------------------------------------------------------
    FC2 u_fc2 (
        .clk        (clk),
        .reset      (rst_act_high),
        .i_start    (w_fc1_done),           // FC1 완료 시 자동 시작
        .o_done     (w_fc2_done),
        .i_pxl_data (w_bram_fc_rdata),
        .o_raddr    (w_bram_fc_raddr),
        .i_weight   (w_w5_rdata),
        .o_w_raddr  (w_w5_raddr),
        .o_fc_data  (w_fc2_data),
        .o_waddr    (w_fc2_waddr),
        .o_we       (w_fc2_we)
    );

    BRAM_WEIGHT5 #(
        .DEPTH(110), .IN_DATA(8), .IN_WADDR(10), .OUT_DATA(64), .IN_RADDR(7)
    ) u_bram_weight5 (
        .clk     (clk),
        .i_we    (i_w5_we),
        .i_data  (i_w5_data),
        .i_waddr (i_w5_waddr),
        .i_raddr (w_w5_raddr),
        .o_data  (w_w5_rdata)
    );

    // -------------------------------------------------------------
    // 8. INF_REG (Argmax 및 최종 추론 레지스터)
    // -------------------------------------------------------------
    INF_REG u_inf_reg (
        .clk        (clk),
        .reset      (rst_act_high),
        .i_cnn_out  (w_fc2_data),           // Signed 8-bit FC2 Output
        .i_cnn_addr (w_fc2_waddr),          // 0 ~ 9 Class Index
        .i_we       (w_fc2_we),
        .i_cnn_done (w_fc2_done),
        .o_inf_out  (o_inf_out),            // Argmax Class ID
        .o_cnn_done (o_cnn_done)            // Control Unit으로의 완료 보고, SR04
    );

endmodule


`timescale 1ns / 1ps

module INF_REG (
    input  logic              clk,
    input  logic              reset,        // Active-High Reset
    
    // FC2에서 BRAM으로 쏘는 신호를 직접 수신
    input  logic signed [7:0] i_cnn_out,   // FC2 output (Signed)
    input  logic [3:0]        i_cnn_addr,  // FC2 o_waddr (0 ~ 9)
    input  logic              i_we,        // FC2 o_we (Write Enable)
    input  logic              i_cnn_done,  // FC2 o_done
    
    output logic [15:0]       o_inf_out,   // 최근 4개의 추론 결과 (4비트 x 4개 = 16-bit)
    output logic              o_cnn_done   // 최종 추론 완료 신호 (7-cycle pulse)
);

    // 10개의 FC2 결과를 저장할 레지스터 (BRAM 역할)
    logic signed [7:0] result_reg [0:9];

    // Argmax (최댓값 추적용)
    logic signed [7:0] max_val;
    logic [3:0]        max_idx;

    // Signed 8-bit 최솟값 (-128)
    localparam logic signed [7:0] MIN_SCORE = 8'sh80;

    // 완료 플래그 전달용 내부 레지스터 및 카운터
    logic       r_cnn_done;
    logic [2:0] cnt; // 0 ~ 7 카운터 (3비트)

    // -------------------------------------------------------------
    // 1. FC2 결과 수신 및 최댓값(Argmax) 실시간 갱신
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            max_val <= MIN_SCORE;
            max_idx <= 4'd0;
            for (int i = 0; i < 10; i++) begin
                result_reg[i] <= 8'd0;
            end
        end 
        // FC2 연산 시작 시점 초기화 (0번 주소가 처음 들어올 때)
        else if (i_we && (i_cnn_addr == 4'd0)) begin
            result_reg[i_cnn_addr] <= i_cnn_out;
            max_val             <= i_cnn_out; // 첫 데이터로 최댓값 초기화
            max_idx             <= 4'd0;
        end
        // FC2에서 주소와 함께 데이터가 적힐 때 (BRAM 쓰기 동작과 동일)
        else if (i_we) begin
            result_reg[i_cnn_addr] <= i_cnn_out; // 레지스터에 저장
            
            // 실시간 최댓값 비교
            if (i_cnn_out > max_val) begin
                max_val <= i_cnn_out;
                max_idx <= i_cnn_addr;           // 가장 점수가 높은 주소(Class) 기억
            end
        end
    end

    // -------------------------------------------------------------
    // 2. FC2 완료 시 4비트 Shift Register로 4개 결과 누적 저장
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            o_inf_out  <= 16'd0;
            r_cnn_done <= 1'b0;
        end else begin
            r_cnn_done <= 1'b0;

            if (i_cnn_done) begin
                // 기존 [11:0] 비트를 상위 [15:4]로 밀고, 하위 [3:0]에 최신 max_idx 삽입
                o_inf_out  <= {o_inf_out[11:0], max_idx};
                r_cnn_done <= 1'b1;             // 1-cycle pulse
            end
        end
    end

    // -------------------------------------------------------------
    // 3. o_cnn_done 7-Cycle Pulse 카운터 블록
    // -------------------------------------------------------------
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            cnt        <= 3'd0;
            o_cnn_done <= 1'b0;
        end else if (r_cnn_done) begin
            cnt        <= 3'd1;     // [Cycle 1] 카운터 시작
            o_cnn_done <= 1'b1;
        end else if (cnt != 3'd0) begin
            if (cnt == 3'd7) begin  // 7번째 클록 달성 시 종료
                cnt        <= 3'd0;
                o_cnn_done <= 1'b0;  // 7클록 완료 후 Low 복귀
            end else begin
                cnt        <= cnt + 3'd1; // 카운터 1씩 증가 (2, 3, 4, 5, 6, 7)
                o_cnn_done <= 1'b1;
            end
        end else begin
            o_cnn_done <= 1'b0;
        end
    end

endmodule