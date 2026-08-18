`timescale 1ns / 1ps

// =============================================================================
// Multi-Image ROM Module (4개 글자 이미지 저장 및 자동 선택 기능)
// =============================================================================
module image_rom (
    input  logic        clk,
    input  logic [1:0]  img_idx, // 현재 추론 중인 자릿수 인덱스 (0~3)
    input  logic [7:0]  addr,    // o_vga_raddr (0~31)
    output logic [31:0] data
);

    // 4개 이미지 x 32행 (32x32) = 총 128행
    logic [31:0] mem [0:127];

    initial begin
         $readmemh("mnist_0815.mem", mem);
        // $readmemh("mnist_0123.mem", mem);
        // $readmemh("mnist_4567.mem", mem); 
        // $readmemh("mnist_6847.mem", mem); 
        // $readmemh("mnist_4667.mem", mem); 
    end

    always_ff @(posedge clk) begin
        // img_idx(0~3)에 따라 32행 오프셋을 적용하여 128행 영역 접근
        data <= mem[{img_idx, addr[4:0]}];
    end

endmodule


// =============================================================================
// Top Testbench for License Plate Recognition (4-Digit Test)
// =============================================================================
module tb_CNN_Top;

    // -------------------------------------------------------------
    // 1. Clock & Reset & Interface Signals
    // -------------------------------------------------------------
    logic        clk;
    logic        reset;      // Active-Low Reset
    logic [1:0]  img_idx;    // 현재 검증할 번호판 이미지 인덱스 (0~3)

    // VGA Control Interface
    logic        i_vga_done;
    logic [31:0] i_pxl_data;
    logic [ 7:0] o_vga_raddr;

    // Weight Load Interfaces
    logic        i_w1_we;   logic [ 7:0] i_w1_data;   logic [ 7:0] i_w1_waddr;
    logic        i_w2_we;   logic [ 7:0] i_w2_data;   logic [11:0] i_w2_waddr;
    logic        i_w3_we;   logic [ 7:0] i_w3_data;   logic [15:0] i_w3_waddr;
    logic        i_w4_we;   logic [ 7:0] i_w4_data;   logic [13:0] i_w4_waddr;
    logic        i_w5_we;   logic [ 7:0] i_w5_data;   logic [ 9:0] i_w5_waddr;

    // Output Interface
    logic [15:0] o_inf_out;  // 16비트 (각 자릿수 4비트 x 4개)
    logic        o_inf_done; // 전체 4자리 완성 펄스
    logic        o_cnn_done; // 각 숫자별 추론 완료 펄스 (7-cycle pulse)

    // 번호판 4자리 최종 인식 결과 저장 배열
    logic [3:0]  plate_result [0:3];

    // -------------------------------------------------------------
    // 2. Memory Depth Constants & Weight Arrays
    // -------------------------------------------------------------
    localparam W1_SIZE = 150;   
    localparam W2_SIZE = 2400;  
    localparam W3_SIZE = 48000; 
    localparam W4_SIZE = 10080; 
    localparam W5_SIZE = 880;   

    logic [7:0] w1_mem [0:W1_SIZE-1];
    logic [7:0] w2_mem [0:W2_SIZE-1];
    logic [7:0] w3_mem [0:W3_SIZE-1];
    logic [7:0] w4_mem [0:W4_SIZE-1];
    logic [7:0] w5_mem [0:W5_SIZE-1];

    // -------------------------------------------------------------
    // 3. DUT Instance
    // -------------------------------------------------------------
    CNN_Top u_cnn_top (
        .clk        (clk),
        .reset      (reset),
        // VGA Interface
        .i_vga_done (i_vga_done),
        .i_pxl_data (i_pxl_data),
        .o_vga_raddr(o_vga_raddr),
        // Weight 1
        .i_w1_we    (i_w1_we),
        .i_w1_data  (i_w1_data),
        .i_w1_waddr (i_w1_waddr),
        // Weight 2
        .i_w2_we    (i_w2_we),
        .i_w2_data  (i_w2_data),
        .i_w2_waddr (i_w2_waddr),
        // Weight 3
        .i_w3_we    (i_w3_we),
        .i_w3_data  (i_w3_data),
        .i_w3_waddr (i_w3_waddr),
        // Weight 4
        .i_w4_we    (i_w4_we),
        .i_w4_data  (i_w4_data),
        .i_w4_waddr (i_w4_waddr),
        // Weight 5
        .i_w5_we    (i_w5_we),
        .i_w5_data  (i_w5_data),
        .i_w5_waddr (i_w5_waddr),
        // Output
        .o_inf_out  (o_inf_out),
        .o_inf_done (o_inf_done),
        .o_cnn_done (o_cnn_done)  // 각 자릿수 완료 신호 연결
    );

    // -------------------------------------------------------------
    // 4. Clock Generation (100MHz / 10ns)
    // -------------------------------------------------------------
    always #5 clk = ~clk;

    // -------------------------------------------------------------
    // 5. Control Unit 내부 digit_cnt 신호 모니터링 및 ROM 전달
    // -------------------------------------------------------------
    assign img_idx = u_cnn_top.u_control.digit_cnt;

    image_rom u_image_rom (
        .clk    (clk),
        .img_idx(img_idx),
        .addr   (o_vga_raddr),
        .data   (i_pxl_data)
    );

    // -------------------------------------------------------------
    // 6. Weight Loading Task (BRAM 전송)
    // -------------------------------------------------------------
    task automatic load_all_weights();
        integer i, node, pxl, bram_addr, mem_idx;

        $display("[TB] --- Starting Weight BRAM Loading ---");

        // Load Weight 1
        for (i = 0; i < W1_SIZE; i++) begin
            @(posedge clk); #1;
            i_w1_we = 1'b1; i_w1_waddr = i[7:0]; i_w1_data = w1_mem[i];
        end
        @(posedge clk); #1; i_w1_we = 1'b0;

        // Load Weight 2
        for (i = 0; i < W2_SIZE; i++) begin
            @(posedge clk); #1;
            i_w2_we = 1'b1; i_w2_waddr = i[11:0]; i_w2_data = w2_mem[i];
        end
        @(posedge clk); #1; i_w2_we = 1'b0;

        // Load Weight 3
        for (i = 0; i < W3_SIZE; i++) begin
            @(posedge clk); #1;
            i_w3_we = 1'b1; i_w3_waddr = i[15:0]; i_w3_data = w3_mem[i];
        end
        @(posedge clk); #1; i_w3_we = 1'b0;

        // Load Weight 4
        for (i = 0; i < W4_SIZE; i++) begin
            @(posedge clk); #1;
            i_w4_we = 1'b1; i_w4_waddr = i[13:0]; i_w4_data = w4_mem[i];
        end
        @(posedge clk); #1; i_w4_we = 1'b0;

        // Load Weight 5
        bram_addr = 0;
        mem_idx   = 0;
        for (node = 0; node < 10; node++) begin
            for (pxl = 0; pxl < 84; pxl++) begin
                @(posedge clk); #1;
                i_w5_we = 1'b1; i_w5_waddr = bram_addr[9:0]; i_w5_data = w5_mem[mem_idx];
                bram_addr++; mem_idx++;
            end
            for (pxl = 0; pxl < 4; pxl++) begin
                @(posedge clk); #1;
                i_w5_we = 1'b1; i_w5_waddr = bram_addr[9:0]; i_w5_data = 8'h00;
                bram_addr++;
            end
        end
        @(posedge clk); #1; i_w5_we = 1'b0;

        $display("[TB] --- Weight Loading Finished! ---\n");
    endtask

    // -------------------------------------------------------------
    // 7. Main Test Scenario
    // -------------------------------------------------------------
    initial begin
        clk        = 0;
        reset      = 0;   // Active-Low
        i_vga_done = 0;

        i_w1_we = 0; i_w1_data = 0; i_w1_waddr = 0;
        i_w2_we = 0; i_w2_data = 0; i_w2_waddr = 0;
        i_w3_we = 0; i_w3_data = 0; i_w3_waddr = 0;
        i_w4_we = 0; i_w4_data = 0; i_w4_waddr = 0;
        i_w5_we = 0; i_w5_data = 0; i_w5_waddr = 0;

        $display("[TB] Loading Weight .mem files...");
        $readmemh("w1_weights.mem", w1_mem);
        $readmemh("w2_weights.mem", w2_mem);
        $readmemh("w3_weights.mem", w3_mem);
        $readmemh("w4_weights.mem", w4_mem);
        $readmemh("w5_weights.mem", w5_mem);

        #50;
        reset = 1; // Reset 해제
        #20;

        load_all_weights();
        #100;

        $display("==================================================");
        $display("   [START LICENSE PLATE RECOGNITION (4 DIGITS)]   ");
        $display("==================================================");

        // 1. VGA 프레임 수신 완료 펄스 1회 전송 (4자리 자동 파이프라인 트리거)
        $display("[TB] Triggering VGA Done Pulse (Start 4-Digit Pipeline)...");
        @(posedge clk);
        i_vga_done <= 1'b1;
        @(posedge clk);
        i_vga_done <= 1'b0;

        // 2. 자릿수별 추론 완료 신호(o_cnn_done) 수신 대기 및 클래스 점수 출력 (총 4회)
        for (int digit_pos = 0; digit_pos < 4; digit_pos++) begin
            @(posedge o_cnn_done);
            #1;

            $display("\n==================================================");
            $display("     [INFERENCE RESULT - DIGIT #%0d]", digit_pos + 1);
            $display("==================================================");
            $display("  Index (Class)  |  Raw Score (Signed 8-bit)      ");
            $display("--------------------------------------------------");

            for (int i = 0; i < 10; i++) begin
                $display("    Class [%0d]   |   %4d  (Hex: 0x%0h)", i,
                         $signed(u_cnn_top.u_inf_reg.result_reg[i]),
                         u_cnn_top.u_inf_reg.result_reg[i]);
            end

            $display("--------------------------------------------------");
            $display("  ==> DIGIT #%0d PREDICTED CLASS : %0d", digit_pos + 1, o_inf_out[3:0]);
            $display("==================================================\n");

            plate_result[digit_pos] = o_inf_out[3:0];
        end

        // 3. 전체 4자리 최종 완료 신호(o_inf_done) 대기
        @(posedge o_inf_done);
        repeat (2) @(posedge clk);

        $display("==================================================");
        $display("          [FINAL LICENSE PLATE RESULT]            ");
        $display("==================================================");
        $display("  RECOGNIZED NUMBER : [ %0d %0d %0d %0d ]", 
                 plate_result[0], plate_result[1], plate_result[2], plate_result[3]);
        $display("  FINAL OUTPUT BUS  : 0x%04X", o_inf_out);
        $display("==================================================\n");

        #100;
        $finish;
    end

endmodule