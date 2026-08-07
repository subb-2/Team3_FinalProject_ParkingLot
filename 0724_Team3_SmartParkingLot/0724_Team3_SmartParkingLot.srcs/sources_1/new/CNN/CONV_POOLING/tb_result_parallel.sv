`timescale 1ns / 1ps

module tb_result_parallel();

    logic clk;
    logic reset;

    logic        i_we_c1;
    logic [ 7:0] i_data_c1;
    logic [ 7:0] i_waddr_c1;
    logic        i_we_c2;
    logic [ 7:0] i_data_c2;
    logic [11:0] i_waddr_c2;
    logic [31:0] i_pxl_data;
    logic [ 7:0] o_raddr;
    logic        i_start_c1;
    logic        o_done_c1;
    logic        i_start_c2;
    logic        o_done_c2;
    logic [63:0] o_data;
    logic [ 5:0] i_raddr;
    logic [ 7:0] pool1_golden[0:1175];
    logic [ 7:0] temp_weights[0:2399];
    logic [ 7:0] conv2_golden[0:1599];
    logic [ 7:0] pool2_golden[0:399];
    logic [ 7:0] conv2_actual[0:1599];
    int conv2_ptr = 0;

    always @(posedge clk) begin
        if (!reset && u_dut.U_CONV2_POOL2.c_done) begin
            #1;
            if (conv2_ptr < 1600) begin
                conv2_actual[conv2_ptr] = u_dut.U_CONV2_POOL2.conv_o_data;
                conv2_ptr++;
            end
        end
    end

    // ---- DUT Instantiation ----
    CONV_POOL_BRAM_TOP u_dut (
        .clk       (clk),
        .reset     (reset),
        .i_we_c1   (i_we_c1),
        .i_data_c1 (i_data_c1),
        .i_waddr_c1(i_waddr_c1),
        .i_we_c2   (i_we_c2),
        .i_data_c2 (i_data_c2),
        .i_waddr_c2(i_waddr_c2),
        .i_pxl_data(i_pxl_data),
        .o_raddr   (o_raddr),
        .i_start_c1(i_start_c1),
        .o_done_c2 (o_done_c2),
        .o_data    (o_data),
        .i_raddr   (i_raddr)
    );

    // ---- Image ROM in Testbench ----
    image_rom1 U_IMG_ROM (
        .clk (clk),
        .addr(o_raddr[4:0]),
        .data(i_pxl_data)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // File Output Descriptors
    integer fp1;
    integer fp2;
    integer fp3;

    initial begin
        fp1 = $fopen("pool1_result_batch.txt", "w");
        fp2 = $fopen("pool2_result_batch.txt", "w");
        fp3 = $fopen("conv2_result_batch.txt", "w");
        $display("[DEBUG] fp1 = %0d, fp2 = %0d, fp3 = %0d", fp1, fp2, fp3);
        
        // Initialize Control Signals
        reset      = 1;
        i_we_c1    = 0;
        i_data_c1  = 0;
        i_waddr_c1 = 0;
        i_we_c2    = 0;
        i_data_c2  = 0;
        i_waddr_c2 = 0;
        i_start_c1 = 0;
        i_start_c2 = 0;
        i_raddr    = 0;

        #200;
        reset = 0;
        @(posedge clk);

        // ---- Direct weight loading from MEM files to BRAM internal memories ----
        $display("[INFO] Loading Conv1 weights directly to BRAM_WEIGHT1...");
        $readmemh("conv1_weight.mem", u_dut.U_CONV1_POOL1.U_BRAM_WEIGHT1.ram);
        
        $display("[INFO] Loading Conv2 weights directly to split BRAM_WEIGHT2...");
        $readmemh("conv2_ch0.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_1.ram);
        $readmemh("conv2_ch1.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_2.ram);
        $readmemh("conv2_ch2.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_3.ram);
        $readmemh("conv2_ch3.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_4.ram);
        $readmemh("conv2_ch4.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_5.ram);
        $readmemh("conv2_ch5.mem", u_dut.U_CONV2_POOL2.U_BRAM_WE2.U_BRAM_WEIGHT2_6.ram);
        
        $display("[INFO] Loading Pool1 golden outputs for verification...");
        $readmemh("pool1_out.mem", pool1_golden);
        $display("[INFO] Loading Conv2 golden outputs for verification...");
        $readmemh("conv2_out.mem", conv2_golden);
        $display("[INFO] Loading Pool2 golden outputs for verification...");
        $readmemh("pool2_out.mem", pool2_golden);
        
        @(posedge clk);

        // ---- Start Pipeline simulation ----
        $display("[INFO] Triggering Conv1/Pool1 via i_start_c1...");
        i_start_c1 = 1;
        @(posedge clk); 
        i_start_c1 = 0;

        // Wait for Conv1/Pool1 to complete (automatic transition)
        $display("[INFO] Waiting for Conv1/Pool1 to complete...");
        wait (u_dut.o_done_c1 === 1'b1);
        @(posedge clk);

        $display("[INFO] Conv1/Pool1 finished. Conv2/Pool2 started automatically.");

        // Wait for Conv2/Pool2 (16 channels)
        $display("[INFO] Waiting for Conv2/Pool2 to complete...");
        wait (o_done_c2 === 1'b1);
        @(posedge clk);
        $display("[INFO] Conv2/Pool2 complete.");

        $display("[INFO] Pipeline simulation completed successfully!");
        #100;

        // ---- Verify BRAM_CONV1 (Pool1 output) against pool1_out.mem ----
        $display("[INFO] Comparing BRAM_CONV1 output against pool1_out.mem...");
        begin : pool1_verify    
            int pool1_mismatch_cnt;
            pool1_mismatch_cnt = 0;
            for (int i = 0; i < 1176; i++) begin
                logic [7:0] actual;
                logic [7:0] expected;
                int ch;
                int addr;
                ch = i / 196;
                addr = i % 196;
                expected = pool1_golden[i];
                case (ch)
                    0: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_1.ram[addr];
                    1: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_2.ram[addr];
                    2: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_3.ram[addr];
                    3: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_4.ram[addr];
                    4: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_5.ram[addr];
                    5: actual = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_6.ram[addr];
                    default: actual = 8'h00;
                endcase
                if (actual !== expected) begin
                    $display("[MISMATCH] Index %0d: Expected %02h, Got %02h", i, expected, actual);
                    pool1_mismatch_cnt++;
                end
            end
            if (pool1_mismatch_cnt == 0) begin
                $display("[SUCCESS] All 1176 outputs of Pool1 match pool1_out.mem completely!");
            end else begin
                $display("[FAIL] Pool1 verification failed with %0d mismatches!", pool1_mismatch_cnt);
            end
        end

        // ---- Read and write the output from BRAM_CONV1 (Pool1 out) to pool1_result_batch.txt ----
        $display("[INFO] Reading BRAM_CONV1 (Pool1 output) and writing to pool1_result_batch.txt...");
        for (int i = 0; i < 1176; i++) begin
            logic [7:0] val;
            int ch;
            int addr;
            ch = i / 196;
            addr = i % 196;
            case (ch)
                0: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_1.ram[addr];
                1: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_2.ram[addr];
                2: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_3.ram[addr];
                3: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_4.ram[addr];
                4: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_5.ram[addr];
                5: val = u_dut.U_CONV1_POOL1.U_BRAM_CONV1_6.ram[addr];
                default: val = 8'h00;
            endcase
            $fwrite(fp1, "%02h\n", val);
        end
        $fclose(fp1);
        $display("[INFO] Pool1 outputs written to pool1_result_batch.txt.");

        // ---- Read and write the output from BRAM_CONV2 (Pool2 out) to pool2_result.txt ----
        $display("[INFO] Reading BRAM_CONV2 (Pool2 output) and comparing against pool2_out.mem...");
        begin : pool2_verify
            int pool2_mismatch_cnt = 0;
            int idx = 0;
            for (int room = 0; room < 50; room++) begin
                i_raddr = room;
                @(posedge clk);
                #1;
                for (int seat = 0; seat < 8; seat++) begin
                    logic [7:0] actual = o_data[seat*8 +: 8];
                    logic [7:0] expected = pool2_golden[idx];
                    if (actual !== expected) begin
                        $display("[MISMATCH] Pool2 Index %0d (Room %0d, Seat %0d): Expected %02h, Got %02h", idx, room, seat, expected, actual);
                        pool2_mismatch_cnt++;
                    end
                    $fwrite(fp2, "%02h\n", actual);
                    idx++;
                end
            end
            if (pool2_mismatch_cnt == 0) begin
                $display("[SUCCESS] All 400 outputs of Pool2 match pool2_out.mem completely!");
            end else begin
                $display("[FAIL] Pool2 verification failed with %0d mismatches!", pool2_mismatch_cnt);
            end
        end
        $fclose(fp2);
        $display("[INFO] Pool2 outputs written to pool2_result_batch.txt.");

        // ---- Compare and write the output from conv2_actual to conv2_result_batch.txt ----
        $display("[INFO] Comparing conv2 actual output against conv2_out.mem...");
        begin : conv2_verify
            int conv2_mismatch_cnt = 0;
            for (int i = 0; i < 1600; i++) begin
                logic [7:0] actual = conv2_actual[i];
                logic [7:0] expected = conv2_golden[i];
                if (actual !== expected) begin
                    $display("[MISMATCH] Conv2 Index %0d: Expected %02h, Got %02h", i, expected, actual);
                    conv2_mismatch_cnt++;
                end
                $fwrite(fp3, "%02h\n", actual);
            end
            if (conv2_mismatch_cnt == 0) begin
                $display("[SUCCESS] All 1600 outputs of Conv2 match conv2_out.mem completely!");
            end else begin
                $display("[FAIL] Conv2 verification failed with %0d mismatches!", conv2_mismatch_cnt);
            end
        end
        $fclose(fp3);
        $display("[INFO] Conv2 outputs written to conv2_result_batch.txt.");

        $display("[INFO] Verification finished successfully!");
        $finish;
    end

endmodule

module image_rom1 (
    input  logic        clk,
    input  logic [4:0]  addr,
    output logic [31:0] data
);

    logic [31:0] mem [0:31];

    initial begin
        $readmemh("mnist_0_label_7_32x32.mem", mem);
    end

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end

endmodule
