`timescale 1ns / 1ps

module cnn_conv1 #(
    parameter IMG_WIDTH = 32
) (
    input  logic        clk,
    input  logic        reset,
    input  logic [ 7:0] i_weight,
    input  logic [31:0] i_pxl_data,
    input  logic        i_start,
    input  logic [ 2:0] ch_cnt,
    output logic [ 7:0] o_raddr,
    output logic [ 7:0] o_pxl_data,
    output logic [ 8:0] o_w_raddr,
    output logic        o_done
);

    typedef enum logic [1:0] {
        IDLE,
        LOAD,
        CALC,
        DONE
    } state_t;

    state_t c_state, n_state;

    localparam logic [7:0] BIAS_PARAM[0:5] = '{
        8'hFF,
        8'hFD,
        8'hFA,
        8'hEF,
        8'h1B,
        8'h01
    };

    logic pxl_start;
    logic [4:0] cnt;
    logic [2:0] con_x;
    logic [2:0] con_y;
    logic [4:0] pxl_x;
    logic [4:0] pxl_y;

    logic [10:0] base_addr;
    logic [10:0] pxl_addr;
    logic [2:0] calc_cnt;

    logic [4:0] bit_sel;
    logic [7:0] selected_pxl;

    logic [7:0] bias;
    logic signed [7:0] pxl_reg[0:24];
    logic signed [7:0] weight_reg[0:24];

    logic signed [20:0] conv_result;
    logic signed [20:0] conv_bias_result;
    logic [2:0] ch_cnt_d;
    logic [7:0] w_raddr_cnt;
    logic       wait_ch_change;

    logic signed [15:0] step0[0:24];
    logic signed [16:0] step1[0:12];
    logic signed [17:0] step2[0:6];
    logic signed [18:0] step3[0:3];
    logic signed [19:0] step4[0:1];

    assign pxl_addr = base_addr + (con_y * IMG_WIDTH) + con_x;

    assign o_raddr = {3'b000, pxl_addr[9:5]};

    assign bias = BIAS_PARAM[ch_cnt];
    assign conv_bias_result = conv_result + $signed({{13{bias[7]}}, bias});
    assign selected_pxl = {7'b0000000, i_pxl_data[bit_sel]};

    // Top 모듈에서 하던 주소 점프를 모듈 내부로 가져옴
    assign o_w_raddr = {1'b0, w_raddr_cnt} + {6'd0, ch_cnt} * 9'd25;

    // FSM State Register
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            c_state <= IDLE;
        end else begin
            c_state <= n_state;
        end
    end

    // Next State Logic
    always_comb begin
        n_state = c_state;
        case (c_state)
            IDLE: begin
                if (i_start || (pxl_start && (!wait_ch_change || (ch_cnt != ch_cnt_d)))) n_state = LOAD;
            end
            LOAD: begin
                if (cnt == 5'd25) n_state = CALC;
            end
            CALC: begin
                if (calc_cnt == 3'd5) n_state = DONE;
            end
            DONE: begin
                n_state = IDLE;
            end
            default: n_state = IDLE;
        endcase
    end

    // Datapath & Control
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            cnt         <= 5'd0;
            calc_cnt    <= 3'd0;
            con_x       <= 3'd0;
            con_y       <= 3'd0;
            pxl_x       <= 5'd0;
            pxl_y       <= 5'd0;
            base_addr   <= 11'd0;
            o_done      <= 1'b0;
            o_pxl_data  <= 8'd0;
            w_raddr_cnt <= 8'd0;
            conv_result <= 21'd0;
            bit_sel     <= 5'd0;
            ch_cnt_d    <= 3'd0;
            pxl_start   <= 0;
            wait_ch_change <= 1'b0;
        end else begin
            ch_cnt_d <= ch_cnt;
            o_done   <= 1'b0;

            if ((c_state == DONE) && (pxl_x == 5'd27) && (pxl_y == 5'd27)) begin
                wait_ch_change <= 1'b1;
            end else if (ch_cnt != ch_cnt_d) begin
                wait_ch_change <= 1'b0;
            end

            // 채널이 바뀌면 모든 스캔 좌표 초기화
            if (ch_cnt != ch_cnt_d) begin
                base_addr <= 11'd0;
                pxl_x     <= 5'd0;
                pxl_y     <= 5'd0;
            end

            bit_sel <= pxl_addr[4:0];

            case (c_state)
                IDLE: begin
                    cnt         <= 5'd0;
                    calc_cnt    <= 3'd0;
                    w_raddr_cnt <= 8'd0;
                    con_x       <= 3'd0;
                    con_y       <= 3'd0;
                    if (i_start) begin
                        pxl_start <= 1;
                    end
                end

                LOAD: begin
                    if (cnt < 5'd25) begin
                        cnt         <= cnt + 1'b1;
                        w_raddr_cnt <= w_raddr_cnt + 1'b1;

                        if (con_x == 3'd4) begin
                            con_x <= 3'd0;
                            con_y <= con_y + 1'b1;
                        end else begin
                            con_x <= con_x + 1'b1;
                        end
                    end

                    if (cnt > 5'd0 && cnt <= 5'd25) begin
                        pxl_reg[cnt-1]    <= selected_pxl;
                        weight_reg[cnt-1] <= i_weight;
                    end
                end

                CALC: begin
                    if (calc_cnt < 3'd5) begin
                        calc_cnt <= calc_cnt + 1'b1;
                    end

                    if (calc_cnt == 3'd0) begin
                        for (int i = 0; i < 25; i++) begin
                            step0[i] <= pxl_reg[i] * weight_reg[i];
                        end
                    end else if (calc_cnt == 3'd1) begin
                        for (int i = 0; i < 12; i++) begin
                            step1[i] <= step0[2*i] + step0[2*i+1];
                        end
                        step1[12] <= step0[24];
                    end else if (calc_cnt == 3'd2) begin
                        for (int i = 0; i < 6; i++) begin
                            step2[i] <= step1[2*i] + step1[2*i+1];
                        end
                        step2[6] <= step1[12];
                    end else if (calc_cnt == 3'd3) begin
                        for (int i = 0; i < 3; i++) begin
                            step3[i] <= step2[2*i] + step2[2*i+1];
                        end
                        step3[3] <= step2[6];
                    end else if (calc_cnt == 3'd4) begin
                        step4[0] <= step3[0] + step3[1];
                        step4[1] <= step3[2] + step3[3];
                    end else if (calc_cnt == 3'd5) begin
                        conv_result <= step4[0] + step4[1];
                    end
                end

                DONE: begin
                    o_done <= 1'b1;

                    // Conv1 Output Shift: SHIFT1 = 1 적용 (ReLU -> Shift -> Saturation)
                    if (conv_bias_result < 0) begin
                        o_pxl_data <= 8'd0;  // ReLU
                    end else if ((conv_bias_result >>> 1) > 21'd255) begin
                        o_pxl_data <= 8'd255;  // Saturation
                    end else begin
                        o_pxl_data <= conv_bias_result[8:1];
                    end

                    // 👈 2D 이미지 스캔을 위한 좌표 래핑 로직 (0~27)
                    if (ch_cnt == ch_cnt_d) begin
                        if (pxl_x == 5'd27) begin
                            pxl_x <= 5'd0;
                            if (pxl_y == 5'd27) begin
                                // 🌟 1채널(28x28) 끝! 무조건 좌표 리셋하고 시동 끈다!
                                pxl_y     <= 5'd0;
                                base_addr <= 11'd0;
                                if(ch_cnt == 5) begin
                                    pxl_start <= 1'b0;
                                end
                            end else begin

                                // 🌟 아직 Y가 끝이 아니면 1 증가! (여기에 있어야 안전)
                                pxl_y     <= pxl_y + 1'b1;
                                base_addr <= base_addr + 11'd5;

                            end
                        end else begin
                            pxl_x     <= pxl_x + 1'b1;
                            base_addr <= base_addr + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

endmodule

