module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
) (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] i_data,
    input  logic       i_valid,
    output logic       o_ready,
    output logic       o_tx
);
    
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    localparam IDLE  = 0,
               START = 1,
               DATA  = 2,
               STOP  = 3;
    logic [1:0] state;
    
    logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
    logic [2:0] bit_idx;
    logic [7:0] shift_reg;

    /******* state update *******/
    always_ff @(posedge clk, posedge reset) begin
        if(reset) state <= IDLE;
        else begin
            case(state)
                IDLE:  if(i_valid)                                        state <= START;
                START: if(clk_cnt == CLKS_PER_BIT-1)                      state <= DATA;
                DATA:  if((clk_cnt == CLKS_PER_BIT-1)&&(bit_idx == 3'd7)) state <= STOP;
                STOP:  if(clk_cnt == CLKS_PER_BIT-1)                      state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end

    /******** output logic *******/
    always_ff @(posedge clk, posedge reset) begin
        if(reset) begin
            clk_cnt   <= 0;
            bit_idx   <= 0;
            shift_reg <= 8'h00;
            o_ready   <= 1'b1;
            o_tx      <= 1'b1;   // IDLE일 때 high
        end else begin
            case(state)
                IDLE: begin
                    o_tx    <= 1'b1;
                    o_ready <= 1'b1;
                    if(i_valid) begin
                        shift_reg <= i_data;
                        clk_cnt   <= 0;
                        o_ready   <= 1'b0;
                    end
                end
                START: begin
                    o_tx <= 1'b0;
                    if(clk_cnt == CLKS_PER_BIT -1) begin
                        clk_cnt <= 0;
                        bit_idx <= 0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                DATA: begin
                    o_tx <= shift_reg[bit_idx];
                    if(clk_cnt == CLKS_PER_BIT -1) begin
                        clk_cnt <= 0;
                        if(bit_idx < 3'd7) begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                STOP: begin
                    o_tx <= 1'b1;
                    if(clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
            endcase
        end
    end
endmodule