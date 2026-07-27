module InfDataController (
    input  logic        clk,
    input  logic        reset,
    input  logic        i_inf_done,
    input  logic [31:0] i_inf_data,
    input  logic        i_fifo_full,
    output logic [ 7:0] o_data,
    output logic        o_push
);

    localparam IDLE = 0,
               POS0 = 1,
               POS1 = 2,
               POS2 = 3,
               POS3 = 4,
               DONE = 5;
    logic [2:0] state, n_state;

    logic [31:0] data_reg;

    /********* state update *********/
    always_ff @(posedge clk, posedge reset) begin
        if(reset) begin
            state  <= IDLE;
        end else begin
            state <= n_state;
        end
    end

    /******* next state logic *******/
    always_comb begin
        n_state = state;
        case(state)
            IDLE: if(i_inf_done)   n_state = POS0;
            POS0: if(!i_fifo_full) n_state = POS1;
            POS1: if(!i_fifo_full) n_state = POS2;
            POS2: if(!i_fifo_full) n_state = POS3;
            POS3: if(!i_fifo_full) n_state = DONE;
            DONE:                  n_state = IDLE;
            default:               n_state = IDLE;
        endcase
    end

    /******** output logic ********/
    always_ff @(posedge clk, posedge reset) begin
        if(reset) begin
            o_data   <= 8'd0;
            o_push   <= 1'b0;
            data_reg <= 32'd0;
        end else begin
            o_push <= 1'b0;
            case(state)
                IDLE: begin
                    if(i_inf_done) begin
                        data_reg <= i_inf_data;
                    end
                end
                POS0: begin
                    if(!i_fifo_full) begin
                        o_data <= data_reg[7:0];
                        o_push <= 1'b1;
                    end
                end
                POS1: begin
                    if(!i_fifo_full) begin
                        o_data <= data_reg[15:8];
                        o_push <= 1'b1;
                    end
                end
                POS2: begin
                    if(!i_fifo_full) begin
                        o_data <= data_reg[23:16];
                        o_push <= 1'b1;
                    end
                end
                POS3: begin
                    if(!i_fifo_full) begin
                        o_data <= data_reg[31:24];
                        o_push <= 1'b1;
                    end
                end
                DONE: begin
                    o_push <= 1'b0;
                end
            endcase
        end
    end
endmodule