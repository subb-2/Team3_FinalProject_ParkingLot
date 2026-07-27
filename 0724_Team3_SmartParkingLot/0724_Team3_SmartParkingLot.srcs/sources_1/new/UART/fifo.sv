module fifo#(
    parameter DEPTH = 4,
    parameter BIT_WIDTH = 8
) (
    input  logic       clk,
    input  logic       reset,
    input  logic       i_push,
    input  logic       i_pop,
    input  logic [7:0] i_data,
    output logic [7:0] o_data,
    output logic       o_full,
    output logic       o_empty
);

    logic [$clog2(DEPTH)-1:0] w_w_ptr, w_r_ptr;

    register_file #(
        .DEPTH(DEPTH),
        .BIT_WIDTH(BIT_WIDTH)
    ) U_FIFO_REGFILE (
        .clk(clk),
        .push_data(i_data),
        .w_addr(w_w_ptr),
        .r_addr(w_r_ptr),
        .we(i_push&(~o_full)),
        .pop_data(o_data)
    );
    fifo_ControlUnit #(
        .DEPTH(DEPTH)
    ) U_FIFO_CTRL (
        .clk(clk),
        .reset(reset),
        .push(i_push),
        .pop(i_pop),
        .w_ptr(w_w_ptr),
        .r_ptr(w_r_ptr),
        .full(o_full),
        .empty(o_empty)
    );
endmodule


module register_file #(
    parameter DEPTH = 4,
    parameter BIT_WIDTH = 8
) (
    input  logic                     clk,
    input  logic [BIT_WIDTH-1:0]     push_data,
    input  logic [$clog2(DEPTH)-1:0] w_addr,
    input  logic [$clog2(DEPTH)-1:0] r_addr,
    input  logic                     we,
    output logic [BIT_WIDTH-1:0]     pop_data
);
    logic [BIT_WIDTH-1:0] fifoRAM[0:DEPTH-1];

    always_ff @(posedge clk) begin
        if(we) fifoRAM[w_addr] <= push_data;
    end
    assign pop_data = fifoRAM[r_addr];
endmodule


module fifo_ControlUnit #(
    parameter DEPTH = 4
) (
    input  logic                     clk,
    input  logic                     reset,
    input  logic                     push,
    input  logic                     pop,
    output logic [$clog2(DEPTH)-1:0] w_ptr,
    output logic [$clog2(DEPTH)-1:0] r_ptr,
    output logic                     full,
    output logic                     empty
);

    localparam PTW_DEPTH = $clog2(DEPTH);

    always_ff @(posedge clk, posedge reset) begin
        if(reset) begin
            w_ptr <= 0;
            r_ptr <= 0;
            full  <= 1'b0;
            empty <= 1'b1;
        end else begin
            case({push, pop})
            // push
            2'b10: begin
                if(!full) begin
                    w_ptr <= w_ptr + 1;
                    empty <= 1'b0;
                end
                if(PTW_DEPTH'(w_ptr + 1'b1) == r_ptr) full <= 1'b1;
            end
            // pop
            2'b01: begin
                if(!empty) begin
                    r_ptr <= r_ptr + 1;
                    full  <= 1'b0;
                end
                if(w_ptr == PTW_DEPTH'(r_ptr + 1'b1)) empty <= 1'b1;
            end
            // push + pop
            2'b11: begin
                if(full == 1'b1) begin
                    r_ptr <= r_ptr + 1;
                    full  <= 1'b0;
                end else if(empty == 1'b1) begin
                    w_ptr <= w_ptr + 1;
                    empty <= 1'b0;
                end else begin
                    w_ptr <= w_ptr + 1;
                    r_ptr <= r_ptr + 1;
                end
            end
            endcase
        end
    end
endmodule