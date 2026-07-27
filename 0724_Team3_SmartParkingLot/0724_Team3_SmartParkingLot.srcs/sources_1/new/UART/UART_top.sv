module UART_top(
    input  logic        clk,
    input  logic        reset,
    input  logic        i_inf_done,
    input  logic [31:0] i_inf_data,
    output logic        o_tx
);

    logic w_fifo_push, w_fifo_pop, w_fifo_full, w_fifo_empty;
    logic [7:0] w_push_data, w_pop_data;
    logic w_uart_ready;
    logic w_trans_req, w_trans_req_reg;

    assign w_tx_start = (!w_fifo_empty) && w_uart_ready;   // fifo가 비어있지 않고, uart가 동작중이지 않을 때 trans_req 발생 => 보낼 데이터가 있는데 tx가 놀고있을 때
    assign w_fifo_pop = w_tx_start;

    InfDataController U_InfDataCtrl(
        .clk(clk),
        .reset(reset),
        .i_inf_done(i_inf_done),
        .i_inf_data(i_inf_data),
        .i_fifo_full(w_fifo_full),
        .o_data(w_push_data),
        .o_push(w_fifo_push)
    );
    fifo #(
        .DEPTH(4),
        .BIT_WIDTH(8)
    ) U_UART_FIFO (
        .clk(clk),
        .reset(reset),
        .i_push(w_fifo_push & !w_fifo_full),
        .i_pop(w_fifo_pop),
        .i_data(w_push_data),
        .o_data(w_pop_data),
        .o_full(w_fifo_full),
        .o_empty(w_fifo_empty)
    );
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115_200)
    ) U_UART_TX (
        .clk(clk),
        .reset(reset),
        .i_data(w_pop_data),
        .i_valid(w_fifo_pop),
        .o_ready(w_uart_ready),
        .o_tx(o_tx)
    );
endmodule