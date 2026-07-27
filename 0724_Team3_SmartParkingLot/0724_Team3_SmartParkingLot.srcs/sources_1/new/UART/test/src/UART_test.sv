module UART_test(
    input  logic clk,
    input  logic reset,
    input  logic btn,
    input  logic sw,
    output logic tx
);

    logic btn_trig;
    logic [31:0] infData_SW;
    assign infData_SW = sw ? 32'hDDCCBBAA : 32'h00112233;
    
    btn_debounce U_BTN_DEB(
        .clk(clk),
        .reset(reset),
        .i_btn(btn),
        .o_btn(btn_trig)
    );
    UART_top U_UART_TOP(
        .clk(clk),
        .reset(reset),
        .i_inf_done(btn_trig),
        .i_inf_data(infData_SW),
        .o_tx(tx)
    );
endmodule