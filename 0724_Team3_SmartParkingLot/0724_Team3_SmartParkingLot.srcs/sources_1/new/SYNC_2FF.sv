module SYNC_2FF #(
    parameter WIDTH = 8
    ) (
    input  logic             clk,      // 수신 측 클록
    input  logic             reset,
    input  logic [WIDTH-1:0] async_in, // 송신 측 비동기 입력 신호
    output logic [WIDTH-1:0] sync_out  // 동기화된 출력 신호
);

    logic [WIDTH-1:0] ff1, ff2;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            ff1 <= 0;
            ff2 <= 0;
        end else begin
            ff1 <= async_in; 
            ff2 <= ff1;
        end
    end

    assign sync_out = ff2;

endmodule