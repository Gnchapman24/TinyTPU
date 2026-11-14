// Formal harness: uart_tx drives uart_rx through a line the solver may corrupt.
//
// The solver chooses every byte, when to send it, whether to force that frame's stop bit low,
// and for how many extra bit times (0-3) to hold the line low afterwards (a "break").
// Proved properties (plus the assertions inside uart_tx and uart_rx):
//   * every good frame is received as exactly the byte that was sent (no corruption, no drops);
//   * a frame whose stop bit was forced low produces frame_err and never a byte;
//   * every frame produces its rx pulse within a fixed bound (nothing is lost or stalls);
//   * rx never produces a pulse when no frame is in flight.
// Environment assumptions: reset in the first cycle; one frame in flight at a time; after a
// corrupted frame, at least one idle bit before the next (the receiver resynchronizes on idle).
module uart_loopback #(
  parameter int CLKS_PER_BIT = 4
) (
  input logic       clk,
  input logic       rst,
  input logic [7:0] in_data,
  input logic       in_valid,
  input logic       corrupt,
  input logic [1:0] brk_len   // extra bit times the line stays low after a corrupted stop bit
);
  localparam int C     = CLKS_PER_BIT;
  localparam int LIMIT = 3 + C / 2 + 9 * C + 1;  // rx pulse deadline, in cycles after acceptance

  logic       ready, line, rx_line, rx_valid, rx_ferr;
  logic [7:0] rx_data;

  uart_tx #(.CLKS_PER_BIT(C)) u_tx (
    .clk, .rst, .data(in_data), .valid(in_valid), .ready, .tx(line)
  );
  uart_rx #(.CLKS_PER_BIT(C)) u_rx (
    .clk, .rst, .rx(rx_line), .data(rx_data), .valid(rx_valid), .frame_err(rx_ferr)
  );

  logic       accept, pending, exp_bad;
  logic [1:0] exp_brk;
  logic [7:0] exp_byte, cyc;
  int         low_end;

  assign accept  = in_valid && ready;
  assign low_end = 10 * C + 32'(exp_brk) * C;  // the forced-low window is [9C, low_end)

  // cyc: cycles since the last acceptance (cycle 0 = first cycle of the start bit), saturating.
  always_ff @(posedge clk) begin
    if (rst) begin
      pending  <= 1'b0;
      exp_bad  <= 1'b0;
      exp_brk  <= '0;
      exp_byte <= '0;
      cyc      <= 8'hFF;
    end else begin
      if (accept) begin
        pending  <= 1'b1;
        exp_bad  <= corrupt;
        exp_brk  <= brk_len;
        exp_byte <= in_data;
        cyc      <= '0;
      end else begin
        if (cyc != 8'hFF) cyc <= cyc + 1'b1;
        if (rx_valid || rx_ferr) pending <= 1'b0;
      end
    end
  end

  // Corrupt the line: force the stop bit (cycles [9C, 10C) of the frame) low, plus brk_len more bits.
  assign rx_line = (exp_bad && 32'(cyc) >= 9 * C && 32'(cyc) < low_end) ? 1'b0 : line;

  logic past_valid = 1'b0;
  always_ff @(posedge clk) past_valid <= 1'b1;

  // Environment.
  always_comb begin
    if (!past_valid) assume (rst);
    if (pending) assume (!in_valid);
    if (exp_bad && 32'(cyc) < low_end + C) assume (!in_valid);  // one idle bit after the break
  end

  // Properties.
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      if (rx_valid) begin
        assert (pending);
        assert (!exp_bad);
        assert (rx_data == exp_byte);
      end
      if (rx_ferr) begin
        assert (pending);
        assert (exp_bad);
      end
      if (pending) assert (32'(cyc) <= LIMIT);
    end
  end

  // Reachability: a good byte, a framing error, and a good byte after a framing error.
  logic seen_ferr;
  always_ff @(posedge clk) begin
    if (rst) seen_ferr <= 1'b0;
    else if (rx_ferr) seen_ferr <= 1'b1;
  end
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      cover (rx_valid && rx_data == 8'hA5);
      cover (rx_ferr);
      cover (seen_ferr && rx_valid);
    end
  end
endmodule
