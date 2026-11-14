// Formal harness for sync_fifo: data integrity and ordering by the "tracked element" method.
//
// The solver may mark any one accepted write (input f_mark). The harness records the marked
// value and how many entries are ahead of it, counts that number down as entries are read, and
// proves that the read which removes the marked entry returns exactly the recorded value. Since
// the marked write can be any write at all, this proves for every entry: it is never lost,
// corrupted, duplicated, or reordered. With sync_fifo's own assertions this also covers
// occupancy, full and empty.
module sync_fifo_formal #(
  parameter int WIDTH = 4,
  parameter int DEPTH = 4
) (
  input logic             clk,
  input logic             rst,
  input logic [WIDTH-1:0] in_data,
  input logic             in_valid,
  input logic             out_ready,
  input logic             f_mark
);
  localparam int AW = $clog2(DEPTH);

  logic             in_ready, out_valid;
  logic [WIDTH-1:0] out_data;
  logic [AW:0]      count;

  sync_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) dut (
    .clk, .rst, .in_data, .in_valid, .in_ready, .out_data, .out_valid, .out_ready, .count
  );

  logic             push, pop;
  logic             tracking, done, seen_full;
  logic [WIDTH-1:0] t_val;
  logic [AW:0]      t_ahead;  // entries in front of the tracked one

  assign push = in_valid && in_ready;
  assign pop  = out_valid && out_ready;

  logic past_valid = 1'b0;
  always_ff @(posedge clk) past_valid <= 1'b1;
  always_comb if (!past_valid) assume (rst);

  always_ff @(posedge clk) begin
    if (rst) begin
      tracking  <= 1'b0;
      done      <= 1'b0;
      seen_full <= 1'b0;
      t_val     <= '0;
      t_ahead   <= '0;
    end else begin
      if (!in_ready) seen_full <= 1'b1;
      if (tracking) begin
        if (pop) begin
          if (t_ahead == '0) begin
            tracking <= 1'b0;   // this read removed the tracked entry (checked below)
            done     <= 1'b1;
          end else begin
            t_ahead <= t_ahead - 1'b1;
          end
        end
      end else if (push && f_mark && !done) begin
        tracking <= 1'b1;
        t_val    <= in_data;
        t_ahead  <= count - (AW + 1)'(pop);  // a same-cycle read removes one entry in front
      end
    end
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      if (tracking) begin
        assert (t_ahead < count);                      // the tracked entry is still inside
        if (pop && t_ahead == '0) assert (out_data == t_val);
      end
    end
  end

  // Reachability: the tracked entry leaves after the FIFO was full, and after it was empty again.
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      cover (tracking && pop && t_ahead == '0 && seen_full);
      cover (done && !out_valid && seen_full);
    end
  end
endmodule
