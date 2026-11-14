/** Synchronous FIFO with valid/ready on both sides and first-word fall-through.
 *
 *  An entry is written on a clock edge where `in_valid` and `in_ready` are both high, and removed
 *  on an edge where `out_valid` and `out_ready` are both high; both may happen in the same cycle.
 *  The oldest entry is always visible on `out_data` while `out_valid` is high (first-word
 *  fall-through), so `out_data` is a combinational read of the storage array: this FIFO is meant
 *  for registers or LUT RAM, not block RAM. `in_ready` is low only when the FIFO is full, even if
 *  an entry is being removed in the same cycle. Storage is not reset; `rst` only empties the FIFO.
 *
 *  Latency: 1 cycle from a write into an empty FIFO to `out_valid`.
 */
module sync_fifo #(
  parameter int WIDTH = 8,  // bits per entry
  parameter int DEPTH = 16  // number of entries; a power of two, at least 2
) (
  input  logic                   clk,        // clock
  input  logic                   rst,        // synchronous active-high reset; empties the FIFO
  input  logic [WIDTH-1:0]       in_data,    // entry to write
  input  logic                   in_valid,   // write in_data this cycle if in_ready is high
  output logic                   in_ready,   // the FIFO is not full
  output logic [WIDTH-1:0]       out_data,   // oldest entry; meaningful while out_valid is high
  output logic                   out_valid,  // the FIFO is not empty
  input  logic                   out_ready,  // remove the oldest entry this cycle if out_valid is high
  output logic [$clog2(DEPTH):0] count       // number of entries stored (0 to DEPTH)
);

  if (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0) begin : g_check_depth
    $error("sync_fifo: DEPTH must be a power of two, at least 2");
  end

  localparam int AW = $clog2(DEPTH);

  logic [WIDTH-1:0] mem [DEPTH];
  logic [AW:0]      wr_ptr, rd_ptr;  // one extra bit tells full from empty
  logic             push, pop;

  assign count     = wr_ptr - rd_ptr;
  assign in_ready  = (count != (AW + 1)'(DEPTH));
  assign out_valid = (count != '0);
  assign out_data  = mem[rd_ptr[AW-1:0]];
  assign push      = in_valid && in_ready;
  assign pop       = out_valid && out_ready;

  always_ff @(posedge clk) begin
    if (push) mem[wr_ptr[AW-1:0]] <= in_data;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      if (push) wr_ptr <= wr_ptr + 1'b1;
      if (pop)  rd_ptr <= rd_ptr + 1'b1;
    end
  end

`ifdef ASSERTIONS
  // Independent occupancy model: the pointer difference must always equal pushes minus pops.
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  logic [AW:0] f_count;

  always_ff @(posedge clk) past_valid <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst) f_count <= '0;
    else     f_count <= f_count + (AW + 1)'(push) - (AW + 1)'(pop);
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      assert (count == f_count);
      assert (count <= (AW + 1)'(DEPTH));
      assert (in_ready == (count < (AW + 1)'(DEPTH)));
      assert (out_valid == (count > '0));
    end
  end
`ifdef FORMAL
  always_comb if (!past_valid) assume (rst);
`endif
`endif

endmodule
