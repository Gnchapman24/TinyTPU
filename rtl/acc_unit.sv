/** Column accumulators and the accumulator buffer (ACC): applies MATMUL results to INT32 rows.
 *
 *  Each cycle with `in_valid` high brings one result vector from the systolic array, the
 *  accumulator row it belongs to, and how to apply it (isa_pkg::acc_mode_t):
 *  - ACC_OVERWRITE:  row = result
 *  - ACC_BIAS:       row = result + bias        (`in_bias`: the N biases of this column tile)
 *  - ACC_ACCUMULATE: row = row + result
 *  All arithmetic is 32-bit two's complement and wraps (AGENTS.md §2). The update is a 3-stage
 *  pipeline: read the row, add, write it back.
 *
 *  Updates of the same row must be at least 3 cycles apart: the pipeline has no forwarding. The
 *  array's tile-spacing rule guarantees this for N >= 4; for N = 2 the controller must space them.
 *  An assertion checks it.
 *
 *  The buffer is a simple dual-port block RAM, N*32 bits wide. The read port also serves REQUANT
 *  and the host (`rd_en`, 1-cycle latency) and the write port also serves the host (`wr_en`);
 *  these must not be used while matrix updates are in flight (`busy`). Assertions check both.
 *
 *  Latency: 2 cycles from an in_valid result to its row being written; rd_data is valid 1 cycle after rd_en.
 */
module acc_unit
  import isa_pkg::*;
#(
  parameter int N     = 8,    // array size: accumulators per row
  parameter int DEPTH = 512   // rows in the buffer; a power of two
) (
  input  logic                     clk,       // clock
  input  logic                     rst,       // synchronous active-high reset; empties the pipeline (not the buffer)
  input  logic                     in_valid,  // a result vector arrives this cycle
  input  logic [$clog2(DEPTH)-1:0] in_addr,   // accumulator row the result belongs to
  input  acc_mode_t                in_mode,   // overwrite, bias, or accumulate
  input  logic [N-1:0][31:0]       in_data,   // result vector from the systolic array
  input  logic [N-1:0][31:0]       in_bias,   // per-column biases, used in ACC_BIAS mode
  input  logic                     rd_en,     // read row rd_addr (REQUANT or host); not while busy
  input  logic [$clog2(DEPTH)-1:0] rd_addr,   // row to read
  output logic [N-1:0][31:0]       rd_data,   // row contents, valid the cycle after rd_en
  input  logic                     wr_en,     // write row wr_addr (host); not while busy
  input  logic [$clog2(DEPTH)-1:0] wr_addr,   // row to write
  input  logic [N-1:0][31:0]       wr_data,   // row contents to write
  output logic                     busy       // updates are in flight (rd_en / wr_en not allowed)
);

  if (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0) begin : g_check_depth
    $error("acc_unit: DEPTH must be a power of two, at least 2");
  end

  localparam int AW = $clog2(DEPTH);
  localparam int RW = N * 32;

  // ---------------------------------------------------------------------------------------------
  // Buffer: simple dual-port RAM (one read port, one write port), inferred as block RAM.
  // ---------------------------------------------------------------------------------------------
  logic [RW-1:0] mem [DEPTH];
  logic          re, we;
  logic [AW-1:0] ra, wa;
  logic [RW-1:0] rq, wd;

  always_ff @(posedge clk) begin
    if (we) mem[wa] <= wd;
  end

  always_ff @(posedge clk) begin
    if (re) rq <= mem[ra];
  end

  // The update pipeline has priority on the read port; external reads only when it is idle.
  assign re = in_valid || rd_en;
  assign ra = in_valid ? in_addr : rd_addr;
  assign rd_data = rq;

  // ---------------------------------------------------------------------------------------------
  // Update pipeline. Stage 1: the row read is in rq; add. Stage 2: write back.
  // ---------------------------------------------------------------------------------------------
  logic              v1, v2;
  logic [AW-1:0]     a1, a2;
  acc_mode_t         mode1;
  logic [N-1:0][31:0] d1, b1, sum, s2;

  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0;
      v2 <= 1'b0;
    end else begin
      v1 <= in_valid;
      v2 <= v1;
    end
    a1    <= in_addr;
    mode1 <= in_mode;
    d1    <= in_data;
    b1    <= in_bias;
    a2    <= a1;
    s2    <= sum;
  end

  always_comb begin
    for (int j = 0; j < N; j++) begin
      unique case (mode1)
        ACC_ACCUMULATE: sum[j] = rq[32*j +: 32] + d1[j];
        ACC_BIAS:       sum[j] = b1[j] + d1[j];
        default:        sum[j] = d1[j];  // ACC_OVERWRITE (ACC_ILLEGAL never reaches here)
      endcase
    end
  end

  assign we   = v2 || wr_en;
  assign wa   = v2 ? a2 : wr_addr;
  assign wd   = v2 ? s2 : wr_data;
  assign busy = v1 || v2;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  logic          h1v, h2v;
  logic [AW-1:0] h1a, h2a;

  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    h1v <= in_valid && !rst;
    h1a <= in_addr;
    h2v <= h1v;
    h2a <= h1a;
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      if (in_valid) begin
        assert (in_mode != ACC_ILLEGAL);
        // Same row again within 2 cycles would read a stale value (no forwarding).
        assert (!(h1v && h1a == in_addr));
        assert (!(h2v && h2a == in_addr));
        assert (!rd_en);       // the read port belongs to the update pipeline
      end
      if (v2) assert (!wr_en); // the write port belongs to the update pipeline
      if (busy) assert (!rd_en && !wr_en);
    end
  end
`endif

endmodule
