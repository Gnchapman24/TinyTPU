/** Address-range check of one instruction before it executes (ISA error 5).
 *
 *  For a decode-legal MATMUL or REQUANT, finds whether the highest address it would use in any
 *  memory it touches is at or beyond that memory's depth (the table in docs/isa.md; reference:
 *  `tinytpu.model.range_error`). With `last = base + product - 1`, the error condition
 *  `last >= depth` is computed as `base + product > depth`; every count is at least 1.
 *    MATMUL:  WBUF w_base + nt*kt,  ABUF a_base + m*kt,  ACC acc_base + m*nt,
 *             and, in bias mode only, PTAB p_base + nt*N.
 *    REQUANT: ACC acc_base + m*nt,  ABUF a_dst + m*nt,   PTAB p_base + nt*N.
 *    HALT, NOP: never an error.
 *
 *  The three products of 12-bit counts are formed by shift-and-add, one multiplier bit per
 *  cycle, so no DSP is used (the DSP budget is exactly 66). The instruction is captured at
 *  `start` (while idle the operands follow `instr` in every cycle, so `start` drives no wide
 *  load enable); the comparison is registered, and the result is ready a fixed number of cycles
 *  later. The input is assumed to be
 *  free of decode errors (instr_decode runs first); otherwise `err` is meaningless.
 *
 *  Latency: COUNT_W + 2 cycles from start to done (14 with isa_pkg::COUNT_W = 12).
 */
module range_check
  import isa_pkg::*;
#(
  parameter int N          = 8,                   // array size: PTAB entries per output-column tile
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,  // weight tiles
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH   // parameter entries
) (
  input  logic               clk,    // clock
  input  logic               rst,    // synchronous active-high reset; abandons a check in progress
  input  logic               start,  // capture instr and begin a check (ignored while busy)
  input  logic [INSTR_W-1:0] instr,  // decode-legal instruction to check
  output logic               busy,   // a check is in progress
  output logic               done,   // one-cycle pulse: err holds the result
  output logic               err     // 1: some address would be out of range (ISA error 5)
);

  localparam int CW = COUNT_W;
  localparam int PW = 2 * CW;           // product width
  localparam int SW = PW + 2;           // base (< 2**16) + product: room for the carry, compared unsigned
  localparam int LN = $clog2(N);
  localparam int CNT_W = $clog2(CW + 1);

  instr_t          in_f;                 // the instruction at `start`
  opcode_t         op;                   // captured fields used by the final compare
  logic            bias;                 // MATMUL acc_mode = bias (PTAB is then used)
  logic [15:0]     w_base, a_base, acc_base, p_base;
  logic [CW-1:0]   nt;
  logic [PW-1:0]   p_nk, p_mk, p_mn;     // nt*kt, m*kt, m*nt (partial during the check)
  logic [PW-1:0]   x_nt, x_m;            // shifted multiplicands
  logic [CW-1:0]   y_kt, y_nt;           // multipliers, consumed one bit per cycle
  logic [CNT_W-1:0] step;
  logic            run, fin, done_q, err_q, cmp;

  assign in_f = instr_t'(instr);

  // The bits of the slots above COUNT_W, and of the fields not needed here, are checked by the
  // decoder; this module only reads the implemented bits.
  logic unused_in;
  assign unused_in = ^{in_f.kt[15:CW], in_f.nt[15:CW], in_f.m[15:CW]};

  // Control: run for CW steps, then one cycle (fin) to compare, then done with the registered result.
  always_ff @(posedge clk) begin
    if (rst) begin
      run    <= 1'b0;
      fin    <= 1'b0;
      done_q <= 1'b0;
    end else begin
      fin    <= 1'b0;
      done_q <= fin;
      if (start && !busy) begin
        run  <= 1'b1;
      end else if (run && step == CNT_W'(CW - 1)) begin
        run <= 1'b0;
        fin <= 1'b1;
      end
    end
  end

  // Datapath. While idle, the operands are loaded from `instr` in every cycle (so `start` does
  // not have to drive a wide load enable; the instruction is stable while it is being checked).
  always_ff @(posedge clk) begin
    if (!run) begin
      step     <= '0;
      op       <= in_f.opcode;
      bias     <= acc_mode_t'(in_f.flags[FLAG_ACC_MODE_LSB +: 2]) == ACC_BIAS;
      w_base   <= in_f.w_base;
      a_base   <= in_f.a_base;
      acc_base <= in_f.acc_base;
      p_base   <= in_f.p_base;
      nt       <= in_f.nt[CW-1:0];
      p_nk     <= '0;
      p_mk     <= '0;
      p_mn     <= '0;
      x_nt     <= PW'(in_f.nt[CW-1:0]);
      x_m      <= PW'(in_f.m[CW-1:0]);
      y_kt     <= in_f.kt[CW-1:0];
      y_nt     <= in_f.nt[CW-1:0];
    end else begin
      if (y_kt[0]) p_nk <= p_nk + x_nt;
      if (y_kt[0]) p_mk <= p_mk + x_m;
      if (y_nt[0]) p_mn <= p_mn + x_m;
      x_nt <= x_nt << 1;
      x_m  <= x_m << 1;
      y_kt <= y_kt >> 1;
      y_nt <= y_nt >> 1;
      step <= step + 1'b1;
    end
    if (fin) err_q <= cmp;
  end

  // Result, from the finished products in `fin`'s cycle; registered into err_q.
  localparam logic [SW-1:0] D_WBUF = SW'(WBUF_DEPTH);
  localparam logic [SW-1:0] D_ABUF = SW'(ABUF_DEPTH);
  localparam logic [SW-1:0] D_ACC  = SW'(ACC_DEPTH);
  localparam logic [SW-1:0] D_PTAB = SW'(PTAB_DEPTH);

  function automatic logic beyond(input logic [15:0] base, input logic [PW-1:0] prod,
                                  input logic [SW-1:0] depth);
    return (SW'(base) + SW'(prod)) > depth;
  endfunction

  logic [PW-1:0] p_ptab;
  assign p_ptab = PW'(nt) << LN;               // nt * N

  always_comb begin
    unique case (op)
      OP_MATMUL:  cmp = beyond(w_base, p_nk, D_WBUF) || beyond(a_base, p_mk, D_ABUF) ||
                        beyond(acc_base, p_mn, D_ACC) || (bias && beyond(p_base, p_ptab, D_PTAB));
      OP_REQUANT: cmp = beyond(acc_base, p_mn, D_ACC) || beyond(a_base, p_mn, D_ABUF) ||
                        beyond(p_base, p_ptab, D_PTAB);
      default:    cmp = 1'b0;
    endcase
  end

  assign busy = run || fin || done_q;
  assign done = done_q;
  assign err  = err_q;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      if (run) assert (step < CNT_W'(CW));
      assert (!(run && fin) && !(run && done_q) && !(fin && done_q));
    end
  end
`endif

endmodule
