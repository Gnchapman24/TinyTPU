// Formal-only stand-in for rtl/range_check.sv, used by sequencer.sby.
//
// Same ports and exactly the same timing (start is ignored while busy; done pulses
// COUNT_W + 2 cycles after start; busy covers the whole check), but the result `err` is the
// parity of the instruction captured at start. In sequencer_formal the instruction comes from
// solver-chosen IMEM data, and a legal MATMUL/REQUANT has free base bits, so the solver can make
// any instruction pass or fail. The control proof therefore holds for any result of the
// check, and the solver does not have to reason about the 24-bit products.
//
// Why not an `(* anyseq *)` wire: yosys-slang does not honour the attribute, and Yosys then
// optimizes the undriven (x) value into each FSM bit independently, which produces
// transitions the real design cannot make. The real arithmetic
// is verified against tinytpu.model.range_error in tb/test_range_check.py (AGENTS.md §3: the
// address arithmetic is checked in cocotb).
module range_check
  import isa_pkg::*;
#(
  parameter int N          = 8,
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               start,
  input  logic [INSTR_W-1:0] instr,
  output logic               busy,
  output logic               done,
  output logic               err
);
  localparam int CNT_W = $clog2(COUNT_W + 1);

  logic             run, fin, done_q, err_q;
  logic [CNT_W-1:0] step;

  always_ff @(posedge clk) begin
    if (rst) begin
      run    <= 1'b0;
      fin    <= 1'b0;
      done_q <= 1'b0;
      step   <= '0;
      err_q  <= 1'b0;
    end else begin
      fin    <= 1'b0;
      done_q <= fin;
      if (start && !busy) begin
        run   <= 1'b1;
        step  <= '0;
        err_q <= ^instr;
      end else if (run) begin
        step <= step + 1'b1;
        if (step == CNT_W'(COUNT_W - 1)) begin
          run <= 1'b0;
          fin <= 1'b1;
        end
      end
    end
  end

  assign busy = run || fin || done_q;
  assign done = done_q;
  assign err  = err_q;
endmodule
