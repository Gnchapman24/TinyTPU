/** Sequencer: program counter, instruction fetch, legality checks, dispatch, HALT, errors, ABORT.
 *
 *  Runs a stored program from `start_pc` (docs/isa.md, execution model): fetch the instruction at
 *  the PC from IMEM, check it, execute it, then PC + 1, until HALT or an error. Instructions run
 *  one at a time; each finishes before the next is fetched.
 *  - Fetch: one IMEM read at a time; the instruction arrives with `imem_rd_valid` (any latency).
 *    If the PC has run past the last instruction, the sequencer halts with error 6 and reports
 *    PC = IMEM_DEPTH, without reading.
 *  - Check: decode errors 1-4 (instr_decode), then for MATMUL / REQUANT the address-range check
 *    (range_check, error 5) before anything executes, so a faulting instruction writes nothing.
 *  - HALT: halted OK, PC of the HALT, its halt code. NOP: nothing.
 *  - MATMUL / REQUANT: `exec_start` pulses with the instruction on `exec_instr`, which then stays
 *    stable until the unit answers with `exec_done` (the instruction has fully finished).
 *  - ABORT (while running): stop at once. The execution units see `exec_abort` and stop issuing;
 *    once they report `exec_idle` and no fetch or check is outstanding, the state is idle with
 *    error 0 and the PC of the instruction that was running. ABORT while stopped does nothing.
 *  - `start` is accepted only while stopped (idle or halted); it clears error and halt code.
 *  Reset (also the protocol's RESET) returns to idle with PC 0; memories are not touched.
 *
 *  Latency: 1 cycle from start to the first fetch; per instruction, fetch + IMEM latency + 1 (decode) + COUNT_W + 2 (range check, MATMUL/REQUANT only) + execution.
 */
module sequencer
  import isa_pkg::*;
#(
  parameter int N          = 8,                   // array size (PTAB entries per output-column tile)
  parameter int IMEM_DEPTH = DEFAULT_IMEM_DEPTH,  // instructions; a power of two, at least 2
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,  // weight tiles
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH   // parameter entries
) (
  input  logic                          clk,            // clock
  input  logic                          rst,            // synchronous active-high reset: idle, PC 0
  input  logic                          start,          // run from start_pc (ignored while running)
  input  logic [$clog2(IMEM_DEPTH)-1:0] start_pc,       // first instruction to run
  input  logic                          abort_req,          // stop a running program (ignored while stopped)
  output seq_state_t                    state,          // idle, running, halted OK, halted error
  output seq_err_t                      err,            // error code while halted with an error, else 0
  output logic [15:0]                   pc,             // HALT / faulting / current instruction; IMEM_DEPTH after overrun
  output logic [15:0]                   halt_code,      // code of the HALT that stopped the program
  output logic                          imem_rd_en,     // read IMEM row imem_rd_addr
  output logic [$clog2(IMEM_DEPTH)-1:0] imem_rd_addr,   // instruction address
  input  logic                          imem_rd_valid,  // imem_rd_data holds the instruction read
  input  logic [INSTR_W-1:0]            imem_rd_data,   // instruction word
  output logic                          exec_start,     // one-cycle pulse: execute exec_instr
  output logic [INSTR_W-1:0]            exec_instr,     // MATMUL / REQUANT being executed; stable until exec_done
  input  logic                          exec_done,      // one-cycle pulse: the instruction has finished
  output logic                          exec_abort,     // stop issuing work and drain (held during ABORT)
  input  logic                          exec_idle       // the execution units have nothing in flight
);

  if (IMEM_DEPTH < 2 || (IMEM_DEPTH & (IMEM_DEPTH - 1)) != 0) begin : g_check_depth
    $error("sequencer: IMEM_DEPTH must be a power of two, at least 2");
  end

  localparam int IAW = $clog2(IMEM_DEPTH);

  typedef enum logic [2:0] {
    S_STOP,     // idle or halted (see stop_state)
    S_FETCH,    // issue the read of the instruction at pc
    S_WAIT,     // wait for the instruction
    S_DECODE,   // decode checks; HALT / NOP / dispatch to the range check
    S_CHECK,    // wait for the range check
    S_EXEC,     // wait for the execution unit
    S_ABORT     // wait for outstanding work to drain
  } fsm_t;

  fsm_t           fsm;
  seq_state_t     stop_state;
  seq_err_t       err_q;
  logic [IAW:0]   pc_q;               // one extra bit: IMEM_DEPTH after overrun
  logic [15:0]    halt_q;
  instr_t         ir;                 // the instruction being decoded / executed
  logic           fetch_pending;      // an IMEM read has been issued and not yet returned

  // Decode checks and range check.
  seq_err_t dec_err;
  logic     rc_start, rc_busy, rc_done, rc_err;

  instr_decode #(
    .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH)
  ) u_decode (
    .instr(ir), .err(dec_err)
  );

  range_check #(
    .N(N), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH)
  ) u_range (
    .clk, .rst, .start(rc_start), .instr(ir), .busy(rc_busy), .done(rc_done), .err(rc_err)
  );

  logic is_exec_op;
  assign is_exec_op = ir.opcode == OP_MATMUL || ir.opcode == OP_REQUANT;
  assign rc_start   = fsm == S_DECODE && dec_err == ERR_NONE && is_exec_op && !abort_req;

  always_ff @(posedge clk) begin
    if (rst) begin
      fsm           <= S_STOP;
      stop_state    <= ST_IDLE;
      err_q         <= ERR_NONE;
      pc_q          <= '0;
      halt_q        <= '0;
      fetch_pending <= 1'b0;
    end else begin
      if (imem_rd_en)    fetch_pending <= 1'b1;
      if (imem_rd_valid) fetch_pending <= 1'b0;

      if (fsm != S_STOP && fsm != S_ABORT && abort_req) begin
        fsm <= S_ABORT;
      end else begin
        unique case (fsm)
          S_STOP: if (start) begin
            fsm    <= S_FETCH;
            pc_q   <= {1'b0, start_pc};
            err_q  <= ERR_NONE;
            halt_q <= '0;
          end
          S_FETCH: begin
            if (pc_q[IAW]) begin                          // ran past the last instruction
              fsm        <= S_STOP;
              stop_state <= ST_HALTED_ERROR;
              err_q      <= ERR_PC_OVERRUN;
            end else begin
              fsm <= S_WAIT;
            end
          end
          S_WAIT: if (imem_rd_valid) begin
            ir  <= instr_t'(imem_rd_data);
            fsm <= S_DECODE;
          end
          S_DECODE: begin
            if (dec_err != ERR_NONE) begin
              fsm        <= S_STOP;
              stop_state <= ST_HALTED_ERROR;
              err_q      <= dec_err;
            end else if (ir.opcode == OP_HALT) begin
              fsm        <= S_STOP;
              stop_state <= ST_HALTED_OK;
              halt_q     <= ir.w_base;
            end else if (ir.opcode == OP_NOP) begin
              fsm  <= S_FETCH;
              pc_q <= pc_q + 1'b1;
            end else begin
              fsm <= S_CHECK;
            end
          end
          S_CHECK: if (rc_done) begin
            if (rc_err) begin
              fsm        <= S_STOP;
              stop_state <= ST_HALTED_ERROR;
              err_q      <= ERR_ADDR_RANGE;
            end else begin
              fsm <= S_EXEC;
            end
          end
          S_EXEC: if (exec_done) begin
            fsm  <= S_FETCH;
            pc_q <= pc_q + 1'b1;
          end
          S_ABORT: if (!fetch_pending && !imem_rd_valid && !rc_busy && exec_idle) begin
            fsm        <= S_STOP;
            stop_state <= ST_IDLE;
            err_q      <= ERR_NONE;
          end
          default: fsm <= S_STOP;
        endcase
      end
    end
  end

  assign imem_rd_en   = fsm == S_FETCH && !pc_q[IAW] && !abort_req;
  assign imem_rd_addr = pc_q[IAW-1:0];
  assign exec_start   = fsm == S_CHECK && rc_done && !rc_err && !abort_req;
  assign exec_instr   = ir;
  assign exec_abort   = fsm == S_ABORT;

  assign state     = fsm == S_STOP ? stop_state : ST_RUNNING;
  assign err       = err_q;
  assign pc        = 16'(pc_q);
  assign halt_code = halt_q;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      // Status is consistent.
      assert ((err_q != ERR_NONE) == (fsm == S_STOP && stop_state == ST_HALTED_ERROR));
      if (fsm == S_STOP && stop_state == ST_HALTED_ERROR && err_q == ERR_PC_OVERRUN)
        assert (pc_q == (IAW + 1)'(IMEM_DEPTH));
      if (fsm != S_STOP) assert (!pc_q[IAW] || fsm == S_FETCH || fsm == S_ABORT);
      // One fetch at a time, only from inside instruction memory.
      if (imem_rd_en) assert (!fetch_pending && !pc_q[IAW]);
      // Execution starts only for a legal, in-range MATMUL / REQUANT, and only from the check.
      if (exec_start) assert (dec_err == ERR_NONE && is_exec_op && !rc_err);
    end
  end
`endif

endmodule
