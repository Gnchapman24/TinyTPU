// Formal harness for sequencer (task A4.2): control properties for any program.
//
// Environment (assumptions):
// - IMEM answers each read one cycle later with any 128-bit word the solver chooses (`f_word`),
//   so every possible program is covered, including illegal instructions.
// - The execution unit is busy from exec_start until it pulses exec_done (at a time of the
//   solver's choosing, at most EXEC_MAX cycles later). On exec_abort it stops and drains, again
//   within EXEC_MAX cycles. exec_idle is high exactly when it is not busy.
// - The host pulses start and abort_req at any time.
//
// Properties (plus the sequencer's own assertions):
// 1. Fetch order: a run's first fetch is at start_pc; every later fetch in the same run is at
//    the previous fetch + 1. Fetches never leave IMEM.
// 2. Dispatch: exec_start only when the unit is idle, never during or after ABORT in a run, and
//    with exec_instr equal to the word fetched last.
// 3. exec_instr (and with it act_signed, acc_mode, ...) is stable while the unit executes.
// 4. Stops: HALTED OK reports the PC of the last fetch; error 6 reports PC = IMEM_DEPTH;
//    any other stop reports the PC of the last fetch. ABORT ends in idle with error 0.
// 5. Termination: every instruction finishes (next fetch, or a stop) within INSTR_BOUND
//    cycles, and an ABORT takes effect within ABORT_BOUND cycles. With property 1 (each fetch
//    is at the previous PC + 1, and fetches never leave IMEM) every run stops: after at most
//    IMEM_DEPTH instructions the PC reaches the end and the run halts with error 6.
//
// sequencer.sby replaces range_check with formal/range_check_stub.sv: same timing, any result.
module sequencer_formal
  import isa_pkg::*;
#(
  parameter int IMEM_DEPTH = 4,
  parameter int EXEC_MAX   = 3
) (
  input logic               clk,
  input logic               rst,
  input logic               start,
  input logic [$clog2(IMEM_DEPTH)-1:0] start_pc,
  input logic               abort_req,
  input logic [INSTR_W-1:0] f_word,      // instruction returned by the next IMEM read
  input logic               f_done       // the execution unit may finish (or finish draining) now
);
  localparam int IAW = $clog2(IMEM_DEPTH);
  // Cycles per instruction: fetch, IMEM, decode, check (COUNT_W + 2), the check-to-exec cycle,
  // execution (EXEC_MAX + 1), plus slack.
  localparam int INSTR_BOUND = 4 + (COUNT_W + 2) + (EXEC_MAX + 1) + 2;
  localparam int ABORT_BOUND = COUNT_W + EXEC_MAX + 7;

  seq_state_t         state;
  seq_err_t           err;
  logic [15:0]        pc, halt_code;
  logic               imem_rd_en;
  logic [IAW-1:0]     imem_rd_addr;
  logic               imem_rd_valid;
  logic [INSTR_W-1:0] imem_rd_data;
  logic               exec_start, exec_done, exec_abort, exec_idle;
  logic [INSTR_W-1:0] exec_instr;

  sequencer #(
    .N(2), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(16), .ABUF_DEPTH(8), .ACC_DEPTH(8), .PTAB_DEPTH(8)
  ) dut (
    .clk, .rst, .start, .start_pc, .abort_req, .state, .err, .pc, .halt_code,
    .imem_rd_en, .imem_rd_addr, .imem_rd_valid, .imem_rd_data,
    .exec_start, .exec_instr, .exec_done, .exec_abort, .exec_idle
  );

  logic past_valid = 1'b0;
  always_ff @(posedge clk) past_valid <= 1'b1;
  always_comb if (!past_valid) assume (rst);

  // ---------------------------------------------------------------------------------------------
  // Environment models.
  // ---------------------------------------------------------------------------------------------
  // IMEM: one-cycle latency, any word.
  always_ff @(posedge clk) begin
    if (rst) imem_rd_valid <= 1'b0;
    else     imem_rd_valid <= imem_rd_en;
  end
  assign imem_rd_data = f_word;

  // Execution unit.
  logic       e_busy, e_drain;
  logic [3:0] e_cnt;
  always_ff @(posedge clk) begin
    if (rst) begin
      e_busy  <= 1'b0;
      e_drain <= 1'b0;
      e_cnt   <= '0;
    end else begin
      if (exec_start) begin
        e_busy <= 1'b1;
        e_cnt  <= '0;
      end else if (e_busy && exec_abort) begin
        e_busy  <= 1'b0;
        e_drain <= 1'b1;
        e_cnt   <= '0;
      end else if (e_busy) begin
        if (exec_done) e_busy <= 1'b0;
        e_cnt <= e_cnt + 1'b1;
      end else if (e_drain) begin
        if (f_done || e_cnt == 4'(EXEC_MAX)) e_drain <= 1'b0;
        e_cnt <= e_cnt + 1'b1;
      end
    end
  end
  assign exec_done = e_busy && !exec_abort && (f_done || e_cnt == 4'(EXEC_MAX));
  assign exec_idle = !e_busy && !e_drain;

  // ---------------------------------------------------------------------------------------------
  // Observers.
  // ---------------------------------------------------------------------------------------------
  logic                 running, aborting;
  logic                 fetched;          // a fetch has been issued in this run
  logic [IAW-1:0]       last_fetch;
  logic [IAW-1:0]       run_pc;           // start_pc of the current run
  logic [INSTR_W-1:0]   last_word;
  logic [5:0]           instr_cycles, abort_cycles;

  assign running = state == ST_RUNNING;

  always_ff @(posedge clk) begin
    if (rst) begin
      fetched      <= 1'b0;
      aborting     <= 1'b0;
      last_fetch   <= '0;
      run_pc       <= '0;
      last_word    <= '0;
      instr_cycles <= '0;
      abort_cycles <= '0;
    end else begin
      if (!running && start) begin
        fetched    <= 1'b0;
        aborting   <= 1'b0;
        run_pc     <= start_pc;
      end
      if (imem_rd_en) begin
        fetched    <= 1'b1;
        last_fetch <= imem_rd_addr;
      end
      if (imem_rd_valid) last_word <= imem_rd_data;
      instr_cycles <= (imem_rd_en || !running) ? '0 : instr_cycles + 1'b1;
      if (running && abort_req) aborting <= 1'b1;
      if (!running) aborting <= 1'b0;
      abort_cycles <= (aborting && running) ? abort_cycles + 1'b1 : '0;
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Properties.
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      // 1. Fetch order.
      if (imem_rd_en && !fetched) assert (imem_rd_addr == run_pc);
      if (imem_rd_en && fetched)  assert (imem_rd_addr == last_fetch + 1'b1);
      if (imem_rd_en) assert (running);

      // 2. Dispatch.
      if (exec_start) begin
        assert (exec_idle);
        assert (!aborting && !abort_req);
        assert (exec_instr == last_word);
      end

      // 3. The instruction is stable while it executes (act_signed cannot change mid-MATMUL).
      if ($past(e_busy) && e_busy) assert ($stable(exec_instr));

      // 4. Stops.
      if ($past(running) && !running) begin
        if (state == ST_HALTED_OK) assert (pc == 16'(last_fetch) && err == ERR_NONE);
        if (state == ST_HALTED_ERROR && err == ERR_PC_OVERRUN) assert (pc == 16'(IMEM_DEPTH));
        if (state == ST_HALTED_ERROR && err != ERR_PC_OVERRUN) assert (pc == 16'(last_fetch));
        if ($past(aborting) || $past(abort_req)) assert (state == ST_IDLE && err == ERR_NONE);
        assert (exec_idle || state == ST_HALTED_OK || state == ST_HALTED_ERROR);
      end
      if (!running) assert (!exec_start && !imem_rd_en);

      // 5. Termination.
      if (!aborting) assert (instr_cycles < 6'(INSTR_BOUND));    // an ABORT is bounded below
      assert (abort_cycles < 6'(ABORT_BOUND));
    end
  end

  // Reachability.
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      cover (state == ST_HALTED_OK && pc == 16'(2));
      cover (state == ST_HALTED_ERROR && err == ERR_ILLEGAL_OPCODE);
      cover (state == ST_HALTED_ERROR && err == ERR_RESERVED_BIT);
      cover (state == ST_HALTED_ERROR && err == ERR_ACC_MODE);
      cover (state == ST_HALTED_ERROR && err == ERR_ZERO_COUNT);
      cover (state == ST_HALTED_ERROR && err == ERR_ADDR_RANGE);
      cover (state == ST_HALTED_ERROR && err == ERR_PC_OVERRUN);
      cover ($past(e_busy) && $past(exec_abort) && state == ST_RUNNING);
      cover ($past(running) && state == ST_IDLE);
    end
  end

endmodule
