/** MATMUL execution unit: the tile-loop address generator that drives WBUF, ABUF, PTAB, the
 *  systolic array and the accumulators for one MATMUL instruction (docs/isa.md).
 *
 *  Loop order n (output-column tile), k (inner tile), m (activation row), as in the ISA. For
 *  every tile (n, k) it issues, counted in cycles c from the tile's start:
 *  - c = 0 .. N-1: the N weight rows, WBUF row (w_base + n*kt + k)*N + c, tagged {load, row};
 *  - c = D .. D+M-1 (D = W_LAT - A_LAT): the M vectors, ABUF a_base + m*kt + k, tagged valid,
 *    so vector 0 enters the array together with weight row 0;
 *  and the next tile starts at c = T = max(3, N + M - 1). That is exactly the array's rule (the
 *  next tile's row 0 enters N cycles after the last vector) and keeps updates of the same ACC row
 *  at least 3 cycles apart, as acc_unit needs (this only binds at N = 2).
 *  Addresses are formed incrementally (no multipliers). The sequencer's range check has already
 *  guaranteed that every one is inside its memory.
 *
 *  Accumulator side: each vector read pushes {ACC row acc_base + m*nt + n, mode} into a queue,
 *  and each array result (`arr_valid`) pops it, so acc_unit receives `acc_valid`, `acc_addr`,
 *  `acc_mode` and `acc_bias` exactly with the result, whatever the array latency. Mode: at k = 0
 *  the instruction's overwrite or bias mode, otherwise accumulate (accumulate mode: always).
 *
 *  Bias mode: before each column tile n, the unit waits until earlier results have drained, then
 *  reads PTAB p_base + n*N + col for col = 0 .. N-1 (tag = col) and keeps the N biases in a
 *  register for that tile's k = 0 results. The tile starts right after the last read: the
 *  biases arrive (P_LAT) long before the first result that uses them.
 *
 *  `done` pulses once every result has been written into ACC. On `abort_req` the unit stops issuing,
 *  waits until everything already issued has finished, and returns to idle without `done`.
 *  `instr` must stay stable from `start` to `done` (the sequencer guarantees it); `act_signed`
 *  for the array comes from it.
 *
 *  Latency: nt*kt*T cycles of issue (plus N + P_LAT + drain per column tile in bias mode), then the array and accumulator latency until done.
 */
module matmul_ctrl
  import isa_pkg::*;
#(
  parameter int N          = 8,                   // array size; a power of two, at least 2
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,  // weight tiles
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH,  // parameter entries
  parameter int W_LAT      = 3,                   // WBUF read latency (buffer_ram 1 + RD_PIPE)
  parameter int A_LAT      = 2,                   // ABUF read latency; W_LAT - A_LAT must be 0 .. N-1
  parameter int P_LAT      = 2                    // PTAB read latency; at most W_LAT + 2*N
) (
  input  logic                                    clk,         // clock
  input  logic                                    rst,         // synchronous active-high reset; abandons the instruction
  input  logic                                    start,       // begin executing instr (a MATMUL)
  input  logic [INSTR_W-1:0]                      instr,       // the MATMUL; stable until done
  input  logic                                    abort_req,       // stop issuing and drain (sequencer exec_abort)
  output logic                                    done,        // one-cycle pulse: all results are in ACC
  output logic                                    idle,        // nothing issued is still in flight
  output logic                                    act_signed,  // activation mode for the array
  output logic                                    w_rd_en,     // WBUF datapath read
  output logic [$clog2(WBUF_DEPTH)+$clog2(N)-1:0] w_rd_addr,   // WBUF row: tile*N + row
  output logic [$clog2(N):0]                      w_rd_tag,    // {w_load, w_row} for the array
  output logic                                    a_rd_en,     // ABUF datapath read
  output logic [$clog2(ABUF_DEPTH)-1:0]           a_rd_addr,   // activation vector
  output logic                                    a_rd_tag,    // a_valid for the array
  output logic                                    p_rd_en,     // PTAB datapath read (bias mode)
  output logic [$clog2(PTAB_DEPTH)-1:0]           p_rd_addr,   // parameter entry
  output logic [$clog2(N)-1:0]                    p_rd_tag,    // column of the entry
  input  logic                                    p_rd_valid,  // PTAB read data is valid
  input  logic [PTAB_ENTRY_W-1:0]                 p_rd_data,   // parameter entry
  input  logic [$clog2(N)-1:0]                    p_rd_col,    // tag returned with it
  input  logic                                    arr_valid,   // the array outputs a result this cycle
  output logic                                    acc_valid,   // apply the result to ACC (to acc_unit in_valid)
  output logic [$clog2(ACC_DEPTH)-1:0]            acc_addr,    // ACC row of the result
  output acc_mode_t                               acc_mode,    // how to apply it
  output logic [N-1:0][31:0]                      acc_bias,    // biases of the current column tile
  input  logic                                    acc_busy     // acc_unit still has updates in flight
);

  if (N < 2 || (N & (N - 1)) != 0) begin : g_check_n
    $error("matmul_ctrl: N must be a power of two, at least 2");
  end
  if (W_LAT - A_LAT < 0 || W_LAT - A_LAT >= N) begin : g_check_lat
    $error("matmul_ctrl: W_LAT - A_LAT must be 0 .. N-1");
  end
  // The last bias of a column tile is read in the cycle before the tile starts; its first result
  // leaves the array W_LAT + 2N + 1 cycles into the tile. The bias must be there by then.
  if (P_LAT > W_LAT + 2 * N) begin : g_check_plat
    $error("matmul_ctrl: P_LAT must be at most W_LAT + 2*N");
  end

  localparam int LN  = $clog2(N);
  localparam int CW  = COUNT_W;
  localparam int WTW = $clog2(WBUF_DEPTH);
  localparam int AAW = $clog2(ABUF_DEPTH);
  localparam int CAW = $clog2(ACC_DEPTH);
  localparam int PAW = $clog2(PTAB_DEPTH);
  localparam int D   = W_LAT - A_LAT;
  localparam int TW  = CW + 2;                   // tile cycle counter: up to N + 4095
  localparam int QD  = 4 * N;                    // queue depth: > ABUF + array latency in flight
  localparam int QW  = CAW + 2;

  typedef enum logic [2:0] {
    S_IDLE,
    S_DRAIN_B,   // bias mode: wait for earlier results before replacing the biases
    S_BIAS,      // read the N biases of column tile n
    S_TILE,      // issue one tile
    S_FINISH,    // wait for the last results
    S_ABORT      // wait for everything issued, then idle
  } fsm_t;

  fsm_t          fsm;
  instr_t        f;
  acc_mode_t     mode;
  logic          bias_mode;
  logic [CW-1:0] n, k, m;                      // loop counters
  logic [TW-1:0] c, t_last;                    // cycle within the tile, last cycle of a tile
  logic [LN-1:0] row, col;
  logic          rows_left, vecs_left;
  logic [WTW-1:0] w_tile;                      // w_base + n*kt + k
  logic [AAW-1:0] a_k, a_cur;                  // a_base + k, then + m*kt
  logic [CAW-1:0] acc_n, acc_cur;              // acc_base + n, then + m*nt
  logic [PAW-1:0] p_n;                         // p_base + n*N
  logic [LN:0]    p_wait;                      // PTAB reads not yet returned

  assign f          = instr_t'(instr);
  assign mode       = acc_mode_t'(f.flags[FLAG_ACC_MODE_LSB +: 2]);
  assign bias_mode  = mode == ACC_BIAS;
  assign act_signed = f.flags[FLAG_ACT_SIGNED];

  // Bits beyond the implemented widths are zero (decoder); the slots are read only in part.
  logic unused_f;
  assign unused_f = ^{f.w_base[15:WTW], f.a_base[15:AAW], f.acc_base[15:CAW], f.p_base[15:PAW],
                      f.kt[15:CW], f.nt[15:CW], f.m[15:CW], f.flags[7:3], f.opcode};

  // ---------------------------------------------------------------------------------------------
  // Result queue: one entry {ACC row, mode} per vector read, popped by each array result.
  // ---------------------------------------------------------------------------------------------
  logic          q_push, q_in_ready, q_out_valid;
  logic [QW-1:0] q_in, q_out;
  logic [$clog2(QD):0] unused_q_count;
  acc_mode_t     v_mode;

  assign v_mode = (k == '0 && mode != ACC_ACCUMULATE) ? mode : ACC_ACCUMULATE;
  assign q_in   = {acc_cur, v_mode};

  sync_fifo #(.WIDTH(QW), .DEPTH(QD)) u_queue (
    .clk, .rst, .in_data(q_in), .in_valid(q_push), .in_ready(q_in_ready),
    .out_data(q_out), .out_valid(q_out_valid), .out_ready(arr_valid), .count(unused_q_count)
  );

  assign acc_valid = arr_valid;
  assign acc_addr  = q_out[QW-1:2];
  assign acc_mode  = acc_mode_t'(q_out[1:0]);

  // ---------------------------------------------------------------------------------------------
  // Issue logic of a tile.
  // ---------------------------------------------------------------------------------------------
  logic issuing, last_tile, drained;

  assign issuing   = fsm == S_TILE && !abort_req;
  assign rows_left = c < TW'(N);
  assign vecs_left = c >= TW'(D) && m < f.m[CW-1:0];
  assign last_tile = n == f.nt[CW-1:0] - 1'b1 && k == f.kt[CW-1:0] - 1'b1;
  assign drained   = !q_out_valid && !acc_busy;

  assign w_rd_en   = issuing && rows_left;
  assign w_rd_addr = {w_tile, row};
  assign w_rd_tag  = {1'b1, row};
  assign a_rd_en   = issuing && vecs_left;
  assign a_rd_addr = a_cur;
  assign a_rd_tag  = 1'b1;
  assign q_push    = a_rd_en;

  assign p_rd_en   = fsm == S_BIAS && !abort_req;
  assign p_rd_addr = p_n + PAW'(col);
  assign p_rd_tag  = col;

  // T - 1 = max(3, N + M - 1) - 1
  always_comb begin
    t_last = TW'(N) + TW'(f.m[CW-1:0]) - TW'(2);
    if (t_last < TW'(2)) t_last = TW'(2);
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      fsm    <= S_IDLE;
      p_wait <= '0;
    end else begin
      if (p_rd_en && !p_rd_valid)      p_wait <= p_wait + 1'b1;
      else if (!p_rd_en && p_rd_valid) p_wait <= p_wait - 1'b1;

      if (abort_req && fsm != S_IDLE && fsm != S_ABORT) begin
        fsm <= S_ABORT;
      end else begin
        unique case (fsm)
          S_IDLE: if (start) begin
            n      <= '0;
            k      <= '0;
            m      <= '0;
            c      <= '0;
            row    <= '0;
            col    <= '0;
            w_tile <= f.w_base[WTW-1:0];
            a_k    <= f.a_base[AAW-1:0];
            a_cur  <= f.a_base[AAW-1:0];
            acc_n  <= f.acc_base[CAW-1:0];
            acc_cur <= f.acc_base[CAW-1:0];
            p_n    <= f.p_base[PAW-1:0];
            fsm    <= bias_mode ? S_DRAIN_B : S_TILE;
          end
          S_DRAIN_B: if (drained) fsm <= S_BIAS;
          S_BIAS: begin
            col <= col + 1'b1;
            if (col == LN'(N - 1)) fsm <= S_TILE;
          end
          S_TILE: begin
            c <= c + 1'b1;
            if (rows_left) row <= row + 1'b1;
            if (vecs_left) begin
              m       <= m + 1'b1;
              a_cur   <= a_cur + AAW'(f.kt[CW-1:0]);
              acc_cur <= acc_cur + CAW'(f.nt[CW-1:0]);
            end
            if (c == t_last) begin                   // tile finished: next (n, k)
              c      <= '0;
              row    <= '0;
              m      <= '0;
              w_tile <= w_tile + 1'b1;
              if (last_tile) begin
                fsm <= S_FINISH;
              end else if (k == f.kt[CW-1:0] - 1'b1) begin
                k       <= '0;
                n       <= n + 1'b1;
                a_k     <= f.a_base[AAW-1:0];
                a_cur   <= f.a_base[AAW-1:0];
                acc_n   <= acc_n + 1'b1;
                acc_cur <= acc_n + 1'b1;
                p_n     <= p_n + PAW'(N);
                col     <= '0;
                if (bias_mode) fsm <= S_DRAIN_B;
              end else begin
                k       <= k + 1'b1;
                a_k     <= a_k + 1'b1;
                a_cur   <= a_k + 1'b1;
                acc_cur <= acc_n;
              end
            end
          end
          S_FINISH: if (drained) fsm <= S_IDLE;
          S_ABORT:  if (drained && p_wait == '0 && !p_rd_valid) fsm <= S_IDLE;
          default:  fsm <= S_IDLE;
        endcase
      end
    end
  end

  // Bias register: filled from PTAB in bias mode, one column per returned read.
  always_ff @(posedge clk) begin
    if (p_rd_valid) acc_bias[p_rd_col] <= p_rd_data[31:0];
  end

  logic unused_p;
  assign unused_p = ^p_rd_data[PTAB_ENTRY_W-1:32];

  assign done = fsm == S_FINISH && drained;
  assign idle = fsm == S_IDLE;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      if (q_push)    assert (q_in_ready);            // the queue never overflows
      if (arr_valid) assert (q_out_valid);           // every result has its entry
      if (start)     assert (fsm == S_IDLE);
      if (fsm != S_IDLE && fsm != S_ABORT) assert ($stable(instr) || $past(fsm) == S_IDLE);
      assert (p_wait <= (LN + 1)'(N));
    end
  end
`endif

endmodule
