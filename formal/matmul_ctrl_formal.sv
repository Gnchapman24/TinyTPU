// Formal harness for matmul_ctrl (task A4.2): array timing, ABORT, loop counts and termination.
//
// Environment: the harness builds each MATMUL from small free inputs (4-bit bases, counts 1..CMAX,
// any act_signed and acc_mode except 11) such that every address is in range, which the
// sequencer's range check guarantees; it is captured at start and held until the unit is idle
// again. PTAB answers each read P_LAT cycles later with the entry's own address as its bias; the
// array returns each vector's result A_LAT + 2N + 1 cycles after the ABUF read; acc_unit is busy
// for the 2 cycles after each result. The solver may abort at any time.
//
// Task `prove` (PDR, for all time):
//  P1 Array timing at the array's inputs: a tile's rows load 0, 1, ..., N-1 on consecutive
//     cycles (row 0 may restart a tile, e.g. after ABORT); no vector enters before row 0 of its
//     tile; a tile's row 0 loads at least N cycles after the previous vector entered.
//  P2 Tile starts (row-0 reads) within an instruction are at least max(3, N + M - 1) cycles
//     apart. With one push per row m per tile, this keeps updates of an ACC row 3 cycles apart
//     (acc_unit has no forwarding; its own assertion checks the same thing in simulation).
//  P3 From the cycle ABORT is requested: no WBUF, ABUF or PTAB read; afterwards no done.
//  plus the assertions of matmul_ctrl and of the real sync_fifo result queue.
// Task `bmc` (from reset, PROP_BMC): properties that relate the loop counters to products of
// the counts, for which PDR finds no invariants. The depth covers the largest instruction
// (counts up to CMAX = 2) from start to done.
//  B1 Exactly nt*kt*N row reads, nt*kt*m vector reads, nt*N PTAB reads in bias mode (else 0),
//     and as many results as vector reads, by done; never more before.
//  B2 Every bias-mode result sees acc_bias[col] == p_base + n*N + col for its column tile n.
//  B3 done within DONE_BOUND cycles of start; idle within ABORT_BOUND cycles of an ABORT.
module matmul_ctrl_formal
  import isa_pkg::*;
#(
  parameter int N    = 2,
  parameter int CMAX = 2
) (
  input logic       clk,
  input logic       rst,
  input logic       start,
  input logic       abort_req,
  input logic [3:0] in_w, in_a, in_c, in_p,       // bases
  input logic [1:0] in_kt, in_nt, in_m,           // counts
  input logic       in_signed,
  input logic [1:0] in_mode
);
  localparam int W_LAT = 3, A_LAT = 2, P_LAT = 2;
  localparam int ARR_LAT = A_LAT + 2 * N + 1;            // ABUF read to array result
  localparam int LN = $clog2(N);
  localparam int DEPTH = 16;                             // every buffer
  localparam int T_MAX = (N + CMAX - 1 > 3) ? N + CMAX - 1 : 3;
  localparam int DONE_BOUND  = CMAX * CMAX * T_MAX + CMAX * (N + ARR_LAT + 6) + ARR_LAT + 8;
  localparam int ABORT_BOUND = ARR_LAT + P_LAT + 6;

  logic                         done, idle, act_signed;
  logic                         w_rd_en, a_rd_en, p_rd_en, a_rd_tag;
  logic [$clog2(DEPTH)+LN-1:0]  w_rd_addr;
  logic [LN:0]                  w_rd_tag;
  logic [$clog2(DEPTH)-1:0]     a_rd_addr, p_rd_addr, acc_addr;
  logic [LN-1:0]                p_rd_tag, p_rd_col;
  logic                         p_rd_valid, arr_valid, acc_valid, acc_busy;
  logic [PTAB_ENTRY_W-1:0]      p_rd_data;
  acc_mode_t                    acc_mode;
  logic [N-1:0][31:0]           acc_bias;
  logic [INSTR_W-1:0]           instr;

  matmul_ctrl #(
    .N(N), .WBUF_DEPTH(DEPTH), .ABUF_DEPTH(DEPTH), .ACC_DEPTH(DEPTH), .PTAB_DEPTH(DEPTH),
    .W_LAT(W_LAT), .A_LAT(A_LAT), .P_LAT(P_LAT)
  ) dut (
    .clk, .rst, .start, .instr, .abort_req, .done, .idle, .act_signed,
    .w_rd_en, .w_rd_addr, .w_rd_tag, .a_rd_en, .a_rd_addr, .a_rd_tag,
    .p_rd_en, .p_rd_addr, .p_rd_tag, .p_rd_valid, .p_rd_data, .p_rd_col,
    .arr_valid, .acc_valid, .acc_addr, .acc_mode, .acc_bias, .acc_busy
  );

  logic past_valid = 1'b0;
  always_ff @(posedge clk) past_valid <= 1'b1;
  always_comb if (!past_valid) assume (rst);

  // ---------------------------------------------------------------------------------------------
  // The instruction: built from the inputs in the start cycle, then held.
  // ---------------------------------------------------------------------------------------------
  logic [3:0] r_w, r_a, r_c, r_p;
  logic [1:0] r_kt, r_nt, r_m, r_mode;
  logic       r_signed, active;
  logic [3:0] i_w, i_a, i_c, i_p;
  logic [1:0] i_kt, i_nt, i_m, i_mode;
  logic       i_signed;
  instr_t     f;

  assign {i_w, i_a, i_c, i_p, i_kt, i_nt, i_m, i_mode, i_signed} = start && idle ?
         {in_w, in_a, in_c, in_p, in_kt, in_nt, in_m, in_mode, in_signed} :
         {r_w, r_a, r_c, r_p, r_kt, r_nt, r_m, r_mode, r_signed};

  always_comb begin
    f          = '0;
    f.opcode   = OP_MATMUL;
    f.flags    = {5'b0, i_mode, i_signed};
    f.w_base   = 16'(i_w);
    f.a_base   = 16'(i_a);
    f.acc_base = 16'(i_c);
    f.p_base   = 16'(i_p);
    f.kt       = 16'(i_kt);
    f.nt       = 16'(i_nt);
    f.m        = 16'(i_m);
  end
  assign instr = f;

  always_ff @(posedge clk) begin
    if (rst) begin
      active <= 1'b0;
      {r_w, r_a, r_c, r_p, r_kt, r_nt, r_m, r_mode, r_signed} <= '0;
    end else if (start && idle) begin
      active <= 1'b1;
      {r_w, r_a, r_c, r_p, r_kt, r_nt, r_m, r_mode, r_signed} <=
        {in_w, in_a, in_c, in_p, in_kt, in_nt, in_m, in_mode, in_signed};
    end else if (idle) begin
      active <= 1'b0;
    end
  end

  always_comb begin
    if (start) begin
      assume (idle);
      assume (in_mode != 2'b11);
      assume (in_kt != 0 && 32'(in_kt) <= CMAX && in_nt != 0 && 32'(in_nt) <= CMAX &&
              in_m != 0 && 32'(in_m) <= CMAX);
      // In range (range_check): base + product <= depth for every memory.
      assume (32'(in_w) + 32'(in_nt) * 32'(in_kt) <= DEPTH);
      assume (32'(in_a) + 32'(in_m) * 32'(in_kt) <= DEPTH);
      assume (32'(in_c) + 32'(in_m) * 32'(in_nt) <= DEPTH);
      assume (32'(in_p) + 32'(in_nt) * 32'(N) <= DEPTH);
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Environment: PTAB, array, acc_unit timing.
  // ---------------------------------------------------------------------------------------------
  logic [P_LAT-1:0]         p_pipe;
  logic [LN-1:0]            p_col [P_LAT];
  logic [$clog2(DEPTH)-1:0] p_adr [P_LAT];
  logic [ARR_LAT-1:0]       a_pipe;
  logic [W_LAT-1:0]         w_pipe;
  logic [LN-1:0]            w_row_p [W_LAT];
  logic [A_LAT-1:0]         av_pipe;
  logic [1:0]               busy_pipe;

  always_ff @(posedge clk) begin
    if (rst) begin
      p_pipe    <= '0;
      a_pipe    <= '0;
      w_pipe    <= '0;
      av_pipe   <= '0;
      busy_pipe <= '0;
    end else begin
      p_pipe    <= {p_pipe[P_LAT-2:0], p_rd_en};
      a_pipe    <= {a_pipe[ARR_LAT-2:0], a_rd_en};
      w_pipe    <= {w_pipe[W_LAT-2:0], w_rd_en && w_rd_tag[LN]};
      av_pipe   <= {av_pipe[A_LAT-2:0], a_rd_en && a_rd_tag};
      busy_pipe <= {busy_pipe[0], arr_valid};
    end
    p_col[0]   <= p_rd_tag;
    p_adr[0]   <= p_rd_addr;
    w_row_p[0] <= w_rd_tag[LN-1:0];
    for (int i = 1; i < P_LAT; i++) begin
      p_col[i] <= p_col[i-1];
      p_adr[i] <= p_adr[i-1];
    end
    for (int i = 1; i < W_LAT; i++) w_row_p[i] <= w_row_p[i-1];
  end

  assign p_rd_valid = p_pipe[P_LAT-1];
  assign p_rd_col   = p_col[P_LAT-1];
  assign p_rd_data  = PTAB_ENTRY_W'(p_adr[P_LAT-1]);      // bias = the entry's own address
  assign arr_valid  = a_pipe[ARR_LAT-1];
  assign acc_busy   = |busy_pipe;

  // What the array sees.
  logic          arr_w_load, arr_a_valid;
  logic [LN-1:0] arr_w_row;
  assign arr_w_load  = w_pipe[W_LAT-1];
  assign arr_w_row   = w_row_p[W_LAT-1];
  assign arr_a_valid = av_pipe[A_LAT-1];

  // ---------------------------------------------------------------------------------------------
  // Observers.
  // ---------------------------------------------------------------------------------------------
  logic        aborted;
  logic        tile_loading, row0_seen, vec_seen;
  logic [LN:0] next_row;
  logic [3:0]  since_vec, since_tile;                   // saturating
  logic        tile_seen;
  logic [3:0]  t_min;

  assign t_min = (4'(N) + 4'(r_m) - 4'd1 > 4'd3) ? 4'(N) + 4'(r_m) - 4'd1 : 4'd3;

  always_ff @(posedge clk) begin
    if (rst) begin
      aborted <= 1'b0;
      tile_loading <= 1'b0; next_row <= '0; row0_seen <= 1'b0;
      since_vec <= '0; vec_seen <= 1'b0; since_tile <= '0; tile_seen <= 1'b0;
    end else begin
      if (start && idle) begin
        aborted   <= 1'b0;
        tile_seen <= 1'b0;
      end else if (abort_req && !idle) begin
        aborted <= 1'b1;
      end

      if (arr_w_load) begin
        next_row     <= (LN + 1)'(arr_w_row) + 1'b1;
        tile_loading <= arr_w_row != LN'(N - 1);
        if (arr_w_row == '0) row0_seen <= 1'b1;
      end
      if (arr_a_valid) begin
        since_vec <= '0;
        vec_seen  <= 1'b1;
      end else if (since_vec != 4'hF) begin
        since_vec <= since_vec + 1'b1;
      end
      if (w_rd_en && w_rd_tag[LN-1:0] == '0) begin
        since_tile <= '0;
        tile_seen  <= 1'b1;
      end else if (since_tile != 4'hF) begin
        since_tile <= since_tile + 1'b1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      // P1 Array timing.
      if (arr_w_load && tile_loading && arr_w_row != '0) assert (arr_w_row == next_row[LN-1:0]);
      if (arr_w_load && arr_w_row != '0) assert (tile_loading);
      if (arr_w_load && arr_w_row == '0 && vec_seen) assert (since_vec >= 4'(N - 1));
      if (arr_a_valid) assert (row0_seen || (arr_w_load && arr_w_row == '0));   // vector 0 may enter with row 0
      // P2 Tile period within one instruction.
      if (w_rd_en && w_rd_tag[LN-1:0] == '0 && tile_seen && active) assert (5'(since_tile) + 5'd1 >= 5'(t_min));
      // P3 Nothing after ABORT.
      if (aborted) assert (!done && !w_rd_en && !a_rd_en && !p_rd_en);
      if (abort_req && !idle) assert (!w_rd_en && !a_rd_en && !p_rd_en);   // from the request's own cycle
    end
  end

  // Invariants linking the unit's bookkeeping to the work in flight. True in every reachable
  // state (so they are checked like any assertion); they also let k-induction rule out
  // unreachable starting states.
  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      assert (dut.p_wait == ($countones(p_pipe)));                       // PTAB reads in flight
      assert (dut.unused_q_count == ($countones(a_pipe)));               // one queue entry per vector in flight
      if (idle) assert (p_pipe == '0 && a_pipe == '0);                    // idle = nothing in flight
      // The latched instruction is a legal one, and a busy unit is always tracked as active.
      if (active) assert (r_kt != 0 && 32'(r_kt) <= CMAX && r_nt != 0 && 32'(r_nt) <= CMAX &&
                          r_m != 0 && 32'(r_m) <= CMAX && r_mode != 2'b11);
      if (!idle) assert (active);
      // Tile counters (fsm encoding: S_TILE = 3): rows at c = 0..N-1, vectors at c = D..D+M-1.
      if (3'(dut.fsm) == 3'd3) begin
        assert (dut.c <= dut.t_last);
        assert (dut.row == ((dut.c < (LN + 2)'(N)) ? LN'(dut.c) : '0));
        assert (32'(dut.m) == ((32'(dut.c) <= 32'(W_LAT - A_LAT)) ? 0 :
                               (32'(dut.c) - 32'(W_LAT - A_LAT) < 32'(r_m)) ? 32'(dut.c) - 32'(W_LAT - A_LAT) : 32'(r_m)));
        if (dut.c != '0 && !aborted) assert (32'(since_tile) == ((32'(dut.c) - 1 < 15) ? 32'(dut.c) - 1 : 15) && tile_seen);
      end
    end
  end

`ifdef PROP_BMC
  logic [5:0] n_w, n_a, n_p, n_res;
  logic [7:0] cyc, abort_cyc;
  logic [5:0] want_w, want_a, want_p;
  logic [3:0] res_n;

  assign want_w = 6'(r_nt) * 6'(r_kt) * 6'(N);
  assign want_a = 6'(r_nt) * 6'(r_kt) * 6'(r_m);
  assign want_p = r_mode == 2'(ACC_BIAS) ? 6'(r_nt) * 6'(N) : '0;
  assign res_n  = 4'((4'(acc_addr) - r_c) % 4'(r_nt));   // column tile of a result

  always_ff @(posedge clk) begin
    if (rst || (start && idle)) begin
      n_w <= '0; n_a <= '0; n_p <= '0; n_res <= '0; cyc <= '0;
    end else begin
      if (w_rd_en)   n_w   <= n_w + 1'b1;
      if (a_rd_en)   n_a   <= n_a + 1'b1;
      if (p_rd_en)   n_p   <= n_p + 1'b1;
      if (arr_valid) n_res <= n_res + 1'b1;
      if (!idle)     cyc   <= cyc + 1'b1;
    end
    if (rst) abort_cyc <= '0;
    else     abort_cyc <= (aborted && !idle) ? abort_cyc + 1'b1 : '0;
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      if (active && !aborted) begin
        assert (n_w <= want_w && n_a <= want_a && n_p <= want_p);                       // B1
        if (done) assert (n_w == want_w && n_a == want_a && n_p == want_p && n_res == n_a);
        assert (cyc < 8'(DONE_BOUND));                                                    // B3
      end
      if (acc_valid && acc_mode == ACC_BIAS && active && !aborted)                         // B2
        for (int col = 0; col < N; col++)
          assert (acc_bias[col] == 32'(r_p) + 32'(res_n) * 32'(N) + 32'(col));
      assert (abort_cyc < 8'(ABORT_BOUND));                                               // B3
    end
  end
`endif

  // Reachability.
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      cover (done && r_kt == 2'd2 && r_nt == 2'd2 && r_m == 2'd2 && r_mode == 2'(ACC_BIAS));
      cover (done && r_m == 2'd2 && r_mode == 2'(ACC_ACCUMULATE));
      cover (aborted && idle);
      cover (arr_w_load && arr_w_row == '0 && vec_seen && since_vec == 4'(N - 1));
    end
  end

endmodule
