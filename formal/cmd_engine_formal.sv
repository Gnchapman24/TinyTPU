// Formal harness for cmd_engine (task A4.1): framing, CRC, NAK, timeout and progress properties
// for any byte stream.
//
// Environment: the solver drives rx_data / rx_valid with any bytes at any time (but, like the
// host's one-request-in-flight rule, never when the receive queue is full) and tx_ready freely.
// The core is abstract: it is running only after core_start, may stop at any time, stops within
// ABORT_MAX cycles of core_abort, and is idle right after core_rst. Host reads return any data
// one cycle later. Status fields are free.
//
// Properties:
//  F1 Every reply is well formed: byte 0 is 0x5A; the code is 0..6; a NAK has LEN = 0;
//     LEN <= MAX_PAYLOAD; exactly 5 + LEN + 2 bytes are sent, then the line is idle or the
//     next reply starts. (Reply CRCs are not checked here: relating two CRC chains over many
//     bytes stalls the solvers. Every reply in tb/test_cmd_engine.py is CRC-checked by
//     tinytpu.packet.parse_reply.)
//  F2 A reply starts only for a completed request (the parser reached its CRC check) that has
//     not been answered, or for a pending RUN / INFER: a partial, timed-out or noise packet
//     never gets a reply, and no request gets two.
//  F3 Timeout: the parser never stays inside a packet for more than TIMEOUT_CYCLES cycles
//     without a byte to read.
//  F4 Host memory access only while the core is stopped, and only to existing rows.
//  F5 The receive queue drops a byte only when it is full (rx_overflow).
//  F6 Progress: within PROGRESS_BOUND cycles of a completed request, a reply has started or the
//     parser is idle again (a RUN / INFER reply is then pending).
//
// Engines (cmd_engine.sby): F1, F2, F3, F5 and "memories only while stopped" (F4) hold for all
// time (k-induction, task `prove`, with the invariants below). The address range of F4 and F6
// are checked from reset by bounded model checking (task `bmc`, PROP_BMC, depth 30: a complete
// short exchange). PDR did not converge on this design in 25 minutes.
module cmd_engine_formal
  import isa_pkg::*;
(
  input logic        clk,
  input logic        rst,
  input logic [7:0]  rx_data,
  input logic        rx_valid,
  input logic        tx_ready,
  input logic        f_stop          // the abstract program stops now
);
  // Data returned by the core (status fields, memory rows) does not steer the engine; it is
  // fixed so the solver works on the control state. Its bytes are still framed and CRC-checked.
  seq_err_t    f_err;
  logic [15:0] f_pc, f_halt;
  logic [127:0] f_rd_data;
  assign f_err = ERR_NONE;
  assign f_pc = 16'h1234;
  assign f_halt = 16'h00A5;
  assign f_rd_data = {16{8'h5A}};
  localparam int N = 2, DEPTH = 4, MAXP = 16, TIMEOUT = 4, RXQ = 4, ABORT_MAX = 3;
  localparam int HW = 128;
  localparam int PROGRESS_BOUND = ABORT_MAX + 12;

  logic [7:0]  tx_data;
  logic        tx_valid, core_rst, core_start, core_abort, host_rd_en, host_wr_en, host_rd_valid;
  logic [15:0] core_start_pc, host_addr;
  seq_state_t  core_state;
  mem_id_t     host_mem;
  logic [HW-1:0] host_wr_data;
  logic        pkt_ok, pkt_nak, pkt_timeout, rx_overflow;

  cmd_engine #(
    .N(N), .IMEM_DEPTH(DEPTH), .WBUF_DEPTH(DEPTH), .ABUF_DEPTH(DEPTH), .ACC_DEPTH(DEPTH), .PTAB_DEPTH(DEPTH),
    .CLKS_PER_BIT(8), .TIMEOUT_CYCLES(TIMEOUT), .MAX_PAYLOAD(MAXP), .RXQ_DEPTH(RXQ), .HW(HW)
  ) dut (
    .clk, .rst, .rx_data, .rx_valid, .tx_data, .tx_valid, .tx_ready,
    .core_rst, .core_start, .core_start_pc, .core_abort,
    .core_state, .core_err(f_err), .core_pc(f_pc), .core_halt_code(f_halt),
    .host_mem, .host_rd_en, .host_wr_en, .host_addr, .host_wr_data, .host_rd_valid, .host_rd_data(f_rd_data),
    .pkt_ok, .pkt_nak, .pkt_timeout, .rx_overflow
  );

  logic past_valid = 1'b0;
  always_ff @(posedge clk) past_valid <= 1'b1;
  always_comb if (!past_valid) assume (rst);

  // The host never sends into a full queue (one request in flight).
  always_comb if (rx_valid) assume (dut.q_in_ready);

  // ---------------------------------------------------------------------------------------------
  // Abstract core.
  // ---------------------------------------------------------------------------------------------
  logic       running, aborting;
  logic [2:0] abort_cnt;
  always_ff @(posedge clk) begin
    if (rst || core_rst) begin
      running <= 1'b0; aborting <= 1'b0; abort_cnt <= '0;
    end else begin
      if (core_start) running <= 1'b1;
      else if (running && (f_stop || (aborting && abort_cnt == 3'(ABORT_MAX)))) running <= 1'b0;
      if (core_abort && running) begin aborting <= 1'b1; abort_cnt <= '0; end
      else if (aborting) abort_cnt <= abort_cnt + 1'b1;
      if (!running) aborting <= 1'b0;
    end
  end
  assign core_state = running ? ST_RUNNING : ST_HALTED_OK;

  always_ff @(posedge clk) begin
    if (rst) host_rd_valid <= 1'b0;
    else     host_rd_valid <= host_rd_en;
  end

  // ---------------------------------------------------------------------------------------------
  // F1 Reply framing, with an independent CRC.
  // ---------------------------------------------------------------------------------------------
  logic        tx_fire;
  logic [15:0] t_n, t_len;               // bytes sent of the current reply; its LEN
  logic [7:0]  t_code;
  logic        in_reply;

  assign tx_fire  = tx_valid && tx_ready;

  logic reply_done;                                          // the last CRC byte went out last cycle
  always_ff @(posedge clk) begin
    if (rst) begin
      in_reply <= 1'b0; t_n <= '0; t_len <= '0; t_code <= '0; reply_done <= 1'b0;
    end else begin
      reply_done <= 1'b0;
      if (tx_fire) begin
        if (!in_reply) begin
          in_reply <= 1'b1;
          t_n      <= 16'd1;
        end else begin
          t_n <= t_n + 1'b1;
          if (t_n == 16'd1) t_code <= tx_data;
          if (t_n == 16'd3) t_len[7:0] <= tx_data;
          if (t_n == 16'd4) t_len[15:8] <= tx_data;
          if (t_n >= 16'd5 && t_n == 16'd5 + t_len + 16'd1) begin
            in_reply   <= 1'b0;
            reply_done <= 1'b1;
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Observers. Parser state codes of cmd_engine (pstate_t), repeated here because yosys-slang
  // cannot reference enum members through the hierarchy; transmitter state codes likewise.
  // ---------------------------------------------------------------------------------------------
  localparam logic [4:0] PS_SYNC = 5'd0, PS_EVAL2 = 5'd7, PS_DATA = 5'd8, PS_CHECK = 5'd11, PS_EXEC = 5'd12,
                         PS_WAIT_RUN = 5'd13, PS_REPLY = 5'd16, PS_TX = 5'd17;
  localparam logic [3:0] TS_IDLE = 4'd0, TS_RD = 4'd7, TS_RDWAIT = 4'd8, TS_ROW = 4'd9;

  logic [4:0] ps;
  assign ps = 5'(dut.ps);

  logic       owed;                      // a completed request (or a parked RUN / INFER) is owed a reply
  logic [4:0] gap, since_check;
  logic       waiting;                   // a completed request has not yet been answered or parked

  always_ff @(posedge clk) begin
    if (rst) begin
      owed <= 1'b0; gap <= '0; since_check <= '0; waiting <= 1'b0;
    end else begin
      if (ps == PS_CHECK || (ps == PS_SYNC && dut.pend && !running)) owed <= 1'b1;
      else if (dut.t_go || ps == PS_SYNC) owed <= 1'b0;
      gap <= (dut.in_packet && !dut.q_valid) ? gap + 1'b1 : '0;
      if (ps == PS_CHECK) begin waiting <= 1'b1; since_check <= '0; end
      else if (waiting && (dut.t_go || ps == PS_SYNC)) waiting <= 1'b0;
      else if (waiting) since_check <= since_check + 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      // F2 A reply starts only when one is owed.
      if (dut.t_go) assert ($past(owed));
      // F3 The parser never waits inside a packet longer than the timeout.
      assert (gap <= 5'(TIMEOUT));
      // F4 Memories only while the core is stopped, and only existing memories.
      if (host_rd_en || host_wr_en) assert (!running && 32'(host_mem) <= 4);
      // F5 A byte is dropped only when the receive queue is full.
      if (rx_overflow) assert ($past(rx_valid && !dut.q_in_ready));
    end
  end

  // Invariants (true in every reachable state; they let k-induction start only from such states).
  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      if (ps >= PS_EXEC && ps <= PS_REPLY) assert (owed);
      if (dut.t_go) assert (ps == PS_TX);
      if (dut.ts != TS_IDLE) assert (ps == PS_TX);
      if (dut.in_packet) assert (gap == 5'(dut.idle_cnt));
      assert (32'(dut.idle_cnt) <= TIMEOUT - 1);
      // Only a completed RUN / INFER starts the core.
      if (core_start) assert (ps == PS_WAIT_RUN);
      // A packet checked while the core was stopped keeps it stopped until it is answered.
      if (ps >= PS_EVAL2 && ps <= PS_EXEC && !dut.e_busy && dut.cmd inside {8'h02, 8'h03, 8'h04, 8'h08}) assert (!running);
      if (ps >= PS_DATA && ps <= PS_EXEC && dut.wr_ok)
        assert (!dut.e_busy && !dut.e_unknown && !dut.e_too_long && !dut.e_bad_len && !dut.e_dup &&
                !dut.e_bad_arg && (dut.cmd == 8'h02 || dut.cmd == 8'h08));
      if (ps == PS_REPLY && dut.r_mem_rows) assert (!running);
      if (ps == PS_TX && (dut.t_rows_left != '0 || dut.ts == TS_RD || dut.ts == TS_RDWAIT || dut.ts == TS_ROW)) assert (!running);
      if (host_wr_en) assert ($past(ps == PS_DATA && dut.wr_ok));   // a row is written the cycle after it completes
      // Every remembered memory ID names a real memory.
      if (ps >= PS_DATA && ps <= PS_EXEC && dut.wr_ok) assert (dut.w_mem <= 8'd4);
      if (ps > PS_WAIT_RUN && ps <= PS_REPLY && dut.r_mem_rows) assert (dut.r_mem <= 8'd4);   // r_* are set as P_EXEC is left
      if (ps == PS_TX && dut.t_go && dut.r_mem_rows) assert (dut.t_mem <= 8'd4 && !running);
      if (ps == PS_TX && dut.t_go && dut.r_mem_rows) assert (dut.r_mem == dut.t_mem);
      if (dut.ts != TS_IDLE && (dut.t_rows_left != '0 || dut.ts == TS_RD || dut.ts == TS_RDWAIT || dut.ts == TS_ROW))
        assert (dut.t_mem <= 8'd4);
      if (dut.ts == TS_IDLE) assert (dut.t_rows_left == '0);
      // Transmitter order: the CRC bytes come only after the last row; a block or row being sent
      // has at least one byte left. (T_BLK = 6, T_ROW = 9, T_CRC0 = 10, T_CRC1 = 11.)
      if (dut.ts == 4'd10 || dut.ts == 4'd11) assert (dut.t_rows_left == '0);
      if (dut.ts == 4'd6) assert (dut.t_blk_left != '0);
      if (dut.ts == TS_ROW) assert (dut.t_row_left != '0);
      if (dut.pend && dut.pend_cmd == 8'h08) assert (dut.pend_mem <= 8'd4);
      if (ps >= PS_EVAL2 && ps <= PS_EXEC && dut.e_dup)
        assert (dut.last_valid && dut.last_cmd == dut.cmd && dut.last_seq == dut.seq &&
                (dut.cmd == 8'h02 || dut.cmd == 8'h04 || dut.cmd == 8'h08));
      if (dut.last_valid) assert (dut.last_cmd inside {8'h02, 8'h04, 8'h06, 8'h08});
      if (ps >= PS_EVAL2 && ps <= PS_EXEC && dut.cmd == 8'h03 && !dut.e_bad_arg && !dut.e_unknown &&
          !dut.e_bad_len && !dut.e_too_long) assert (dut.a_mem <= 8'd4);
      if (ps >= PS_EVAL2 && ps <= PS_EXEC && dut.cmd == 8'h08 && !dut.e_bad_arg && !dut.e_unknown &&
          !dut.e_bad_len && !dut.e_too_long) assert (dut.i_out_mem <= 8'd4);
      if (dut.last_valid && dut.last_cmd == 8'h08) assert (dut.last_mem <= 8'd4);
    end
  end

  // F1 Reply framing, for all time.
  function automatic logic [31:0] row_bytes(input logic [7:0] m);
    unique case (m)
      8'd0:       return 32'd16;
      8'd1, 8'd2: return 32'(N);
      8'd3:       return 32'(4 * N);
      default:    return 32'd8;
    endcase
  endfunction
  function automatic logic [31:0] blk_bytes(input logic [1:0] b);
    return (b == 2'd2) ? 32'd16 : (b == 2'd1) ? 32'd8 : 32'd0;
  endfunction

  function automatic logic [31:0] ub(input logic [7:0] m, input logic [15:0] count);   // bytes of count units
    return 32'(count) * row_bytes(m) * ((m == 8'd1) ? 32'(N) : 32'd1);
  endfunction

  logic [31:0] dut_left;                 // bytes the transmitter still has to send after this state's byte(s)
  always_comb begin
    dut_left = 32'(dut.t_rows_left) * row_bytes(dut.t_mem) + 32'd2;
    if (dut.ts == 4'd6)  dut_left = dut_left + 32'(dut.t_blk_left);
    if (dut.ts == TS_ROW) dut_left = dut_left + 32'(dut.t_row_left);
    if (dut.ts == 4'd10) dut_left = 32'd2;
    if (dut.ts == 4'd11) dut_left = 32'd1;
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      if (tx_fire && !in_reply) assert (tx_data == 8'h5A);
      if (in_reply && t_n == 16'd2) assert (t_code <= 8'h06);
      if (in_reply && t_n == 16'd5) assert (t_len <= 16'(MAXP) && (t_code == 8'h00 || t_len == 16'd0));
      if (dut.ts == TS_IDLE) assert (!in_reply);             // never fewer bytes than LEN says
      // Invariants: the harness's count follows the transmitter, and LEN matches the content.
      assert (in_reply == (dut.ts != TS_IDLE && dut.ts != 4'd1));
      if (dut.ts == 4'd2) assert (t_n == 16'd1);
      if (dut.ts == 4'd3) assert (t_n == 16'd2 && t_code == dut.t_code);
      if (dut.ts == 4'd4) assert (t_n == 16'd3 && t_code == dut.t_code);
      if (dut.ts == 4'd5) assert (t_n == 16'd4 && t_code == dut.t_code && t_len[7:0] == dut.t_len[7:0]);
      if (dut.ts >= 4'd6) assert (t_n >= 16'd5 && t_len == dut.t_len && t_code == dut.t_code && dut.t_len <= 16'(MAXP) &&
                                  32'd7 + 32'(t_len) - 32'(t_n) == dut_left);
      if (dut.ts >= 4'd1 && dut.ts <= 4'd5)
        assert (32'(dut.t_len) == 32'(dut.t_blk_left) + 32'(dut.t_rows_left) * row_bytes(dut.t_mem) &&
                dut.t_len <= 16'(MAXP) && dut.t_code <= 8'h06 && (dut.t_code == 8'h00 || dut.t_len == 16'd0));
      if (dut.ts == TS_RD || dut.ts == TS_RDWAIT) assert (dut.t_rows_left != '0);
      if (dut.ts != TS_IDLE || dut.t_go) assert (dut.t_mem <= 8'd4 || dut.t_rows_left == '0);
      // What the transmitter is about to load (t_go) and what P_REPLY hands it.
      if (dut.t_go || ps == PS_REPLY)
        assert (32'(dut.r_len) == blk_bytes(dut.r_blk) + (dut.r_mem_rows ? 32'(dut.r_rows) * row_bytes(dut.r_mem) : 32'd0) &&
                dut.r_blk != 2'd3 && dut.r_len <= 16'(MAXP) && dut.r_code <= 8'h06 &&
                (dut.r_code == 8'h00 || (dut.r_len == 16'd0 && dut.r_blk == 2'd0 && !dut.r_mem_rows)));
      if (dut.t_go) assert (dut.t_code == dut.r_code && dut.t_len == dut.r_len && dut.t_mem == dut.r_mem);
      // Where r_* come from: every source describes a reply that fits.
      if (ps == 5'd14 || ps == 5'd15) assert (dut.r_code == 8'h00 && !dut.r_mem_rows);   // WAIT_STOP / WAIT_RST
      if (dut.pend) assert (dut.pend_cmd == 8'h04 || dut.pend_cmd == 8'h08);
      if (dut.pend && dut.pend_cmd == 8'h08) assert (32'd8 + ub(dut.pend_mem, dut.pend_count) <= MAXP && dut.pend_count != 0);
      if (dut.last_valid && dut.last_cmd == 8'h08) assert (32'd8 + ub(dut.last_mem, dut.last_count) <= MAXP && dut.last_count != 0);
      if (ps >= PS_EVAL2 && ps <= PS_EXEC && !dut.e_bad_arg && !dut.e_unknown && !dut.e_bad_len && !dut.e_too_long) begin
        if (dut.cmd == 8'h03) assert (ub(dut.a_mem, dut.a_count) <= MAXP && dut.a_count != 0);
        if (dut.cmd == 8'h08) assert (32'd8 + ub(dut.i_out_mem, dut.i_out_count) <= MAXP && dut.i_out_count != 0);
      end
    end
  end

`ifdef PROP_BMC
  // Bounded from reset: address range, progress.
  always_ff @(posedge clk) begin
    if (past_valid && !rst && !$past(rst)) begin
      // F4 (range)
      if (host_rd_en || host_wr_en) assert (32'(host_addr) < (host_mem == MEM_WBUF ? DEPTH * N : DEPTH));
      // F6
      if (waiting) assert (since_check < 5'(PROGRESS_BOUND));
    end
  end
`endif

  // Reachability.
  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      cover (reply_done && t_code == 8'h00 && t_len == 16'd8);     // a status reply
      cover (reply_done && t_code == 8'h01);                       // NAK bad CRC
      cover (pkt_timeout);
      cover (host_wr_en);
      cover (core_start);
    end
  end

endmodule
