/** Packet engine: the host protocol of docs/uart_protocol.md on top of tpu_core.
 *
 *  Receive: bytes from uart_rx go through a small queue (RXQ_DEPTH) into the parser, so the
 *  receiver keeps listening while a reply is being sent. A request is `A5 CMD SEQ LEN payload
 *  CRC`; the CRC unit sees every byte after the sync byte, including the two CRC bytes, and the
 *  packet is intact exactly when its result is zero. A gap of more than TIMEOUT_CYCLES between
 *  bytes inside a request discards it without a reply.
 *
 *  Checks: after the header and the command's fixed fields (WRITE / READ 6 bytes, RUN 2,
 *  INFER 10), one evaluation cycle decides, in the protocol's priority order, unknown command,
 *  payload too long, bad length, duplicate (same CMD and SEQ as the last WRITE / RUN / INFER /
 *  ABORT executed), busy (a program is running) and bad argument. Only if all pass are the
 *  payload bytes of a WRITE or INFER written through to memory as they arrive; the reply
 *  (after the CRC) is NAK 0x01 if the CRC failed, else the first problem found, else ACK.
 *  A PTAB entry with non-zero reserved bits is not written and turns the reply into NAK 0x05.
 *
 *  Memories are handled as runs of physical rows (a WBUF tile is N rows; a PTAB entry is 8
 *  wire bytes for 54 bits), read and written through tpu_core's host port while the sequencer
 *  is stopped. Replies are built from a status or info block followed by memory rows, with the
 *  CRC computed as the bytes go out.
 *
 *  Programs: RUN and INFER start the sequencer only after a good CRC and leave a pending reply
 *  (status, plus INFER's output) that is sent once the program stops. ABORT and RESET replace a
 *  pending reply with their own. A duplicate RUN / INFER while its program still runs re-arms
 *  the pending reply instead of replying.
 *
 *  Latency: a reply starts a few cycles after the last CRC byte of its request is taken from the queue (pending replies: after the program stops).
 */
module cmd_engine
  import isa_pkg::*;
#(
  parameter int N              = 8,                   // array size
  parameter int IMEM_DEPTH     = DEFAULT_IMEM_DEPTH,  // instructions
  parameter int WBUF_DEPTH     = DEFAULT_WBUF_DEPTH,  // weight tiles
  parameter int ABUF_DEPTH     = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH      = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH     = DEFAULT_PTAB_DEPTH,  // parameter entries
  parameter int CLKS_PER_BIT   = 100,                 // reported by INFO
  parameter int TIMEOUT_CYCLES = 500_000,             // inter-byte timeout inside a request (5 ms)
  parameter int MAX_PAYLOAD    = 4096,                // largest payload, request or reply
  parameter int RXQ_DEPTH      = 32,                  // receive queue (bytes); a power of two
  parameter int HW             = (N * 32 > INSTR_W) ? N * 32 : INSTR_W  // tpu_core host data width
) (
  input  logic              clk,            // clock
  input  logic              rst,            // synchronous active-high reset
  input  logic [7:0]        rx_data,        // received byte (from uart_rx)
  input  logic              rx_valid,       // rx_data holds a new byte this cycle
  output logic [7:0]        tx_data,        // byte to send (to uart_tx)
  output logic              tx_valid,       // tx_data is valid
  input  logic              tx_ready,       // the transmitter takes tx_data this cycle
  output logic              core_rst,       // one-cycle pulse: reset the core (RESET command)
  output logic              core_start,     // one-cycle pulse: run from core_start_pc
  output logic [15:0]       core_start_pc,  // first instruction
  output logic              core_abort,     // one-cycle pulse: abort the running program
  input  seq_state_t        core_state,     // sequencer state
  input  seq_err_t          core_err,       // sequencer error code
  input  logic [15:0]       core_pc,        // sequencer PC
  input  logic [15:0]       core_halt_code, // HALT code
  output mem_id_t           host_mem,       // memory of the host access
  output logic              host_rd_en,     // host read of row host_addr
  output logic              host_wr_en,     // host write of row host_addr
  output logic [15:0]       host_addr,      // physical row
  output logic [HW-1:0]     host_wr_data,   // row to write, right-aligned
  input  logic              host_rd_valid,  // host_rd_data holds the row read
  input  logic [HW-1:0]     host_rd_data,   // row read, right-aligned
  output logic              pkt_ok,         // pulse: an ACK reply was started
  output logic              pkt_nak,        // pulse: a NAK reply was started
  output logic              pkt_timeout,    // pulse: a partial request was discarded
  output logic              rx_overflow     // pulse: a received byte was dropped (queue full)
);

  if (N < 2 || (N & (N - 1)) != 0) begin : g_check_n
    $error("cmd_engine: N must be a power of two, at least 2");
  end

  localparam int LN   = $clog2(N);
  localparam int MRB  = (4 * N > 16) ? 4 * N : 16;   // most bytes in one row (ACC or IMEM)
  localparam int IW   = $clog2(MRB);                // byte index within a row
  localparam int TW   = $clog2(TIMEOUT_CYCLES + 1);

  localparam logic [7:0] SYNC_REQ = 8'hA5, SYNC_RSP = 8'h5A;
  localparam logic [7:0] C_INFO = 8'h01, C_WRITE = 8'h02, C_READ = 8'h03, C_RUN = 8'h04,
                         C_STATUS = 8'h05, C_ABORT = 8'h06, C_RESET = 8'h07, C_INFER = 8'h08;
  localparam logic [7:0] ACK = 8'h00, NAK_CRC = 8'h01, NAK_UNKNOWN = 8'h02, NAK_LENGTH = 8'h03,
                         NAK_TOO_LONG = 8'h04, NAK_ARG = 8'h05, NAK_BUSY = 8'h06;

  // ---------------------------------------------------------------------------------------------
  // Memory geometry, per memory ID (0 IMEM, 1 WBUF, 2 ABUF, 3 ACC, 4 PTAB).
  // ---------------------------------------------------------------------------------------------
  function automatic logic mem_ok(input logic [7:0] m);
    return m <= 8'd4;
  endfunction
  function automatic logic [31:0] depth_of(input logic [7:0] m);          // in ISA units
    unique case (m)
      8'd0:    return 32'(IMEM_DEPTH);
      8'd1:    return 32'(WBUF_DEPTH);
      8'd2:    return 32'(ABUF_DEPTH);
      8'd3:    return 32'(ACC_DEPTH);
      default: return 32'(PTAB_DEPTH);
    endcase
  endfunction
  function automatic logic [4:0] row_shift(input logic [7:0] m);         // log2 bytes per row
    unique case (m)
      8'd0:    return 5'd4;
      8'd1:    return 5'(LN);
      8'd2:    return 5'(LN);
      8'd3:    return 5'(LN + 2);
      default: return 5'd3;
    endcase
  endfunction
  function automatic logic [4:0] rows_shift(input logic [7:0] m);        // log2 rows per unit
    return (m == 8'd1) ? 5'(LN) : 5'd0;
  endfunction
  function automatic logic [31:0] unit_bytes(input logic [7:0] m, input logic [15:0] count);
    return 32'(count) << (row_shift(m) + rows_shift(m));
  endfunction

  // ---------------------------------------------------------------------------------------------
  // Receive queue and request CRC.
  // ---------------------------------------------------------------------------------------------
  logic       q_in_ready, q_valid, q_pop;
  logic [7:0] q_data;
  logic [$clog2(RXQ_DEPTH):0] unused_q_count;

  sync_fifo #(.WIDTH(8), .DEPTH(RXQ_DEPTH)) u_rxq (
    .clk, .rst, .in_data(rx_data), .in_valid(rx_valid), .in_ready(q_in_ready),
    .out_data(q_data), .out_valid(q_valid), .out_ready(q_pop), .count(unused_q_count)
  );

  logic        rc_clear, rc_valid, rc_zero;
  logic [15:0] unused_rc_crc;
  crc16_ccitt u_rx_crc (
    .clk, .rst, .clear(rc_clear), .data(q_data), .valid(rc_valid), .crc(unused_rc_crc), .zero(rc_zero)
  );

  // ---------------------------------------------------------------------------------------------
  // Parser state.
  // ---------------------------------------------------------------------------------------------
  typedef enum logic [4:0] {
    P_SYNC, P_CMD, P_SEQ, P_LEN0, P_LEN1, P_FIX, P_EVAL, P_EVAL2, P_DATA, P_CRC0, P_CRC1,
    P_CHECK, P_EXEC, P_WAIT_RUN, P_WAIT_STOP, P_WAIT_RST, P_REPLY, P_TX
  } pstate_t;

  pstate_t     ps;
  logic [7:0]  cmd, seq;
  logic [15:0] len, remain;
  logic [3:0]  fix_n, fix_i;
  logic [7:0]  args [10];
  logic [TW-1:0] idle_cnt;

  // Fixed fields, by name.
  logic [7:0]  a_mem, a_rsv, i_out_mem, i_rsv;
  logic [15:0] a_addr, a_count, r_pc, i_in_addr, i_pc, i_out_addr, i_out_count;
  assign a_mem       = args[0];
  assign a_rsv       = args[1];
  assign a_addr      = {args[3], args[2]};
  assign a_count     = {args[5], args[4]};
  assign r_pc        = {args[1], args[0]};
  assign i_in_addr   = {args[1], args[0]};
  assign i_pc        = {args[3], args[2]};
  assign i_out_mem   = args[4];
  assign i_rsv       = args[5];
  assign i_out_addr  = {args[7], args[6]};
  assign i_out_count = {args[9], args[8]};

  // Decision, latched in P_EVAL.
  logic        e_unknown, e_too_long, e_bad_len, e_dup, e_busy, e_bad_arg, e_ptab_bad, wr_ok;

  // Remembered last executed request (WRITE / RUN / INFER / ABORT).
  logic        last_valid;
  logic [7:0]  last_cmd, last_seq;
  logic [7:0]  last_mem;                   // output region of the last executed INFER: a duplicate
  logic [15:0] last_addr, last_count;      // answers from these (already checked), not its own payload

  // Pending RUN / INFER reply.
  logic        pend;
  logic [7:0]  pend_cmd, pend_seq, pend_mem;
  logic [15:0] pend_addr, pend_count;

  logic running;
  assign running = core_state == ST_RUNNING;

  // ---------------------------------------------------------------------------------------------
  // Evaluation (combinational, from the header and fixed fields; used in P_EVAL).
  // ---------------------------------------------------------------------------------------------
  logic        v_unknown, v_too_long, v_bad_len, v_dup, v_busy, v_bad_arg;
  logic [31:0] v_in_bytes;

  always_comb begin
    v_unknown  = !(cmd inside {C_INFO, C_WRITE, C_READ, C_RUN, C_STATUS, C_ABORT, C_RESET, C_INFER});
    v_too_long = 32'(len) > 32'(MAX_PAYLOAD);
    v_in_bytes = 32'(len) - 32'd10;
    v_bad_len  = 1'b0;
    v_bad_arg  = 1'b0;
    unique case (cmd)
      C_INFO, C_STATUS, C_ABORT, C_RESET: v_bad_len = len != 16'd0;
      C_RUN:  begin
        v_bad_len = len != 16'd2;
        v_bad_arg = 32'(r_pc) >= 32'(IMEM_DEPTH);
      end
      C_READ: begin
        v_bad_len = len != 16'd6;
        v_bad_arg = !mem_ok(a_mem) || a_rsv != 8'd0 || a_count == 16'd0 ||
                    32'(a_addr) + 32'(a_count) > depth_of(a_mem) ||
                    unit_bytes(a_mem, a_count) > 32'(MAX_PAYLOAD);
      end
      C_WRITE: begin
        // The length depends on the memory's unit; an unknown memory is a bad argument instead.
        v_bad_len = len < 16'd6 || (mem_ok(a_mem) && 32'(len) != 32'd6 + unit_bytes(a_mem, a_count));
        v_bad_arg = !mem_ok(a_mem) || a_rsv != 8'd0 || a_count == 16'd0 ||
                    32'(a_addr) + 32'(a_count) > depth_of(a_mem);
      end
      C_INFER: begin
        v_bad_len = len < 16'd10 || v_in_bytes[LN-1:0] != '0;
        v_bad_arg = 32'(i_in_addr) + (v_in_bytes >> LN) > 32'(ABUF_DEPTH) ||
                    32'(i_pc) >= 32'(IMEM_DEPTH) ||
                    !mem_ok(i_out_mem) || i_rsv != 8'd0 || i_out_count == 16'd0 ||
                    32'(i_out_addr) + 32'(i_out_count) > depth_of(i_out_mem) ||
                    32'd8 + unit_bytes(i_out_mem, i_out_count) > 32'(MAX_PAYLOAD);
      end
      default: ;
    endcase
    v_dup  = last_valid && cmd == last_cmd && seq == last_seq && (cmd inside {C_WRITE, C_RUN, C_INFER});
    v_busy = running && (cmd inside {C_WRITE, C_READ, C_RUN, C_INFER});
  end

  // ---------------------------------------------------------------------------------------------
  // Write-through: assemble rows from payload bytes and write each complete row.
  // ---------------------------------------------------------------------------------------------
  logic [7:0]    w_mem;
  logic [15:0]   w_row;                    // next physical row
  logic [IW-1:0] w_idx;                    // byte within the row
  logic [HW-9:0] w_buf;                    // bytes received so far, at the top end
  logic [HW-1:0] w_shift, w_next;
  logic          w_last_byte, w_fire, w_skip;

  // Bytes shift in from the top; once a row is complete it is right-aligned (a row is 8, 16,
  // N or 4N bytes, so this is a small mux, not a byte-wide decoder).
  assign w_shift = {q_data, w_buf};
  always_comb begin
    unique case (w_mem)
      8'd0:       w_next = w_shift >> (HW - 128);         // IMEM: 16 bytes
      8'd1, 8'd2: w_next = w_shift >> (HW - 8 * N);       // WBUF row, ABUF vector: N bytes
      8'd3:       w_next = w_shift >> (HW - 32 * N);      // ACC: 4N bytes
      default:    w_next = w_shift >> (HW - 64);          // PTAB: 8 bytes
    endcase
  end
  assign w_last_byte = 32'(w_idx) == (32'd1 << row_shift(w_mem)) - 1;

  // PTAB rows: 8 wire bytes {bias LE, M LE, s, pad} -> 54 bits {s, M, bias}.
  logic [HW-1:0] w_data;
  // Wire byte 6 is the shift (bits 7:6 reserved), byte 7 the pad byte (reserved).
  assign w_skip = w_mem == 8'd4 && (w_next[55:54] != 2'b00 || w_next[63:56] != 8'd0);
  always_comb begin
    w_data = w_next;
    if (w_mem == 8'd4) w_data = HW'({w_next[53:48], w_next[47:32], w_next[31:0]});
  end

  // ---------------------------------------------------------------------------------------------
  // Transmit side.
  // ---------------------------------------------------------------------------------------------
  typedef enum logic [3:0] {
    T_IDLE, T_SYNC, T_CODE, T_SEQ, T_LEN0, T_LEN1, T_BLK, T_RD, T_RDWAIT, T_ROW, T_CRC0, T_CRC1
  } tstate_t;

  tstate_t       ts;
  logic [7:0]    t_code, t_seq;
  logic [15:0]   t_len;
  logic [127:0]  t_blk;                    // status or info block, byte 0 in bits 7:0
  logic [4:0]    t_blk_left;
  logic [7:0]    t_mem;
  logic [15:0]   t_row, t_rows_left;
  logic [HW-1:0] t_buf;
  logic [IW:0]   t_row_left;               // bytes left in t_buf
  logic          t_go;                     // pulse: start the reply described by t_*
  logic          tx_fire;
  logic          tc_clear, tc_valid, unused_tc_zero;
  logic [15:0]   tc_crc;

  crc16_ccitt u_tx_crc (
    .clk, .rst, .clear(tc_clear), .data(tx_data), .valid(tc_valid), .crc(tc_crc), .zero(unused_tc_zero)
  );

  assign tx_fire  = tx_valid && tx_ready;
  assign tc_clear = t_go;
  assign tc_valid = tx_fire && (ts inside {T_CODE, T_SEQ, T_LEN0, T_LEN1, T_BLK, T_ROW});

  always_comb begin
    tx_valid = 1'b1;
    tx_data  = 8'h00;
    unique case (ts)
      T_SYNC:  tx_data = SYNC_RSP;
      T_CODE:  tx_data = t_code;
      T_SEQ:   tx_data = t_seq;
      T_LEN0:  tx_data = t_len[7:0];
      T_LEN1:  tx_data = t_len[15:8];
      T_BLK:   tx_data = t_blk[7:0];
      T_ROW:   tx_data = t_buf[7:0];
      T_CRC0:  tx_data = tc_crc[15:8];
      T_CRC1:  tx_data = tc_crc[7:0];
      default: tx_valid = 1'b0;
    endcase
  end

  // Status and info blocks.
  logic [127:0] status_blk, info_blk;
  assign status_blk = 128'({16'd0, core_halt_code, core_pc, 8'(core_err), 8'(core_state)});
  assign info_blk   = {16'(MAX_PAYLOAD), 16'(PTAB_DEPTH), 16'(ACC_DEPTH), 16'(ABUF_DEPTH),
                       16'(WBUF_DEPTH), 16'(IMEM_DEPTH), 16'(CLKS_PER_BIT), 8'(N), 8'd1};

  // A row read for the reply, as wire bytes (PTAB: 54 bits -> 8 bytes).
  logic [HW-1:0] rd_bytes;
  always_comb begin
    rd_bytes = host_rd_data;
    if (t_mem == 8'd4) rd_bytes = HW'({8'd0, 2'b00, host_rd_data[53:48], host_rd_data[47:32], host_rd_data[31:0]});
  end

  // ---------------------------------------------------------------------------------------------
  // Main state machine (parser and command execution).
  // ---------------------------------------------------------------------------------------------
  logic in_packet, rx_byte;
  assign in_packet = ps inside {P_CMD, P_SEQ, P_LEN0, P_LEN1, P_FIX, P_DATA, P_CRC0, P_CRC1};
  assign rx_byte   = q_valid && (ps inside {P_SYNC, P_CMD, P_SEQ, P_LEN0, P_LEN1, P_FIX, P_DATA, P_CRC0, P_CRC1})
                     && !(ps == P_SYNC && pend && !running);
  assign q_pop     = rx_byte;
  assign rc_clear  = rx_byte && ps == P_SYNC && q_data == SYNC_REQ;
  assign rc_valid  = rx_byte && in_packet;
  assign w_fire    = rx_byte && ps == P_DATA && wr_ok;

  // Host port: writes from the parser, reads from the transmitter (never at the same time).
  assign host_wr_en   = w_fire && w_last_byte && !w_skip;
  assign host_rd_en   = ts == T_RD;
  assign host_mem     = mem_id_t'(host_rd_en ? t_mem : w_mem);
  assign host_addr    = host_rd_en ? t_row : w_row;
  assign host_wr_data = w_data;

  // Reply set-up shared by every path into P_REPLY.
  logic [7:0]  r_code;
  logic [15:0] r_len;
  logic [1:0]  r_blk;                       // 0 none, 1 status, 2 info
  logic        r_mem_rows;                  // memory rows follow the block
  logic [7:0]  r_mem;
  logic [15:0] r_row0, r_rows;

  always_ff @(posedge clk) begin
    core_rst    <= 1'b0;
    core_start  <= 1'b0;
    core_abort  <= 1'b0;
    pkt_timeout <= 1'b0;
    rx_overflow <= rx_valid && !q_in_ready;
    t_go        <= 1'b0;
    if (rst) begin
      ps         <= P_SYNC;
      idle_cnt   <= '0;
      last_valid <= 1'b0;
      pend       <= 1'b0;
      wr_ok      <= 1'b0;
    end else begin
      // Inter-byte timeout inside a request.
      if (in_packet && !q_valid) begin
        if (32'(idle_cnt) >= TIMEOUT_CYCLES - 1) begin
          ps          <= P_SYNC;
          pkt_timeout <= 1'b1;
          idle_cnt    <= '0;
        end else begin
          idle_cnt <= idle_cnt + 1'b1;
        end
      end else begin
        idle_cnt <= '0;
      end

      if (!(in_packet && !q_valid && 32'(idle_cnt) >= TIMEOUT_CYCLES - 1)) begin
        unique case (ps)
          P_SYNC: begin
            if (pend && !running) begin                    // a program has stopped: send its reply
              pend       <= 1'b0;
              r_code     <= ACK;
              t_seq      <= pend_seq;
              r_blk      <= 2'd1;
              r_mem_rows <= pend_cmd == C_INFER;
              r_mem      <= pend_mem;
              r_row0     <= pend_addr << rows_shift(pend_mem);
              r_rows     <= pend_count << rows_shift(pend_mem);
              r_len      <= (pend_cmd == C_INFER) ? 16'(32'd8 + unit_bytes(pend_mem, pend_count)) : 16'd8;
              ps         <= P_REPLY;
            end else if (rx_byte && q_data == SYNC_REQ) begin
              ps <= P_CMD;
            end
          end
          P_CMD:  if (rx_byte) begin cmd <= q_data; ps <= P_SEQ; end
          P_SEQ:  if (rx_byte) begin seq <= q_data; ps <= P_LEN0; end
          P_LEN0: if (rx_byte) begin len[7:0] <= q_data; ps <= P_LEN1; end
          P_LEN1: if (rx_byte) begin
            len[15:8] <= q_data;
            remain    <= {q_data, len[7:0]};
            fix_i     <= '0;
            unique case (cmd)
              C_WRITE, C_READ: fix_n <= 4'd6;
              C_RUN:           fix_n <= 4'd2;
              C_INFER:         fix_n <= 4'd10;
              default:         fix_n <= 4'd0;
            endcase
            for (int i = 0; i < 10; i++) args[i] <= 8'd0;
            ps <= ({q_data, len[7:0]} == 16'd0 || !(cmd inside {C_WRITE, C_READ, C_RUN, C_INFER})) ? P_EVAL : P_FIX;
          end
          P_FIX: if (rx_byte) begin
            args[fix_i] <= q_data;
            fix_i       <= fix_i + 1'b1;
            remain      <= remain - 1'b1;
            if (fix_i == fix_n - 1'b1 || remain == 16'd1) ps <= P_EVAL;
          end
          P_EVAL: begin
            e_unknown  <= v_unknown;
            e_too_long <= v_too_long;
            e_bad_len  <= v_bad_len;
            e_dup      <= v_dup;
            e_busy     <= v_busy;
            e_bad_arg  <= v_bad_arg;
            e_ptab_bad <= 1'b0;
            w_mem      <= (cmd == C_INFER) ? 8'd2 : a_mem;
            w_row      <= (cmd == C_INFER) ? i_in_addr : 16'(a_addr << rows_shift(a_mem));
            w_idx      <= '0;
            ps         <= P_EVAL2;
          end
          P_EVAL2: begin                                   // second cycle: from the registered checks
            wr_ok <= !e_unknown && !e_too_long && !e_bad_len && !e_dup && !e_busy && !e_bad_arg &&
                     (cmd inside {C_WRITE, C_INFER});
            ps    <= (remain == 16'd0) ? P_CRC0 : P_DATA;
          end
          P_DATA: if (rx_byte) begin
            remain <= remain - 1'b1;
            if (wr_ok) begin
              w_buf <= w_shift[HW-1:8];
              if (w_last_byte) begin
                w_idx <= '0;
                w_row <= w_row + 1'b1;
                if (w_skip) e_ptab_bad <= 1'b1;
              end else begin
                w_idx <= w_idx + 1'b1;
              end
            end
            if (remain == 16'd1) ps <= P_CRC0;
          end
          P_CRC0: if (rx_byte) ps <= P_CRC1;
          P_CRC1: if (rx_byte) ps <= P_CHECK;
          P_CHECK: ps <= P_EXEC;                           // rc_zero now covers the CRC bytes
          P_EXEC: begin
            t_seq      <= seq;
            r_blk      <= 2'd0;
            r_mem_rows <= 1'b0;
            r_len      <= 16'd0;
            r_code     <= ACK;
            ps         <= P_REPLY;
            if (!rc_zero)        r_code <= NAK_CRC;
            else if (e_unknown)  r_code <= NAK_UNKNOWN;
            else if (e_too_long) r_code <= NAK_TOO_LONG;
            else if (e_bad_len)  r_code <= NAK_LENGTH;
            else if (e_dup) begin
              // WRITE: ACK without writing. RUN / INFER: re-arm while running, else reply now.
              if (cmd != C_WRITE) begin
                if (running) begin
                  pend       <= 1'b1;
                  pend_cmd   <= cmd;
                  pend_seq   <= seq;
                  pend_mem   <= last_mem;
                  pend_addr  <= last_addr;
                  pend_count <= last_count;
                  ps         <= P_SYNC;
                end else begin
                  pend       <= 1'b0;
                  r_blk      <= 2'd1;
                  r_mem_rows <= cmd == C_INFER;
                  r_mem      <= last_mem;
                  r_row0     <= last_addr << rows_shift(last_mem);
                  r_rows     <= last_count << rows_shift(last_mem);
                  r_len      <= (cmd == C_INFER) ? 16'(32'd8 + unit_bytes(last_mem, last_count)) : 16'd8;
                end
              end
            end
            else if (e_busy)                    r_code <= NAK_BUSY;
            else if (e_bad_arg || e_ptab_bad)   r_code <= NAK_ARG;
            else begin
              unique case (cmd)
                C_INFO:   begin r_blk <= 2'd2; r_len <= 16'd16; end
                C_STATUS: begin r_blk <= 2'd1; r_len <= 16'd8; end
                C_READ: begin
                  r_mem_rows <= 1'b1;
                  r_mem      <= a_mem;
                  r_row0     <= a_addr << rows_shift(a_mem);
                  r_rows     <= a_count << rows_shift(a_mem);
                  r_len      <= 16'(unit_bytes(a_mem, a_count));
                end
                C_WRITE: begin
                  last_valid <= 1'b1; last_cmd <= cmd; last_seq <= seq;
                end
                C_RUN, C_INFER: begin
                  last_valid    <= 1'b1; last_cmd <= cmd; last_seq <= seq;
                  last_mem      <= i_out_mem;
                  last_addr     <= i_out_addr;
                  last_count    <= i_out_count;
                  core_start    <= 1'b1;
                  core_start_pc <= (cmd == C_RUN) ? r_pc : i_pc;
                  pend          <= 1'b1;
                  pend_cmd      <= cmd;
                  pend_seq      <= seq;
                  pend_mem      <= i_out_mem;
                  pend_addr     <= i_out_addr;
                  pend_count    <= i_out_count;
                  ps            <= P_WAIT_RUN;
                end
                C_ABORT: begin
                  last_valid <= 1'b1; last_cmd <= cmd; last_seq <= seq;
                  pend       <= 1'b0;
                  core_abort <= running;
                  ps         <= P_WAIT_STOP;
                end
                C_RESET: begin
                  last_valid <= 1'b0;
                  pend       <= 1'b0;
                  core_rst   <= 1'b1;
                  ps         <= P_WAIT_RST;
                end
                default: ;
              endcase
            end
          end
          P_WAIT_RUN: if (!core_start) ps <= P_SYNC;        // the sequencer is running from here on
          P_WAIT_STOP: if (!running && !core_abort) begin   // the abort has taken effect
            r_blk <= 2'd1;
            r_len <= 16'd8;
            ps    <= P_REPLY;
          end
          P_WAIT_RST: if (!core_rst) begin
            r_blk <= 2'd1;
            r_len <= 16'd8;
            ps    <= P_REPLY;
          end
          P_REPLY: begin                                   // describe the reply to the transmitter
            t_code      <= r_code;
            t_len       <= r_len;
            t_mem       <= r_mem;
            t_go        <= 1'b1;                           // the transmitter loads the rest
            ps          <= P_TX;
          end
          P_TX: if (ts == T_IDLE && !t_go) ps <= P_SYNC;
          default: ps <= P_SYNC;
        endcase
      end
    end
  end

  assign pkt_ok  = t_go && t_code == ACK;
  assign pkt_nak = t_go && t_code != ACK;

  // ---------------------------------------------------------------------------------------------
  // Transmitter: header, block, memory rows, CRC.
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      ts          <= T_IDLE;
      t_rows_left <= '0;
    end else begin
      unique case (ts)
        T_IDLE: if (t_go) begin                            // r_* are stable while ps == P_TX
          t_blk       <= (r_blk == 2'd2) ? info_blk : status_blk;
          t_blk_left  <= (r_blk == 2'd2) ? 5'd16 : (r_blk == 2'd1) ? 5'd8 : 5'd0;
          t_row       <= r_row0;
          t_rows_left <= r_mem_rows ? r_rows : 16'd0;
          ts          <= T_SYNC;
        end
        T_SYNC: if (tx_fire) ts <= T_CODE;
        T_CODE: if (tx_fire) ts <= T_SEQ;
        T_SEQ:  if (tx_fire) ts <= T_LEN0;
        T_LEN0: if (tx_fire) ts <= T_LEN1;
        T_LEN1: if (tx_fire) ts <= (t_blk_left != '0) ? T_BLK : (t_rows_left != '0) ? T_RD : T_CRC0;
        T_BLK: if (tx_fire) begin
          t_blk      <= t_blk >> 8;
          t_blk_left <= t_blk_left - 1'b1;
          if (t_blk_left == 5'd1) ts <= (t_rows_left != '0) ? T_RD : T_CRC0;
        end
        T_RD:     ts <= T_RDWAIT;
        T_RDWAIT: if (host_rd_valid) begin
          t_buf       <= rd_bytes;
          t_row_left  <= (IW + 1)'(32'd1 << row_shift(t_mem));
          t_row       <= t_row + 1'b1;
          t_rows_left <= t_rows_left - 1'b1;
          ts          <= T_ROW;
        end
        T_ROW: if (tx_fire) begin
          t_buf      <= t_buf >> 8;
          t_row_left <= t_row_left - 1'b1;
          if (t_row_left == (IW + 1)'(1)) ts <= (t_rows_left != '0) ? T_RD : T_CRC0;
        end
        T_CRC0: if (tx_fire) ts <= T_CRC1;
        T_CRC1: if (tx_fire) ts <= T_IDLE;
        default: ts <= T_IDLE;
      endcase
    end
  end

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      if (host_wr_en || host_rd_en) assert (!running);       // memories only while stopped
      assert (!(host_wr_en && host_rd_en));
      if (core_start) assert (!running);
      if (ts != T_IDLE) assert (ps == P_TX);                   // one reply at a time
    end
  end
`endif

endmodule
