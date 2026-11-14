/** Bring-up echo engine: checks request packets and echoes their payload back (task A2.4).
 *
 *  Uses the real packet format of docs/uart_protocol.md with one bring-up-only command, ECHO
 *  (`CMD` = 0x7E). A request is `A5 CMD SEQ LEN(u16 LE) payload CRC(hi, lo)`. Once the whole
 *  request has arrived, the engine replies `5A CODE SEQ LEN payload CRC`:
 *  - `CODE` 0x00 (ACK) with the payload echoed, if the CRC is good, CMD is ECHO and LEN is at
 *    most MAX_PAYLOAD;
 *  - otherwise a NAK with LEN 0: 0x01 bad CRC, 0x04 payload too long, 0x02 unknown command
 *    (checked in that order).
 *  A gap longer than TIMEOUT_CYCLES between bytes inside a request discards it with no reply.
 *  Bytes that arrive while a reply is being sent are dropped and reported on `rx_overflow`.
 *
 *  With `raw` high the packet logic is bypassed: every received byte is echoed back as is, which
 *  is useful for a first test with a serial terminal. Change `raw` only while the link is idle.
 *  This module is for board bring-up only; the real command engine is task A4.1.
 *
 *  Latency: the reply starts 2 cycles after the last CRC byte of a request is received.
 */
module loopback_echo #(
  parameter int MAX_PAYLOAD    = 256,     // largest echoed payload in bytes; a power of two
  parameter int TIMEOUT_CYCLES = 500_000  // inter-byte timeout inside a request (5 ms at 100 MHz)
) (
  input  logic       clk,           // clock
  input  logic       rst,           // synchronous active-high reset
  input  logic       raw,           // 1 = echo every byte unchanged, 0 = packet mode
  input  logic [7:0] rx_data,       // received byte (from uart_rx)
  input  logic       rx_valid,      // rx_data holds a new byte this cycle
  output logic [7:0] tx_data,       // byte to transmit (to uart_tx)
  output logic       tx_valid,      // tx_data holds a byte to send
  input  logic       tx_ready,      // the transmitter accepts tx_data this cycle
  output logic       pkt_ok,        // pulse: an ACK reply was started
  output logic       pkt_bad_crc,   // pulse: a request failed its CRC check
  output logic       pkt_bad_cmd,   // pulse: a request had an unknown command
  output logic       pkt_too_long,  // pulse: a request's payload exceeded MAX_PAYLOAD
  output logic       pkt_timeout,   // pulse: a partial request was discarded after a timeout
  output logic       rx_overflow    // pulse: a received byte was dropped
);

  localparam logic [7:0] SYNC_REQ = 8'hA5;
  localparam logic [7:0] SYNC_RSP = 8'h5A;
  localparam logic [7:0] CMD_ECHO = 8'h7E;
  localparam logic [7:0] ACK          = 8'h00;
  localparam logic [7:0] NAK_BAD_CRC  = 8'h01;
  localparam logic [7:0] NAK_UNKNOWN  = 8'h02;
  localparam logic [7:0] NAK_TOO_LONG = 8'h04;

  localparam int FCW = $clog2(MAX_PAYLOAD) + 1;
  localparam int TW  = $clog2(TIMEOUT_CYCLES + 1);

  // ---------------------------------------------------------------------------------------------
  // Payload buffer and the two CRC units (request check, reply generation)
  // ---------------------------------------------------------------------------------------------
  logic [7:0]     fifo_in, fifo_out;
  logic           fifo_push, fifo_in_ready, fifo_out_valid, fifo_pop, fifo_flush;
  logic [FCW-1:0] unused_fifo_count;  // occupancy is tracked by the state machine instead

  sync_fifo #(.WIDTH(8), .DEPTH(MAX_PAYLOAD)) u_fifo (
    .clk, .rst(rst || fifo_flush),
    .in_data(fifo_in), .in_valid(fifo_push), .in_ready(fifo_in_ready),
    .out_data(fifo_out), .out_valid(fifo_out_valid), .out_ready(fifo_pop),
    .count(unused_fifo_count)
  );

  logic        rxc_clear, rxc_valid, rxc_zero;
  logic [15:0] unused_rxc_crc;  // requests are checked with the zero-residue flag only
  crc16_ccitt u_rx_crc (
    .clk, .rst, .clear(rxc_clear), .data(rx_data), .valid(rxc_valid), .crc(unused_rxc_crc),
    .zero(rxc_zero)
  );

  logic        txc_clear, txc_valid, txc_zero;
  logic [15:0] txc_crc;
  crc16_ccitt u_tx_crc (
    .clk, .rst, .clear(txc_clear), .data(tx_data), .valid(txc_valid), .crc(txc_crc), .zero(txc_zero)
  );

  // ---------------------------------------------------------------------------------------------
  // Receive side
  // ---------------------------------------------------------------------------------------------
  typedef enum logic [3:0] {
    R_SYNC, R_CMD, R_SEQ, R_LEN0, R_LEN1, R_PAY, R_CRC0, R_CRC1, R_CHECK, R_REPLY
  } rstate_t;

  typedef enum logic [3:0] {
    T_IDLE, T_SYNC, T_CODE, T_SEQ, T_LEN0, T_LEN1, T_PAY, T_CRC0, T_CRC1
  } tstate_t;

  rstate_t       rstate;
  tstate_t       tstate;
  logic [7:0]    cmd, seq, code;
  logic [15:0]   len, remain, reply_len;
  logic          too_long, start_reply, reply_done;
  logic [TW-1:0] idle_cnt;

  assign rxc_clear   = (rstate == R_SYNC) && rx_valid && (rx_data == SYNC_REQ) && !raw;
  assign rxc_valid   = rx_valid && !raw && (rstate inside {R_CMD, R_SEQ, R_LEN0, R_LEN1, R_PAY, R_CRC0, R_CRC1});
  assign fifo_in     = rx_data;
  assign fifo_push   = rx_valid && fifo_in_ready && (raw || (rstate == R_PAY && !too_long));
  assign reply_done  = (tstate == T_CRC1) && tx_ready;

  always_ff @(posedge clk) begin
    pkt_ok       <= 1'b0;
    pkt_bad_crc  <= 1'b0;
    pkt_bad_cmd  <= 1'b0;
    pkt_too_long <= 1'b0;
    pkt_timeout  <= 1'b0;
    rx_overflow  <= 1'b0;
    start_reply  <= 1'b0;
    fifo_flush   <= 1'b0;
    if (rst) begin
      rstate    <= R_SYNC;
      cmd       <= '0;
      seq       <= '0;
      code      <= '0;
      len       <= '0;
      remain    <= '0;
      reply_len <= '0;
      too_long  <= 1'b0;
      idle_cnt  <= '0;
    end else if (raw) begin
      rstate   <= R_SYNC;
      idle_cnt <= '0;
      if (rx_valid && !fifo_in_ready) rx_overflow <= 1'b1;
    end else begin
      // Inter-byte timeout while a request is partly received.
      if (rstate inside {R_CMD, R_SEQ, R_LEN0, R_LEN1, R_PAY, R_CRC0, R_CRC1}) begin
        if (rx_valid) begin
          idle_cnt <= '0;
        end else if (32'(idle_cnt) == TIMEOUT_CYCLES - 1) begin
          idle_cnt    <= '0;
          rstate      <= R_SYNC;
          fifo_flush  <= 1'b1;
          pkt_timeout <= 1'b1;
        end else begin
          idle_cnt <= idle_cnt + 1'b1;
        end
      end else begin
        idle_cnt <= '0;
      end

      if (rx_valid && rstate inside {R_CHECK, R_REPLY}) rx_overflow <= 1'b1;

      unique case (rstate)
        R_SYNC:  if (rx_valid && rx_data == SYNC_REQ) rstate <= R_CMD;
        R_CMD:   if (rx_valid) begin cmd <= rx_data; rstate <= R_SEQ; end
        R_SEQ:   if (rx_valid) begin seq <= rx_data; rstate <= R_LEN0; end
        R_LEN0:  if (rx_valid) begin len[7:0] <= rx_data; rstate <= R_LEN1; end
        R_LEN1: begin
          if (rx_valid) begin
            len[15:8] <= rx_data;
            remain    <= {rx_data, len[7:0]};
            too_long  <= ({rx_data, len[7:0]} > 16'(MAX_PAYLOAD));
            rstate    <= ({rx_data, len[7:0]} == 16'd0) ? R_CRC0 : R_PAY;
          end
        end
        R_PAY: begin
          if (rx_valid) begin
            remain <= remain - 1'b1;
            if (remain == 16'd1) rstate <= R_CRC0;
          end
        end
        R_CRC0:  if (rx_valid) rstate <= R_CRC1;
        R_CRC1:  if (rx_valid) rstate <= R_CHECK;
        R_CHECK: begin  // the CRC unit has now absorbed both CRC bytes
          if (!rxc_zero) begin
            code <= NAK_BAD_CRC;  pkt_bad_crc  <= 1'b1;
          end else if (too_long) begin
            code <= NAK_TOO_LONG; pkt_too_long <= 1'b1;
          end else if (cmd != CMD_ECHO) begin
            code <= NAK_UNKNOWN;  pkt_bad_cmd  <= 1'b1;
          end else begin
            code <= ACK;          pkt_ok       <= 1'b1;
          end
          reply_len   <= (rxc_zero && !too_long && cmd == CMD_ECHO) ? len : 16'd0;
          fifo_flush  <= !(rxc_zero && !too_long && cmd == CMD_ECHO);
          start_reply <= 1'b1;
          rstate      <= R_REPLY;
        end
        R_REPLY: if (reply_done) rstate <= R_SYNC;
        default: rstate <= R_SYNC;
      endcase
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Transmit side
  // ---------------------------------------------------------------------------------------------
  logic        tx_fire;
  logic [15:0] sent;

  assign tx_fire   = tx_valid && tx_ready;
  assign txc_clear = start_reply;
  assign txc_valid = tx_fire && (tstate inside {T_CODE, T_SEQ, T_LEN0, T_LEN1, T_PAY});
  assign fifo_pop  = raw ? tx_fire : (tx_fire && tstate == T_PAY);

  always_comb begin
    tx_valid = 1'b1;
    unique case (tstate)
      T_SYNC:  tx_data = SYNC_RSP;
      T_CODE:  tx_data = code;
      T_SEQ:   tx_data = seq;
      T_LEN0:  tx_data = reply_len[7:0];
      T_LEN1:  tx_data = reply_len[15:8];
      T_PAY:   tx_data = fifo_out;
      T_CRC0:  tx_data = txc_crc[15:8];
      T_CRC1:  tx_data = txc_crc[7:0];
      default: begin
        tx_data  = fifo_out;
        tx_valid = raw && fifo_out_valid;
      end
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst || raw) begin
      tstate <= T_IDLE;
      sent   <= '0;
    end else begin
      unique case (tstate)
        T_IDLE:  if (start_reply) tstate <= T_SYNC;
        T_SYNC:  if (tx_fire) tstate <= T_CODE;
        T_CODE:  if (tx_fire) tstate <= T_SEQ;
        T_SEQ:   if (tx_fire) tstate <= T_LEN0;
        T_LEN0:  if (tx_fire) tstate <= T_LEN1;
        T_LEN1: begin
          if (tx_fire) begin
            sent   <= '0;
            tstate <= (reply_len == 16'd0) ? T_CRC0 : T_PAY;
          end
        end
        T_PAY: begin
          if (tx_fire) begin
            sent <= sent + 1'b1;
            if (sent + 1'b1 == reply_len) tstate <= T_CRC0;
          end
        end
        T_CRC0:  if (tx_fire) tstate <= T_CRC1;
        T_CRC1:  if (tx_fire) tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  // The reply CRC is consumed only through txc_crc; its zero flag is not needed here.
  logic unused_txc_zero;
  assign unused_txc_zero = txc_zero;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) past_valid <= 1'b1;

  always_ff @(posedge clk) begin
    if (past_valid && !rst && !raw) begin
      // A payload byte that should be stored always fits (LEN <= MAX_PAYLOAD and the FIFO starts empty).
      if (rx_valid && rstate == R_PAY && !too_long) assert (fifo_in_ready);
      // While echoing, the payload byte to send is always available.
      if (tstate == T_PAY) assert (fifo_out_valid);
      // The reply side is busy exactly while the receive side waits for it.
      assert ((tstate != T_IDLE) == (rstate == R_REPLY) || start_reply);
    end
  end
`endif

endmodule
