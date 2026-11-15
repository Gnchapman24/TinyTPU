/** Basys3 top level of TinyTPU: UART host link, packet engine and the TPU core (task A5.1).
 *
 *  RsRx -> uart_rx -> cmd_engine <-> tpu_core, cmd_engine -> uart_tx -> RsTx, as in
 *  docs/uart_protocol.md. `btnC` resets everything (memories keep their contents); the design
 *  also comes out of reset by itself after configuration. The protocol's RESET command resets
 *  the core (sequencer and execution units) the same way. Port names match syn/basys3.xdc.
 *
 *  LEDs: led[0] heartbeat; led[1] receive activity; led[2] transmit activity; led[3] a program is
 *  running; led[4] halted OK; led[5] halted with an error; led[6] a NAK was sent; led[7] a
 *  partial request timed out; led[8] UART framing error; led[9] a received byte was dropped
 *  (receive queue full). led[6]-led[9] stay on until btnC. led[15:12] show the ISA error code
 *  while halted with an error. Switches are unused.
 *
 *  Latency: see uart_rx, cmd_engine, tpu_core and uart_tx.
 */
module tinytpu_top
  import isa_pkg::*;
#(
  parameter int N                = 8,                   // array size
  parameter int IMEM_DEPTH       = DEFAULT_IMEM_DEPTH,  // instructions
  parameter int WBUF_DEPTH       = DEFAULT_WBUF_DEPTH,  // weight tiles
  parameter int ABUF_DEPTH       = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH        = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH       = DEFAULT_PTAB_DEPTH,  // parameter entries
  parameter int CLKS_PER_BIT     = 100,                 // UART bit length in clocks (1 Mbaud at 100 MHz)
  parameter int TIMEOUT_CYCLES   = 500_000,             // inter-byte timeout inside a request (5 ms)
  parameter int LED_STRETCH_BITS = 22,                  // activity LEDs stay on 2**this cycles (~42 ms)
  parameter int BLINK_BIT        = 26                   // heartbeat LED toggles every 2**this cycles (~0.67 s)
) (
  input  logic        clk,   // 100 MHz board clock
  input  logic        btnC,  // centre button: reset
  input  logic        RsRx,  // USB-UART receive line (from the PC)
  output logic        RsTx,  // USB-UART transmit line (to the PC)
  input  logic [15:0] sw,    // slide switches (unused)
  output logic [15:0] led    // status LEDs
);

  localparam int HW = (N * 32 > INSTR_W) ? N * 32 : INSTR_W;

  logic rst;
  reset_gen u_reset (.clk, .btn(btnC), .rst);

  logic unused_sw;
  assign unused_sw = ^sw;

  // UART.
  logic [7:0] rx_data, tx_data;
  logic       rx_valid, frame_err, tx_valid, tx_ready;

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk, .rst, .rx(RsRx), .data(rx_data), .valid(rx_valid), .frame_err
  );

  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk, .rst, .data(tx_data), .valid(tx_valid), .ready(tx_ready), .tx(RsTx)
  );

  // Packet engine and core.
  logic          core_rst, core_start, core_abort, host_rd_en, host_wr_en, host_rd_valid;
  logic [15:0]   core_start_pc, core_pc, core_halt_code, host_addr;
  seq_state_t    core_state;
  seq_err_t      core_err;
  mem_id_t       host_mem;
  logic [HW-1:0] host_wr_data, host_rd_data;
  logic          pkt_ok, pkt_nak, pkt_timeout, rx_overflow;

  cmd_engine #(
    .N(N), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH),
    .PTAB_DEPTH(PTAB_DEPTH), .CLKS_PER_BIT(CLKS_PER_BIT), .TIMEOUT_CYCLES(TIMEOUT_CYCLES), .HW(HW)
  ) u_engine (
    .clk, .rst, .rx_data, .rx_valid, .tx_data, .tx_valid, .tx_ready,
    .core_rst, .core_start, .core_start_pc, .core_abort,
    .core_state, .core_err, .core_pc, .core_halt_code,
    .host_mem, .host_rd_en, .host_wr_en, .host_addr, .host_wr_data, .host_rd_valid, .host_rd_data,
    .pkt_ok, .pkt_nak, .pkt_timeout, .rx_overflow
  );

  tpu_core #(
    .N(N), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH),
    .PTAB_DEPTH(PTAB_DEPTH), .HW(HW)
  ) u_core (
    .clk, .rst(rst || core_rst),
    .start(core_start), .start_pc(core_start_pc[$clog2(IMEM_DEPTH)-1:0]), .abort_req(core_abort),
    .state(core_state), .err(core_err), .pc(core_pc), .halt_code(core_halt_code),
    .host_mem, .host_rd_en, .host_wr_en, .host_addr, .host_wr_data, .host_rd_valid, .host_rd_data
  );

  // start_pc bits beyond IMEM are checked by the engine (NAK 0x05); pkt_ok is not shown.
  logic unused_status;
  assign unused_status = ^{core_start_pc[15:$clog2(IMEM_DEPTH)], pkt_ok};

  // Status LEDs: heartbeat, stretched activity, sequencer state, sticky error flags.
  logic [BLINK_BIT:0]        blink;
  logic [LED_STRETCH_BITS:0] rx_act, tx_act;
  logic                      seen_nak, seen_timeout, seen_frame, seen_overflow;

  always_ff @(posedge clk) begin
    if (rst) begin
      blink         <= '0;
      rx_act        <= '0;
      tx_act        <= '0;
      seen_nak      <= 1'b0;
      seen_timeout  <= 1'b0;
      seen_frame    <= 1'b0;
      seen_overflow <= 1'b0;
    end else begin
      blink <= blink + 1'b1;
      if (rx_valid)             rx_act <= '1;
      else if (rx_act != '0)    rx_act <= rx_act - 1'b1;
      if (tx_valid && tx_ready) tx_act <= '1;
      else if (tx_act != '0)    tx_act <= tx_act - 1'b1;
      if (pkt_nak)              seen_nak      <= 1'b1;
      if (pkt_timeout)          seen_timeout  <= 1'b1;
      if (frame_err)            seen_frame    <= 1'b1;
      if (rx_overflow)          seen_overflow <= 1'b1;
    end
  end

  assign led[0]     = blink[BLINK_BIT];
  assign led[1]     = rx_act != '0;
  assign led[2]     = tx_act != '0;
  assign led[3]     = core_state == ST_RUNNING;
  assign led[4]     = core_state == ST_HALTED_OK;
  assign led[5]     = core_state == ST_HALTED_ERROR;
  assign led[6]     = seen_nak;
  assign led[7]     = seen_timeout;
  assign led[8]     = seen_frame;
  assign led[9]     = seen_overflow;
  assign led[11:10] = 2'b00;
  assign led[15:12] = (core_state == ST_HALTED_ERROR) ? core_err[3:0] : 4'd0;

  logic unused_err;
  assign unused_err = ^core_err[7:4];

endmodule
