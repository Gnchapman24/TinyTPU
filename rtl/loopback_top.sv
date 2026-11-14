/** Basys3 top level for UART bring-up: echo engine plus status LEDs (task A2.4).
 *
 *  Connects the USB-UART (RsRx/RsTx) through uart_rx and uart_tx to loopback_echo. Port names
 *  match syn/basys3.xdc. `btnC` resets the design; it also comes out of reset by itself after
 *  configuration.
 *
 *  Switches: sw[0] selects raw byte echo (on) or packet echo (off). sw[1] selects what led[15:8]
 *  shows: the good-packet count (off) or the error count (on). Other switches are unused.
 *
 *  LEDs: led[0] heartbeat; led[1] receive activity; led[2] transmit activity; led[3] raw mode;
 *  led[4] CRC error seen; led[5] framing error seen; led[6] timeout seen; led[7] other error seen
 *  (dropped byte, unknown command, payload too long). led[4]-led[7] stay on until reset.
 *  led[15:8] is an 8-bit counter (see sw[1]).
 *
 *  Latency: see uart_rx, loopback_echo and uart_tx.
 */
module loopback_top #(
  parameter int CLKS_PER_BIT     = 100,      // UART bit length in clocks (100 = 1 Mbaud at 100 MHz)
  parameter int TIMEOUT_CYCLES   = 500_000,  // inter-byte timeout inside a request (5 ms)
  parameter int LED_STRETCH_BITS = 22,       // activity LEDs stay on 2**this cycles (~42 ms)
  parameter int BLINK_BIT        = 26        // heartbeat LED toggles every 2**this cycles (~0.67 s)
) (
  input  logic        clk,   // 100 MHz board clock
  input  logic        btnC,  // centre button: reset
  input  logic        RsRx,  // USB-UART receive line (from the PC)
  output logic        RsTx,  // USB-UART transmit line (to the PC)
  input  logic [15:0] sw,    // slide switches (sw[0] raw echo, sw[1] LED counter select)
  output logic [15:0] led    // status LEDs
);

  logic rst;
  reset_gen u_reset (.clk, .btn(btnC), .rst);

  // Switches are asynchronous: synchronize the two that are used.
  (* ASYNC_REG = "TRUE" *) logic [1:0] sw_q0, sw_q1;
  logic raw, show_errors;
  always_ff @(posedge clk) begin
    sw_q0 <= sw[1:0];
    sw_q1 <= sw_q0;
  end
  assign raw         = sw_q1[0];
  assign show_errors = sw_q1[1];

  logic unused_sw;
  assign unused_sw = ^sw[15:2];

  // UART and echo engine.
  logic [7:0] rx_data, tx_data;
  logic       rx_valid, frame_err, tx_valid, tx_ready;
  logic       pkt_ok, pkt_bad_crc, pkt_bad_cmd, pkt_too_long, pkt_timeout, rx_overflow;

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk, .rst, .rx(RsRx), .data(rx_data), .valid(rx_valid), .frame_err
  );

  loopback_echo #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) u_echo (
    .clk, .rst, .raw,
    .rx_data, .rx_valid,
    .tx_data, .tx_valid, .tx_ready,
    .pkt_ok, .pkt_bad_crc, .pkt_bad_cmd, .pkt_too_long, .pkt_timeout, .rx_overflow
  );

  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk, .rst, .data(tx_data), .valid(tx_valid), .ready(tx_ready), .tx(RsTx)
  );

  // Status: heartbeat, stretched activity, sticky error flags, counters.
  logic [BLINK_BIT:0]        blink;
  logic [LED_STRETCH_BITS:0] rx_act, tx_act;
  logic                      seen_crc, seen_frame, seen_timeout, seen_other;
  logic [7:0]                good_cnt, err_cnt;
  logic                      any_err;

  assign any_err = pkt_bad_crc || pkt_bad_cmd || pkt_too_long || pkt_timeout || rx_overflow || frame_err;

  always_ff @(posedge clk) begin
    if (rst) begin
      blink        <= '0;
      rx_act       <= '0;
      tx_act       <= '0;
      seen_crc     <= 1'b0;
      seen_frame   <= 1'b0;
      seen_timeout <= 1'b0;
      seen_other   <= 1'b0;
      good_cnt     <= '0;
      err_cnt      <= '0;
    end else begin
      blink <= blink + 1'b1;
      if (rx_valid)                  rx_act <= '1;
      else if (rx_act != '0)         rx_act <= rx_act - 1'b1;
      if (tx_valid && tx_ready)      tx_act <= '1;
      else if (tx_act != '0)         tx_act <= tx_act - 1'b1;
      if (pkt_bad_crc)               seen_crc     <= 1'b1;
      if (frame_err)                 seen_frame   <= 1'b1;
      if (pkt_timeout)               seen_timeout <= 1'b1;
      if (pkt_bad_cmd || pkt_too_long || rx_overflow) seen_other <= 1'b1;
      if (pkt_ok)                    good_cnt <= good_cnt + 1'b1;
      if (any_err)                   err_cnt  <= err_cnt + 1'b1;
    end
  end

  assign led[0]    = blink[BLINK_BIT];
  assign led[1]    = rx_act != '0;
  assign led[2]    = tx_act != '0;
  assign led[3]    = raw;
  assign led[4]    = seen_crc;
  assign led[5]    = seen_frame;
  assign led[6]    = seen_timeout;
  assign led[7]    = seen_other;
  assign led[15:8] = show_errors ? err_cnt : good_cnt;

endmodule
