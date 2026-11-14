/** UART receiver: 8N1, mid-bit sampling with a cycle counter (no oversampling).
 *
 *  The asynchronous `rx` pin passes through a two-flop synchronizer. A low level starts a frame;
 *  the start bit is re-checked half a bit later, so glitches shorter than about half a bit are
 *  ignored. The 8 data bits (LSB first) and the stop bit are then sampled at mid-bit.
 *
 *  A good frame gives a one-cycle `valid` pulse, with the byte on `data`. A low stop bit gives a
 *  one-cycle `frame_err` pulse instead, the byte is discarded, and the receiver waits for the line
 *  to return high before it looks for the next start bit. `valid` and `frame_err` are never high
 *  together. The receiver cannot apply back-pressure; the consumer must accept every pulse.
 *
 *  Latency: 3 + CLKS_PER_BIT/2 + 9*CLKS_PER_BIT clock edges, counting the first edge that samples the start bit low, until `valid` or `frame_err` is high (953 at CLKS_PER_BIT = 100).
 */
module uart_rx #(
  parameter int CLKS_PER_BIT = 100  // clock cycles per bit (100 = 1 Mbaud at 100 MHz); at least 4
) (
  input  logic       clk,       // clock
  input  logic       rst,       // synchronous active-high reset
  input  logic       rx,        // serial input; asynchronous to clk, idles high
  output logic [7:0] data,      // last good byte; updated with each valid pulse
  output logic       valid,     // one-cycle pulse: a good frame was received into data
  output logic       frame_err  // one-cycle pulse: the stop bit was low and the frame was discarded
);

  if (CLKS_PER_BIT < 4) begin : g_check_clks
    $error("uart_rx: CLKS_PER_BIT must be at least 4");
  end

  localparam int              CW   = $clog2(CLKS_PER_BIT);
  localparam logic [CW-1:0]   FULL = CW'(CLKS_PER_BIT - 1);
  localparam logic [CW-1:0]   HALF = CW'(CLKS_PER_BIT / 2 - 1);

  typedef enum logic [2:0] {
    S_IDLE,
    S_START,
    S_DATA,
    S_STOP,
    S_WAIT_HIGH
  } state_t;

  (* ASYNC_REG = "TRUE" *) logic [1:0] sync_q;
  logic          rx_s;
  state_t        state;
  logic [CW-1:0] cnt;
  logic [2:0]    bit_idx;
  logic [7:0]    shreg;

  assign rx_s = sync_q[1];

  always_ff @(posedge clk) begin
    if (rst) sync_q <= 2'b11;
    else     sync_q <= {sync_q[0], rx};
  end

  always_ff @(posedge clk) begin
    valid     <= 1'b0;
    frame_err <= 1'b0;
    if (rst) begin
      state   <= S_IDLE;
      cnt     <= '0;
      bit_idx <= '0;
      shreg   <= '0;
      data    <= '0;
    end else begin
      unique case (state)
        S_IDLE: begin
          if (!rx_s) begin
            cnt   <= HALF;
            state <= S_START;
          end
        end
        S_START: begin  // wait to the middle of the start bit and confirm it is still low
          if (cnt == '0) begin
            if (!rx_s) begin
              cnt     <= FULL;
              bit_idx <= '0;
              state   <= S_DATA;
            end else begin
              state <= S_IDLE;  // glitch, not a start bit
            end
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        S_DATA: begin
          if (cnt == '0) begin
            shreg <= {rx_s, shreg[7:1]};  // LSB first
            cnt   <= FULL;
            if (bit_idx == 3'd7) state <= S_STOP;
            else                 bit_idx <= bit_idx + 1'b1;
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        S_STOP: begin
          if (cnt == '0) begin
            if (rx_s) begin
              data  <= shreg;
              valid <= 1'b1;
              state <= S_IDLE;
            end else begin
              frame_err <= 1'b1;
              state     <= S_WAIT_HIGH;
            end
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        S_WAIT_HIGH: begin  // after a framing error, resynchronize on an idle (high) line
          if (rx_s) state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) past_valid <= 1'b1;

  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      assert (!(valid && frame_err));
      assert (state inside {S_IDLE, S_START, S_DATA, S_STOP, S_WAIT_HIGH});
      if ($past(valid))     assert (!valid);      // single-cycle pulses
      if ($past(frame_err)) assert (!frame_err);
    end
  end

  // Only meaningful when CLKS_PER_BIT is not a power of two; otherwise cnt cannot exceed FULL.
  if (CLKS_PER_BIT != (1 << CW)) begin : g_cnt_range
    always_ff @(posedge clk) if (past_valid && !rst) assert (cnt <= FULL);
  end
`ifdef FORMAL
  always_comb if (!past_valid) assume (rst);
`endif
`endif

endmodule
