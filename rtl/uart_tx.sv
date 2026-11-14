/** UART transmitter: 8N1 with a valid/ready handshake.
 *
 *  A byte is accepted on a clock edge where `valid` and `ready` are both high. The frame then
 *  starts on the next cycle: one start bit (low), 8 data bits LSB first, one stop bit (high),
 *  each exactly CLKS_PER_BIT cycles. `ready` is high while idle and also in the last cycle of the
 *  stop bit, so a byte held on `valid` follows back-to-back at exactly 10*CLKS_PER_BIT cycles per
 *  byte. `tx` is a registered output and idles high.
 *
 *  Latency: 1 cycle from acceptance to the start bit on `tx`; 10*CLKS_PER_BIT cycles per frame.
 */
module uart_tx #(
  parameter int CLKS_PER_BIT = 100  // clock cycles per bit (100 = 1 Mbaud at 100 MHz); at least 4
) (
  input  logic       clk,    // clock
  input  logic       rst,    // synchronous active-high reset
  input  logic [7:0] data,   // byte to send; sampled when valid and ready are both high
  input  logic       valid,  // data holds a byte to send
  output logic       ready,  // the transmitter can accept a byte this cycle
  output logic       tx      // serial output; idles high
);

  if (CLKS_PER_BIT < 4) begin : g_check_clks
    $error("uart_tx: CLKS_PER_BIT must be at least 4");
  end

  localparam int            CW   = $clog2(CLKS_PER_BIT);
  localparam logic [CW-1:0] FULL = CW'(CLKS_PER_BIT - 1);

  typedef enum logic [1:0] {
    S_IDLE,
    S_START,
    S_DATA,
    S_STOP
  } state_t;

  state_t        state;
  logic [CW-1:0] cnt;
  logic [2:0]    bit_idx;
  logic [7:0]    shreg;

  assign ready = (state == S_IDLE) || (state == S_STOP && cnt == '0);

  always_ff @(posedge clk) begin
    if (rst) begin
      state   <= S_IDLE;
      cnt     <= '0;
      bit_idx <= '0;
      shreg   <= '0;
      tx      <= 1'b1;
    end else if (valid && ready) begin
      shreg <= data;
      tx    <= 1'b0;  // start bit
      cnt   <= FULL;
      state <= S_START;
    end else begin
      unique case (state)
        S_IDLE: tx <= 1'b1;
        S_START: begin
          if (cnt == '0) begin
            tx      <= shreg[0];
            shreg   <= shreg >> 1;
            bit_idx <= '0;
            cnt     <= FULL;
            state   <= S_DATA;
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        S_DATA: begin
          if (cnt == '0) begin
            cnt <= FULL;
            if (bit_idx == 3'd7) begin
              tx    <= 1'b1;  // stop bit
              state <= S_STOP;
            end else begin
              tx      <= shreg[0];
              shreg   <= shreg >> 1;
              bit_idx <= bit_idx + 1'b1;
            end
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        S_STOP: begin
          if (cnt == '0) state <= S_IDLE;
          else           cnt   <= cnt - 1'b1;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

`ifdef ASSERTIONS
  // Reference model of the line: after each accepted byte, tx must follow the exact 8N1 waveform,
  // and ready must be low until the last cycle of the stop bit.
  // f_bit = which bit of the frame (0 start, 1-8 data, 9 stop); f_sub = cycle within that bit.
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  logic          f_active;
  logic [3:0]    f_bit;
  logic [CW-1:0] f_sub;
  logic [7:0]    f_byte;
  logic          f_exp;
  logic          f_last;

  always_ff @(posedge clk) past_valid <= 1'b1;

  assign f_last = (f_bit == 4'd9) && (f_sub == FULL);

  always_ff @(posedge clk) begin
    if (rst) begin
      f_active <= 1'b0;
      f_bit    <= '0;
      f_sub    <= '0;
      f_byte   <= '0;
    end else if (valid && ready) begin
      f_active <= 1'b1;
      f_bit    <= '0;
      f_sub    <= '0;
      f_byte   <= data;
    end else if (f_active) begin
      if (f_sub == FULL) begin
        f_sub <= '0;
        if (f_bit == 4'd9) f_active <= 1'b0;
        else               f_bit    <= f_bit + 1'b1;
      end else begin
        f_sub <= f_sub + 1'b1;
      end
    end
  end

  always_comb begin
    if (f_bit == 4'd0)      f_exp = 1'b0;
    else if (f_bit <= 4'd8) f_exp = f_byte[3'(f_bit - 4'd1)];
    else                    f_exp = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (past_valid && !rst) begin
      assert (state inside {S_IDLE, S_START, S_DATA, S_STOP});
      if (f_active) begin
        assert (f_bit <= 4'd9);
        assert (tx == f_exp);
        assert (ready == f_last);
      end else begin
        assert (tx);
        assert (ready);
      end
    end
  end
  // Only meaningful when CLKS_PER_BIT is not a power of two; otherwise the counters cannot exceed FULL.
  if (CLKS_PER_BIT != (1 << CW)) begin : g_cnt_range
    always_ff @(posedge clk) begin
      if (past_valid && !rst) begin
        assert (cnt <= FULL);
        if (f_active) assert (f_sub <= FULL);
      end
    end
  end
`ifdef FORMAL
  always_comb if (!past_valid) assume (rst);
`endif
`endif

endmodule
