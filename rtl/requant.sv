/** Requantizer: turns one INT32 accumulator into an 8-bit activation per cycle (REQUANT).
 *
 *  Computes `y = clamp((acc * M + R) >>> s, lo, hi)` with `R = (s == 0) ? 0 : 2**(s-1)` (round
 *  half up), exactly as `tinytpu.model.requant` (AGENTS.md §2). `M` (u16) and `s` (0..63) are the
 *  per-channel parameters of the value's output column; `relu` selects the clamp range: [0, 255]
 *  (UINT8) when set, [-128, 127] (INT8) when clear. `out_data` is the low byte of `y`, i.e. what
 *  REQUANT writes into the activation buffer.
 *
 *  The product `acc * M` is kept at full width (48 bits signed: |acc| <= 2**31, M < 2**16) and
 *  maps onto two cascaded DSP48E1s. The rounding add is folded into the shift:
 *  `(P + 2**(s-1)) >>> s == ((2P >>> s) + 1) >>> 1` for every s, including s = 0, so no
 *  variable-width R is ever built. One value enters per cycle; there are no stalls. `in_tag` is
 *  carried along unchanged so the controller can tell which column and row a result belongs to.
 *
 *  Raw (unrequantized) results do not pass through here: the host reads them straight from the
 *  accumulator buffer (acc_unit) while the sequencer is halted.
 *
 *  Pipeline: inputs (DSP A/B registers), product (MREG), sum (PREG), one cascade-alignment
 *  stage, shift, round + clamp. Only the valid bits are reset.
 *
 *  Latency: 6 cycles from in_valid to out_valid.
 */
module requant #(
  parameter int TAG_W = 8   // width of the side-band tag carried from in_tag to out_tag
) (
  input  logic                    clk,        // clock
  input  logic                    rst,        // synchronous active-high reset; empties the pipeline
  input  logic                    in_valid,   // a value enters this cycle
  input  logic signed [31:0]      in_acc,     // INT32 accumulator
  input  logic        [15:0]      in_mult,    // M: unsigned multiplier of this value's column
  input  logic        [5:0]       in_shift,   // s: right shift of this value's column, 0..63
  input  logic                    in_relu,    // 1: clamp to [0, 255]; 0: clamp to [-128, 127]
  input  logic        [TAG_W-1:0] in_tag,     // side-band tag, returned with the result
  output logic                    out_valid,  // out_data / out_tag hold a result
  output logic        [7:0]       out_data,   // requantized value: UINT8 (relu) or INT8 (two's complement)
  output logic        [TAG_W-1:0] out_tag     // in_tag of this result
);

  localparam int PW = 48;           // product width: 32-bit signed x 17-bit signed fits in 48
  localparam int LAT = 6;          // stages in total
  localparam int SH_STAGE = 4;     // stage that uses the shift
  localparam int RL_STAGE = 5;     // stage that uses the relu bit

  // Side band (valid, tag, shift, relu) travels in delay lines beside the data, each only as far
  // as it is used. Entry i holds the value of stage i + 1.
  logic             v   [LAT];
  logic [TAG_W-1:0] tg  [LAT];
  logic [5:0]       sh  [SH_STAGE];
  logic             rl  [RL_STAGE];

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < LAT; i++) v[i] <= 1'b0;
    end else begin
      v[0] <= in_valid;
      for (int i = 1; i < LAT; i++) v[i] <= v[i-1];
    end
    tg[0] <= in_tag;
    sh[0] <= in_shift;
    rl[0] <= in_relu;
    for (int i = 1; i < LAT; i++)      tg[i] <= tg[i-1];
    for (int i = 1; i < SH_STAGE; i++) sh[i] <= sh[i-1];
    for (int i = 1; i < RL_STAGE; i++) rl[i] <= rl[i-1];
  end

  // ---------------------------------------------------------------------------------------------
  // Stages 1-4: full-width product in the DSPs. M is zero-extended to 17 bits so both operands
  // are signed; the operands are widened before the multiply so the product is not truncated.
  // ---------------------------------------------------------------------------------------------
  logic signed [31:0]   a_q;
  logic signed [16:0]   b_q;
  logic signed [PW-1:0] m_q, p_q, p2_q;

  always_ff @(posedge clk) begin
    a_q  <= in_acc;
    b_q  <= $signed({1'b0, in_mult});
    m_q  <= PW'(a_q) * PW'(b_q);
    p_q  <= m_q;
    p2_q <= p_q;
  end

  // ---------------------------------------------------------------------------------------------
  // Stage 5: t = (2P) >>> s, so t = floor(P / 2**(s-1)) for s >= 1 and t = 2P for s = 0.
  // A 49-bit value shifted by 48 or more is all sign bits, which still rounds correctly.
  // ---------------------------------------------------------------------------------------------
  logic signed [PW:0] t_q;

  always_ff @(posedge clk) begin
    t_q <= $signed({p2_q, 1'b0}) >>> sh[SH_STAGE-1];
  end

  // ---------------------------------------------------------------------------------------------
  // Stage 6: y = (t + 1) >>> 1 (round half up), then clamp. t + 1 cannot overflow: |2P| < 2**48.
  // ---------------------------------------------------------------------------------------------
  logic signed [PW:0]   t1;
  logic signed [PW-1:0] y;
  logic        [7:0]    y8;
  logic        [7:0]    d_q;

  always_comb begin
    t1 = t_q + 1;
    y  = PW'(t1 >>> 1);
    if (rl[RL_STAGE-1]) begin
      if (y < 0)                   y8 = 8'd0;
      else if (y > PW'(255))       y8 = 8'd255;
      else                         y8 = y[7:0];
    end else begin
      if (y < -PW'(128))           y8 = 8'h80;
      else if (y > PW'(127))       y8 = 8'h7F;
      else                         y8 = y[7:0];
    end
  end

  always_ff @(posedge clk) begin
    d_q <= y8;
  end

  assign out_valid = v[LAT-1];
  assign out_data  = d_q;
  assign out_tag   = tg[LAT-1];

endmodule
