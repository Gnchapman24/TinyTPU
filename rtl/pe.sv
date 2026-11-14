/** Processing element: one multiply-accumulate of the weight-stationary systolic array, sized
 *  to map onto exactly one DSP48E1.
 *
 *  The weight is stationary: it is captured from the column's weight bus `w_in` on a cycle with
 *  `w_load` high (the array raises it for one row at a time) and then held. Each cycle the
 *  activation `a_in` arrives from the left and is passed on to the right one cycle later on
 *  `a_out`; the product of that activation and the weight is added to the partial sum arriving
 *  from above on `psum_in` and passed down on `psum_out`.
 *
 *  Operands are 9-bit signed: the array edge sign- or zero-extends 8-bit values once
 *  (AGENTS.md §3). Pipeline, chosen so Vivado can place it inside the DSP: `a_q` (activation
 *  register, also drives `a_out`), `w_q` (weight, BREG with clock enable), `m_q` (product, MREG)
 *  and `p_q` (sum, PREG). The 48-bit partial sum matches the DSP's P / PCIN / PCOUT width so that
 *  a column of PEs can use the dedicated cascade. No register is reset: the array controller
 *  only uses sums whose inputs were valid.
 *
 *  `use_dsp = "yes"` on the product and sum registers is required: without it Vivado builds a
 *  9x9 multiplier from about 136 LUTs (syn/ooc.tcl checks this; see AGENTS.md §3).
 *
 *  Latency: 1 cycle a_in to a_out; 3 cycles a_in to its product in psum_out; 1 cycle psum_in to psum_out.
 */
module pe #(
  parameter int AW = 9,   // activation width (signed); 9 = 8-bit value plus extension bit
  parameter int WW = 9,   // weight width (signed)
  parameter int PW = 48   // partial-sum width; 48 = the DSP48E1 P/PCIN/PCOUT width
) (
  input  logic                 clk,       // clock
  input  logic signed [AW-1:0] a_in,      // activation from the left neighbour (or array edge)
  output logic signed [AW-1:0] a_out,     // activation to the right neighbour, delayed one cycle
  input  logic signed [WW-1:0] w_in,      // weight bus of this column
  input  logic                 w_load,    // capture w_in as the stationary weight this cycle
  input  logic signed [PW-1:0] psum_in,   // partial sum from the PE above (0 at the top row)
  output logic signed [PW-1:0] psum_out   // psum_in + a*w, to the PE below
);

  logic signed [AW-1:0]    a_q;
  logic signed [WW-1:0]    w_q;
  (* use_dsp = "yes" *) logic signed [AW+WW-1:0] m_q;
  (* use_dsp = "yes" *) logic signed [PW-1:0]    p_q;

  always_ff @(posedge clk) begin
    a_q <= a_in;
    if (w_load) w_q <= w_in;
    m_q <= a_q * w_q;
    p_q <= psum_in + PW'(m_q);
  end

  assign a_out    = a_q;
  assign psum_out = p_q;

endmodule
