/** N x N weight-stationary systolic array: computes one N-element output vector, x . W, per cycle.
 *
 *  Weights: a tile is loaded one row per cycle. On a cycle with `w_load` high, every PE in row
 *  `w_row` captures `w_in` (column j's weight on `w_in[j]`); rows may be loaded in any order.
 *  Weights are sign-extended to 9 bits at the edge.
 *
 *  Activations: on a cycle with `a_valid` high, `a_in` is one activation vector (element i
 *  multiplies weight row i). It is extended to 9 bits at the edge (sign-extended if `act_signed`,
 *  else zero-extended), then row i is delayed i cycles (input skew). Partial sums flow down the
 *  columns through the PEs, so column j finishes j cycles after column 0; column j is then delayed
 *  N-1-j cycles (output de-skew), so all N results of a vector appear together on `out_data`, with
 *  `out_valid`, a fixed latency after the vector entered. Vectors may enter every cycle, with or
 *  without gaps. Each result is the 32-bit truncation of the column's 48-bit partial sum (exact:
 *  an N-term INT8 sum fits in 32 bits).
 *
 *  Weight timing rules (the controller must follow them; the array does not check):
 *  - Loading may overlap the vector stream: a vector may enter in the same cycle as row 0 of
 *    its tile is loaded, provided rows are loaded in order 0, 1, ..., N-1 on consecutive cycles.
 *  - A new tile's row 0 may be loaded no earlier than N cycles after the previous tile's last
 *    vector entered (that vector must have passed every PE first).
 *
 *  Latency: 2*N + 1 cycles from a vector on a_in (a_valid high) to its result on out_data (out_valid high).
 */
module systolic_array #(
  parameter int N = 8  // array size (rows = columns); a power of two, at least 2
) (
  input  logic                     clk,         // clock
  input  logic                     rst,         // synchronous active-high reset; clears out_valid tracking
  input  logic                     act_signed,  // activations are INT8 (1) or UINT8 (0)
  input  logic [N-1:0][7:0]        a_in,        // activation vector: a_in[i] multiplies weight row i
  input  logic                     a_valid,     // a_in holds a vector to process this cycle
  input  logic [N-1:0][7:0]        w_in,        // one weight-tile row: w_in[j] is column j's weight
  input  logic                     w_load,      // load w_in into the PEs of row w_row this cycle
  input  logic [$clog2(N)-1:0]     w_row,       // which row to load
  output logic [N-1:0][31:0]       out_data,    // result vector: out_data[j] = sum_i a[i]*W[i][j]
  output logic                     out_valid    // out_data holds the result of one input vector
);

  if (N < 2 || (N & (N - 1)) != 0) begin : g_check_n
    $error("systolic_array: N must be a power of two, at least 2");
  end

  localparam int PW  = 48;
  localparam int LAT = 2 * N + 1;

  // Edge extension to 9 bits (AGENTS.md §3).
  logic signed [8:0] a_ext [N];
  logic signed [8:0] w_ext [N];
  for (genvar i = 0; i < N; i++) begin : g_ext
    assign a_ext[i] = {act_signed & a_in[i][7], a_in[i]};
    assign w_ext[i] = {w_in[i][7], w_in[i]};
  end

  // Grid wiring: a_h[i][j] enters PE(i,j) from the left; psum_v[i][j] enters PE(i,j) from above.
  logic signed [8:0]    a_h    [N][N+1];
  logic signed [PW-1:0] psum_v [N+1][N];

  // Input skew: row i is delayed i cycles.
  for (genvar i = 0; i < N; i++) begin : g_skew
    if (i == 0) begin : g_none
      assign a_h[0][0] = a_ext[0];
    end else begin : g_delay
      logic signed [8:0] sr [i];
      always_ff @(posedge clk) begin
        sr[0] <= a_ext[i];
        for (int k = 1; k < i; k++) sr[k] <= sr[k-1];
      end
      assign a_h[i][0] = sr[i-1];
    end
  end

  for (genvar j = 0; j < N; j++) begin : g_top
    assign psum_v[0][j] = '0;
  end

  for (genvar i = 0; i < N; i++) begin : g_row
    for (genvar j = 0; j < N; j++) begin : g_col
      pe #(.AW(9), .WW(9), .PW(PW)) u_pe (
        .clk,
        .a_in    (a_h[i][j]),
        .a_out   (a_h[i][j+1]),
        .w_in    (w_ext[j]),
        .w_load  (w_load && (w_row == i[$clog2(N)-1:0])),
        .psum_in (psum_v[i][j]),
        .psum_out(psum_v[i+1][j])
      );
    end
  end

  // The activations leaving the right-hand column are not needed.
  logic unused_a_right;
  always_comb begin
    unused_a_right = 1'b0;
    for (int i = 0; i < N; i++) unused_a_right ^= ^a_h[i][N];
  end

  // Output de-skew: each column is truncated to 32 bits, then column j is delayed N-1-j cycles.
  logic [PW-1:32] unused_psum_high [N];
  for (genvar j = 0; j < N; j++) begin : g_deskew
    localparam int D = N - 1 - j;
    assign unused_psum_high[j] = psum_v[N][j][PW-1:32];
    if (D == 0) begin : g_none
      assign out_data[j] = psum_v[N][j][31:0];
    end else begin : g_delay
      logic [31:0] sr [D];
      always_ff @(posedge clk) begin
        sr[0] <= psum_v[N][j][31:0];
        for (int k = 1; k < D; k++) sr[k] <= sr[k-1];
      end
      assign out_data[j] = sr[D-1];
    end
  end

  // Valid tracking: out_valid is a_valid delayed by the array latency.
  logic [LAT-1:0] vpipe;
  always_ff @(posedge clk) begin
    if (rst) vpipe <= '0;
    else     vpipe <= {vpipe[LAT-2:0], a_valid};
  end
  assign out_valid = vpipe[LAT-1];

endmodule
