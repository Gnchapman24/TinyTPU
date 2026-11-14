/** Reset generator: power-on reset plus a synchronized push-button reset.
 *
 *  `rst` is high after FPGA configuration and while the (asynchronous) button is pressed, and stays
 *  high for STAGES cycles after power-up or release. Power-on reset relies on Xilinx 7-series
 *  flip-flops starting at 0 after configuration (the default INIT), so no register here has an
 *  initial value or a reset: the `run` shift register starts all-zero (in reset) and fills with
 *  ones. The button passes through a two-flop synchronizer first.
 *
 *  Latency: STAGES + 2 cycles from the button's release to `rst` going low.
 */
module reset_gen #(
  parameter int STAGES = 4  // cycles rst stays high after power-up or release; at least 2
) (
  input  logic clk,  // clock
  input  logic btn,  // reset button, active high, asynchronous to clk
  output logic rst   // synchronous active-high reset for the rest of the design
);

  if (STAGES < 2) begin : g_check_stages
    $error("reset_gen: STAGES must be at least 2");
  end

  (* ASYNC_REG = "TRUE" *) logic [1:0] btn_q;
  logic [STAGES-1:0] run_q;  // all ones = running; all zeros at power-up

  always_ff @(posedge clk) btn_q <= {btn_q[0], btn};

  always_ff @(posedge clk) begin
    if (btn_q[1]) run_q <= '0;
    else          run_q <= {run_q[STAGES-2:0], 1'b1};
  end

  assign rst = ~run_q[STAGES-1];

endmodule
