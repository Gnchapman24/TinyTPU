// Test-only wrapper: uart_tx and uart_rx side by side. With `loopback` high the receiver hears the
// transmitter; otherwise it hears `rx_pin`, which cocotb drives bit by bit.
module uart_tb #(
  parameter int CLKS_PER_BIT = 8
) (
  input  logic       clk,
  input  logic       rst,
  input  logic       loopback,
  input  logic [7:0] tx_data,
  input  logic       tx_valid,
  output logic       tx_ready,
  output logic       tx_line,
  input  logic       rx_pin,
  output logic [7:0] rx_data,
  output logic       rx_valid,
  output logic       rx_frame_err
);
  logic rx_in;

  assign rx_in = loopback ? tx_line : rx_pin;

  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk, .rst, .data(tx_data), .valid(tx_valid), .ready(tx_ready), .tx(tx_line)
  );

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk, .rst, .rx(rx_in), .data(rx_data), .valid(rx_valid), .frame_err(rx_frame_err)
  );
endmodule
