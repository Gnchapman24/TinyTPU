// Formal-only stand-in for rtl/crc16_ccitt.sv, used by cmd_engine.sby.
//
// Same ports and the same timing (1 cycle from a valid byte to the updated value; `clear`
// restarts, and may carry the first byte), but the checksum is the XOR of the bytes, so `zero`
// (packet intact) is something the solver can choose freely through the last byte. The packet
// engine's control behaviour is thereby checked for every CRC outcome without the solver having
// to invert a real CRC. The real module is verified against tinytpu.crc and binascii.crc_hqx in
// tb/test_crc16.py.
module crc16_ccitt (
  input  logic        clk,
  input  logic        rst,
  input  logic        clear,
  input  logic [7:0]  data,
  input  logic        valid,
  output logic [15:0] crc,
  output logic        zero
);
  logic [7:0] x;
  always_ff @(posedge clk) begin
    if (rst)        x <= '0;
    else if (clear) x <= valid ? data : '0;
    else if (valid) x <= x ^ data;
  end
  assign crc  = {x, x};
  assign zero = x == '0;
endmodule
