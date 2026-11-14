/** CRC-16/CCITT-FALSE, one byte per clock cycle.
 *
 *  Polynomial 0x1021, initial value 0xFFFF, no reflection, final XOR 0x0000 (check value: the
 *  ASCII string "123456789" gives 0x29B1). Each byte with `valid` high updates `crc`, MSB first;
 *  cycles without `valid` hold it. `clear` restarts at 0xFFFF; if `valid` is also high in the same
 *  cycle, that byte is the first byte of the new message.
 *
 *  Checking a received message: run the message bytes *and* its two CRC bytes (high byte first, as
 *  sent on the wire) through the module. The result is zero exactly when the CRC matches, which
 *  `zero` reports, so the receiver needs no separate compare. The Python reference is
 *  `tinytpu.crc`.
 *
 *  Latency: 1 cycle from a valid byte to the updated `crc`.
 */
module crc16_ccitt (
  input  logic        clk,    // clock
  input  logic        rst,    // synchronous active-high reset; crc becomes 0xFFFF
  input  logic        clear,  // restart at 0xFFFF (the initial value)
  input  logic [7:0]  data,   // message byte
  input  logic        valid,  // fold data into the CRC this cycle
  output logic [15:0] crc,    // CRC of all bytes since the last clear
  output logic        zero    // crc is 0x0000: a message followed by its correct CRC has been seen
);

  localparam logic [15:0] POLY = 16'h1021;
  localparam logic [15:0] INIT = 16'hFFFF;

  function automatic logic [15:0] crc_byte(input logic [15:0] c, input logic [7:0] d);
    logic [15:0] x;
    x = c ^ {d, 8'h00};
    for (int i = 0; i < 8; i++) begin
      x = x[15] ? ((x << 1) ^ POLY) : (x << 1);
    end
    return x;
  endfunction

  always_ff @(posedge clk) begin
    if (rst)        crc <= INIT;
    else if (clear) crc <= valid ? crc_byte(INIT, data) : INIT;
    else if (valid) crc <= crc_byte(crc, data);
  end

  assign zero = (crc == 16'h0000);

endmodule
