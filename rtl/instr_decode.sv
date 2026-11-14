/** Instruction decoder: the ISA's decode-time legality checks (errors 1-4) on one 128-bit word.
 *
 *  Reports the first error in the order of docs/isa.md and `tinytpu.isa.decode_error` (the
 *  reference model):
 *  1. illegal opcode (anything but HALT, NOP, MATMUL, REQUANT);
 *  2. a non-zero reserved bit: a flag the opcode does not use, a slot the opcode does not use,
 *     or an address/count bit beyond the implemented width (`$clog2` of the buffer depth for
 *     addresses, isa_pkg::COUNT_W for counts);
 *  3. MATMUL with acc_mode = 11;
 *  4. a count (`kt`, `nt`, `m`) of 0.
 *  Address-range errors (5) need the instruction's products and are checked separately before
 *  execution; PC overrun (6) is the sequencer's.
 *
 *  Latency: combinational.
 */
module instr_decode
  import isa_pkg::*;
#(
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,  // weight tiles; sets the implemented w_base bits
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors; sets a_base / a_dst bits
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows; sets acc_base bits
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH   // parameter entries; sets p_base bits
) (
  input  logic [INSTR_W-1:0] instr,  // instruction word, bit 0 = least-significant bit of byte 0
  output seq_err_t           err     // ERR_NONE if legal, else the first decode error (codes 1-4)
);

  localparam logic [15:0] W_MASK = 16'((1 << $clog2(WBUF_DEPTH)) - 1);
  localparam logic [15:0] A_MASK = 16'((1 << $clog2(ABUF_DEPTH)) - 1);
  localparam logic [15:0] C_MASK = 16'((1 << $clog2(ACC_DEPTH)) - 1);
  localparam logic [15:0] P_MASK = 16'((1 << $clog2(PTAB_DEPTH)) - 1);
  localparam logic [15:0] N_MASK = 16'((1 << COUNT_W) - 1);

  instr_t f;
  assign f = instr_t'(instr);

  // Bits of a slot outside its allowed mask.
  function automatic logic over(input logic [15:0] v, input logic [15:0] mask);
    return (v & ~mask) != '0;
  endfunction

  logic legal_op, reserved, bad_mode, zero_count;

  always_comb begin
    legal_op   = 1'b1;
    reserved   = 1'b0;
    bad_mode   = 1'b0;
    zero_count = 1'b0;
    unique case (f.opcode)
      OP_HALT: begin
        reserved = over(16'(f.flags), 16'(FLAGS_MASK_HALT)) || f.a_base != '0 || f.acc_base != '0 ||
                   f.p_base != '0 || f.kt != '0 || f.nt != '0 || f.m != '0;
      end
      OP_NOP: begin
        reserved = over(16'(f.flags), 16'(FLAGS_MASK_NOP)) || f.w_base != '0 || f.a_base != '0 ||
                   f.acc_base != '0 || f.p_base != '0 || f.kt != '0 || f.nt != '0 || f.m != '0;
      end
      OP_MATMUL: begin
        reserved   = over(16'(f.flags), 16'(FLAGS_MASK_MATMUL)) || over(f.w_base, W_MASK) ||
                     over(f.a_base, A_MASK) || over(f.acc_base, C_MASK) || over(f.p_base, P_MASK) ||
                     over(f.kt, N_MASK) || over(f.nt, N_MASK) || over(f.m, N_MASK);
        bad_mode   = acc_mode_t'(f.flags[FLAG_ACC_MODE_LSB +: 2]) == ACC_ILLEGAL;
        zero_count = f.kt == '0 || f.nt == '0 || f.m == '0;
      end
      OP_REQUANT: begin
        reserved   = over(16'(f.flags), 16'(FLAGS_MASK_REQUANT)) || f.w_base != '0 ||
                     over(f.a_base, A_MASK) || over(f.acc_base, C_MASK) || over(f.p_base, P_MASK) ||
                     f.kt != '0 || over(f.nt, N_MASK) || over(f.m, N_MASK);
        zero_count = f.nt == '0 || f.m == '0;
      end
      default: legal_op = 1'b0;
    endcase

    if      (!legal_op) err = ERR_ILLEGAL_OPCODE;
    else if (reserved)  err = ERR_RESERVED_BIT;
    else if (bad_mode)  err = ERR_ACC_MODE;
    else if (zero_count) err = ERR_ZERO_COUNT;
    else                err = ERR_NONE;
  end

endmodule
