// Test-only wrapper: exposes every field of isa_pkg::instr_t and ptab_entry_t as ports, so cocotb
// can check that the SystemVerilog packed-struct layout matches the Python mirror bit for bit.
module isa_pkg_wrap
  import isa_pkg::*;
(
  input  logic [INSTR_W-1:0]      instr_raw,
  output logic [7:0]              opcode,
  output logic [7:0]              flags,
  output logic [15:0]             w_base,
  output logic [15:0]             a_base,
  output logic [15:0]             acc_base,
  output logic [15:0]             p_base,
  output logic [15:0]             kt,
  output logic [15:0]             nt,
  output logic [15:0]             m,
  output logic                    act_signed,
  output logic [1:0]              acc_mode,
  output logic                    relu,
  input  logic [PTAB_ENTRY_W-1:0] ptab_raw,
  output logic signed [31:0]      bias,
  output logic [15:0]             mult,
  output logic [5:0]              shift
);
  instr_t      instr;
  ptab_entry_t entry;

  assign instr      = instr_t'(instr_raw);
  assign opcode     = instr.opcode;
  assign flags      = instr.flags;
  assign w_base     = instr.w_base;
  assign a_base     = instr.a_base;
  assign acc_base   = instr.acc_base;
  assign p_base     = instr.p_base;
  assign kt         = instr.kt;
  assign nt         = instr.nt;
  assign m          = instr.m;
  assign act_signed = instr.flags[FLAG_ACT_SIGNED];
  assign acc_mode   = instr.flags[FLAG_ACC_MODE_LSB +: 2];
  assign relu       = instr.flags[FLAG_RELU];

  assign entry = ptab_entry_t'(ptab_raw);
  assign bias  = entry.bias;
  assign mult  = entry.mult;
  assign shift = entry.s;
endmodule
