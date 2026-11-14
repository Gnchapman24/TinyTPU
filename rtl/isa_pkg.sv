/** TinyTPU ISA definitions shared by the RTL and the testbenches.
 *
 *  Mirrors docs/isa.md (the source of truth) and the memory defaults in docs/memory_map.md.
 *  The Python mirror is host/tinytpu/isa.py; tb/test_isa.py checks the two stay identical.
 *  Change nothing here without changing docs/isa.md first (AGENTS.md §8).
 */
package isa_pkg;
  // Package constants exist for importers; any one module uses only some of them.
  /* verilator lint_off UNUSEDPARAM */

  // ---------------------------------------------------------------------------------------------
  // Instruction format: 128 bits = eight 16-bit slots, least-significant byte first on the wire.
  // ---------------------------------------------------------------------------------------------
  localparam int INSTR_W = 128;
  localparam int SLOT_W  = 16;

  typedef enum logic [7:0] {
    OP_HALT    = 8'h00,
    OP_NOP     = 8'h01,
    OP_MATMUL  = 8'h10,
    OP_REQUANT = 8'h20
  } opcode_t;

  // MATMUL flags[2:1]
  typedef enum logic [1:0] {
    ACC_OVERWRITE  = 2'b00,
    ACC_ACCUMULATE = 2'b01,
    ACC_BIAS       = 2'b10,
    ACC_ILLEGAL    = 2'b11
  } acc_mode_t;

  // Bit positions within the 8-bit flags field (instruction bits [15:8]).
  localparam int FLAG_ACT_SIGNED    = 0;  // MATMUL: activations are INT8 (else UINT8)
  localparam int FLAG_ACC_MODE_LSB  = 1;  // MATMUL: acc_mode in flags[2:1]
  localparam int FLAG_RELU          = 0;  // REQUANT: clamp to [0, 255] (else [-128, 127])

  // Flag bits each opcode may set; every other flag bit is reserved and must be zero.
  localparam logic [7:0] FLAGS_MASK_HALT    = 8'h00;
  localparam logic [7:0] FLAGS_MASK_NOP     = 8'h00;
  localparam logic [7:0] FLAGS_MASK_MATMUL  = 8'h07;
  localparam logic [7:0] FLAGS_MASK_REQUANT = 8'h01;

  // Generic view of one instruction. Packed structs list the most-significant field first, so
  // `opcode` is bits [7:0] and `m` is bits [127:112]. Per-opcode field meanings:
  //   slot 1 `w_base`   : MATMUL w_base, HALT halt code
  //   slot 2 `a_base`   : MATMUL a_base (source), REQUANT a_dst (destination)
  typedef struct packed {
    logic [15:0] m;         // slot 7, [127:112]: activation rows
    logic [15:0] nt;        // slot 6, [111:96]:  output-column tiles
    logic [15:0] kt;        // slot 5, [95:80]:   inner-dimension tiles (MATMUL only)
    logic [15:0] p_base;    // slot 4, [79:64]:   parameter-table base
    logic [15:0] acc_base;  // slot 3, [63:48]:   accumulator base
    logic [15:0] a_base;    // slot 2, [47:32]:   activation base (MATMUL) / a_dst (REQUANT)
    logic [15:0] w_base;    // slot 1, [31:16]:   weight base (MATMUL) / halt code (HALT)
    logic [7:0]  flags;     // slot 0, [15:8]
    opcode_t     opcode;    // slot 0, [7:0]
  } instr_t;

  // ---------------------------------------------------------------------------------------------
  // Sequencer status (reported in the protocol status block)
  // ---------------------------------------------------------------------------------------------
  typedef enum logic [7:0] {
    ST_IDLE         = 8'd0,
    ST_RUNNING      = 8'd1,
    ST_HALTED_OK    = 8'd2,
    ST_HALTED_ERROR = 8'd3
  } seq_state_t;

  typedef enum logic [7:0] {
    ERR_NONE           = 8'd0,
    ERR_ILLEGAL_OPCODE = 8'd1,
    ERR_RESERVED_BIT   = 8'd2,  // includes field bits beyond the implemented width
    ERR_ACC_MODE       = 8'd3,  // MATMUL acc_mode = 11
    ERR_ZERO_COUNT     = 8'd4,  // kt, nt or m = 0
    ERR_ADDR_RANGE     = 8'd5,  // a generated address at or beyond its memory's depth
    ERR_PC_OVERRUN     = 8'd6   // PC ran past the end of IMEM without HALT
  } seq_err_t;

  // ---------------------------------------------------------------------------------------------
  // Memories (IDs as used by the host protocol) and default depths (docs/memory_map.md)
  // ---------------------------------------------------------------------------------------------
  typedef enum logic [7:0] {
    MEM_IMEM = 8'd0,
    MEM_WBUF = 8'd1,
    MEM_ABUF = 8'd2,
    MEM_ACC  = 8'd3,
    MEM_PTAB = 8'd4
  } mem_id_t;

  localparam int DEFAULT_IMEM_DEPTH = 128;   // instructions (LUT RAM)
  localparam int DEFAULT_WBUF_DEPTH = 2048;  // N x N weight tiles
  localparam int DEFAULT_ABUF_DEPTH = 1024;  // N-element activation vectors
  localparam int DEFAULT_ACC_DEPTH  = 512;   // rows of N x INT32
  localparam int DEFAULT_PTAB_DEPTH = 512;   // output channels

  localparam int COUNT_W = 12;  // implemented bits of kt, nt and m

  // One parameter-table entry: bits [31:0] bias, [47:32] M, [53:48] s.
  typedef struct packed {
    logic [5:0]         s;     // [53:48] requantization shift
    logic [15:0]        mult;  // [47:32] requantization multiplier M (unsigned)
    logic signed [31:0] bias;  // [31:0]  bias added in MATMUL bias mode
  } ptab_entry_t;

  localparam int PTAB_ENTRY_W = 54;

  /* verilator lint_on UNUSEDPARAM */
endpackage
