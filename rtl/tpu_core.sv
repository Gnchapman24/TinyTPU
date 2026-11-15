/** TinyTPU core without the host link: sequencer, MATMUL and REQUANT units, the five memories,
 *  the systolic array, the accumulators and the requantizer.
 *
 *  Runs stored programs (docs/isa.md). The host side (the packet engine, A4.1) starts and
 *  aborts programs, reads the status, and reads and writes every memory through one port while
 *  the sequencer is stopped (state != running), which is the protocol's busy rule; an assertion
 *  checks it.
 *
 *  Host port: `host_mem` selects the memory (isa_pkg::mem_id_t), `host_addr` is the physical
 *  row (WBUF: tile*N + row; others: their ISA unit), data is right-aligned in `host_wr_data` /
 *  `host_rd_data`. A read returns `host_rd_valid` a fixed number of cycles later (IMEM 1, WBUF
 *  1 + W_PIPE, ABUF and PTAB 2, ACC 1); reads of different memories must not overlap in flight.
 *
 *  Sharing: WBUF and ABUF are read by MATMUL, ABUF is written by REQUANT (one byte per cycle),
 *  PTAB is read by both units (the read tag carries the owner, so returned entries go only to
 *  the unit that asked), ACC is updated by MATMUL and read by REQUANT. Only one instruction
 *  runs at a time, so no port is ever contended.
 *
 *  Latency: per instruction as in matmul_ctrl and requant_ctrl, plus the sequencer's fetch and checks.
 */
module tpu_core
  import isa_pkg::*;
#(
  parameter int N          = 8,                   // array size; a power of two, at least 2
  parameter int IMEM_DEPTH = DEFAULT_IMEM_DEPTH,  // instructions (LUT RAM)
  parameter int WBUF_DEPTH = DEFAULT_WBUF_DEPTH,  // weight tiles (N rows each)
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH,  // parameter entries
  parameter int W_PIPE     = 2,                   // WBUF read pipeline stages (buffer_ram RD_PIPE)
  parameter int HW         = (N * 32 > INSTR_W) ? N * 32 : INSTR_W  // host data width: the widest unit
) (
  input  logic                          clk,            // clock
  input  logic                          rst,            // synchronous active-high reset: idle; memories kept
  input  logic                          start,          // run from start_pc (only while stopped)
  input  logic [$clog2(IMEM_DEPTH)-1:0] start_pc,       // first instruction
  input  logic                          abort_req,      // stop a running program
  output seq_state_t                    state,          // idle, running, halted OK, halted error
  output seq_err_t                      err,            // error code when halted with an error
  output logic [15:0]                   pc,             // PC of HALT / fault / current instruction
  output logic [15:0]                   halt_code,      // code of the HALT
  input  mem_id_t                       host_mem,       // memory addressed by the host port
  input  logic                          host_rd_en,     // host read (only while stopped)
  input  logic                          host_wr_en,     // host write (only while stopped)
  input  logic [15:0]                   host_addr,      // physical row in the selected memory
  input  logic [HW-1:0]                 host_wr_data,   // row to write, right-aligned
  output logic                          host_rd_valid,  // host_rd_data holds the row read
  output logic [HW-1:0]                 host_rd_data    // row read, right-aligned
);

  localparam int LN  = $clog2(N);
  localparam int IAW = $clog2(IMEM_DEPTH);
  localparam int WRW = $clog2(WBUF_DEPTH) + LN;
  localparam int AAW = $clog2(ABUF_DEPTH);
  localparam int CAW = $clog2(ACC_DEPTH);
  localparam int PAW = $clog2(PTAB_DEPTH);
  localparam int VW  = N * 8;                     // WBUF row / ABUF vector width
  localparam int RW  = N * 32;                    // ACC row width
  localparam int PTW = 1 + AAW + LN;              // PTAB tag: {owner (1 = REQUANT), payload}
  localparam int W_LAT = 1 + W_PIPE, A_LAT = 2, P_LAT = 2;

  logic host_sel;
  assign host_sel = state != ST_RUNNING;

  // Host port decode.
  logic hr_imem, hr_wbuf, hr_abuf, hr_acc, hr_ptab;
  logic hw_imem, hw_wbuf, hw_abuf, hw_acc, hw_ptab;
  assign hr_imem = host_rd_en && host_mem == MEM_IMEM;
  assign hr_wbuf = host_rd_en && host_mem == MEM_WBUF;
  assign hr_abuf = host_rd_en && host_mem == MEM_ABUF;
  assign hr_acc  = host_rd_en && host_mem == MEM_ACC;
  assign hr_ptab = host_rd_en && host_mem == MEM_PTAB;
  assign hw_imem = host_wr_en && host_mem == MEM_IMEM;
  assign hw_wbuf = host_wr_en && host_mem == MEM_WBUF;
  assign hw_abuf = host_wr_en && host_mem == MEM_ABUF;
  assign hw_acc  = host_wr_en && host_mem == MEM_ACC;
  assign hw_ptab = host_wr_en && host_mem == MEM_PTAB;

  // ---------------------------------------------------------------------------------------------
  // Sequencer and IMEM.
  // ---------------------------------------------------------------------------------------------
  logic               imem_rd_en, imem_rd_valid, exec_start, exec_done, exec_abort, exec_idle;
  logic [IAW-1:0]     imem_rd_addr;
  logic [INSTR_W-1:0] imem_rd_data, exec_instr;
  logic [0:0]         imem_tag;

  sequencer #(
    .N(N), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH),
    .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH)
  ) u_seq (
    .clk, .rst, .start(start && host_sel), .start_pc, .abort_req, .state, .err, .pc, .halt_code,
    .imem_rd_en, .imem_rd_addr, .imem_rd_valid, .imem_rd_data,
    .exec_start, .exec_instr, .exec_done, .exec_abort, .exec_idle
  );

  buffer_ram #(.WIDTH(INSTR_W), .DEPTH(IMEM_DEPTH), .LANES(1), .RD_PIPE(0), .TAG_W(1), .LUTRAM(1'b1)) u_imem (
    .clk, .rst, .host_sel,
    .dp_rd_en(imem_rd_en), .dp_rd_addr(imem_rd_addr), .dp_rd_tag(1'b1),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(hr_imem), .host_rd_addr(host_addr[IAW-1:0]),
    .host_wr_en(hw_imem), .host_wr_addr(host_addr[IAW-1:0]), .host_wr_data(host_wr_data[INSTR_W-1:0]),
    .rd_valid(imem_rd_valid), .rd_data(imem_rd_data), .rd_tag(imem_tag)
  );

  // Dispatch.
  opcode_t ei_op;
  logic    mm_start, rq_start, mm_done, rq_done, mm_idle, rq_idle;
  assign ei_op     = opcode_t'(exec_instr[7:0]);
  assign mm_start  = exec_start && ei_op == OP_MATMUL;
  assign rq_start  = exec_start && ei_op == OP_REQUANT;
  assign exec_done = mm_done || rq_done;
  assign exec_idle = mm_idle && rq_idle;

  // ---------------------------------------------------------------------------------------------
  // MATMUL unit, WBUF, ABUF, array, accumulators.
  // ---------------------------------------------------------------------------------------------
  logic                 w_rd_en, a_rd_en, a_rd_tag, mm_p_rd_en, act_signed;
  logic [WRW-1:0]       w_rd_addr;
  logic [LN:0]          w_rd_tag, w_tag;
  logic [AAW-1:0]       a_rd_addr;
  logic [PAW-1:0]       mm_p_rd_addr;
  logic [LN-1:0]        mm_p_rd_tag;
  logic                 w_valid, a_valid, a_tag;
  logic [VW-1:0]        w_data, a_data;
  logic                 arr_valid, acc_valid, acc_busy;
  logic [N-1:0][31:0]   arr_data, acc_bias;
  logic [CAW-1:0]       acc_addr;
  acc_mode_t            acc_mode;

  // PTAB read path (shared).
  logic                    p_rd_en, p_valid, mm_p_valid, rq_p_valid;
  logic [PAW-1:0]          p_rd_addr;
  logic [PTW-1:0]          p_rd_tag, p_tag;
  logic [PTAB_ENTRY_W-1:0] p_data;

  matmul_ctrl #(
    .N(N), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH),
    .W_LAT(W_LAT), .A_LAT(A_LAT), .P_LAT(P_LAT)
  ) u_mm (
    .clk, .rst, .start(mm_start), .instr(exec_instr), .abort_req(exec_abort), .done(mm_done), .idle(mm_idle),
    .act_signed, .w_rd_en, .w_rd_addr, .w_rd_tag, .a_rd_en, .a_rd_addr, .a_rd_tag,
    .p_rd_en(mm_p_rd_en), .p_rd_addr(mm_p_rd_addr), .p_rd_tag(mm_p_rd_tag),
    .p_rd_valid(mm_p_valid), .p_rd_data(p_data), .p_rd_col(p_tag[LN-1:0]),
    .arr_valid, .acc_valid, .acc_addr, .acc_mode, .acc_bias, .acc_busy
  );

  logic [VW-1:0] w_host_data, a_host_data;
  logic          w_host_valid, a_host_valid;

  buffer_ram #(.WIDTH(VW), .DEPTH(WBUF_DEPTH * N), .LANES(1), .RD_PIPE(W_PIPE), .TAG_W(LN + 1)) u_wbuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(w_rd_en), .dp_rd_addr(w_rd_addr), .dp_rd_tag(w_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(hr_wbuf), .host_rd_addr(host_addr[WRW-1:0]),
    .host_wr_en(hw_wbuf), .host_wr_addr(host_addr[WRW-1:0]), .host_wr_data(host_wr_data[VW-1:0]),
    .rd_valid(w_valid), .rd_data(w_data), .rd_tag(w_tag)
  );
  assign w_host_valid = w_valid && host_sel;
  assign w_host_data  = w_data;

  // ABUF: MATMUL reads, REQUANT writes.
  logic           rq_a_wr_en;
  logic [AAW-1:0] rq_a_wr_addr;
  logic [VW-1:0]  rq_a_wr_data;
  logic [N-1:0]   rq_a_wr_be;

  buffer_ram #(.WIDTH(VW), .DEPTH(ABUF_DEPTH), .LANES(N), .RD_PIPE(A_LAT - 1), .TAG_W(1)) u_abuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(a_rd_en), .dp_rd_addr(a_rd_addr), .dp_rd_tag(a_rd_tag),
    .dp_wr_en(rq_a_wr_en), .dp_wr_addr(rq_a_wr_addr), .dp_wr_data(rq_a_wr_data), .dp_wr_be(rq_a_wr_be),
    .host_rd_en(hr_abuf), .host_rd_addr(host_addr[AAW-1:0]),
    .host_wr_en(hw_abuf), .host_wr_addr(host_addr[AAW-1:0]), .host_wr_data(host_wr_data[VW-1:0]),
    .rd_valid(a_valid), .rd_data(a_data), .rd_tag(a_tag)
  );
  assign a_host_valid = a_valid && host_sel;
  assign a_host_data  = a_data;

  systolic_array #(.N(N)) u_array (
    .clk, .rst, .act_signed,
    .a_in(a_data), .a_valid(a_tag),
    .w_in(w_data), .w_load(w_tag[LN]), .w_row(w_tag[LN-1:0]),
    .out_data(arr_data), .out_valid(arr_valid)
  );

  // ACC: MATMUL updates, REQUANT reads, host reads and writes.
  logic               rq_acc_rd_en, c_rd_en, c_host_valid;
  logic [CAW-1:0]     rq_acc_rd_addr, c_rd_addr;
  logic [N-1:0][31:0] c_rd_data;

  assign c_rd_en   = host_sel ? hr_acc : rq_acc_rd_en;
  assign c_rd_addr = host_sel ? host_addr[CAW-1:0] : rq_acc_rd_addr;

  acc_unit #(.N(N), .DEPTH(ACC_DEPTH)) u_acc (
    .clk, .rst,
    .in_valid(acc_valid), .in_addr(acc_addr), .in_mode(acc_mode), .in_data(arr_data), .in_bias(acc_bias),
    .rd_en(c_rd_en), .rd_addr(c_rd_addr), .rd_data(c_rd_data),
    .wr_en(hw_acc), .wr_addr(host_addr[CAW-1:0]), .wr_data(host_wr_data[RW-1:0]),
    .busy(acc_busy)
  );

  always_ff @(posedge clk) begin
    if (rst) c_host_valid <= 1'b0;
    else     c_host_valid <= hr_acc && host_sel;
  end

  // ---------------------------------------------------------------------------------------------
  // REQUANT unit and PTAB.
  // ---------------------------------------------------------------------------------------------
  logic                rq_p_rd_en;
  logic [PAW-1:0]      rq_p_rd_addr;
  logic [AAW+LN-1:0]   rq_p_rd_tag;

  requant_ctrl #(
    .N(N), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH), .P_LAT(P_LAT)
  ) u_rq (
    .clk, .rst, .start(rq_start), .instr(exec_instr), .abort_req(exec_abort), .done(rq_done), .idle(rq_idle),
    .acc_rd_en(rq_acc_rd_en), .acc_rd_addr(rq_acc_rd_addr), .acc_rd_data(c_rd_data),
    .p_rd_en(rq_p_rd_en), .p_rd_addr(rq_p_rd_addr), .p_rd_tag(rq_p_rd_tag),
    .p_rd_valid(rq_p_valid), .p_rd_data(p_data), .p_rd_ret(p_tag[AAW+LN-1:0]),
    .a_wr_en(rq_a_wr_en), .a_wr_addr(rq_a_wr_addr), .a_wr_data(rq_a_wr_data), .a_wr_be(rq_a_wr_be)
  );

  assign p_rd_en    = mm_p_rd_en || rq_p_rd_en;
  assign p_rd_addr  = rq_p_rd_en ? rq_p_rd_addr : mm_p_rd_addr;
  assign p_rd_tag   = rq_p_rd_en ? {1'b1, rq_p_rd_tag} : {1'b0, (PTW - 1)'(mm_p_rd_tag)};
  assign mm_p_valid = p_valid && !host_sel && !p_tag[PTW-1];
  assign rq_p_valid = p_valid && !host_sel && p_tag[PTW-1];

  buffer_ram #(.WIDTH(PTAB_ENTRY_W), .DEPTH(PTAB_DEPTH), .LANES(1), .RD_PIPE(P_LAT - 1), .TAG_W(PTW)) u_ptab (
    .clk, .rst, .host_sel,
    .dp_rd_en(p_rd_en), .dp_rd_addr(p_rd_addr), .dp_rd_tag(p_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(hr_ptab), .host_rd_addr(host_addr[PAW-1:0]),
    .host_wr_en(hw_ptab), .host_wr_addr(host_addr[PAW-1:0]), .host_wr_data(host_wr_data[PTAB_ENTRY_W-1:0]),
    .rd_valid(p_valid), .rd_data(p_data), .rd_tag(p_tag)
  );

  // ---------------------------------------------------------------------------------------------
  // Host read data.
  // ---------------------------------------------------------------------------------------------
  logic i_host_valid, p_host_valid;
  assign i_host_valid = imem_rd_valid && host_sel;
  assign p_host_valid = p_valid && host_sel;

  always_comb begin
    host_rd_valid = i_host_valid || w_host_valid || a_host_valid || c_host_valid || p_host_valid;
    host_rd_data  = '0;
    if (i_host_valid) host_rd_data = HW'(imem_rd_data);
    if (w_host_valid) host_rd_data = HW'(w_host_data);
    if (a_host_valid) host_rd_data = HW'(a_host_data);
    if (c_host_valid) host_rd_data = HW'(c_rd_data);
    if (p_host_valid) host_rd_data = HW'(p_data);
  end

  // Bits of the host address beyond the addressed memory, the IMEM tag, and the buffers' valid
  // outputs that the datapath does not need.
  logic unused;
  assign unused = ^{host_addr, imem_tag, a_valid};

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      if (host_rd_en || host_wr_en) assert (host_sel);       // the protocol's busy rule
      assert ($onehot0({i_host_valid, w_host_valid, a_host_valid, c_host_valid, p_host_valid}));
      assert (!(mm_p_rd_en && rq_p_rd_en));
      assert (mm_idle || rq_idle);
    end
  end
`endif

endmodule
