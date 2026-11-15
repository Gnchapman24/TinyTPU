// Test-only wrapper: matmul_ctrl with the real datapath around it (WBUF, ABUF, PTAB as
// buffer_ram, systolic_array, acc_unit). cocotb loads the memories through the host ports while
// host_sel is high, starts one MATMUL, and reads ACC back afterwards.
module matmul_tb
  import isa_pkg::*;
#(
  parameter int N          = 4,
  parameter int WBUF_DEPTH = 64,   // tiles
  parameter int ABUF_DEPTH = 64,
  parameter int ACC_DEPTH  = 32,
  parameter int PTAB_DEPTH = 64,
  parameter int W_PIPE     = 2,
  parameter int A_PIPE     = 1,
  parameter int P_PIPE     = 1
) (
  input  logic                                    clk,
  input  logic                                    rst,
  input  logic                                    host_sel,
  // instruction
  input  logic                                    start,
  input  logic [INSTR_W-1:0]                      instr,
  input  logic                                    abort_req,
  output logic                                    done,
  output logic                                    idle,
  // host access
  input  logic                                    w_wr_en,
  input  logic [$clog2(WBUF_DEPTH)+$clog2(N)-1:0] w_wr_addr,
  input  logic [N*8-1:0]                          w_wr_data,
  input  logic                                    a_wr_en,
  input  logic [$clog2(ABUF_DEPTH)-1:0]           a_wr_addr,
  input  logic [N*8-1:0]                          a_wr_data,
  input  logic                                    p_wr_en,
  input  logic [$clog2(PTAB_DEPTH)-1:0]           p_wr_addr,
  input  logic [PTAB_ENTRY_W-1:0]                 p_wr_data,
  input  logic                                    c_wr_en,
  input  logic [$clog2(ACC_DEPTH)-1:0]            c_wr_addr,
  input  logic [N-1:0][31:0]                      c_wr_data,
  input  logic                                    c_rd_en,
  input  logic [$clog2(ACC_DEPTH)-1:0]            c_rd_addr,
  output logic [N-1:0][31:0]                      c_rd_data,
  output logic                                    acc_busy
);
  localparam int LN  = $clog2(N);
  localparam int WRW = $clog2(WBUF_DEPTH) + LN;

  logic                          w_rd_en, a_rd_en, p_rd_en, a_rd_tag;
  logic [WRW-1:0]                w_rd_addr;
  logic [LN:0]                   w_rd_tag, w_tag;
  logic [$clog2(ABUF_DEPTH)-1:0] a_rd_addr;
  logic [$clog2(PTAB_DEPTH)-1:0] p_rd_addr;
  logic [LN-1:0]                 p_rd_tag, p_tag;
  logic [N*8-1:0]                w_data, a_data;
  logic [PTAB_ENTRY_W-1:0]       p_data;
  logic                          a_tag, p_valid, act_signed;
  logic                          arr_valid, acc_valid;
  logic [N-1:0][31:0]            arr_data, acc_bias;
  logic [$clog2(ACC_DEPTH)-1:0]  acc_addr;
  acc_mode_t                     acc_mode;
  logic                          unused_valid;
  logic                          w_valid, a_valid;

  assign unused_valid = w_valid ^ a_valid;

  matmul_ctrl #(
    .N(N), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH),
    .W_LAT(1 + W_PIPE), .A_LAT(1 + A_PIPE), .P_LAT(1 + P_PIPE)
  ) u_ctrl (
    .clk, .rst, .start, .instr, .abort_req, .done, .idle, .act_signed,
    .w_rd_en, .w_rd_addr, .w_rd_tag, .a_rd_en, .a_rd_addr, .a_rd_tag,
    .p_rd_en, .p_rd_addr, .p_rd_tag, .p_rd_valid(p_valid), .p_rd_data(p_data), .p_rd_col(p_tag),
    .arr_valid, .acc_valid, .acc_addr, .acc_mode, .acc_bias, .acc_busy
  );

  buffer_ram #(.WIDTH(N*8), .DEPTH(WBUF_DEPTH*N), .LANES(1), .RD_PIPE(W_PIPE), .TAG_W(LN+1)) u_wbuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(w_rd_en), .dp_rd_addr(w_rd_addr), .dp_rd_tag(w_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(1'b0), .host_rd_addr('0), .host_wr_en(w_wr_en), .host_wr_addr(w_wr_addr), .host_wr_data(w_wr_data),
    .rd_valid(w_valid), .rd_data(w_data), .rd_tag(w_tag)
  );

  buffer_ram #(.WIDTH(N*8), .DEPTH(ABUF_DEPTH), .LANES(N), .RD_PIPE(A_PIPE), .TAG_W(1)) u_abuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(a_rd_en), .dp_rd_addr(a_rd_addr), .dp_rd_tag(a_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be('0),
    .host_rd_en(1'b0), .host_rd_addr('0), .host_wr_en(a_wr_en), .host_wr_addr(a_wr_addr), .host_wr_data(a_wr_data),
    .rd_valid(a_valid), .rd_data(a_data), .rd_tag(a_tag)
  );

  buffer_ram #(.WIDTH(PTAB_ENTRY_W), .DEPTH(PTAB_DEPTH), .LANES(1), .RD_PIPE(P_PIPE), .TAG_W(LN)) u_ptab (
    .clk, .rst, .host_sel,
    .dp_rd_en(p_rd_en), .dp_rd_addr(p_rd_addr), .dp_rd_tag(p_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(1'b0), .host_rd_addr('0), .host_wr_en(p_wr_en), .host_wr_addr(p_wr_addr), .host_wr_data(p_wr_data),
    .rd_valid(p_valid), .rd_data(p_data), .rd_tag(p_tag)
  );

  systolic_array #(.N(N)) u_array (
    .clk, .rst, .act_signed,
    .a_in(a_data), .a_valid(a_tag),
    .w_in(w_data), .w_load(w_tag[LN]), .w_row(w_tag[LN-1:0]),
    .out_data(arr_data), .out_valid(arr_valid)
  );

  acc_unit #(.N(N), .DEPTH(ACC_DEPTH)) u_acc (
    .clk, .rst,
    .in_valid(acc_valid), .in_addr(acc_addr), .in_mode(acc_mode), .in_data(arr_data), .in_bias(acc_bias),
    .rd_en(c_rd_en), .rd_addr(c_rd_addr), .rd_data(c_rd_data),
    .wr_en(c_wr_en), .wr_addr(c_wr_addr), .wr_data(c_wr_data),
    .busy(acc_busy)
  );
endmodule
