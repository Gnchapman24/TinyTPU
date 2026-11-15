// Test-only wrapper: requant_ctrl with the real memories (acc_unit for ACC, PTAB and ABUF as
// buffer_ram). cocotb fills ACC / PTAB / ABUF through the host ports while host_sel is high,
// starts one REQUANT, and reads ABUF back afterwards.
module requant_tb
  import isa_pkg::*;
#(
  parameter int N          = 4,
  parameter int ABUF_DEPTH = 64,
  parameter int ACC_DEPTH  = 32,
  parameter int PTAB_DEPTH = 64,
  parameter int P_PIPE     = 1
) (
  input  logic                          clk,
  input  logic                          rst,
  input  logic                          host_sel,
  input  logic                          start,
  input  logic [INSTR_W-1:0]            instr,
  input  logic                          abort_req,
  output logic                          done,
  output logic                          idle,
  input  logic                          c_wr_en,
  input  logic [$clog2(ACC_DEPTH)-1:0]  c_wr_addr,
  input  logic [N-1:0][31:0]            c_wr_data,
  input  logic                          p_wr_en,
  input  logic [$clog2(PTAB_DEPTH)-1:0] p_wr_addr,
  input  logic [PTAB_ENTRY_W-1:0]       p_wr_data,
  input  logic                          a_wr_en,
  input  logic [$clog2(ABUF_DEPTH)-1:0] a_wr_addr,
  input  logic [N*8-1:0]                a_wr_data,
  input  logic                          a_rd_en,
  input  logic [$clog2(ABUF_DEPTH)-1:0] a_rd_addr,
  output logic [N*8-1:0]                a_rd_data
);
  localparam int LN  = $clog2(N);
  localparam int TGW = $clog2(ABUF_DEPTH) + LN;

  logic                          acc_rd_en, p_rd_en, p_valid, dp_wr_en, acc_busy;
  logic [$clog2(ACC_DEPTH)-1:0]  acc_rd_addr;
  logic [N-1:0][31:0]            acc_rd_data;
  logic [$clog2(PTAB_DEPTH)-1:0] p_rd_addr;
  logic [TGW-1:0]                p_rd_tag, p_ret;
  logic [PTAB_ENTRY_W-1:0]       p_data;
  logic [$clog2(ABUF_DEPTH)-1:0] dp_wr_addr;
  logic [N*8-1:0]                dp_wr_data;
  logic [N-1:0]                  dp_wr_be;
  logic                          a_valid, unused;
  logic [0:0]                    a_tag;

  assign unused = acc_busy ^ a_valid ^ a_tag[0];

  requant_ctrl #(.N(N), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH), .PTAB_DEPTH(PTAB_DEPTH), .P_LAT(1 + P_PIPE)) u_ctrl (
    .clk, .rst, .start, .instr, .abort_req, .done, .idle,
    .acc_rd_en, .acc_rd_addr, .acc_rd_data,
    .p_rd_en, .p_rd_addr, .p_rd_tag, .p_rd_valid(p_valid), .p_rd_data(p_data), .p_rd_ret(p_ret),
    .a_wr_en(dp_wr_en), .a_wr_addr(dp_wr_addr), .a_wr_data(dp_wr_data), .a_wr_be(dp_wr_be)
  );

  acc_unit #(.N(N), .DEPTH(ACC_DEPTH)) u_acc (
    .clk, .rst,
    .in_valid(1'b0), .in_addr('0), .in_mode(ACC_OVERWRITE), .in_data('0), .in_bias('0),
    .rd_en(acc_rd_en), .rd_addr(acc_rd_addr), .rd_data(acc_rd_data),
    .wr_en(c_wr_en), .wr_addr(c_wr_addr), .wr_data(c_wr_data),
    .busy(acc_busy)
  );

  buffer_ram #(.WIDTH(PTAB_ENTRY_W), .DEPTH(PTAB_DEPTH), .LANES(1), .RD_PIPE(P_PIPE), .TAG_W(TGW)) u_ptab (
    .clk, .rst, .host_sel,
    .dp_rd_en(p_rd_en), .dp_rd_addr(p_rd_addr), .dp_rd_tag(p_rd_tag),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(1'b0), .host_rd_addr('0), .host_wr_en(p_wr_en), .host_wr_addr(p_wr_addr), .host_wr_data(p_wr_data),
    .rd_valid(p_valid), .rd_data(p_data), .rd_tag(p_ret)
  );

  buffer_ram #(.WIDTH(N*8), .DEPTH(ABUF_DEPTH), .LANES(N), .RD_PIPE(1), .TAG_W(1)) u_abuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(1'b0), .dp_rd_addr('0), .dp_rd_tag(1'b0),
    .dp_wr_en(dp_wr_en), .dp_wr_addr(dp_wr_addr), .dp_wr_data(dp_wr_data), .dp_wr_be(dp_wr_be),
    .host_rd_en(a_rd_en), .host_rd_addr(a_rd_addr), .host_wr_en(a_wr_en), .host_wr_addr(a_wr_addr), .host_wr_data(a_wr_data),
    .rd_valid(a_valid), .rd_data(a_rd_data), .rd_tag(a_tag)
  );
endmodule
