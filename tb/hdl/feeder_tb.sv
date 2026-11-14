// Test-only wrapper: WBUF and ABUF (buffer_ram) feeding the systolic array through their read
// paths alone. The array's control inputs come only from the read tags: WBUF's tag is
// {w_load, w_row}, ABUF's tag is a_valid. cocotb plays the address generator (A4.2): it loads
// both buffers through the host ports, then issues datapath reads.
module feeder_tb #(
  parameter int N       = 4,
  parameter int WROWS   = 128,  // WBUF rows (tiles * N)
  parameter int AROWS   = 64,   // ABUF vectors
  parameter int W_PIPE  = 2,    // WBUF RD_PIPE
  parameter int A_PIPE  = 1     // ABUF RD_PIPE
) (
  input  logic                     clk,
  input  logic                     rst,
  input  logic                     host_sel,
  input  logic                     act_signed,
  // host loading
  input  logic                     w_host_wr_en,
  input  logic [$clog2(WROWS)-1:0] w_host_wr_addr,
  input  logic [N*8-1:0]           w_host_wr_data,
  input  logic                     a_host_wr_en,
  input  logic [$clog2(AROWS)-1:0] a_host_wr_addr,
  input  logic [N*8-1:0]           a_host_wr_data,
  // datapath reads (the address generator's side)
  input  logic                     w_rd_en,
  input  logic [$clog2(WROWS)-1:0] w_rd_addr,
  input  logic [$clog2(N)-1:0]     w_rd_row,
  input  logic                     a_rd_en,
  input  logic [$clog2(AROWS)-1:0] a_rd_addr,
  // array results
  output logic [N-1:0][31:0]       out_data,
  output logic                     out_valid
);
  localparam int RW = $clog2(N);

  logic [N*8-1:0] w_data, a_data;
  logic [RW:0]    w_tag;
  logic           a_tag;
  logic           unused_valid;

  // rd_tag is 0 except on datapath reads, so the tags alone drive the array.
  logic           w_valid, a_valid;
  assign unused_valid = w_valid ^ a_valid;

  buffer_ram #(.WIDTH(N*8), .DEPTH(WROWS), .LANES(1), .RD_PIPE(W_PIPE), .TAG_W(RW + 1)) u_wbuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(w_rd_en), .dp_rd_addr(w_rd_addr), .dp_rd_tag({1'b1, w_rd_row}),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be(1'b0),
    .host_rd_en(1'b0), .host_rd_addr('0),
    .host_wr_en(w_host_wr_en), .host_wr_addr(w_host_wr_addr), .host_wr_data(w_host_wr_data),
    .rd_valid(w_valid), .rd_data(w_data), .rd_tag(w_tag)
  );

  buffer_ram #(.WIDTH(N*8), .DEPTH(AROWS), .LANES(N), .RD_PIPE(A_PIPE), .TAG_W(1)) u_abuf (
    .clk, .rst, .host_sel,
    .dp_rd_en(a_rd_en), .dp_rd_addr(a_rd_addr), .dp_rd_tag(1'b1),
    .dp_wr_en(1'b0), .dp_wr_addr('0), .dp_wr_data('0), .dp_wr_be('0),
    .host_rd_en(1'b0), .host_rd_addr('0),
    .host_wr_en(a_host_wr_en), .host_wr_addr(a_host_wr_addr), .host_wr_data(a_host_wr_data),
    .rd_valid(a_valid), .rd_data(a_data), .rd_tag(a_tag)
  );

  systolic_array #(.N(N)) u_array (
    .clk, .rst, .act_signed,
    .a_in(a_data), .a_valid(a_tag),
    .w_in(w_data), .w_load(w_tag[RW]), .w_row(w_tag[RW-1:0]),
    .out_data, .out_valid
  );
endmodule
