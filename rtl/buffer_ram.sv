/** On-chip buffer (WBUF, ABUF, PTAB): a block RAM with a fixed-latency read path and host access.
 *
 *  One read port and one write port (simple dual-port BRAM). Each port is a 2:1 mux between the
 *  datapath and the host, selected by `host_sel` (high while the sequencer is halted). The host
 *  and the datapath never use the buffer at the same time, so there is no arbitration;
 *  assertions check that the unselected side is idle.
 *
 *  Reads have a fixed latency of 1 + RD_PIPE cycles: the BRAM's own output register plus RD_PIPE
 *  pipeline stages, which Vivado uses for the BRAM output register and the read-data mux of a
 *  buffer built from many BRAMs. Nothing ever stalls. A datapath read carries a TAG_W-bit tag
 *  (e.g. `w_load`/`w_row` or `a_valid` for the systolic array) that comes out on `rd_tag` with
 *  its data; `rd_tag` is 0 in every other cycle (idle or host read), so its bits can drive the
 *  array's load/valid inputs directly. The read path itself is the feeder: no FIFO is needed
 *  (AGENTS.md §3).
 *
 *  Writes are split into LANES equal lanes with their own enables (`dp_wr_be`), so REQUANT can
 *  write one byte of an activation vector per cycle (ABUF: LANES = N). Host writes write all
 *  lanes. Reading a row in the same cycle it is written is not allowed (asserted): the BRAM's
 *  result would be undefined.
 *
 *  Latency: 1 + RD_PIPE cycles from rd_en to rd_valid (with rd_data and rd_tag); writes take effect at the clock edge.
 */
module buffer_ram #(
  parameter int WIDTH   = 64,    // bits per row
  parameter int DEPTH   = 1024,  // rows; a power of two, at least 2
  parameter int LANES   = 1,     // write-enable lanes per row; WIDTH must be a multiple of LANES
  parameter int RD_PIPE = 0,     // extra read pipeline stages after the BRAM output register
  parameter int TAG_W   = 1      // width of the side-band tag of a datapath read
) (
  input  logic                     clk,           // clock
  input  logic                     rst,           // synchronous active-high reset; empties the read pipeline (not the memory)
  input  logic                     host_sel,      // 1: the host ports drive the RAM; 0: the datapath ports do
  input  logic                     dp_rd_en,      // datapath: read row dp_rd_addr
  input  logic [$clog2(DEPTH)-1:0] dp_rd_addr,    // datapath: row to read
  input  logic [TAG_W-1:0]         dp_rd_tag,     // datapath: tag returned on rd_tag with the data
  input  logic                     dp_wr_en,      // datapath: write the enabled lanes of row dp_wr_addr
  input  logic [$clog2(DEPTH)-1:0] dp_wr_addr,    // datapath: row to write
  input  logic [WIDTH-1:0]         dp_wr_data,    // datapath: data to write
  input  logic [LANES-1:0]         dp_wr_be,      // datapath: lane enables; lane l is bits [l*WIDTH/LANES +: WIDTH/LANES]
  input  logic                     host_rd_en,    // host: read row host_rd_addr
  input  logic [$clog2(DEPTH)-1:0] host_rd_addr,  // host: row to read
  input  logic                     host_wr_en,    // host: write row host_wr_addr (all lanes)
  input  logic [$clog2(DEPTH)-1:0] host_wr_addr,  // host: row to write
  input  logic [WIDTH-1:0]         host_wr_data,  // host: data to write
  output logic                     rd_valid,      // rd_data holds the row of a read issued 1 + RD_PIPE cycles ago
  output logic [WIDTH-1:0]         rd_data,       // row contents
  output logic [TAG_W-1:0]         rd_tag         // dp_rd_tag of a datapath read issued 1 + RD_PIPE cycles ago; 0 otherwise
);

  if (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0) begin : g_check_depth
    $error("buffer_ram: DEPTH must be a power of two, at least 2");
  end
  if (LANES < 1 || WIDTH % LANES != 0) begin : g_check_lanes
    $error("buffer_ram: WIDTH must be a multiple of LANES");
  end
  if (RD_PIPE < 0) begin : g_check_pipe
    $error("buffer_ram: RD_PIPE must be at least 0");
  end

  localparam int AW  = $clog2(DEPTH);
  localparam int LW  = WIDTH / LANES;
  localparam int LAT = 1 + RD_PIPE;

  // ---------------------------------------------------------------------------------------------
  // Port muxes.
  // ---------------------------------------------------------------------------------------------
  logic             re, we;
  logic [AW-1:0]    ra, wa;
  logic [WIDTH-1:0] wd;
  logic [LANES-1:0] wbe;

  assign re  = host_sel ? host_rd_en   : dp_rd_en;
  assign ra  = host_sel ? host_rd_addr : dp_rd_addr;
  assign we  = host_sel ? host_wr_en   : dp_wr_en;
  assign wa  = host_sel ? host_wr_addr : dp_wr_addr;
  assign wd  = host_sel ? host_wr_data : dp_wr_data;
  assign wbe = host_sel ? '1           : dp_wr_be;

  // ---------------------------------------------------------------------------------------------
  // RAM (Vivado simple dual-port template with byte-wide write enables).
  // ---------------------------------------------------------------------------------------------
  logic [WIDTH-1:0] mem [DEPTH];
  logic [WIDTH-1:0] rq;

  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++) begin
      if (we && wbe[l]) mem[wa][l*LW +: LW] <= wd[l*LW +: LW];
    end
  end

  always_ff @(posedge clk) begin
    if (re) rq <= mem[ra];
  end

  // ---------------------------------------------------------------------------------------------
  // Read pipeline: data stages after the BRAM, and the valid/tag delay line beside them.
  // ---------------------------------------------------------------------------------------------
  logic             v  [LAT];
  logic [TAG_W-1:0] tg [LAT];

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < LAT; i++) v[i] <= 1'b0;
    end else begin
      v[0] <= re;
      for (int i = 1; i < LAT; i++) v[i] <= v[i-1];
    end
    tg[0] <= (!host_sel && dp_rd_en) ? dp_rd_tag : '0;
    for (int i = 1; i < LAT; i++) tg[i] <= tg[i-1];
  end

  if (RD_PIPE == 0) begin : g_no_pipe
    assign rd_data = rq;
  end else begin : g_pipe
    logic [WIDTH-1:0] pq [RD_PIPE];
    always_ff @(posedge clk) begin
      pq[0] <= rq;
      for (int i = 1; i < RD_PIPE; i++) pq[i] <= pq[i-1];
    end
    assign rd_data = pq[RD_PIPE-1];
  end

  assign rd_valid = v[LAT-1];
  assign rd_tag   = tg[LAT-1];

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */

  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      // The side that is not selected must be idle (no arbitration).
      if (host_sel) assert (!dp_rd_en && !dp_wr_en);
      else          assert (!host_rd_en && !host_wr_en);
      // No read of a row in the cycle it is written.
      if (re && we && wbe != '0) assert (ra != wa);
    end
  end
`endif

endmodule
