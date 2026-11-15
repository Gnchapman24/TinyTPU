/** REQUANT execution unit: streams accumulators through the requantizer into the activation
 *  buffer, one value per cycle (docs/isa.md, REQUANT).
 *
 *  Loop order n (column tile), m (row), col, as in the ISA. In each issue cycle it reads ACC row
 *  acc_base + m*nt + n (acc_unit read port, C_LAT = 1) and PTAB entry p_base + n*N + col, tagged
 *  {ABUF vector a_dst + m*nt + n, col}. When the PTAB entry returns (P_LAT), the ACC value of
 *  that column, delayed to match, enters the requantizer together with the entry's M and s and
 *  the instruction's relu bit; the requantizer returns the tag with the result byte, which is
 *  written into lane `col` of that ABUF vector (byte write enable). Addresses are formed
 *  incrementally; the sequencer's range check has guaranteed that all are in range.
 *
 *  `done` pulses once the last byte has been written. On `abort_req` the unit stops issuing,
 *  waits until everything issued has been written, and returns to idle without `done`.
 *  `instr` must stay stable from `start` to `done`.
 *
 *  Latency: nt*m*N issue cycles, then P_LAT + 6 (requant) + 1 cycles until done.
 */
module requant_ctrl
  import isa_pkg::*;
#(
  parameter int N          = 8,                   // array size; a power of two, at least 2
  parameter int ABUF_DEPTH = DEFAULT_ABUF_DEPTH,  // activation vectors
  parameter int ACC_DEPTH  = DEFAULT_ACC_DEPTH,   // accumulator rows
  parameter int PTAB_DEPTH = DEFAULT_PTAB_DEPTH,  // parameter entries
  parameter int P_LAT      = 2                    // PTAB read latency, at least 1 (ACC read latency is 1)
) (
  input  logic                                       clk,         // clock
  input  logic                                       rst,         // synchronous active-high reset; abandons the instruction
  input  logic                                       start,       // begin executing instr (a REQUANT)
  input  logic [INSTR_W-1:0]                         instr,       // the REQUANT; stable until done
  input  logic                                       abort_req,   // stop issuing and drain (sequencer exec_abort)
  output logic                                       done,        // one-cycle pulse: every byte is in ABUF
  output logic                                       idle,        // nothing issued is still in flight
  output logic                                       acc_rd_en,   // ACC read (acc_unit rd_en)
  output logic [$clog2(ACC_DEPTH)-1:0]               acc_rd_addr, // ACC row
  input  logic [N-1:0][31:0]                         acc_rd_data, // row contents, 1 cycle after acc_rd_en
  output logic                                       p_rd_en,     // PTAB datapath read
  output logic [$clog2(PTAB_DEPTH)-1:0]              p_rd_addr,   // parameter entry
  output logic [$clog2(ABUF_DEPTH)+$clog2(N)-1:0]    p_rd_tag,    // {ABUF vector, col}
  input  logic                                       p_rd_valid,  // PTAB read data is valid
  input  logic [PTAB_ENTRY_W-1:0]                    p_rd_data,   // parameter entry {s, M, bias}
  input  logic [$clog2(ABUF_DEPTH)+$clog2(N)-1:0]    p_rd_ret,    // tag returned with it
  output logic                                       a_wr_en,     // ABUF datapath write
  output logic [$clog2(ABUF_DEPTH)-1:0]              a_wr_addr,   // ABUF vector
  output logic [N*8-1:0]                             a_wr_data,   // the result byte in every lane
  output logic [N-1:0]                               a_wr_be      // lane enable: the result's column
);

  if (N < 2 || (N & (N - 1)) != 0) begin : g_check_n
    $error("requant_ctrl: N must be a power of two, at least 2");
  end
  if (P_LAT < 1) begin : g_check_plat
    $error("requant_ctrl: P_LAT must be at least 1");
  end

  localparam int LN  = $clog2(N);
  localparam int CW  = COUNT_W;
  localparam int AAW = $clog2(ABUF_DEPTH);
  localparam int CAW = $clog2(ACC_DEPTH);
  localparam int PAW = $clog2(PTAB_DEPTH);
  localparam int TGW = AAW + LN;
  localparam int RQ_LAT = 6;                      // requant latency
  localparam int OW  = $clog2(P_LAT + RQ_LAT + 4) + 1;

  typedef enum logic [1:0] {
    S_IDLE,
    S_ISSUE,
    S_FINISH,
    S_ABORT
  } fsm_t;

  fsm_t           fsm;
  instr_t         f;
  logic [CW-1:0]  n, m;
  logic [LN-1:0]  col;
  logic [CAW-1:0] acc_n, acc_cur;                 // acc_base + n, then + m*nt
  logic [AAW-1:0] a_n, a_cur;                     // a_dst + n, then + m*nt
  logic [PAW-1:0] p_n;                            // p_base + n*N
  logic [OW-1:0]  outstanding;                    // issued values not yet written
  logic           relu;

  assign f    = instr_t'(instr);
  assign relu = f.flags[FLAG_RELU];

  logic unused_f;
  assign unused_f = ^{f.w_base, f.kt, f.a_base[15:AAW], f.acc_base[15:CAW], f.p_base[15:PAW],
                      f.nt[15:CW], f.m[15:CW], f.flags[7:1], f.opcode};

  // ---------------------------------------------------------------------------------------------
  // Issue.
  // ---------------------------------------------------------------------------------------------
  logic issuing, last;

  assign issuing     = fsm == S_ISSUE && !abort_req;
  assign last        = n == f.nt[CW-1:0] - 1'b1 && m == f.m[CW-1:0] - 1'b1 && col == LN'(N - 1);
  assign acc_rd_en   = issuing;
  assign acc_rd_addr = acc_cur;
  assign p_rd_en     = issuing;
  assign p_rd_addr   = p_n + PAW'(col);
  assign p_rd_tag    = {a_cur, col};

  always_ff @(posedge clk) begin
    if (rst) begin
      fsm <= S_IDLE;
    end else if (abort_req && (fsm == S_ISSUE || fsm == S_FINISH)) begin
      fsm <= S_ABORT;
    end else begin
      unique case (fsm)
        S_IDLE: if (start) begin
          n       <= '0;
          m       <= '0;
          col     <= '0;
          acc_n   <= f.acc_base[CAW-1:0];
          acc_cur <= f.acc_base[CAW-1:0];
          a_n     <= f.a_base[AAW-1:0];
          a_cur   <= f.a_base[AAW-1:0];
          p_n     <= f.p_base[PAW-1:0];
          fsm     <= S_ISSUE;
        end
        S_ISSUE: begin
          col <= col + 1'b1;
          if (col == LN'(N - 1)) begin                // row done: next m, or next n
            if (last) begin
              fsm <= S_FINISH;
            end else if (m == f.m[CW-1:0] - 1'b1) begin
              m       <= '0;
              n       <= n + 1'b1;
              acc_n   <= acc_n + 1'b1;
              acc_cur <= acc_n + 1'b1;
              a_n     <= a_n + 1'b1;
              a_cur   <= a_n + 1'b1;
              p_n     <= p_n + PAW'(N);
            end else begin
              m       <= m + 1'b1;
              acc_cur <= acc_cur + CAW'(f.nt[CW-1:0]);
              a_cur   <= a_cur + AAW'(f.nt[CW-1:0]);
            end
          end
        end
        S_FINISH: if (outstanding == '0) fsm <= S_IDLE;
        S_ABORT:  if (outstanding == '0) fsm <= S_IDLE;
        default:  fsm <= S_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Value path: select the column 1 cycle after the ACC read, delay to meet the PTAB entry.
  // ---------------------------------------------------------------------------------------------
  logic [LN-1:0]        col_d1;
  logic signed [31:0]   val [P_LAT];              // val[0] = selected value (1 cycle after issue)

  always_ff @(posedge clk) begin
    col_d1 <= col;
  end
  assign val[0] = acc_rd_data[col_d1];
  for (genvar i = 1; i < P_LAT; i++) begin : g_val
    always_ff @(posedge clk) val[i] <= val[i-1];
  end

  ptab_entry_t       entry;
  logic              rq_valid;
  logic [7:0]        rq_data;
  logic [TGW-1:0]    rq_tag;
  logic signed [31:0] unused_bias;

  assign entry       = ptab_entry_t'(p_rd_data);
  assign unused_bias = entry.bias;

  requant #(.TAG_W(TGW)) u_requant (
    .clk, .rst,
    .in_valid(p_rd_valid), .in_acc(val[P_LAT-1]), .in_mult(entry.mult), .in_shift(entry.s),
    .in_relu(relu), .in_tag(p_rd_ret),
    .out_valid(rq_valid), .out_data(rq_data), .out_tag(rq_tag)
  );

  assign a_wr_en   = rq_valid;
  assign a_wr_addr = rq_tag[TGW-1:LN];
  assign a_wr_data = {N{rq_data}};
  assign a_wr_be   = N'(1) << rq_tag[LN-1:0];

  // Values issued and not yet written.
  always_ff @(posedge clk) begin
    if (rst) outstanding <= '0;
    else outstanding <= outstanding + OW'(p_rd_en) - OW'(a_wr_en);
  end

  assign done = fsm == S_FINISH && outstanding == '0;
  assign idle = fsm == S_IDLE;

`ifdef ASSERTIONS
  /* verilator lint_off PROCASSINIT */
  logic past_valid = 1'b0;
  /* verilator lint_on PROCASSINIT */
  always_ff @(posedge clk) begin
    past_valid <= 1'b1;
    if (past_valid && !rst) begin
      if (start) assert (fsm == S_IDLE);
      assert (outstanding <= OW'(P_LAT + RQ_LAT + 1));
      if (a_wr_en) assert (outstanding != '0);
    end
  end
`endif

endmodule
