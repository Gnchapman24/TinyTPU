// Test-only wrapper: cmd_engine driving the real tpu_core. cocotb feeds request bytes on
// rx_data/rx_valid and takes reply bytes from tx_data/tx_valid/tx_ready (no UART), so tests run
// at full speed. RESET from the engine resets the core, as on the board.
module engine_tb
  import isa_pkg::*;
#(
  parameter int N              = 4,
  parameter int IMEM_DEPTH     = 16,
  parameter int WBUF_DEPTH     = 64,
  parameter int ABUF_DEPTH     = 64,
  parameter int ACC_DEPTH      = 32,
  parameter int PTAB_DEPTH     = 64,
  parameter int TIMEOUT_CYCLES = 300
) (
  input  logic       clk,
  input  logic       rst,
  input  logic [7:0] rx_data,
  input  logic       rx_valid,
  output logic [7:0] tx_data,
  output logic       tx_valid,
  input  logic       tx_ready,
  output logic       pkt_ok,
  output logic       pkt_nak,
  output logic       pkt_timeout,
  output logic       rx_overflow
);
  localparam int HW = (N * 32 > INSTR_W) ? N * 32 : INSTR_W;

  logic        core_rst, core_start, core_abort, host_rd_en, host_wr_en, host_rd_valid;
  logic [15:0] core_start_pc, core_pc, core_halt_code, host_addr;
  seq_state_t  core_state;
  seq_err_t    core_err;
  mem_id_t     host_mem;
  logic [HW-1:0] host_wr_data, host_rd_data;

  cmd_engine #(
    .N(N), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH),
    .PTAB_DEPTH(PTAB_DEPTH), .CLKS_PER_BIT(100), .TIMEOUT_CYCLES(TIMEOUT_CYCLES)
  ) u_engine (
    .clk, .rst, .rx_data, .rx_valid, .tx_data, .tx_valid, .tx_ready,
    .core_rst, .core_start, .core_start_pc, .core_abort,
    .core_state, .core_err, .core_pc, .core_halt_code,
    .host_mem, .host_rd_en, .host_wr_en, .host_addr, .host_wr_data, .host_rd_valid, .host_rd_data,
    .pkt_ok, .pkt_nak, .pkt_timeout, .rx_overflow
  );

  tpu_core #(
    .N(N), .IMEM_DEPTH(IMEM_DEPTH), .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH), .ACC_DEPTH(ACC_DEPTH),
    .PTAB_DEPTH(PTAB_DEPTH)
  ) u_core (
    .clk, .rst(rst || core_rst),
    .start(core_start), .start_pc(core_start_pc[$clog2(IMEM_DEPTH)-1:0]), .abort_req(core_abort),
    .state(core_state), .err(core_err), .pc(core_pc), .halt_code(core_halt_code),
    .host_mem, .host_rd_en, .host_wr_en, .host_addr, .host_wr_data, .host_rd_valid, .host_rd_data
  );

  logic unused_pc;
  assign unused_pc = ^core_start_pc[15:$clog2(IMEM_DEPTH)];
endmodule
