# TinyTPU — Agent Context

A minimal INT8 Tensor Processing Unit on a Digilent Basys3 FPGA. Hardware is written in SystemVerilog. Host tooling and tests are in Python.

> **Status:** early stage. Tooling, CI, `rtl/isa_pkg.sv`, the UART (`uart_rx`, `uart_tx`), `sync_fifo`, `crc16_ccitt`, `reset_gen`, the reference model `tinytpu.model`, the PE (`pe`), the systolic array (`systolic_array`), the accumulators (`acc_unit`), the requantizer (`requant`), the buffers (`buffer_ram`), the decoder (`instr_decode`), range checker (`range_check`), sequencer (`sequencer`), execution units (`matmul_ctrl`, `requant_ctrl`), the core without the host link (`tpu_core`), the packet engine (`cmd_engine`), the board top level (`tinytpu_top`), the host API (`tinytpu.device`), the compiler's float layers, quantizer and code generation (`tinytpu.nn`, `tinytpu.quant`, `tinytpu.compiler`), and the bring-up design (`loopback_echo`, `loopback_top`, which builds in Vivado) exist; see `TASKS.md` for progress. Anything below marked **(planned)** does not exist yet. Create it to match this file; don't assume it's already there. Items marked **OPEN** are undecided. Ask before choosing, and record the answer here.

## 1. Target hardware (fixed)

| Item | Value |
|---|---|
| Device | Xilinx Artix-7 `xc7a35tcpg236-1` |
| DSP | 90 × DSP48E1 (25×18 multiplier, 48-bit accumulator) |
| BRAM | 50 × RAMB36 = 1,800 Kb (each can split into 2 × RAMB18) |
| Clock | 100 MHz oscillator on pin `W5`. Single clock domain unless agreed otherwise. |
| UART | On-board FTDI USB-UART: `RsRx` = `B18` (FPGA input), `RsTx` = `A18` (FPGA output) |
| Reset | Center button `btnC` = `U18`. Synchronize it before use. |

All pins above, plus the 16 switches and 16 LEDs, were checked against Digilent's Basys-3-Master.xdc (commit `c8644a9`). The board constraints are in `syn/basys3.xdc`, which keeps Digilent's port names (`clk`, `btnC`, `RsRx`, `RsTx`, `sw[15:0]`, `led[15:0]`).

## 2. Numeric format

- Weights and activations: **8-bit integers**. Signedness is **decided ("D-lite")**:
  - Weights are always signed INT8. The host quantizer restricts weights to −127…127 (symmetric).
  - Activation signedness is selected per matmul by a mode bit `act_signed`: signed INT8 (−128…127) or unsigned UINT8 (0…255).
  - Under every allowed mode, every product fits in 16 signed bits. Unsigned × unsigned is not supported.
  - `act_signed` must not change while the array is busy. The formal tier asserts this.
- Accumulators are **decided: signed INT32, wrapping on overflow (no saturation).**
  - The width depends on K, the full inner dimension across all tiles, not on the array size N.
  - Inside the array, partial sums use the 48-bit DSP cascade. They are truncated to 32 bits at the column-bottom adders and in the accumulator buffer.
  - The host driver must reject any job where `K * 32385 + max|bias| >= 2**31`. The largest possible product is 127 × 255. This check guarantees wraparound never happens in valid use.
  - The NumPy reference model uses `np.int32` throughout.
- Output requantization is **decided (Option D)**:
  - **Unit:** one shared, pipelined requantization unit sits between the accumulator buffer and the output. It handles one value per cycle and uses about 2 DSP48E1s.
  - **Modes:**
    - **raw:** the host reads INT32 accumulators directly from the accumulator buffer while the sequencer is halted (debugging, host-side schemes).
    - **requant** (the `REQUANT` instruction): `y = clamp((acc * M + R) >>> s, lo, hi)`, where:
      - `M` is a 16-bit unsigned multiplier and `s` is a 6-bit shift (0–63). Both are set per output column (per-channel scaling) in the per-channel parameter table (§3).
      - `R = (s == 0) ? 0 : 2**(s-1)`, i.e. round half up.
      - The product `acc * M` is computed at full width (48 bits signed). Never truncate before the shift.
      - The clamp range comes from a ReLU bit: ReLU on gives UINT8 [0, 255], which feeds the next layer with `act_signed = 0`. ReLU off gives INT8 [−128, 127], which feeds the next layer with `act_signed = 1`.
  - **No zero-points:** quantization is symmetric throughout.
  - **No TFLite compatibility:** the design does not reproduce TFLite's Q31 rounding behaviour. The NumPy reference model implements exactly the formula above.
  - **Tests:** cover negative values, rounding ties, both clamp edges, and s = 0.
  - **Implemented (A3.5, `rtl/requant.sv`):**
    - One value per cycle (`in_valid`, `in_acc`, `in_mult`, `in_shift`, `in_relu`), no stalls; 6-cycle latency. `out_data` is the low byte of `y`, as written into ABUF.
    - A `TAG_W`-bit `in_tag` travels with each value to `out_tag`, so the controller can tell which column and row a result belongs to.
    - Rounding is folded into the shift: `(P + R) >>> s == ((2P >>> s) + 1) >>> 1` for every s (including 0 and s ≥ 48), so no variable R is built.
    - **Raw mode needs no hardware here:** the host reads ACC through `acc_unit`'s read port.
    - **Vivado OOC:** 2 DSP48E1 joined by 1 PCOUT→PCIN cascade, MREG = PREG = 1, 284 LUTs, 110 FFs, about +3.9 ns post-synthesis slack at 100 MHz.
- No floating point anywhere in the RTL.

## 3. Architecture

- **Dataflow:** weight-stationary 2D systolic array, N×N.
  - **Decided:** square N×N with a single parameter `N`. The build and synthesis target is **N = 8**: 64 processing-element DSPs plus 2 for the requantizer = 66 of 90.
    - No separate row/column parameters.
    - RTL must work for any N ≥ 2, but only powers of two are supported.
    - Partial sums cascade down columns over the DSP's dedicated PCOUT→PCIN connections.
  - Activations enter the array skewed diagonally: row *i* is delayed *i* cycles. Partial sums flow down the columns and come out skewed, so de-skew them before writing results.
- **Processing element (PE):** one MAC per DSP48E1.
  - Extend operands to 9 bits once, at the array edge, not inside each PE:
    - activations: `{act_signed & a[7], a}`
    - weights: always `{w[7], w}`
  - The 9-bit values travel through the PEs and are used as `signed` in the DSP.
  - Never mix signed and unsigned operands in one SystemVerilog expression.
  - Register inputs, product, and output so Vivado can absorb them into AREG/BREG/MREG/PREG.
  - `(* use_dsp = "yes" *)` is allowed when inference falls back to LUTs.
  - **Implemented (A3.2, `rtl/pe.sv`):**
    - Ports: `a_in` / `a_out` (horizontal, 1-cycle delay), `w_in` / `w_load`, `psum_in` / `psum_out` (48-bit, vertical).
    - Registers: `a_q`, `w_q`, `m_q = a_q*w_q`, `p_q = psum_in + m_q`. Latency: a_in to its product in psum_out is 3 cycles; psum_in to psum_out is 1 cycle. Hence the row skew of 1 cycle per row.
    - **Weight loading is decided: a per-row load enable on a shared column bus, not a shift chain.** On load cycle *i* the array raises `w_load` for row *i*, and every PE in that row captures tile row *i* from its column's `w_in`. That is the WBUF row order, and it lets the stationary weight live in the DSP's BREG.
    - `use_dsp = "yes"` sits on `m_q`/`p_q`, because without it Vivado builds the 9x9 multiplier from about 136 LUTs. The attribute goes on signals, not on `module`, because the doc header must sit directly above `module`.
    - **Vivado OOC result:** 1 DSP48E1, 0 LUTs, 9 FFs, AREG = BREG = MREG = PREG = 1. The activation register is duplicated: one copy is the DSP's AREG, the other is the fabric copy for `a_out`.
- **Systolic array (A3.3, `rtl/systolic_array.sv`):**
  - **Interface:**
    - `a_in` / `a_valid`: one activation vector per cycle, unskewed.
    - `w_in` / `w_load` / `w_row`: one weight-tile row per cycle.
    - `out_data` / `out_valid`: result vectors, de-skewed and truncated to 32 bits.
  - **Latency:** exactly 2N+1 cycles.
  - **Timing rules the controller must follow:**
    - A tile's rows are loaded 0..N-1 on consecutive cycles, and its vectors may start in the same cycle as row 0.
    - The next tile's row 0 must be loaded **at least N cycles** after the previous tile's last vector entered. `tb/test_systolic_array.py` shows N is the exact minimum: one cycle less corrupts a result.
  - **Vivado OOC at N = 8:**
    - 64 DSP48E1, 56 PCOUT→PCIN cascade links (all column partial sums on the dedicated cascade), BREG = MREG = PREG = 1.
    - AREG is 1 or 2: Vivado pulls one skew register into some DSPs, which is fine.
    - 11 LUTs and 871 FFs, the skew and de-skew delay lines.
    - The 64 `Synth 8-3936` warnings (each PE's sum trimmed from 48 to 32 bits) are expected: only 32 bits reach the output, and the cascade itself stays 48 bits.
- **Accumulators (A3.4, `rtl/acc_unit.sv`):**
  - **Matmul side:** one array result per cycle (`in_valid`, `in_addr`, `in_mode`, `in_data`, `in_bias`), applied by a 3-stage read-add-write pipeline: overwrite, bias, or accumulate, with 32-bit wraparound.
  - **No forwarding:** updates of the same row must be ≥ 3 cycles apart. The array's N-cycle tile spacing guarantees this for N ≥ 4. **At N = 2 the controller must insert spacing.** An assertion checks it.
  - **Other ports:** a read port (REQUANT/host, 1-cycle latency) and a write port (host), both forbidden while `busy` (asserted).
  - The controller must present `in_addr` / `in_mode` / `in_bias` in the same cycle as the array's `out_valid`, i.e. delayed by the array latency.
  - **Vivado OOC at N = 8, DEPTH = 512:** 4 BRAM tiles, 0 DSPs, 540 LUTs, 794 FFs. The one `Netlist 29-101` warning (flattened netlist not ideal for floorplanning) is harmless.
- **Memory:** separate buffers for weights, activations, and accumulators, plus the parameter table and instruction memory.
  - **No FIFOs between the buffers and the array (decided in A3.6).** The address generator is deterministic, BRAM reads have a fixed latency, and nothing downstream stalls, so the buffer read paths feed the array directly. A tag issued with each read comes out with its data and drives the array's `w_load`/`w_row`/`a_valid`.
  - **Sequencing lives in the address generator (A4.2), not in the buffers:** tile-row order, the N-cycle tile spacing, and the offset between WBUF and ABUF reads (below).
  - **`buffer_ram` (A3.6, `rtl/buffer_ram.sv`) is used for WBUF, ABUF, and PTAB:**
    - Simple dual-port BRAM: one read and one write port, each a 2:1 mux between datapath and host, selected by `host_sel`. Assertions check that the unselected side is idle and that no row is read in the cycle it is written.
    - Read latency is fixed at 1 + `RD_PIPE`. The datapath read's `TAG_W`-bit tag comes out on `rd_tag` with the data. `rd_tag` is 0 in every other cycle, including host reads, so its bits can drive load and valid signals directly.
    - `LANES` gives per-lane write enables. ABUF uses LANES = N, so REQUANT writes one byte of a vector per cycle and needs no packer. The lanes map onto the BRAM's own byte write enables.
    - **Settings at N = 8:**

      | Buffer | `RD_PIPE` | Read latency | BRAM tiles | LUTs | FFs |
      |---|---|---|---|---|---|
      | WBUF | 2 | 3 | 32 | 198 | 79 |
      | ABUF | 1 | 2 | 2 | 96 | 4 |
      | PTAB | 1 | 2 | 1 | 77 | 10 |

      - `RD_PIPE = 1` lets Vivado use the BRAM's output register.
      - WBUF's second stage registers the 32-way read mux, the path to watch once placed. The post-synthesis estimate meets 100 MHz with any setting.
    - **WBUF and ABUF latencies differ by 1:** the address generator issues a tile's vector reads 1 cycle after its first row read, so vector 0 enters the array with row 0. The next tile's first row read is at least N cycles after the previous tile's last vector entered the array. `tb/test_feeder.py` confirms this is exactly the minimum.
    - **PTAB holds one channel per row.** The address generator must collect a column tile's N biases into a register for `acc_unit.in_bias` before the first k-tile's results arrive. REQUANT reads one entry per value, matching the requantizer's one value per cycle.
  - **`sync_fifo` (A2.2):** parameterized `WIDTH` and power-of-two `DEPTH`, valid/ready on both sides, first-word fall-through, and a `count` output.
    - Its combinational read means it is for registers or LUT RAM, not BRAM.
    - `in_ready` depends only on full, even during a same-cycle read.
    - Reuse it for every small FIFO (e.g. the UART receive FIFO).
  - **Memory map is decided (J1.4); full spec in `docs/memory_map.md`.** Defaults at N = 8:

    | Memory | Depth | Width | Built from | BRAM36 |
    |---|---|---|---|---|
    | `IMEM` | 128 instructions | 128 b | LUT RAM | 0 |
    | `PTAB` | 512 channels | 54 b | BRAM | 1 |
    | `ACC` | 512 rows | 256 b | BRAM | 4 |
    | `ABUF` | 1024 vectors | 64 b | BRAM | 2 |
    | `WBUF` | 2048 tiles (16384 rows) | 64 b | BRAM | 32 |

    Total: 39 BRAM36 (78%).
  - All depths are parameters, and powers of two. Address-field widths are `$clog2(depth)`; counts are 12 bits.
  - WBUF and ABUF are N bytes wide: a weight tile loads in N cycles, one row per cycle.
  - **Byte order:**
    - Weight tile row *i* (multiplies activation element *i*) is physical WBUF row `tile*N + i`; byte *j* is output column *j*. So `W[k*N:(k+1)*N, n*N:(n+1)*N].tobytes()` is a tile.
    - Vector byte *j* = element *j*.
    - ACC word *j* = column *j*.
  - Each memory has a datapath port plus host access through a 2:1 mux. The host has access only while halted, so no arbitration is needed.
  - **Model limits:** weights ≤ 128 KB, total channels ≤ 512, ABUF ≤ 8 KB, per layer M·NT ≤ 512.
- **Host interface:** UART.
  - Baud rate is **decided: 1,000,000 baud**.
    - Set by the build-time parameter `CLKS_PER_BIT` (= 100 at 100 MHz). The rate is exact on both the FPGA and FTDI sides.
    - The receiver samples at mid-bit using a cycle counter. No 16× oversampling.
    - There is no hardware flow control (only TX and RX are wired). The protocol must ensure the host never sends faster than the FPGA can accept, e.g. by waiting for an acknowledgement per command or block. Keep a small receive FIFO.
    - cocotb tests override `CLKS_PER_BIT` with a small value (e.g. 8) for speed, plus at least one test at 100. Formal proofs use `CLKS_PER_BIT` of 4–8.
    - **Implemented (A2.1):**
      - `uart_tx`: valid/ready, back-to-back at exactly 10 bits per byte.
      - `uart_rx`: 2-flop synchronizer, half-bit start-bit check that rejects glitches, one-cycle `valid` or `frame_err` pulses, no back-pressure.
      - After a framing error, `uart_rx` waits for an idle (high) line before it looks for the next start bit.
      - Tested at `CLKS_PER_BIT` 8 and 100, including ±3% baud error at 100.
    - Host docs tell Windows users to set the FTDI latency timer to 1 ms (Device Manager → Port Settings → Advanced).
    - **Packet engine (A4.1, `rtl/cmd_engine.sv`):** the whole protocol on top of `tpu_core`'s host port.
      - **Receive path:** a 32-byte receive queue, then a parser: header → fixed fields → two evaluation cycles (NAK priority; the busy decision is latched) → write-through of WRITE / INFER rows → CRC.
      - **Memory traffic:** memories are runs of physical rows (WBUF tile = N rows; PTAB entry = 8 wire bytes for 54 bits).
      - **Replies:** a status or info block plus streamed rows, with the CRC computed as bytes go out.
      - **RUN / INFER:** they start the core and leave a pending reply, sent when the core stops.
      - **Duplicates:** a duplicate INFER answers from the remembered region of the original, never from its own payload (found by formal).
      - **Vivado OOC (N = 8):** 1,620 LUTs, 1,097 FFs, 0 DSPs / BRAM, +2.07 ns post-synthesis. The worst path is the argument checks in the evaluation cycle. With `tpu_core` (2,497 LUTs) that is about 4,100 of the 5,200-LUT ceiling.
      - **Formal (`formal/cmd_engine.sby`):** any byte stream, with an abstract core.
        - `prove` (k-induction, about 30 s) holds for all time:
          - reply framing (sync, code, NAK ⇒ LEN 0, LEN ≤ max, exactly LEN payload bytes)
          - a reply only for a completed request or a pending RUN / INFER
          - the inter-byte timeout
          - memories only while the core is stopped
          - a byte is dropped only when the queue is full
        - `bmc` (depth 30) adds the address range and a progress bound.
        - The harness invariants (TX byte accounting, remembered-state validity) are what make induction work; PDR did not converge.
        - `formal/crc16_stub.sv` replaces the CRC (XOR checksum, same timing): the real CRC is verified in `tb/test_crc16.py`, and relating CRC chains stalls every solver.
        - Parser state codes are repeated as numbers in the harness: yosys-slang crashes on hierarchical enum-member references.
        - For abc engines, techmap then `setundef -anyseq` in the script: variable part-selects leave x bits the AIG writer rejects.
      - **cocotb (`tb/test_cmd_engine.py`, through `tb/hdl/engine_tb.sv` with the real `tpu_core`):** every command and NAK code, PTAB reserved bits, RUN / INFER against `Machine`, duplicates, STATUS / ABORT / RESET while running, the timeout, noise, and back-to-back bytes.
      - **Found while testing A4.1:** `buffer_ram`'s `rd_tag` was not gated after a reset. Stale tags in the delay line drove the array's `a_valid` after a RESET in mid-MATMUL. `rd_tag` is now gated by the reset valid bit.
    - **Host link (A2.5):** `tinytpu.link.Link` implements the protocol's host rules: one request in flight, single `write()` per packet, resend with the **same** SEQ after a timeout, bad reply, or NAK BAD_CRC, 10 ms silence before each resend, and `RESET` + `LinkError` after 3 resends. Other NAKs are returned, not resent.
      - It is tested against `tb/fake_echo.py`, a Python model of the bring-up board with fault injection.
      - **Never change the port's `timeout` while a packet may be in flight (found on the board).** On Windows, pyserial applies each assignment by reconfiguring the port (`SetCommState`), which corrupts or drops bytes the FTDI chip is still sending. `Link` reads with the fixed `link.POLL` set at open and checks its own deadlines; `tb/test_link.py` asserts no assignment happens.
      - **Injectable clock (A5.1):** `Link(..., clock=...)` measures every deadline and quiet time with `clock()`; the default is `time.monotonic`. The end-to-end simulation passes the simulation time, so the real resend rules run there unchanged.
    - **Host API (A5.1, `tinytpu.device.Device`):** every command as a method, on a `Link`.
      - `write` / `read` take each memory's unit and split transfers into packets automatically.
      - Helpers load programs, weight tiles, vectors and per-channel parameters in canonical layouts, and read vectors and ACC rows back.
      - A NAK raises `DeviceError`.
    - **Board top level (A5.1, `rtl/tinytpu_top.sv`):** `reset_gen` → `uart_rx` → `cmd_engine` ↔ `tpu_core` → `uart_tx`. The engine's RESET also resets the core.
      - LEDs: heartbeat; RX / TX activity; running / halted OK / halted error; sticky NAK / timeout / framing / overflow; the ISA error code on led[15:12].
      - **Full build (A5.4, Windows: `.\syn\vivado.ps1 -Top tinytpu_top`):** timing met after routing at 100 MHz, WNS +1.075 ns, WHS +0.035 ns; 3,702 LUTs (17.8%), 4,185 FFs (10.1%), BRAM 39, DSP 66; DRC clean. The bitstream is `%LOCALAPPDATA%\tinytpu\vivado\tinytpu_top\tinytpu_top.bit`; program it with `.\syn\vivado.ps1 -Program -Top tinytpu_top`.
        - `syn/build.tcl` also writes the routed checkpoint (`<top>_routed.dcp`) and the 20 worst setup paths (`<top>_paths.rpt`).
        - **Host writes are registered (A5.4):** `cmd_engine` writes a completed row one cycle later, from flip-flops. The first routed build (WNS +0.731 ns) had all 20 worst paths from the engine's row-assembly logic to the write enables of WBUF's 32 BRAMs, 79% routing delay.
        - **Worst paths now:** the host read side into WBUF (the engine's TX state → read enable, and the sequencer's state → `host_sel` → the read-address mux), from +1.08 ns.
      - **Vivado OOC at N = 8, memory-map depths:** 3,904 LUTs (19%), 3,906 FFs (9.4%), DSP 66, BRAM 39, +1.5 ns post-synthesis. `syn/ooc.tcl` checks the ceilings (new `ceiling` table).
    - **End-to-end test (A5.1, `tb/test_tinytpu_top.py`):** board pins only. The host side is the real `Device` + `Link`, run as blocking code in a `cocotb.task.bridge` thread, on `SimSerial` (`tb/uart_bfm.py` models, `cocotb.task.resume`) with the simulation time as the link's clock.
      - Covers INFO, a two-layer network in both activation modes with sizes that aren't multiples of N (RUN and INFER, against the plain matmul / requant math), random matrices against `tinytpu.model`, RESET while running, and link recovery through the real engine (a corrupted request, lost WRITE and RUN replies; the resent RUN must not run twice).
      - Runs at `CLKS_PER_BIT` 8 (everything) and 100 (INFO and one network).
    - **Board tools (A6.2):** user guide `docs/host.md`.
      - **`python -m tinytpu.selftest`:**
        - checks: info, memories, matmul, network (RUN and INFER), errors (ISA errors 1 / 5 / 6, a NAK), speed
        - everything compared with `tinytpu.model`; shapes fit any reported depths
        - `--probe` finds the UART port with INFO, and recognises the bring-up design
      - **`python -m tinytpu.example`:** a commented matmul on the board, checked against NumPy.
      - **Tested in simulation:** both run against the simulated board in `tb/test_tinytpu_top.py`. The selftest must pass, and must report a wrong result as FAIL. Probe and CLI are tested in `tb/test_selftest.py` with fake ports.
      - **`--manual`:** interactive checks at the board: `leds` (errors 1 / 4 / 6 and OK on `led[4]`, `led[5]`, `led[15:12]`) and `button` (`btnC` returns the core to idle and keeps every memory). The simulation answers from the LED pins and pulses `btnC`.
      - **Board test plan:** `docs/board_test.md`, a step-by-step first session (H0.3, H0.4, H2.6, H5.5) with expected results.
    - **Fallback baud rate:** `syn/build.tcl` builds named variants (`variants` table; outputs named `<top>_<variant>`; `vivado.ps1 -Variant`). `115k` sets `CLKS_PER_BIT = 868` (115,207 baud) for both board tops. The host tools take `--baud`, and `Link.open` scales its default reply timeout to the baud rate. `tinytpu_top` passed end to end in simulation at 868. Variants are not compared with the baseline.
    - **Port discovery:** Windows can show **two** COM ports for the Basys3's FT2232H (JTAG and UART channels) with identical USB details, and other FT2232H boards match too.
      - Never pick one by position. Use `--port`, or `python -m tinytpu.bringup --probe`, which sends one ECHO to each and reports which answers.
      - Seen on the dev machine: COM6 and COM7, both `0403:6010`, serial "8".
  - Framing and error detection are **decided (Option D)**:
    - **Bytes:** 8N1. The receiver raises a framing-error flag when the stop bit is low.
    - **Packets:** every command or data packet ends with a CRC-16/CCITT-FALSE checksum, sent high byte first:
      - polynomial 0x1021, initial value 0xFFFF, no reflection, final XOR 0x0000
      - check value for ASCII `"123456789"` = 0x29B1
    - **Replies and recovery:** the receiver answers ACK or NAK, and the host resends on NAK. Packets carry a length field, and the receiver discards a partial packet after a timeout.
    - **Testing:** check the CRC module in cocotb against the Python implementation; it is not formally verified. Inject flipped bits, dropped bytes, and bad CRCs.
    - **Implemented (A2.3):** `rtl/crc16_ccitt.sv` handles one byte per cycle, with `clear` (which may carry the first byte) and a `zero` output. The Python reference is `tinytpu.crc` (`crc16`, `crc_bytes`, `check`).
      - **Packet check:** feed the CRC the packet bytes after the sync byte *and* the two CRC bytes, high byte first. The packet is intact exactly when `zero` is high; no separate compare is needed. A low-byte-first CRC fails this check.
      - `tinytpu.crc` is cross-checked against Python's `binascii.crc_hqx`.
    - **Packet format is decided (J1.3); full spec in `docs/uart_protocol.md`:**
      - **Shape:** request `A5 CMD SEQ LEN(u16) payload CRC`; reply `5A CODE SEQ LEN(u16) payload CRC`.
        - The CRC covers everything after the sync byte, and is sent high byte first.
        - All other multi-byte fields are little-endian.
        - Maximum payload is 4096 bytes.
      - **Duplicates:** the FPGA remembers the last executed `SEQ`. A repeat is answered without being re-executed.
      - **Write-through:** `WRITE`/`INFER` data streams straight into memory, with no packet buffer. A bad CRC gives a NAK, and the host resends. Nothing executes before its CRC passes.
      - **Timeouts and resync:**
        - On the FPGA, a gap of more than 5 ms inside a packet discards it, with no reply.
        - The host sends each packet in one `write()`, waits 10 ms of silence before any resend, and after 3 failures sends `RESET` and reports an error.
  - UART bandwidth, not compute, limits end-to-end throughput. Don't over-optimize the array at the expense of simplicity.
- **Control:** custom minimal ISA (load weights, load activations, matmul, store/read results).
  - **Execution model is decided: stored program.**
    - The host uploads a program into an on-chip instruction memory (BRAM), plus the model's weights, then sends RUN. An on-chip sequencer executes the program until HALT, without host round trips.
    - Goal: real-time inference, where only each inference's input and output cross the UART, and weights stay resident on-chip.
    - Instructions execute in order, one at a time; each completes before the next starts. No pipelining between instructions, so no hazards between them.
    - **Control flow is decided: macro (tiled) instructions in a straight-line program.**
      - The program counter only advances, until HALT. There are no branches, loops, or scalar registers.
      - One MATMUL instruction covers a whole layer. Its fields give base addresses (weights, activations, accumulators), K-tile and N-tile counts, the number of activation rows, and accumulate/overwrite.
      - A hardware **address generator** runs the nested tile loops for each instruction.
      - Data-dependent steps (e.g. argmax) are done on the host.
    - **Encoding is decided: fixed 128-bit instructions**, one per instruction-memory row, sent to the FPGA as 16 bytes.
      - Reserved bits must be zero.
      - An illegal opcode or a non-zero reserved bit halts the sequencer with an error status and the faulting PC.
      - Instruction-memory depth is a parameter, default 128. Whether it is LUT RAM or BRAM is decided in J1.4.
      - Field positions are fixed in `docs/isa.md`.
    - **Addressing is decided: canonical layouts, N-relative units, base addresses only (no strides).**
      - **Units:** weights in N×N tiles, activations in N-element vectors, accumulators in rows of N × INT32.
      - **Layouts**, with KT = K/N and NT = Nout/N:
        - weights `w_base + n*KT + k`
        - activations `a_base + m*KT + k`
        - accumulators `acc_base + m*NT + n`
      - The host compiler rearranges data into these layouts on upload.
      - **Padding:** the host zero-pads K and Nout to multiples of N, in weights and biases. M is not padded. Padded columns stay 0 through requantization.
      - A requantized accumulator row `(m, n)` is exactly one activation vector at `m*NT + n`, i.e. already the canonical activation layout for the next layer, whose KT is this layer's NT.
      - Stride fields may be added later as a new opcode variant; programs that don't use them are unaffected.
    - **Instruction set is decided:**
      - **Opcodes:** `0x00 HALT` (slot 1 = 16-bit halt code; all-zero memory therefore halts), `0x01 NOP`, `0x10 MATMUL`, `0x20 REQUANT`. All others are illegal.
      - A typical layer is `MATMUL` (bias mode) + `REQUANT`.
      - **Layout:** eight 16-bit slots, sent least-significant byte first.
        - Slot 0: `[7:0]` opcode, `[15:8]` flags.
        - Slots 1–7: `[31:16]` `w_base` or halt code; `[47:32]` `a_base` (MATMUL source) or `a_dst` (REQUANT destination); `[63:48]` `acc_base`; `[79:64]` `p_base`; `[95:80]` `kt`; `[111:96]` `nt`; `[127:112]` `m`.
        - MATMUL flags: `[8]` act_signed, `[10:9]` acc_mode (00 overwrite, 01 accumulate, 10 bias, 11 illegal). REQUANT flags: `[8]` relu.
        - Field bits beyond what a buffer needs, and unused slots or flags, are reserved and must be zero.
      - **Per-channel parameter table:** entries of `{bias INT32, M u16, s u6}`, at `p_base + n*N + col`.
        - MATMUL bias mode starts the first k-tile from `bias[col]`.
        - REQUANT reads `M`/`s` from the same entries.
      - **Write-back (closes J1.2):** REQUANT writes 8-bit results into the activation buffer at `a_dst + m*NT + n`. That is its only destination. The next layer reads from there, and the host reads final outputs from there.
      - **Host I/O is not part of the ISA.** While the sequencer is halted, the host reads and writes buffers directly through protocol commands (raw INT32 results are read straight from the accumulator buffer).
      - **Host commands (decided; byte layout is J1.3):**
        - `INFO`: protocol version, N, `CLKS_PER_BIT`, buffer depths. The host compiler checks a model fits before uploading.
        - `WRITE` / `READ` of the memories (instruction memory, weights, activations, accumulators, parameter table), in each memory's ISA unit.
        - `RUN` (start PC): replies once the program halts, with the final status.
        - `STATUS`, `ABORT`, `RESET`: `RESET` returns to idle and keeps memory; `btnC` does the same.
        - **`ABORT` is decided (A4.2):** it stops issuing work at once, lets issued datapath work drain, then goes idle with error 0 and PC = the instruction that was running. No new error code.
        - `INFER`: writes the input, runs, and replies with status plus output data, in one round trip. It can be built after the other commands work.
        - **While running:** only `STATUS`, `ABORT`, `INFO` and `RESET` are accepted (decided in A4.1: RESET too, as the ISA table said). Anything else gets a NAK with a busy code. The host never accesses a memory while the sequencer runs.
        - **Decided in A4.1** (full rules in `docs/uart_protocol.md`):
          - An ABORT or RESET reply replaces a pending RUN/INFER reply, so every request gets at most one reply.
          - A READ/INFER reply over 4096 bytes is NAK 0x05.
          - A PTAB entry with non-zero reserved bits is skipped and the reply is NAK 0x05; earlier entries may already be written.
          - A duplicate RUN/INFER whose program still runs re-arms the pending reply.
          - NAK priority: CRC, unknown, too long, length, duplicate rule, busy, argument.
          - Only WRITE/RUN/INFER/ABORT update the remembered SEQ.
      - **Error halts** record an error code and the faulting PC, for any of:
        - illegal opcode, non-zero reserved bit, acc_mode 11, or a count of 0
        - any generated address at or beyond its buffer's depth
        - the PC running past the end of instruction memory
      - **Address-range errors (5) are checked before an instruction starts (decided, A4.2):** from the highest address per memory (table in `docs/isa.md`), so a faulting instruction writes nothing, in the hardware and in `tinytpu.model`. The address generators therefore never produce an out-of-range address.
    - The sequencer (program counter, fetch, decode) and the address-generator FSM are in scope for the formal tier, in particular loop bounds and termination. The address arithmetic itself is checked in cocotb.
    - **Implemented (A4.2, part 1):**
      - **`instr_decode`:** combinational errors 1–4, in the reference order. Tested against `tinytpu.isa.decode_error` with every single-bit flip of legal words, at default and small depths.
      - **`range_check`:** error 5 before execution, using `tinytpu.model.range_error`'s condition `base + product > depth`.
        - The products `nt·kt`, `m·kt`, `m·nt` are shift-and-add over 12 cycles, so it uses no DSPs. Latency is COUNT_W + 2 = 14 cycles.
        - Operands follow the instruction while idle, and the compare is registered; both are for timing.
      - **`sequencer`:** fetch over a valid handshake (any IMEM latency), decode, range check, then dispatch with `exec_start` / `exec_instr` / `exec_done`.
        - It also handles HALT, NOP, PC overrun (6), and ABORT via `exec_abort` / `exec_idle`.
        - `exec_instr` is stable from `exec_start` to `exec_done`; this is how "`act_signed` is stable while busy" is guaranteed.
        - The port is `abort_req`, because Verilator rejects `abort` (a C++ word).
      - **Formal (`formal/sequencer.sby`):** proved for any program (IMEM returns solver-chosen words), with an execution unit that finishes or drains within 3 cycles. Properties:
        - fetch order: the first fetch is at `start_pc`, then +1 each time
        - dispatch only when idle and never during ABORT
        - `exec_instr` stable while executing
        - correct PC and status on every stop
        - ABORT ends idle with error 0
        - per-instruction and ABORT cycle bounds; with the PC bounded by IMEM depth, these give termination
      - **`formal/range_check_stub.sv`** replaces `range_check` in that proof: same timing, and the result is the instruction's parity (solver-controlled).
        - Don't use `(* anyseq *)` with yosys-slang. The attribute is ignored, and Yosys then folds the undriven X into each FSM bit independently, producing impossible transitions.
      - **cocotb (`tb/test_sequencer.py`):** environment models with IMEM latency 1 and 3. Every run's executed-instruction trace and final status must equal `tinytpu.model.Machine`'s, covering every error code, ABORT at every cycle of a program, and reset.
      - **Vivado OOC:** `sequencer` (with decoder and range check) has 0 DSPs, 362 LUTs, about 400 FFs, and +3.3 ns post-synthesis slack. The worst path is decode → halt-code load.
    - **Implemented (A4.2, parts 2–4):**
      - **`matmul_ctrl`** (MATMUL address generator):
        - Loop order n, k, m. Tile period `T = max(3, N + M − 1)`: rows at c = 0…N−1, vectors at c = D…D+M−1 (D = WBUF − ABUF latency = 1). This is the array's exact rule plus `acc_unit`'s 3-cycle spacing, which binds only at N = 2.
        - Addresses are incremental, with no multipliers.
        - Each vector pushes {ACC row, mode} into a `sync_fifo` (depth 4N); each array result pops it, so `acc_unit` is driven at whatever the array latency is.
        - Bias mode: drain, read the N PTAB biases (tag = column), then start the column tile; the biases arrive long before the first result (checked: P_LAT ≤ W_LAT + 2N).
        - `done` waits until ACC is written. ABORT stops issuing and drains.
      - **`requant_ctrl`:**
        - Loop order n, m, col, one value per cycle.
        - Each cycle it reads the ACC row and the column's PTAB entry, tagged {ABUF vector, col}. The selected ACC value is delayed to meet the PTAB data, then goes through `requant`.
        - Each result byte is written into its ABUF lane (byte write enable), so no packer is needed.
      - **`tpu_core`:** sequencer + both units + IMEM (`buffer_ram` with `LUTRAM = 1`) + WBUF / ABUF / PTAB + array + `acc_unit`, behind one host port (`host_mem`, `host_addr` = physical row, right-aligned data, fixed per-memory read latency), allowed only while stopped (asserted).
        - PTAB is shared by both units. Its read tag carries the owner, so a MATMUL bias read can never reach the requantizer.
      - **Formal (`formal/matmul_ctrl.sby`):**
        - `prove` / `prove_n4` (k-induction, N = 2 and 4): array timing at the array's inputs, tile period, and nothing issued from the cycle of an ABORT request.
          - PDR did not converge in 15 minutes. Harness invariants tying the unit's counters to the work in flight make k-induction pass in seconds.
          - Use hierarchical references (`dut.c`) for such invariants; yosys-slang accepts them.
        - `bmc` (from reset, depth 72): exact read and result counts, per-tile bias values, and `done` / ABORT cycle bounds. These relate counters to products of the counts, which induction and PDR can't capture.
      - **cocotb:**
        - `test_matmul_ctrl.py` / `test_requant_ctrl.py`: each unit on the real datapath, every ACC / ABUF word compared with `Machine`.
        - `test_tpu_core.py`, end to end:
          - a two-layer network with sizes that aren't multiples of N, also checked against the plain math
          - the exact `docs/isa.md` program at N = 8
          - random programs, including error halts, where memories also match
          - ABORT followed by a normal run
      - **Vivado OOC at N = 8, memory-map depths:**
        - `matmul_ctrl`: 0 DSPs, 257 LUTs, +3.9 ns. `requant_ctrl`: 2 DSPs, 508 LUTs, +3.9 ns.
        - **`tpu_core`:** exactly 66 DSPs (57 cascade links) and 39 BRAM tiles, 2,495 LUTs (12%), 2,668 FFs (6.4%), +2.97 ns post-synthesis slack. This is the first measurement of the whole core.
  - The single source of truth is `docs/isa.md`. It has full semantics, the error codes (1–6, confirmed in J1.3), and a worked example program. The summary below must stay consistent with it.
  - Don't change opcodes or encodings without asking.
  - The matmul instruction carries an `act_signed` mode field (see §2).
  - Functional tests run in both activation modes.
- **Host compiler (planned, Phase 7 in `TASKS.md`).** Scope decided 2026-11-16:
  - **Input:** the project's own Python layer API: a list of `Dense` layers with float NumPy weights and biases and an optional ReLU, plus calibration inputs. The compiler quantizes. Pre-quantized integer layers are accepted too, so tests can skip quantization. No framework dependency; importers (PyTorch, ONNX) may be added later on top.
  - **Layers:** Dense + optional ReLU only. Each hidden layer is one `MATMUL` (bias mode) plus one `REQUANT`. No convolutions: the ISA has no strides.
  - **Output:** the last layer is `MATMUL` only, and `INFER` reads its raw INT32 logits from `ACC`. The host dequantizes them, and does argmax.
  - **Demo:** MNIST, with a small MLP trained in pure NumPy. The dataset is downloaded once into a cache outside the repo. Tests and CI never download it.
  - **Design approved 2026-11-16 (J7.1); full spec in `docs/compiler.md`:**
    - Quantization: per-output-channel weight scales (`max|w| / 127`), one activation scale per layer input from calibration (maximum by default, percentile optional), UINT8 after a ReLU and INT8 otherwise. `M`/`s`: the largest `s` ≤ 63 with `round(r · 2**s)` ≤ 65535. Float-to-integer rounding is ties away from zero.
    - Memory: weights and PTAB packed by layer from 0; every layer uses ACC from row 0; ABUF alternates between two regions, with the input in the first.
    - Batch size is fixed at compile time; the default is the largest that fits. Version 1 runs every batch with one `INFER`.
    - The `HALT` code is a 16-bit hash of the compiled model, checked after every `INFER`.
    - A compiled model is one `.npz` (`allow_pickle=False`) with a JSON metadata string.
    - MNIST demo: 784-128-10 (196-64-10 optional), downloaded from torchvision's mirror `https://ossci-datasets.s3.amazonaws.com/mnist/` with pinned SHA-256 hashes.
  - **Implemented (A7.2):** `tinytpu.nn` (`Dense`, `Model`) and `tinytpu.quant` (`quantize`, `QModel`, `QLayer`, `mult_shift`, `round_half_away`).
    - `QModel.forward_int` chains `model.matmul` / `model.requant`: the integer reference that compiled programs must match bit-exactly. `QModel` can also be built from integer layers directly.
    - **Never round through an int64 cast of a large float.** `r · 2**63` overflowed and wrapped negative, so `M` passed the `≤ 65535` test (found by the tests). `mult_shift` works in Python integers, and `round_half_away` rejects magnitudes of `2**62` or more.
    - **Never round with `floor(x + 0.5)`:** the addition itself rounds `0.49999999999999994` up to 1. Use the fractional part, `x - floor(x) >= 0.5`, which is exact.
  - **Implemented (A7.3):** `tinytpu.compiler`: `Target` (N, depths, max payload; `default(n)` or `from_info(info)`), `compile_model`, `max_batch`, `check_fit`, `FitError` (lists every exceeded limit), `Compiled` (program, tiles, PTAB arrays, I/O addresses, tag, scales).
    - Generated words are checked with `isa.decode_error` and `model.range_error` before they are returned.
    - `tb/test_compiler.py` runs images on `model.Machine`; for 784-128-10 at N = 8 the default batch is 5 (1,600 tiles, 144 channels), as `docs/compiler.md` says.
    - ABUF regions are reused by design: after a run, each region holds only the last hidden layer written to it.
  - **Implemented (A7.4):** the runtime, in `tinytpu.compiler`.
    - `Compiled.load(dev_or_machine)` writes the image and returns a `Runner`; `Compiled.simulator()` does the same on a fresh `model.Machine`. Both backends share one interface, which is how board and reference results are compared.
    - `Runner.infer` / `infer_int` split any number of rows into batches (the last one zero-padded), and raise `RunError` unless every run halts OK with the program's tag. `Runner.verify()` reads the image back.
    - `Compiled.save` / `from_file`: one `.npz`, `allow_pickle=False`, format 1; loading recomputes the tag and refuses a corrupted file.
    - `quant.quantize_input` / `quant.dequantize` are shared by `QModel` and the runtime, so the two can't drift apart. Inputs are clipped before rounding, so huge values saturate instead of failing.

## 4. Repository layout (planned)

```
rtl/          SystemVerilog sources (one module per file, file name = module name); rtl/isa_pkg.sv
tb/           cocotb testbenches (test_*.py) + Python reference models; tb/hdl/ test-only HDL wrappers;
              tb/uart_bfm.py (UartSource/UartSink serial-line models for end-to-end tests)
host/tinytpu/ Python package `tinytpu` (installed editable by uv sync): isa.py, crc.py, packet.py,
              link.py (serial link with resend rules), device.py (host API: every command),
              bringup.py (loopback board test CLI), selftest.py (full-design board
              self-test CLI), example.py (example matmul script), baseline.py (build
              metrics vs syn/baseline.json),
              model.py (bit-exact reference model); planned (Phase 7): nn.py (float layer
              API), quant.py (quantizer), compiler.py (code generation + runtime), mnist.py (demo)
formal/       SymbiYosys .sby files and property wrappers
syn/          Vivado: basys3.xdc, build.tcl (batch build of a top: reports, metrics JSON, ceiling
              check, bitstream), program.tcl (JTAG), ooc.tcl (per-module OOC synthesis + resource
              expectations), vivado.ps1 (Windows wrapper; runs the baseline check), baseline.json
              (approved in J5.3)
docs/         MkDocs site (architecture, memory map, ISA, UART protocol) + gen_rtl_docs.py
tools/        versions.env (pinned tool versions), setup notes
pyproject.toml, uv.lock   Python 3.12 project and locked dependencies
mkdocs.yml    MkDocs config
```

Keep generated output out of git: `sim_build/`, `*.vcd`/`*.fst`, `results.xml`, `formal/*/` run dirs, Vivado `.Xil/`, `*.jou`, `*.log`, `*.dcp`, report dumps, `site/`, `docs/rtl/` (generated), `.venv/`.

## 5. Verification (three tiers, planned)

**Entry points:** the top-level `Makefile`, run in WSL. Run `make help` for the list.
- `make lint`: the `rtl/` packages on their own, then every `rtl/` module, through Verilator `-Wall` and yosys-slang with `ASSERTIONS` defined. Must pass with zero warnings.
- `make sim`: tier 1 at N = 2 and 4. Add `K=<pytest filter>` to select tests.
- `make sim-full`: tier 1 at N = 8.
- `make formal`: tier 2.
- `make docs`, `make all`, `make clean`.

The Makefile puts the OSS CAD Suite `bin/` on `PATH`, and runs Python tools via `uv run --locked`. It refuses to run if the installed suite release doesn't match `tools/versions.env`.

**CI:** `.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual dispatch.
- It runs the same `make` targets:
  - `rtl` job: `lint`, `sim`, `formal`
  - `docs` job: `docs`
- It reads the OSS CAD Suite release and uv version from `tools/versions.env`, so changing a pin there changes CI too.
- A manual run with `full = true` also runs `make sim-full` (N = 8). Use it before merging changes to the array, feeders, or accumulators.
- Tier 3 (Vivado) never runs in CI.

**File conventions (`make` discovers these automatically):**
- `rtl/*_pkg.sv` are packages; they are compiled first.
- `tb/test_*.py` are pytest files. Each one builds with `cocotb_tools.runner` and runs its own cocotb tests. It reads the array size from the `TINYTPU_N` environment variable, passes it as the `N` parameter, and uses the build directory `sim_build/<name>_N<n>`.
- `formal/<name>.sby` lists sources as `../rtl/...` in `[files]`.
- **All cocotb tests in one file share one simulation.** DUT state carries over from test to test, so each test must start by resetting the DUT. Only a deliberate power-up test (placed first) may skip that.
- **End-to-end tests** drive a top level only through its board pins, using `tb/uart_bfm.py` and the host's own `tinytpu.packet`, so they exercise the same path as the board.

**Lint exceptions (no others without asking):**
- **Formal `past_valid` idiom:** its declaration-with-initializer triggers Verilator `PROCASSINIT`. Wrap only that declaration in `/* verilator lint_off PROCASSINIT */ ... /* verilator lint_on PROCASSINIT */`.
- **Packages:** wrap a package's body in `/* verilator lint_off UNUSEDPARAM */ ... lint_on`. Package constants exist for importers, and any one module uses only some of them.

**ISA consistency:**
- `rtl/isa_pkg.sv` and `host/tinytpu/isa.py` both mirror `docs/isa.md`.
- `tb/test_isa.py` checks, with pyslang and Verilator, that every constant, enum, struct layout, and the doc's example program agree. A change to one of the three must update all of them in the same commit.
- `tinytpu.isa.decode_error()` is the reference model for the hardware decoder's legality checks (ISA errors 1–4).

1. **Functional: cocotb + Verilator**
   - Reference results come from NumPy using the same integer semantics as the RTL (same signedness, accumulator width, and requantization).
     - **The reference is `tinytpu.model` (A3.1):**
       - `matmul` / `requant` / `wrap32` for datapath tests.
       - `Machine` runs ISA programs instruction by instruction, with canonical layouts, all three acc modes, write-back, and every error halt, for sequencer and end-to-end tests.
       - `pack_weights` / `pack_activations` / `unpack_*` for memory images.
     - Never re-derive expected values in a testbench; call the model.
     - The model itself is tested against naive pure-Python references in `tb/test_model.py`, including the `docs/isa.md` example program. If the ISA changes, the model changes in the same commit.
   - Compare bit-exactly; no tolerance.
   - Include edge cases: all −128 or all 127, zeros, identity matrices, random.
   - Array-size coverage:
     - routine runs at N = 2 and N = 4
     - full suite at N = 8, before merging any change to the array, feeders, or accumulators
   - The host driver pads and tiles matrices to multiples of N, and tests cover dimensions that are not multiples of N.
2. **Formal: SymbiYosys, control logic only**
   - In scope: FIFOs, UART, the instruction decoder and FSMs. Never the MAC datapath.
   - Use `assert property`, `assume`, and `cover`.
   - Required UART properties:
     - the framing-error flag rises when the stop bit is low
     - the receiver never drops a byte while its FIFO has space
   - **UART proof (done, `formal/uart.sby` + `formal/uart_loopback.sv`):** `uart_tx` drives `uart_rx` over a line where the solver can force any stop bit low and hold a break for 0–3 extra bits.
     - Proved: good frames arrive as exactly the byte sent; corrupted frames give `frame_err` and never a byte; every frame gives its pulse within a fixed bound; no pulse without a frame in flight; plus each module's own assertions.
     - The FIFO half of "never drops a byte" is proved with `sync_fifo` (A2.2, below) and the packet engine (A4.1).
   - **FIFO proof (done, `formal/sync_fifo.sby` + `formal/sync_fifo_formal.sv`):** proved at depths 2, 4, and 8.
     - Uses the **tracked-element** method: the solver marks any one accepted write, and the harness proves that the read removing it returns exactly that value at exactly its queue position. So no entry is ever lost, corrupted, duplicated, or reordered.
     - The module's own assertions add the occupancy model, full/empty, and `ready`/`valid`.
     - Use the same tracked-element pattern for any future queue or buffer.
     - For PDR, designs with memory arrays need `memory_map` after `prep` in the `.sby` script.
   - **Harnesses and engines:**
     - Multi-module properties live in harness modules in `formal/`, which the `.sby` file names as top. Harnesses are formal-only, so their `assume`s need no `ifdef`.
     - Prove with `abc pdr`: it proves for all time without hand-written induction invariants. Cover with `smtbmc`.
     - Use the smallest legal parameters (e.g. `CLKS_PER_BIT = 4`).
   - **Check every proof and testbench against injected bugs** before calling it done: break the RTL in a few plausible ways and confirm both formal and simulation fail. A proof that never fails proves nothing.
     - Never run other tests while a bug is injected. Sources are edited in place, so a simulation or test started at the same time can import the broken file and report a false failure (seen in A7.4).
  - For registered outputs, also check that driving new inputs does *not* change the outputs before the clock edge. Otherwise a combinational pass-through bug can look correct.
   - **Parameter-dependent assertions:** an assertion that is always true for some parameter values (e.g. a counter range check when the limit is a power of two) triggers Verilator `-Wall` warnings in test builds. Put it in a generate-`if` that only exists when it can fail.
   - **Decided:** the formal front-end is **yosys-slang**, which reads SystemVerilog through the slang compiler and is bundled in the OSS CAD Suite.
     - `.sby` files load the plugin with `plugin -i slang` and read sources with `read_slang`, instead of `read -formal`.
     - If a module hits a yosys-slang bug, it may fall back to Yosys's built-in reader. Note why in the `.sby` file.
   - **Property style:** write properties without SVA sequence operators (`|->`, `|=>`, `##`). Use clocked assertions with `$past`, `$stable`, `$rose`, and `$fell`, e.g. `always_ff @(posedge clk) if (!rst && $past(!rst) && $past(cond)) assert (expr);`. This keeps them valid in yosys-slang, Yosys's built-in reader, and Verilator.
   - Verilator runs with `--assert`, so every property is also checked during cocotb simulation.
3. **Implementation metrics: Vivado TCL, run locally or on a self-hosted runner**
   - Vivado is too large to run on GitHub-hosted CI.
   - Run out-of-context `synth_design`, then `report_utilization`, `report_timing_summary`, and `report_power`.
   - **Per-module OOC checks (`syn/ooc.tcl`, Windows: `.\syn\vivado.ps1 -Ooc -Top <module>`):**
     - Synthesizes one module out of context and writes `<top>_ooc.json` with cell counts (LUT, FF, DSP, BRAM) and the DSP register settings (AREG, BREG, MREG, PREG).
     - Fails if the module's entry in the script's `expect` table does not match.
     - Every datapath module that must map to DSPs gets an entry. Cells are counted by `REF_NAME` (`LUT*`, `FD*`, `DSP48E1`, `RAMB*`).
     - A named configuration (e.g. `wbuf`, `abuf`, `ptab`) synthesizes a module with given parameters (`module` and `generics` tables).
     - `WNS` is the post-synthesis worst setup slack, with zero input and output delay on the ports. `SLACK_MET` is 1 when it is ≥ 0. It is an estimate only; real timing comes from the full build.
     - `BRAM` is in tiles (RAMB36 = 1, RAMB18 = ½), matching the memory map.
     - `CASCADE` counts DSPs whose PCIN is driven by another DSP's PCOUT. Nets that cross module boundaries must be followed with `get_nets -segments` and `get_pins -leaf`, or the count is wrongly 0.
   - **Resource budgets are decided (fixed ceilings + baseline check).** Measure at N = 8 with no debug cores (ILA) in the design.
     - **Hard ceilings:**
       - LUT ≤ 25% (5,200)
       - FF ≤ 15% (6,240)
       - BRAM36 ≤ 80% (40). Expected exactly 39 (J1.4 memory map). It was raised from 50% so weights can stay on-chip; the remaining ~10 blocks are for ILA debug cores.
       - DSP **exactly 66**: 65 means a MAC fell into LUTs; 67 or more means an unexpected DSP was inferred
       - worst setup slack ≥ 0 at 100 MHz
     - **Implemented (A5.2):**
       - **`syn/build.tcl`** writes `<top>_metrics.json` after routing:
         - `lut` (Slice LUTs), `lut_memory` and `ff` (Slice Registers), parsed from `report_utilization`: the device's own counts, which the percentage ceilings refer to. LUT cells before packing (`lut_cells`, informational) are higher, about 4,060 vs 3,700.
         - `dsp` and `bram` (tiles), counted like `ooc.tcl`
         - `wns_ns`, `whs_ns`
         - `power_total_w` and `power_dynamic_w`, parsed from `report_power` (vectorless)
         - plus `top`, `part` and the Vivado version
       - For `tinytpu_top` it fails, with no bitstream, on a ceiling breach (LUT > 5,200, FF > 6,240, BRAM > 40, DSP ≠ 66) or negative slack. The metrics file is written first, so a failing build still leaves its numbers behind.
       - **`python -m tinytpu.baseline <metrics.json>`** applies the regression rules below and prints a table. `--create REASON` writes a baseline, and refuses to replace one without `--force`. It is tested in `tb/test_baseline.py`.
       - **`syn\vivado.ps1 -Top tinytpu_top`** runs that comparison after the build, once `syn/baseline.json` exists. It skips the comparison, printing the command instead, if `UV_PROJECT_ENVIRONMENT` isn't set: uv must not create a Windows venv in the repo's `.venv`.
     - **Regression check** against the committed `syn/baseline.json`:
       - fail if LUT or FF grows more than 10%, or if the DSP or BRAM count changes at all
       - create the baseline at the first full synthesis; update it only deliberately, in its own reviewed change, with the reason in the commit message
     - **Per-module figures:** report them via `report_utilization -hierarchical` for diagnosis only, not as pass/fail.
     - The ceilings deliberately leave room for ILA debug cores.
   - Optional: sweep N = 2, 4, 8 and report LUT, DSP, Fmax, and power against N.
   - Without simulation activity data, `report_power` is only a vectorless estimate. Use it to compare design versions, not as an absolute figure.

## 5a. Platform and tool versions (decided)

The development machine runs Windows with WSL2 (Ubuntu 26.04). Each tool runs on the side where it is best supported. Exact versions are recorded in `tools/versions.env` **(planned)**. Check that file instead of assuming versions. Upgrades are deliberate, single changes.

| Side | Runs | Pinned version |
|---|---|---|
| **WSL2 (Ubuntu)** | Tier 1 (cocotb + Verilator), Tier 2 (SymbiYosys + yosys-slang), MkDocs, CI parity | OSS CAD Suite **dated release**, recorded at install. CI downloads the same date. |
| **Windows** | Tier 3 (Vivado TCL), bitstream build, board programming, host driver (`host/`, pyserial on a COM port) | Vivado **2025.2** at `C:\AMDDesignTools\2025.2\Vivado`. 2026.1 is also installed; don't use it unless the pin is changed. |
| **Both** | Python | **3.12**, installed per project with `uv` (`uv python install 3.12`). Packages are locked in `uv.lock`. Don't use the system Pythons (3.14 in WSL, 3.15 on Windows). |

- **Repo location:** inside the WSL filesystem, so simulation and formal runs aren't slowed by cross-filesystem I/O. Windows tools reach it via `\\wsl.localhost\Ubuntu\home\<user>\...`.
- **Vivado and UNC paths (confirmed in J0.7):** Vivado 2025.2 reads sources and constraints directly from `\\wsl.localhost\...`; no drive letter is needed.
  - Run directories go to a local Windows disk, never into the repo: `syn/vivado.ps1` uses `%LOCALAPPDATA%\tinytpu\vivado\<top>` and runs Vivado from there, so `.Xil/` and logs land there too.
  - `syn/build.tcl` takes the top and the output directory as arguments, and fails if setup or hold timing is not met.
- **Use the OSS CAD Suite tools only, with a project-only PATH:**
  - The suite is installed at `~/tools/oss-cad-suite` and is **not** on the default shell `PATH`.
  - WSL also has Ubuntu's own packaged Verilator (5.032 in `/usr/bin`). Don't use it.
  - In WSL, run `source tools/env.sh` first. It puts only the suite's `bin/` first on `PATH` and warns if the installed release differs from `tools/versions.env`. Makefile targets do the same.
  - Never source the suite's own `environment` script, and never add its `py3bin/`: that would put the suite's Python 3.11 first. The project uses Python 3.12 via `uv`.
- **Formal reset assumption:** every `.sby` proof must assume reset on the first cycle (e.g. `always_comb if (!past_valid) assume (rst);`). Otherwise the solver starts from arbitrary register states and reports failures that can't happen in hardware.
- **Guard properties by where they run:**
  - Put `assert` and `cover` under `` `ifdef ASSERTIONS ``, which both the `.sby` files and the Verilator builds define.
  - Put `assume` statements about the environment, including the reset assumption, under `` `ifdef FORMAL ``, which only the `.sby` files define.
  - Reason: Verilator with `--assert` checks `assume` like `assert`. The reset assumption fails at time 0 in simulation, before the testbench drives `rst`. This was seen in the setup smoke test.
- **Commands say which side they run on**, e.g. "WSL: `make sim`", "Windows: `vivado -mode batch -source syn/ooc.tcl`".

## 6. Coding guidelines

- SystemVerilog IEEE 1800-2012, with one coding standard for all RTL. Packages, enums, and packed structs are encouraged; e.g. an `isa_pkg` holds opcode enums and instruction structs, shared by the decoder and the testbenches. Code must be accepted by both Verilator and yosys-slang (see §5).
- Parameterize array size and data and accumulator widths. Derive dependent widths with `$clog2`, not magic numbers.
- Keep datapath and control in separate modules.
- Use synchronous, active-high internal reset `rst`. Synchronize the button input once at the top level.
  - `rtl/reset_gen.sv` does this, and also gives a power-on reset with no initial values: it relies on 7-series flip-flops starting at 0 after configuration. Every top level uses it.
- **Unused inputs or outputs:** name the sink signal with `unused_` (e.g. `assign unused_sw = ^sw[15:2];`). Verilator's default `--unused-regexp` then skips it. This is a naming convention, not a lint exception. Vivado still reports unused top-level ports as warnings; that is expected.
- **Bring-up design:**
  - `rtl/loopback_echo.sv` and `rtl/loopback_top.sv` (A2.4) are for board bring-up only.
  - They use the real packet format plus a bring-up-only `ECHO` command (0x7E), with a 256-byte payload limit.
  - User guide: `docs/bringup.md`. The real command engine (A4.1) replaces `loopback_echo`.
  - Bring-up modules are tested end to end in cocotb but have no formal proofs.
- Use `always_ff` / `always_comb`. No latches. No initial blocks in synthesizable RTL, except BRAM init files.
- Write for synthesis and routing: pipeline long paths, avoid wide muxes across the array, and infer BRAM with the standard Vivado templates.
- Python: 3.12 (see §5a). All packages are pinned in `uv.lock`.
  - Host code is the `tinytpu` package in `host/tinytpu/`. It is built with `uv_build` and installed editable by `uv sync`, so `import tinytpu` works in tests, docs, and on Windows.
  - Runtime dependencies (`numpy`, `pyserial`) go in `[project] dependencies`.
  - Test tools go in the `sim` group; docs tools go in the `docs` group.
  - Add packages with `uv add` (`--group sim|docs`); never use `pip install`.
  - On Windows, use `uv sync --no-default-groups` with `UV_PROJECT_ENVIRONMENT` pointing outside the repo (see `tools/setup.md`).

## 7. Documentation

- **Decided: one MkDocs site. Module docs are generated from the source. No Natural Docs.**
- **SystemVerilog:**
  - Every module has a `/** ... */` header comment directly above `module`; only blank lines may separate them.
    - Its first line is a one-line summary, followed by optional details.
    - It must contain a line starting with `Latency:`, e.g. `Latency: 3 cycles from in_valid to out_valid`, or `Latency: combinational`.
  - Use ANSI port lists and `#( ... )` parameter lists, with **one parameter or port per line**. Each line carries a trailing `// description` comment.
  - `docs/gen_rtl_docs.py` reads the actual parameters and ports with `pyslang`, takes descriptions from those comments, and writes `docs/rtl/index.md` plus `docs/rtl/<module>.md`.
    - It fails, listing every problem with file and line, if any rule above is broken or the RTL doesn't compile.
    - Packages (`*_pkg.sv`) are compiled but don't get pages.
  - Don't hand-write port or parameter tables anywhere. `docs/rtl/` is generated; don't edit it.
- **Python:** Google-style docstrings, rendered by `mkdocstrings`.
- **Build:** `make docs` runs the generator, then `mkdocs build --strict`, in WSL and in CI. `make docs-serve` gives a live preview.
  - `mkdocs.yml` has an explicit `nav`. Add each new hand-written page there.
  - Python API pages use mkdocstrings `::: module.path` blocks.
- `pyslang`, `mkdocs`, and `mkdocstrings` are pinned in `uv.lock`.
- **Project-level:** MkDocs Markdown in `docs/` covering architecture, the memory map, the ISA, and the UART protocol. Update docs in the same change as the RTL they describe.

## 8. Working rules for agents

- Before resolving an **OPEN** item, ask. After a decision, update this file.
- Work is tracked in `TASKS.md`. Update a task's status in the same change that does the work.
  - Never do a **Human** task: it needs the board, admin rights, accounts, or a design decision.
  - For a **Joint** task, do the agent part, then stop and ask.
- Don't claim something passes without running it. If a tool (Verilator, SBY, Vivado) isn't installed, say so instead of guessing results.
- Keep changes small and reviewable: one module plus its testbench per change where possible.
