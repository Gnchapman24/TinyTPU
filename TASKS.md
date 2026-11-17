# TinyTPU — Task List

Owners:
- **Human:** needs the board, admin rights, accounts, or a design decision.
- **Agent:** can be done entirely from the repo and the command line.
- **Joint:** an agent drafts, the human decides or signs off.

Status: `[ ]` todo · `[~]` in progress · `[x]` done.

Agents: update status here as part of the same change. Never do a **Human** task. For a **Joint** task, do the agent part, then stop and ask. Design rules live in `AGENTS.md`.

---

## Phase 0 — Environment and repo setup

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] H0.1 | Human | Install the latest OSS CAD Suite **dated release** in WSL at `~/tools/oss-cad-suite`. Installed release 20261112 and smoke-tested. It goes on PATH per project via `source tools/env.sh`, not in the shell profile. | — |
| [x] H0.2 | Human | Install `uv` in WSL and on Windows. Both are at 0.13.8. The Windows host venv was created at `%LOCALAPPDATA%\tinytpu\venv` (Python 3.12.10, numpy + pyserial only). The WSL `.venv` is unaffected. | — |
| [ ] H0.3 | Human | Set the FTDI latency timer to 1 ms: Windows Device Manager → the Basys3 COM port → Port Settings → Advanced. Still open (2026-11-16): the registry shows `LatencyTimer = 16` for COM7, and the board runs show it (16 ms per 256-byte echo, 16 ms per INFER at both baud rates). | — |
| [x] H0.4 | Human | Check that Vivado 2025.2 Hardware Manager detects the Basys3. Install the cable drivers if it doesn't. Done 2026-11-16: `vivado.ps1 -Program` loaded all four bitstreams. | — |
| [ ] H0.5 | Human | Delete the empty leftover folder `C:\Users\Grant\HOME\PROJECTS\TPU\TinyTPU` once no app has it open. | — |
| [x] A0.1 | Agent | Add `.gitignore` (per AGENTS.md §4), `.gitattributes` (`* text=auto eol=lf`; binary files marked as binary), and a short `README.md`. | — |
| [x] A0.2 | Agent | Record the installed tool versions in `tools/versions.env`. Write `tools/setup.md` with the WSL and Windows setup steps. Also added `tools/env.sh`. | H0.1, H0.2 |
| [x] A0.3 | Agent | Create `pyproject.toml` (Python 3.12) and `uv.lock` with cocotb, numpy, pyserial, pyslang, mkdocs, mkdocstrings[python]. | H0.2 |
| [x] A0.4 | Agent | Create a top-level `Makefile` with targets `sim`, `sim-full` (N = 8), `formal`, `docs`, `lint` (`verilator --lint-only`). Each target names the side it runs on. | A0.3 |
| [x] A0.5 | Agent | Create the `mkdocs.yml` skeleton, `docs/index.md`, and the `docs/gen_rtl_docs.py` generator (pyslang; strict failure on missing descriptions). | A0.3 |
| [x] A0.6 | Agent | Fetch Digilent's official Basys3 master XDC and create a trimmed `syn/basys3.xdc` (clock, UART, reset, LEDs, switches). Confirm the pins against AGENTS.md §1. | — |
| [x] J0.7 | Joint | Confirm how Vivado handles the repo's WSL path. Agent: run a trivial batch synthesis from the `\\wsl.localhost` path. Human: map a drive letter if it fails. Update AGENTS.md §5a. | A0.6 |
| [x] J0.8 | Joint | First commit and push. Agent prepares the commit; human approves the push. | A0.1 |
| [x] A0.9 | Agent | Add a GitHub Actions workflow running Tier 1 (N = 2, 4), Tier 2, and `mkdocs build --strict`, with the OSS CAD Suite pinned to the date in `tools/versions.env`. First run (commit 52b40d3) passed: both jobs green. | A0.2–A0.5, J0.8 |
| [ ] H0.10 | Human | GitHub settings: enable Actions, branch protection on `main` requiring CI. | A0.9 |

## Phase 1 — ISA, protocol, memory map

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] J1.1 | Joint | ISA design. Agent drafts options with trade-offs: instruction format, opcodes, matmul `act_signed` field, STORE raw/requant/ReLU fields, scale-table load, tile sizes and strides. Human decides. Result: `docs/isa.md`. | — |
| [x] J1.2 | Joint | Decide on-chip write-back of requantized results to the activation buffer (the last OPEN item in AGENTS.md). | J1.1 |
| [x] J1.3 | Joint | Packet format. Agent drafts header, length field, opcode/payload layout, ACK/NAK codes, and the timeout rule. Human decides. Result: `docs/uart_protocol.md`. | J1.1 |
| [x] J1.4 | Joint | Memory map: buffer depths for weights, activations, and accumulators, checked against the BRAM ceiling (≤ 25 BRAM36). Result: `docs/memory_map.md`. | J1.1 |
| [x] A1.5 | Agent | Write `rtl/isa_pkg.sv` (opcode enums, instruction structs) and a matching Python module in `host/`, both generated from or checked against `docs/isa.md`. | J1.1–J1.4 |

## Phase 2 — Serial link (UART, CRC, packets)

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] A2.1 | Agent | `uart_rx`, `uart_tx` (`CLKS_PER_BIT` parameter, mid-bit sampling, framing-error flag) + cocotb tests + formal properties. | A0.4 |
| [x] A2.2 | Agent | `sync_fifo` + cocotb tests + formal properties (no overflow or underflow, count invariants). | A0.4 |
| [x] A2.3 | Agent | `crc16_ccitt` (CCITT-FALSE) + cocotb test against the Python reference, including check value 0x29B1. | A0.4 |
| [x] A2.4 | Agent | Loopback top-level for bring-up (echo with CRC check, LED status) + Vivado build script. | A2.1–A2.3, A0.6 |
| [x] A2.5 | Agent | Host driver basics: open port at 1 Mbaud, CRC framing, ACK/NAK and resend, timeout. | A2.3, J1.3 |
| [x] H2.6 | Human | Board bring-up: program the loopback bitstream, run the host loopback test at 1 Mbaud, and report results or errors. Done 2026-11-16 (UART on COM7): raw echo and 7/7 packet checks at 1 Mbaud and 115k, 0 resends. The first runs failed from a host bug, now fixed: `Link` changed the port timeout during each transfer, which on Windows reconfigures the port and corrupts bytes in flight (`791be6c`). The longer run (board test 3.3) was not logged. | A2.4, A2.5, H0.3, H0.4 |

## Phase 3 — Compute datapath

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] A3.1 | Agent | NumPy reference model: INT8 D-lite semantics, INT32 wraparound, tiling, requantization formula. Unit-tested in isolation. | — |
| [x] A3.2 | Agent | PE module + standalone Vivado synthesis check that it maps to exactly 1 DSP48E1 with pipeline registers absorbed. | A0.4 |
| [x] A3.3 | Agent | `systolic_array` (N parameter, edge sign extension, input skew, output de-skew, PCOUT→PCIN cascade) + tests at N = 2, 4, 8. | A3.1, A3.2 |
| [x] A3.4 | Agent | Column accumulators + accumulator buffer + bias preload (if in the ISA) + tests. | A3.3, J1.4 |
| [x] A3.5 | Agent | Requantizer (shared, pipelined, per-column M/s, ReLU clamp, raw bypass) + edge-case tests. Raw bypass is the ACC read port; no requantizer hardware needed. | A3.1, J1.1 |
| [x] A3.6 | Agent | Weight and activation buffers + feeders/FIFOs + tests. Decided: no FIFOs, fixed-latency tagged read paths (`buffer_ram`, also used for PTAB); sequencing is in A4.2. | J1.4 |

## Phase 4 — Control

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] A4.1 | Agent | Packet parser (length, CRC check, ACK/NAK, timeout) + formal properties. Done: `cmd_engine` (full protocol on `tpu_core`), formal proof (framing, no reply without a request, timeout, busy rule, no dropped byte); open protocol points decided (ABORT/RESET replace a pending reply, RESET while running, big replies, PTAB reserved bits, duplicate RUN while running). | A2.1–A2.3, J1.3 |
| [x] A4.2 | Agent | Instruction decoder + control FSMs + formal properties (including: `act_signed` is stable while the array is busy). Done: `instr_decode`, `range_check` (error 5 before execution, decided), `sequencer`, `matmul_ctrl`, `requant_ctrl`, `tpu_core`; formal proofs for the sequencer and the MATMUL address generator; end-to-end tests against the model; ABORT semantics decided. | A1.5 |

## Phase 5 — Integration and implementation

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] A5.1 | Agent | Top-level `tinytpu_top` + end-to-end cocotb test through a UART driver model (random matrices, both activation modes, non-multiple-of-N sizes). Done, plus the host API `tinytpu.device` and an injectable clock in `tinytpu.link`, so the test runs the real host code. | Phases 2–4 |
| [x] A5.2 | Agent | Tier 3 TCL: out-of-context synthesis, utilization/timing/power reports to JSON, ceiling check, baseline compare script. Done: `build.tcl` writes `<top>_metrics.json` and checks the ceilings; `python -m tinytpu.baseline` compares with `syn/baseline.json` (run by `vivado.ps1`). | A0.6 |
| [x] J5.3 | Joint | First full synthesis. Agent creates `syn/baseline.json`; human reviews and approves it. Done: approved 2026-11-15 (LUT 3702, FF 4185, BRAM 39, DSP 66, WNS +1.075 ns, WHS +0.035 ns). | A5.1, A5.2 |
| [x] A5.4 | Agent | Full implementation: build the bitstream; meet slack ≥ 0 at 100 MHz. Done: WNS +1.075 ns, WHS +0.035 ns after routing (host writes registered to get there); LUT 17.8%, FF 10.1%, BRAM 39, DSP 66. | A5.1 |
| [x] H5.5 | Human | On-board end-to-end test: host driver runs matmuls and compares with NumPy. Report mismatches. Done 2026-11-16 at 1 Mbaud: self-test 6/6, the four example matmuls match NumPy, manual LED and `btnC` checks 8/8, and 20 soak seeds all pass, with 0 resends throughout. No mismatches. The 115k build also passes the self-test 6/6. | A5.4, H2.6 |
| [ ] A5.6 | Agent | Optional: N = 2, 4, 8 sweep with a results table and plots in the docs. | A5.2 |

## Phase 6 — Documentation

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] A6.1 | Agent | Architecture page with block and dataflow diagrams (array skew, tiling, requantization path). Done: `docs/architecture.md` with five hand-drawn SVGs in `docs/img/` (block diagram, array dataflow, tiling, tile timing, requantization path), plus throughput, resources and a verification map. | Phase 3 |
| [x] A6.2 | Agent | Host driver user guide and an example script. Done: `docs/host.md`, `python -m tinytpu.selftest` (board self-test for H5.5), `python -m tinytpu.example`; both tested against the simulated board. | A2.5, A5.1 |
| [ ] H6.3 | Human | Final review of the docs site; decide what to publish (README, GitHub Pages). | A6.1, A6.2 |

## Phase 7 — Host compiler

Scope decided 2026-11-16 (AGENTS.md §3, "Host compiler"): own Dense + ReLU layer API with float weights and calibration data, raw INT32 logits from the last layer, MNIST demo trained in NumPy.

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [x] J7.1 | Joint | Compiler design page `docs/compiler.md`. Done: approved 2026-11-16, all seven choices as proposed (784-128-10, max calibration, INFER-only batches of the largest size that fits, hashed HALT tag, `.npz` file, torchvision's MNIST mirror). Agent drafts: the layer API, the quantization math (per-channel weight scales, activation scales from calibration, bias rounding, choosing `M`/`s`, dequantizing logits), memory allocation, batch handling, the limit checks (each `docs/memory_map.md` limit, the K overflow rule, INFER's 4096-byte payloads), the compiled-file format, and the MNIST source and model size. Human approves the open choices. | — |
| [x] A7.2 | Agent | Done: `tinytpu.nn` (float layers) and `tinytpu.quant` (`quantize`, `QModel.forward_int` as the integer reference, `mult_shift`), 31 tests in `tb/test_quant.py`, each injected bug of 12 caught. Quantizer `tinytpu.quant`: float `nn.Model` + calibration data → integer layers (INT8 weights, INT32 biases, `M`, `s`, ReLU, `act_signed` per layer). Tests: the `M`/`s` choice at its edges (s = 0, M at 65535, a too-large scale rejected), all-zero channels, bias rounding, K overflow rejected, quantized vs float accuracy on synthetic data. | J7.1 |
| [x] A7.3 | Agent | Done: `tinytpu.compiler` (`Target`, `compile_model`, `max_batch`, `check_fit`, `FitError`, `Compiled`), 79 tests in `tb/test_compiler.py`, each injected bug of 15 caught. Code generation `tinytpu.compiler`: integer layers + target (N and depths, from `Info` or the defaults) + batch size → program, WBUF tiles, PTAB entries, and the input / output addresses. Rejects models that don't fit, with a message naming the limit. Tests: random models with sizes that aren't multiples of N, both input signedness, hidden layers with and without ReLU, at N = 2, 4, 8, run on `model.Machine` and compared bit-exactly with the chained `model.matmul` / `model.requant`; each limit just met and just exceeded. | A7.2 |
| [x] A7.4 | Agent | Done: `Compiled.load` / `simulator` → `Runner` (`infer`, `infer_int`, `verify`; `RunError`), `Compiled.save` / `from_file`; 30 tests in `tb/test_runtime.py`, each injected bug of 14 caught; `compiled_model` in `tb/test_tinytpu_top.py` passes at N = 2, 4 and 8. Runtime: load a compiled model onto a `Device` (checks N and depths against `INFO`), `infer()` with input quantization, batching and padding, and logit dequantization; save and load compiled models. Tests: on `Machine`, and an end-to-end test in `tb/test_tinytpu_top.py` (the simulated board must match `Machine` bit-exactly). | A7.3 |
| [x] A7.5 | Agent | Done: `tinytpu.mnist`. 784-128-10 reaches 97.50% float and 97.48% integer accuracy on the MNIST test set; 14 tests in `tb/test_mnist.py`, each injected bug of 19 caught; `mnist_demo` passes on the simulated board at N = 2, 4 and 8. User guide: `docs/host.md` §6. MNIST demo `python -m tinytpu.mnist`: download with SHA-256 check into a cache outside the repo, NumPy MLP training, compile, then board accuracy and speed, with a check that the board matches `Machine` and a comparison with the float model. Tests use a tiny synthetic dataset, with no download. Docs: a user-guide section. | A7.4 |
| [ ] H7.6 | Human | Run the MNIST demo on the board; report accuracy, agreement with `Machine`, and ms per image. Do H0.3 (1 ms latency timer) first, so the speed figure is meaningful. | A7.5, H0.3 |
| [ ] A7.7 | Agent | Optional: rebuild `selftest`'s network check and `tinytpu.example` on the compiler, and remove their hand-written programs. | A7.4 |
