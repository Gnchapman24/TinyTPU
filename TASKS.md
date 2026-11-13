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
| [ ] H0.3 | Human | Set the FTDI latency timer to 1 ms: Windows Device Manager → the Basys3 COM port → Port Settings → Advanced. | — |
| [ ] H0.4 | Human | Check that Vivado 2025.2 Hardware Manager detects the Basys3. Install the cable drivers if it doesn't. | — |
| [ ] H0.5 | Human | Delete the empty leftover folder `C:\Users\Grant\HOME\PROJECTS\TPU\TinyTPU` once no app has it open. | — |
| [x] A0.1 | Agent | Add `.gitignore` (per AGENTS.md §4), `.gitattributes` (`* text=auto eol=lf`; binary files marked as binary), and a short `README.md`. | — |
| [x] A0.2 | Agent | Record the installed tool versions in `tools/versions.env`. Write `tools/setup.md` with the WSL and Windows setup steps. Also added `tools/env.sh`. | H0.1, H0.2 |
| [x] A0.3 | Agent | Create `pyproject.toml` (Python 3.12) and `uv.lock` with cocotb, numpy, pyserial, pyslang, mkdocs, mkdocstrings[python]. | H0.2 |
| [x] A0.4 | Agent | Create a top-level `Makefile` with targets `sim`, `sim-full` (N = 8), `formal`, `docs`, `lint` (`verilator --lint-only`). Each target names the side it runs on. | A0.3 |
| [x] A0.5 | Agent | Create the `mkdocs.yml` skeleton, `docs/index.md`, and the `docs/gen_rtl_docs.py` generator (pyslang; strict failure on missing descriptions). | A0.3 |
| [x] A0.6 | Agent | Fetch Digilent's official Basys3 master XDC and create a trimmed `syn/basys3.xdc` (clock, UART, reset, LEDs, switches). Confirm the pins against AGENTS.md §1. | — |
| [ ] J0.7 | Joint | Confirm how Vivado handles the repo's WSL path. Agent: run a trivial batch synthesis from the `\\wsl.localhost` path. Human: map a drive letter if it fails. Update AGENTS.md §5a. | A0.6 |
| [x] J0.8 | Joint | First commit and push. Agent prepares the commit; human approves the push. | A0.1 |
| [x] A0.9 | Agent | Add a GitHub Actions workflow running Tier 1 (N = 2, 4), Tier 2, and `mkdocs build --strict`, with the OSS CAD Suite pinned to the date in `tools/versions.env`. First run (commit 52b40d3) passed: both jobs green. | A0.2–A0.5, J0.8 |
| [ ] H0.10 | Human | GitHub settings: enable Actions, branch protection on `main` requiring CI. | A0.9 |

## Phase 1 — ISA, protocol, memory map

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] J1.1 | Joint | ISA design. Agent drafts options with trade-offs: instruction format, opcodes, matmul `act_signed` field, STORE raw/requant/ReLU fields, scale-table load, tile sizes and strides. Human decides. Result: `docs/isa.md`. | — |
| [ ] J1.2 | Joint | Decide on-chip write-back of requantized results to the activation buffer (the last OPEN item in AGENTS.md). | J1.1 |
| [ ] J1.3 | Joint | Packet format. Agent drafts header, length field, opcode/payload layout, ACK/NAK codes, and the timeout rule. Human decides. Result: `docs/uart_protocol.md`. | J1.1 |
| [ ] J1.4 | Joint | Memory map: buffer depths for weights, activations, and accumulators, checked against the BRAM ceiling (≤ 25 BRAM36). Result: `docs/memory_map.md`. | J1.1 |
| [ ] A1.5 | Agent | Write `rtl/isa_pkg.sv` (opcode enums, instruction structs) and a matching Python module in `host/`, both generated from or checked against `docs/isa.md`. | J1.1–J1.4 |

## Phase 2 — Serial link (UART, CRC, packets)

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] A2.1 | Agent | `uart_rx`, `uart_tx` (`CLKS_PER_BIT` parameter, mid-bit sampling, framing-error flag) + cocotb tests + formal properties. | A0.4 |
| [ ] A2.2 | Agent | `sync_fifo` + cocotb tests + formal properties (no overflow or underflow, count invariants). | A0.4 |
| [ ] A2.3 | Agent | `crc16_ccitt` (CCITT-FALSE) + cocotb test against the Python reference, including check value 0x29B1. | A0.4 |
| [ ] A2.4 | Agent | Loopback top-level for bring-up (echo with CRC check, LED status) + Vivado build script. | A2.1–A2.3, A0.6 |
| [ ] A2.5 | Agent | Host driver basics: open port at 1 Mbaud, CRC framing, ACK/NAK and resend, timeout. | A2.3, J1.3 |
| [ ] H2.6 | Human | Board bring-up: program the loopback bitstream, run the host loopback test at 1 Mbaud, and report results or errors. | A2.4, A2.5, H0.3, H0.4 |

## Phase 3 — Compute datapath

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] A3.1 | Agent | NumPy reference model: INT8 D-lite semantics, INT32 wraparound, tiling, requantization formula. Unit-tested in isolation. | — |
| [ ] A3.2 | Agent | PE module + standalone Vivado synthesis check that it maps to exactly 1 DSP48E1 with pipeline registers absorbed. | A0.4 |
| [ ] A3.3 | Agent | `systolic_array` (N parameter, edge sign extension, input skew, output de-skew, PCOUT→PCIN cascade) + tests at N = 2, 4, 8. | A3.1, A3.2 |
| [ ] A3.4 | Agent | Column accumulators + accumulator buffer + bias preload (if in the ISA) + tests. | A3.3, J1.4 |
| [ ] A3.5 | Agent | Requantizer (shared, pipelined, per-column M/s, ReLU clamp, raw bypass) + edge-case tests. | A3.1, J1.1 |
| [ ] A3.6 | Agent | Weight and activation buffers + feeders/FIFOs + tests. | J1.4 |

## Phase 4 — Control

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] A4.1 | Agent | Packet parser (length, CRC check, ACK/NAK, timeout) + formal properties. | A2.1–A2.3, J1.3 |
| [ ] A4.2 | Agent | Instruction decoder + control FSMs + formal properties (including: `act_signed` is stable while the array is busy). | A1.5 |

## Phase 5 — Integration and implementation

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] A5.1 | Agent | Top-level `tinytpu_top` + end-to-end cocotb test through a UART driver model (random matrices, both activation modes, non-multiple-of-N sizes). | Phases 2–4 |
| [ ] A5.2 | Agent | Tier 3 TCL: out-of-context synthesis, utilization/timing/power reports to JSON, ceiling check, baseline compare script. | A0.6 |
| [ ] J5.3 | Joint | First full synthesis. Agent creates `syn/baseline.json`; human reviews and approves it. | A5.1, A5.2 |
| [ ] A5.4 | Agent | Full implementation: build the bitstream; meet slack ≥ 0 at 100 MHz. | A5.1 |
| [ ] H5.5 | Human | On-board end-to-end test: host driver runs matmuls and compares with NumPy. Report mismatches. | A5.4, H2.6 |
| [ ] A5.6 | Agent | Optional: N = 2, 4, 8 sweep with a results table and plots in the docs. | A5.2 |

## Phase 6 — Documentation

| ID | Owner | Task | Depends on |
|---|---|---|---|
| [ ] A6.1 | Agent | Architecture page with block and dataflow diagrams (array skew, tiling, requantization path). | Phase 3 |
| [ ] A6.2 | Agent | Host driver user guide and an example script. | A2.5, A5.1 |
| [ ] H6.3 | Human | Final review of the docs site; decide what to publish (README, GitHub Pages). | A6.1, A6.2 |
