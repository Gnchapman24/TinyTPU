# TinyTPU — Agent Context

A minimal INT8 Tensor Processing Unit on a Digilent Basys3 FPGA. Hardware is written in SystemVerilog. Host tooling and tests are in Python.

> **Status:** early stage. The repo has no RTL, tests, or CI yet. Anything below marked **(planned)** does not exist yet. Create it to match this file; don't assume it's already there. Items marked **OPEN** are undecided. Ask before choosing, and record the answer here.

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
  - **STORE modes:**
    - **raw:** stream the INT32 accumulator unchanged (debugging, host-side schemes).
    - **requant:** `y = clamp((acc * M + R) >>> s, lo, hi)`, where:
      - `M` is a 16-bit unsigned multiplier and `s` is a 6-bit shift (0–63). Both are set per output column (per-channel scaling) in a scale table loaded by an instruction.
      - `R = (s == 0) ? 0 : 2**(s-1)`, i.e. round half up.
      - The product `acc * M` is computed at full width (48 bits signed). Never truncate before the shift.
      - The clamp range comes from a ReLU bit: ReLU on gives UINT8 [0, 255], which feeds the next layer with `act_signed = 0`. ReLU off gives INT8 [−128, 127], which feeds the next layer with `act_signed = 1`.
  - **No zero-points:** quantization is symmetric throughout.
  - **No TFLite compatibility:** the design does not reproduce TFLite's Q31 rounding behaviour. The NumPy reference model implements exactly the formula above.
  - **Tests:** cover negative values, rounding ties, both clamp edges, and s = 0.
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
- **Memory:** separate BRAM buffers for weights, activations, and accumulators/results. FIFOs sit between the buffers and the array edges.
- **Host interface:** UART.
  - Baud rate is **decided: 1,000,000 baud**.
    - Set by the build-time parameter `CLKS_PER_BIT` (= 100 at 100 MHz). The rate is exact on both the FPGA and FTDI sides.
    - The receiver samples at mid-bit using a cycle counter. No 16× oversampling.
    - There is no hardware flow control (only TX and RX are wired). The protocol must ensure the host never sends faster than the FPGA can accept, e.g. by waiting for an acknowledgement per command or block. Keep a small receive FIFO.
    - cocotb tests override `CLKS_PER_BIT` with a small value (e.g. 8) for speed, plus at least one test at 100. Formal proofs use `CLKS_PER_BIT` of 4–8.
    - Host docs tell Windows users to set the FTDI latency timer to 1 ms (Device Manager → Port Settings → Advanced).
  - Framing and error detection are **decided (Option D)**:
    - **Bytes:** 8N1. The receiver raises a framing-error flag when the stop bit is low.
    - **Packets:** every command or data packet ends with a CRC-16/CCITT-FALSE checksum, sent high byte first:
      - polynomial 0x1021, initial value 0xFFFF, no reflection, final XOR 0x0000
      - check value for ASCII `"123456789"` = 0x29B1
    - **Replies and recovery:** the receiver answers ACK or NAK, and the host resends on NAK. Packets carry a length field, and the receiver discards a partial packet after a timeout.
    - **Testing:** check the CRC module in cocotb against the Python implementation; it is not formally verified. Inject flipped bits, dropped bytes, and bad CRCs.
    - **Still open:** header, opcode, and payload layout are decided with the ISA.
  - UART bandwidth, not compute, limits end-to-end throughput. Don't over-optimize the array at the expense of simplicity.
- **Control:** custom minimal ISA (load weights, load activations, matmul, store/read results).
  - The single source of truth is `docs/isa.md` **(planned)**.
  - Don't change opcodes or encodings without asking.
  - The matmul instruction carries an `act_signed` mode field (see §2).
  - The ISA needs:
    - a scale-table load instruction (per-column `M`, `s`)
    - STORE fields for raw/requant mode and the ReLU bit
  - **OPEN, to decide during ISA design:** whether requantized results can be written back to the activation buffer (on-chip layer chaining) instead of only going to the host.
  - Functional tests run in both activation modes.

## 4. Repository layout (planned)

```
rtl/          SystemVerilog sources (one module per file, file name = module name)
tb/           cocotb testbenches + Python reference models
formal/       SymbiYosys .sby files and property wrappers
syn/          Vivado TCL (OOC synthesis, reports), XDC constraints, baseline.json
host/         Python host driver (pyserial)
docs/         MkDocs site (architecture, memory map, ISA, UART protocol) + gen_rtl_docs.py
tools/        versions.env (pinned tool versions), setup notes
pyproject.toml, uv.lock   Python 3.12 project and locked dependencies
mkdocs.yml    MkDocs config
```

Keep generated output out of git: `sim_build/`, `*.vcd`/`*.fst`, `results.xml`, `formal/*/` run dirs, Vivado `.Xil/`, `*.jou`, `*.log`, `*.dcp`, report dumps, `site/`, `docs/rtl/` (generated), `.venv/`.

## 5. Verification (three tiers, planned)

**Entry points:** the top-level `Makefile`, run in WSL. Run `make help` for the list.
- `make lint`: every `rtl/` module through Verilator `-Wall` and yosys-slang, with `ASSERTIONS` defined. Must pass with zero warnings.
- `make sim`: tier 1 at N = 2 and 4. Add `K=<pytest filter>` to select tests.
- `make sim-full`: tier 1 at N = 8.
- `make formal`: tier 2.
- `make docs`, `make all`, `make clean`.

The Makefile puts the OSS CAD Suite `bin/` on `PATH`, and runs Python tools via `uv run --locked`. It refuses to run if the installed suite release doesn't match `tools/versions.env`.

**File conventions (`make` discovers these automatically):**
- `rtl/*_pkg.sv` are packages; they are compiled first.
- `tb/test_*.py` are pytest files. Each one builds with `cocotb_tools.runner` and runs its own cocotb tests. It reads the array size from the `TINYTPU_N` environment variable, passes it as the `N` parameter, and uses the build directory `sim_build/<name>_N<n>`.
- `formal/<name>.sby` lists sources as `../rtl/...` in `[files]`.

**Lint exception for the formal `past_valid` idiom:** its declaration-with-initializer triggers Verilator `PROCASSINIT`. Wrap only that declaration in `/* verilator lint_off PROCASSINIT */ ... /* verilator lint_on PROCASSINIT */`. No other lint suppressions without asking.

1. **Functional: cocotb + Verilator**
   - Reference results come from NumPy using the same integer semantics as the RTL (same signedness, accumulator width, and requantization).
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
   - **Decided:** the formal front-end is **yosys-slang**, which reads SystemVerilog through the slang compiler and is bundled in the OSS CAD Suite.
     - `.sby` files load the plugin with `plugin -i slang` and read sources with `read_slang`, instead of `read -formal`.
     - If a module hits a yosys-slang bug, it may fall back to Yosys's built-in reader. Note why in the `.sby` file.
   - **Property style:** write properties without SVA sequence operators (`|->`, `|=>`, `##`). Use clocked assertions with `$past`, `$stable`, `$rose`, and `$fell`, e.g. `always_ff @(posedge clk) if (!rst && $past(!rst) && $past(cond)) assert (expr);`. This keeps them valid in yosys-slang, Yosys's built-in reader, and Verilator.
   - Verilator runs with `--assert`, so every property is also checked during cocotb simulation.
3. **Implementation metrics: Vivado TCL, run locally or on a self-hosted runner**
   - Vivado is too large to run on GitHub-hosted CI.
   - Run out-of-context `synth_design`, then `report_utilization`, `report_timing_summary`, and `report_power`.
   - **Resource budgets are decided (fixed ceilings + baseline check).** Measure at N = 8 with no debug cores (ILA) in the design.
     - **Hard ceilings:**
       - LUT ≤ 25% (5,200)
       - FF ≤ 15% (6,240)
       - BRAM36 ≤ 50% (25)
       - DSP **exactly 66**: 65 means a MAC fell into LUTs; 67 or more means an unexpected DSP was inferred
       - worst setup slack ≥ 0 at 100 MHz
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
- **Vivado and UNC paths:** Vivado may misbehave with UNC paths. Map a drive letter to the WSL home if needed, and write Vivado run directories to a local Windows disk, never into the repo. TCL scripts take the output directory as an argument. Confirm this on the first Tier 3 run, then update this note.
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
- Use `always_ff` / `always_comb`. No latches. No initial blocks in synthesizable RTL, except BRAM init files.
- Write for synthesis and routing: pipeline long paths, avoid wide muxes across the array, and infer BRAM with the standard Vivado templates.
- Python: 3.12 (see §5a). All packages are pinned in `uv.lock`.
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
