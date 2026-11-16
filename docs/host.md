# Using the board from a PC

This page takes you from a built bitstream to running your own matrix products on the Basys3. Everything here runs on **Windows**, in PowerShell, from the repo folder (`\\wsl.localhost\Ubuntu\home\<user>\Projects\FPGA\TinyTPU`).

If the serial link has never been tested on this board, run the [loopback bring-up](bringup.md) first. It checks the USB-UART on its own, so link problems don't look like TPU problems.

## 1. One-time setup

From [`tools/setup.md`](https://github.com/Gnchapman24/TinyTPU/blob/main/tools/setup.md):

- Vivado, with the cable drivers, so Hardware Manager sees the board.
- FTDI latency timer = **1 ms**: Device Manager → Ports → the Basys3 USB Serial Port → Properties → Port Settings → Advanced. Without it, every round trip gains about 16 ms.
- The host Python environment, outside the repo:

  ```powershell
  $env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\tinytpu\venv"   # or set it permanently
  uv sync --no-default-groups
  ```

## 2. Build and program

```powershell
.\syn\vivado.ps1 -Top tinytpu_top            # build: several minutes; fails if timing is not met
.\syn\vivado.ps1 -Top tinytpu_top -Program   # program the board over USB/JTAG
```

- The build writes to `%LOCALAPPDATA%\tinytpu\vivado\tinytpu_top\`:
  - the bitstream
  - `vivado.log`
  - the utilization, timing, worst-path and DRC reports
  - the routed checkpoint
- Programming is volatile: power the board off and the design is gone.
- After programming, `led[0]` blinks (about every 0.7 s). Press `btnC` to reset at any time. Memories keep their contents across a reset.

## 3. Find the serial port

The Basys3's FTDI chip shows up as **two** COM ports (JTAG and UART) with identical details. Ask the board which is which:

```powershell
uv run --no-default-groups python -m tinytpu.selftest --probe
```

```text
COM6: no valid answer (LinkError)
COM7: TinyTPU answered (N = 8)
```

- A port that answers "the bring-up design" still has `loopback_top` loaded.
- Close Vivado's Hardware Manager first; it can hold the port.

## 4. Run the self-test

```powershell
uv run --no-default-groups python -m tinytpu.selftest --port COM7
```

**It overwrites every memory on the board.** It prints one line per check and a summary, and exits with status 0 only if everything passed:

| Check | What it does |
|---|---|
| `info` | Reads INFO: protocol version, N, `CLKS_PER_BIT`, the depth of every memory |
| `memories` | Writes random data to every memory and reads it back |
| `matmul` | Random matrix products, both activation modes, sizes that are not multiples of N, raw INT32 results compared with the model |
| `network` | A two-layer network (MATMUL + REQUANT, twice) with RUN and with INFER, outputs compared with the model |
| `errors` | Programs that must halt with ISA errors 1, 5 and 6; a READ the board must refuse (lights `led[6]`) |
| `speed` | 20 INFER round trips; prints the time per inference |

- Every comparison is bit-exact, against `tinytpu.model` (the same reference the simulations use).
- The summary line counts the link's resends and timeouts. On a healthy link they are all 0.
- **`--manual`** adds two checks that need you at the board: confirming the status and error-code LEDs, and pressing `btnC`, which must leave every memory unchanged.
- Options: `--count` (matmuls), `--rounds` (INFERs), `--seed`, and `--baud` for the 115,200-baud fallback build (`-Variant 115k`). Use `--help` for details.
- The [board test plan](board_test.md) is a complete first session: link, self-test, manual checks and a longer run.
- The same script runs in simulation against the RTL before each change is merged (`tb/test_tinytpu_top.py`), so a failure on the board points at the board, the link or the build, not at the script.

## 5. Your own computation

`python -m tinytpu.example` multiplies two random matrices on the board and checks the result against NumPy:

```powershell
uv run --no-default-groups python -m tinytpu.example --port COM7 --m 4 --k 50 --n 20
uv run --no-default-groups python -m tinytpu.example --port COM7 --signed
```

Its source ([`host/tinytpu/example.py`](https://github.com/Gnchapman24/TinyTPU/blob/main/host/tinytpu/example.py)) is the starting point for your own code. The core of it:

```python
from tinytpu import isa, model
from tinytpu.device import Device
from tinytpu.isa import AccMode
from tinytpu.link import Link

with Link.open("COM7") as link:
    dev = Device(link)
    n = dev.info().n
    kt, nt = model.ceil_div(K, n), model.ceil_div(Nout, n)
    dev.load_weights(model.pack_weights(w, n))         # W: (K, Nout) INT8, tiled and zero-padded
    dev.load_vectors(model.pack_activations(x, n))     # X: (M, K) UINT8 (or INT8 with act_signed)
    dev.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, kt=kt, nt=nt, m=M,
                   act_signed=False, acc_mode=AccMode.OVERWRITE),
        isa.halt(1),
    ])
    status = dev.run()                                 # waits until the program halts
    y = model.unpack_acc(dev.read_acc(0, M * nt), M, Nout)   # (M, Nout) INT32
```

For a whole network:

- Chain `MATMUL` (bias mode) and `REQUANT` per layer, and upload each layer's per-channel parameters with `dev.load_params(bias, mult, shift)`.
- `tinytpu.selftest.two_layer_network` and the [ISA](isa.md#example-a-196-64-10-network) example show the layout.
- `dev.infer(...)` then does one inference per round trip.

The [`tinytpu.device` API](api/device.md) lists every method. The [ISA](isa.md) and the [memory map](memory_map.md) define the layouts.

## LEDs

| LED | Meaning |
|---|---|
| `led[0]` | Heartbeat: the design is loaded and running |
| `led[1]` / `led[2]` | Receive / transmit activity |
| `led[3]` | A program is running |
| `led[4]` | The last program halted OK |
| `led[5]` | The last program halted with an error; `led[15:12]` show the [ISA error code](isa.md#errors) |
| `led[6]` | A request was refused (NAK) |
| `led[7]` | A partial request timed out |
| `led[8]` | UART framing error |
| `led[9]` | A received byte was dropped (receive queue full: the host sent without waiting for a reply) |

`led[6]`–`led[9]` stay on until `btnC`. After a passing self-test, `led[4]` is on (the last inference halted OK), and so is `led[6]`, because the `errors` check provokes a NAK on purpose.

## When something fails

| Symptom | Likely cause |
|---|---|
| `--probe` finds nothing | Bitstream not loaded (`led[0]` not blinking), wrong board, or Hardware Manager holding the port |
| `LinkError` with timeouts | Wrong port; or the latency timer is not 1 ms and `RUN` needs a longer `timeout` |
| Resends in the summary, but checks pass | A noisy link: the protocol recovered. Note the counts. |
| A wrong result in `matmul` / `network` | A real mismatch. Record the printed shape and position, and the `--seed`. |
| `led[8]` on | Baud rate mismatch: the bitstream's `CLKS_PER_BIT` (see `info`) does not give 1 Mbaud |
