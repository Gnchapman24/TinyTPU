# Board test plan

A step-by-step session for the first time on the Basys3. Allow about 1 to 1½ hours. It covers TASKS.md H0.3, H0.4, H2.6 and H5.5.

The stages build on each other, and each one adds one layer:

1. **JTAG:** the PC can program the board.
2. **Serial link:** the UART link works on its own, with the loopback design.
3. **Full design:** TinyTPU computes correctly over that link.

A failure at one stage then points at that layer, not at everything at once.

Everything runs on **Windows**, in PowerShell.

## Bitstreams

There are four bitstreams. All are built at 100 MHz and met timing. They are written to `%LOCALAPPDATA%\tinytpu\vivado\<name>\<name>.bit`:

| Name | Design | Baud rate | Use it for |
|---|---|---|---|
| `loopback_top` | Bring-up echo design | 1,000,000 | Stage 3: the serial link on its own |
| `tinytpu_top` | The full TPU | 1,000,000 | Stage 4: everything |
| `loopback_top_115k` | Bring-up echo design | 115,200 | Fallback, only if stage 3 fails at 1 Mbaud |
| `tinytpu_top_115k` | The full TPU | 115,200 | Fallback, only if 1 Mbaud is unusable |

- **Fallback builds:** the same designs with `CLKS_PER_BIT = 868` (100 MHz / 868 = 115,207 baud). Every host command then needs `--baud 115200`.
- **Programming:** the `-Program` commands below load these files and don't rebuild anything. To rebuild, see [the end of this page](#rebuilding-the-bitstreams).
- **Volatile:** programming lasts until the board is powered off. After a power cycle, program again.

## 0. Set up PowerShell (every new window)

Paste this, with your WSL user name in the path:

```powershell
Set-Location \\wsl.localhost\Ubuntu\home\<user>\Projects\FPGA\TinyTPU
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\tinytpu\venv"
function tpy { uv run --no-default-groups python @args }
$log = (New-Item -ItemType Directory -Force "$env:USERPROFILE\Desktop\tinytpu-board-test").FullName
tpy -m tinytpu.selftest --help | Select-Object -First 3
```

- `tpy` is a short name for the project's Python. The last line prints the self-test's usage, which checks that it works.
- `$log` is a folder on your desktop for the results. Each command below ends with `*>> "$log\<file>.txt"`, which saves everything it prints there.
  - Leave that ending off if you'd rather watch the output live.
  - The manual checks in step 4.4 need the console, so they use `Tee-Object` instead, which shows the output and saves it.

## 1. Connect and program (H0.4)

1. Connect the Basys3's micro-USB port (labelled PROG / UART) to the PC.
   - Use a **data** cable: some cables only charge.
   - Leave the power-select jumper on USB, the factory setting.
2. Switch the board on. Its power LED lights. A factory demo design may start running, which is fine.
3. Close any open Vivado Hardware Manager and any serial terminal, then program the loopback design:

   ```powershell
   .\syn\vivado.ps1 -Top loopback_top -Program
   ```

**Expected:**
- The command ends with `TinyTPU: programmed xc7a35t_0 with ...loopback_top.bit`.
- The board's DONE LED is on.
- `led[0]` (the rightmost LED) blinks about every 0.7 s. All other LEDs are off.

**This completes H0.4.** If it fails, see [Troubleshooting](#troubleshooting): usually the cable, the driver, or another program holding the JTAG port.

## 2. Find the serial port and set the latency timer (H0.3)

1. Find the board's UART. Windows often shows **two** COM ports for the board, JTAG and UART, with identical details.

   ```powershell
   tpy -m tinytpu.bringup --list-ports
   tpy -m tinytpu.bringup --probe
   ```

   `--probe` sends one packet to each port. Exactly one should answer: `COM7: TinyTPU bring-up design answered`. Keep that port for the rest of the session:

   ```powershell
   $port = "COM7"     # the port that answered
   ```

2. Set the FTDI latency timer to **1 ms** for that port. Without it, every round trip gains about 16 ms.
   - Open Device Manager → Ports (COM & LPT) → the USB Serial Port with that number.
   - Go to Properties → Port Settings → Advanced → Latency Timer (msec) = 1 → OK.
   - Setting the other port too does no harm.
3. Run `--probe` again: the same port must still answer.
   - If Windows needed a replug for the setting, the board was power-cycled and lost its design. Program it again (step 1.3).

**This completes H0.3.**

## 3. The serial link at 1 Mbaud (H2.6)

### 3.1 Raw byte echo

Set **`sw[0]` on** (the rightmost switch, up). `led[3]` lights.

```powershell
tpy -m tinytpu.bringup --port $port --raw *>> "$log\3-link.txt"; Get-Content "$log\3-link.txt" -Tail 5
```

**Expected:** `PASS  raw  1000 bytes echoed unchanged`, and the exit status is 0.

### 3.2 Packet echo

Set **`sw[0]` off** and press **`btnC`** (the centre button).

```powershell
tpy -m tinytpu.bringup --port $port *>> "$log\3-link.txt"; Get-Content "$log\3-link.txt" -Tail 12
```

**Expected:**
- Seven `PASS` lines: `ping`, `sizes`, `throughput`, `bad_crc`, `unknown_cmd`, `too_long`, `timeout`.
- The link summary shows **0 resends and 0 timeouts**.
- The `throughput` line shows about **5–8 ms per 256-byte round trip**. Note the figure.
- **LEDs afterwards:** `led[4]`, `led[6]` and `led[7]` are on, because the error checks cause them on purpose. `led[5]` (framing error) is **off**. Press `btnC` and all three go out.
- **Optional:** with `sw[1]` off, `led[15:8]` count good packets. With `sw[1]` on, they count errors.

### 3.3 Longer run

```powershell
tpy -m tinytpu.bringup --port $port --count 2000 --rounds 1000 --seed 1 *>> "$log\3-link.txt"; Get-Content "$log\3-link.txt" -Tail 12
```

**Expected:** seven `PASS` lines, with 0 resends and 0 timeouts. A few resends with every check passing means the protocol recovered from a noisy link. Note the counts.

**If stage 3 passes, H2.6 is done:** go to stage 4. If 3.1 or 3.2 fails, try stage 3F before anything else.

### 3F. Fallback: the link at 115,200 baud

```powershell
.\syn\vivado.ps1 -Top loopback_top -Variant 115k -Program
tpy -m tinytpu.bringup --port $port --baud 115200 *>> "$log\3F-link-115k.txt"; Get-Content "$log\3F-link-115k.txt" -Tail 12
```

- Repeat 3.1 with `--raw --baud 115200`. Expect about 46 ms per 256-byte round trip.
- **115k passes but 1M fails:** the PC, driver or cable can't keep up at 1 Mbaud. Continue with stage 4F instead of stage 4, and tell me: the full design still works, just about 9 times slower.
- **Both fail:** it isn't the baud rate. Check the port, cable and driver ([Troubleshooting](#troubleshooting)), and send me the logs.

## 4. The full design (H5.5)

### 4.1 Program and identify

```powershell
.\syn\vivado.ps1 -Top tinytpu_top -Program
tpy -m tinytpu.selftest --probe
```

**Expected:**
- `led[0]` blinks, and all other LEDs are off. Switches have no effect in this design.
- The probe prints `COMx: TinyTPU answered (N = 8)`.
- If it says "the bring-up design answered" instead, the loopback design is still loaded.

### 4.2 Self-test

**It overwrites every memory on the board.**

```powershell
tpy -m tinytpu.selftest --port $port *>> "$log\4-selftest.txt"; Get-Content "$log\4-selftest.txt" -Tail 12
```

**Expected:** six `PASS` lines and `6/6 checks passed`, with 0 resends and 0 timeouts:

| Check | Expected detail |
|---|---|
| `info` | `N = 8, CLKS_PER_BIT = 100, IMEM 128, WBUF 2048, ABUF 1024, ACC 512, PTAB 512; state IDLE` |
| `memories` | `5 memories, ... bytes written and read back` |
| `matmul` | `6 random matmuls, both activation modes, all results exact` |
| `network` | `... inferences (RUN and INFER, UINT8 and INT8 inputs), all outputs exact` |
| `errors` | `illegal opcode, address range and PC overrun halt correctly; bad READ gets NAK` |
| `speed` | `20 INFERs of a ... network: X ms each`. Note X. A few ms is expected; well over 20 ms suggests the latency timer is not 1 ms. |

**LEDs afterwards:**
- On: `led[4]` (the last program halted OK) and `led[6]` (the `errors` check provoked a NAK on purpose).
- Off: `led[3]`, `led[5]`, `led[7]`, `led[8]`, `led[9]` and `led[15:12]`.

### 4.3 Example matmuls

```powershell
tpy -m tinytpu.example --port $port *>> "$log\4-example.txt"
tpy -m tinytpu.example --port $port --signed *>> "$log\4-example.txt"
tpy -m tinytpu.example --port $port --m 16 --k 300 --n 100 --seed 2 *>> "$log\4-example.txt"
tpy -m tinytpu.example --port $port --m 8 --k 1000 --n 64 --seed 3 --signed *>> "$log\4-example.txt"
Select-String "matches NumPy|MISMATCH" "$log\4-example.txt"
```

**Expected:** four lines of `matches NumPy`.
- The last run uses most of the weight and activation buffers: 1,000 weight tiles and 1,000 vectors.
- It takes a few seconds, most of it uploading 64 KB of weights.

### 4.4 Manual checks: LEDs and the reset button

These need you at the board. The tool asks questions; answer `y` or `n`, then press `btnC` when asked.

```powershell
tpy -m tinytpu.selftest --port $port --manual --seed 4 | Tee-Object -Append "$log\4-manual.txt"
```

**Expected:** eight `PASS` lines.

- **`leds`:** the tool runs programs that halt with ISA errors 1, 4 and 6, then one that halts OK. For each, it asks whether `led[5]` / `led[4]` and the 4-bit error code on `led[15:12]` match.
  - `led[15]` is the leftmost LED, so the code reads left to right.
  - For example, error 6 is `0110`: `led[14]` and `led[13]` on.
- **`button`:** the tool leaves data in every memory and a halted program, then asks you to press `btnC`. Afterwards the core must be idle, and **every memory must still hold its data**. That is the design's reset rule.
- **LEDs afterwards:** only `led[0]` blinks. The reset cleared `led[4]` and the sticky `led[6]`.

### 4.5 Longer run

Run the self-test with 20 different seeds and more work per run. It takes a few minutes, and also lets the board warm up:

```powershell
1..20 | ForEach-Object {
    tpy -m tinytpu.selftest --port $port --seed $_ --count 20 --rounds 100 *>> "$log\4-soak.txt"
    "seed $_ exit $LASTEXITCODE" | Tee-Object -Append "$log\4-soak.txt"
}
```

**Expected:** `exit 0` for all 20 seeds, and 0 resends in every summary.

A failure that only appears here, especially after the board has warmed up, matters most to report. The design's hold-time margin is small (+0.035 ns), though Vivado signs it off.

**If stage 4 passes, H5.5 is done.**

### 4F. Full design at 115,200 baud (only after stage 3F)

```powershell
.\syn\vivado.ps1 -Top tinytpu_top -Variant 115k -Program
tpy -m tinytpu.selftest --port $port --baud 115200 *>> "$log\4F-selftest-115k.txt"; Get-Content "$log\4F-selftest-115k.txt" -Tail 12
```

- Repeat 4.3 to 4.5 with `--baud 115200` added to every command.
- The `info` line must show `CLKS_PER_BIT = 868`.
- Everything is about 9 times slower.
- This build's hold margin is +0.010 ns.

## 5. What to send back

- The whole `tinytpu-board-test` folder from your desktop, or just paste me the files.
- Anything that differed from **Expected**, especially:
  - any `FAIL` line, with its command (the `--seed` matters)
  - LEDs that didn't match
  - non-zero resend, timeout or NAK counts
- The numbers: ms per round trip (3.2) and ms per INFER (4.2).
- Which COM port the UART was, and whether the 1 ms latency timer is set.
- Whether you needed the 115k fallback.

Then tick H0.3, H0.4, H2.6 and H5.5 in `TASKS.md`, or tell me to.

## Troubleshooting

| Symptom | Likely cause and what to try |
|---|---|
| `-Program`: "no xc7a35t found" | The board is off, the cable only charges, or the cable driver is missing (Vivado's `install_drivers`, see `tools/setup.md`). Close other Vivado / Hardware Manager windows. |
| Programmed, but `led[0]` doesn't blink | Wrong bitstream, or a reset is stuck: check that `btnC` isn't held. Program again. |
| `--probe`: no port answers | Wrong design loaded (bring-up vs full: use the matching tool's `--probe`), `sw[0]` on for the loopback design, or the port is held by Hardware Manager or a terminal. Close them and probe again. |
| `LinkError: no reply after 3 resends` | Wrong port; a 115k build with no `--baud 115200` (or the reverse); or the design is not loaded. |
| `led[5]` (loopback) or `led[8]` (full design) on | UART framing errors: a baud mismatch between the host and the bitstream. |
| `led[9]` on (full design) | A byte was dropped because the receive queue was full. The host tools never cause this. Report it. |
| Resends or timeouts, but checks pass | A noisy link; the protocol recovered. Try another USB port or cable, and note the counts. |
| A wrong result (`matmul`, `network`, `example`) | A real mismatch. Keep the full output and the seed. Run the same command again to see whether it repeats. |
| Failures that appear only after warming up | Possibly the small hold margin. Note how long the board had been on, and report it. |

## Rebuilding the bitstreams

Only needed if the files are missing or the RTL changed. Each build takes a few minutes and fails, without a bitstream, if timing or a resource ceiling is not met:

```powershell
.\syn\vivado.ps1 -Top loopback_top
.\syn\vivado.ps1 -Top tinytpu_top
.\syn\vivado.ps1 -Top loopback_top -Variant 115k
.\syn\vivado.ps1 -Top tinytpu_top -Variant 115k
```

Variants are defined in `syn/build.tcl` (table `variants`). Only the default `tinytpu_top` build is compared with `syn/baseline.json`.
