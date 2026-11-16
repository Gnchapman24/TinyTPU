# Board bring-up: UART loopback

The bring-up design `loopback_top` (task A2.4) tests the whole serial path on the Basys3 before any compute hardware exists:

PC → USB-UART → `uart_rx` → echo engine with CRC check → `uart_tx` → PC

It speaks the real [packet format](uart_protocol.md) with one bring-up-only command, `ECHO` (0x7E), which returns the payload unchanged. It also has a raw mode that echoes every byte, for a first check with a serial terminal.

## Build and program (Windows PowerShell)

From the repo folder (`\\wsl.localhost\Ubuntu\home\<user>\Projects\FPGA\TinyTPU`):

```powershell
.\syn\vivado.ps1 -Top loopback_top            # build: about 2-4 minutes
.\syn\vivado.ps1 -Top loopback_top -Program   # program the board over USB/JTAG
```

- Builds go to `%LOCALAPPDATA%\tinytpu\vivado\loopback_top\`: the bitstream, `vivado.log`, and the utilization, timing and DRC reports. The build fails if timing is not met at 100 MHz.
- Programming is volatile: the design is lost when the board is powered off.

## Switches and LEDs

| Control | Meaning |
|---|---|
| `btnC` | Reset. The design also resets itself after programming. |
| `sw[0]` | **On:** raw byte echo. **Off:** packet echo. Change it only while nothing is being sent. |
| `sw[1]` | What `led[15:8]` shows. **Off:** good-packet count. **On:** error count. |

| LED | Meaning |
|---|---|
| `led[0]` | Heartbeat: blinks about every 0.7 s while the design is running |
| `led[1]` | Receive activity |
| `led[2]` | Transmit activity |
| `led[3]` | Raw mode is on |
| `led[4]` | A CRC error was seen (stays on until reset) |
| `led[5]` | A UART framing error was seen (stays on until reset) |
| `led[6]` | A partial request timed out (stays on until reset) |
| `led[7]` | Other error: dropped byte, unknown command, or payload too long (stays on until reset) |
| `led[15:8]` | 8-bit counter, selected by `sw[1]` |

## Test 1: raw echo with a terminal

1. Set `sw[0]` on (`led[3]` lights).
2. Open the board's COM port in a serial terminal at **1,000,000 baud, 8N1, no flow control**. PuTTY accepts a custom speed.
3. Type: every character comes straight back. `led[1]` and `led[2]` flicker.

If nothing comes back:
- check the COM port number in Device Manager
- check the baud rate
- check that `led[0]` is blinking, which means the bitstream is loaded

## Finding the COM port

The Basys3's FTDI chip has two channels, JTAG and UART. Windows may create a COM port for **both**, with identical USB details. Other FT2232H boards, such as a Nandland Go Board, also match. In PowerShell, from the repo folder, with the host environment set up as in `tools/setup.md`:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\tinytpu\venv"   # unless set permanently
uv run --no-default-groups python -m tinytpu.bringup --list-ports
uv run --no-default-groups python -m tinytpu.bringup --probe
```

- `--list-ports` lists every FT2232H port.
- `--probe` sends one short ECHO packet to each and reports which one answers. It needs the bitstream loaded and `sw[0]` off.
- Close Vivado's Hardware Manager first, or it may hold the port.

## Test 2: packet echo

Set `sw[0]` off, press `btnC`, then run (with the port found above):

```powershell
uv run --no-default-groups python -m tinytpu.bringup --port COM7
```

It runs seven checks and prints `PASS`/`FAIL` for each, then a summary of link statistics (resends, timeouts):

| Check | What it does | LED afterwards |
|---|---|---|
| `ping` | 4 short echoes | — |
| `sizes` | every interesting payload length from 0 to 256, plus 100 random ones | — |
| `throughput` | 50 back-to-back 256-byte echoes; prints ms per round trip and kB/s | — |
| `bad_crc` | a deliberately corrupted request must get NAK `BAD_CRC` | `led[4]` on |
| `unknown_cmd` | `INFO` (not in the bring-up design) must get NAK `UNKNOWN_CMD` | `led[7]` on |
| `too_long` | a 300-byte payload must get NAK `TOO_LONG` | `led[7]` on |
| `timeout` | a request cut off part-way must get no reply; the next echo must work | `led[6]` on |

- The exit status is 0 only if every check passed.
- **Expected round trip:** about 5–8 ms per 256-byte echo, of which about 5.3 ms is wire time at 1 Mbaud. The rest is USB and driver latency, which is why the FTDI latency timer should be 1 ms.
- **`--raw`** (with `sw[0]` on) tests raw byte echo instead.
- Options: `--count`, `--rounds`, `--seed`, and `--baud` for the 115,200-baud fallback build (`.\syn\vivado.ps1 -Top loopback_top -Variant 115k`). Use `--help` for details.
- The [board test plan](board_test.md) runs these tests as part of a full first session on the board.

## Limits of this design

- Payloads above 256 bytes are rejected with NAK `0x04`. The final protocol allows 4096.
- Only `ECHO` is implemented. The full design (`tinytpu_top`, with every command) and its self-test are in [Using the board from a PC](host.md).
- Bytes sent while a reply is in progress are dropped (`led[7]`): the host must wait for each reply, as the protocol requires.
