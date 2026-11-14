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

## Test 2: packet echo

Set `sw[0]` off. Each `ECHO` request (payload up to 256 bytes) gets an ACK reply with the same payload and a valid CRC.
- A corrupted request gets a NAK (`led[4]`).
- An unknown command gets a NAK (`led[7]`).
- A request cut off part-way gets no reply after 5 ms (`led[6]`).

The host test script for this is task A2.5. Until it exists, a packet can be built in Python with `tinytpu.packet.build_request(Cmd.ECHO, seq, payload)` and sent with pyserial.

## Limits of this design

- Payloads above 256 bytes are rejected with NAK `0x04`. The final protocol allows 4096.
- Only `ECHO` is implemented. The real command engine is task A4.1.
- Bytes sent while a reply is in progress are dropped (`led[7]`): the host must wait for each reply, as the protocol requires.
