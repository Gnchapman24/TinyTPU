# UART protocol

This page defines the bytes exchanged between the host and TinyTPU over the Basys3 USB-UART. What each command *does* is defined in the [ISA](isa.md#host-interaction). This page is the wire format.

## Link

| Property | Value |
|---|---|
| Rate | 1,000,000 baud (`CLKS_PER_BIT` = 100 at 100 MHz) |
| Byte framing | 8N1. The receiver flags a framing error when a stop bit is low. |
| Flow control | none in hardware; **one request in flight at a time**: the host waits for each reply before sending the next request |
| Byte order | little-endian for every multi-byte field, **except the CRC**, which is sent high byte first |
| Windows host | set the FTDI latency timer to 1 ms (see `tools/setup.md`) |

## Packet layout

### Request (host → FPGA)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `SYNC` | `0xA5` |
| 1 | 1 | `CMD` | command code (see [Commands](#commands)) |
| 2 | 1 | `SEQ` | sequence number, 0–255; the host increments it for each new command |
| 3 | 2 | `LEN` | payload length in bytes, 0–4096 |
| 5 | `LEN` | payload | depends on the command |
| 5 + `LEN` | 2 | `CRC` | CRC-16/CCITT-FALSE over offsets 1 … 4 + `LEN`, **high byte first** |

### Reply (FPGA → host)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `SYNC` | `0x5A`, different from requests so a reply can never be taken for a request |
| 1 | 1 | `CODE` | `0x00` ACK, otherwise a NAK code (see [Reply codes](#reply-codes)) |
| 2 | 1 | `SEQ` | copied from the request |
| 3 | 2 | `LEN` | payload length in bytes. NAK replies have `LEN = 0`. |
| 5 | `LEN` | payload | depends on the command |
| 5 + `LEN` | 2 | `CRC` | as for requests |

**The CRC is CRC-16/CCITT-FALSE:** polynomial `0x1021`, initial value `0xFFFF`, no reflection, final XOR `0x0000`. Check value: `"123456789"` → `0x29B1`.

## Memories and units

| ID | Memory | Bytes per unit on the wire (N = 8) | Unit contents |
|---|---|---|---|
| 0 | `IMEM` | 16 | one instruction, least-significant byte first |
| 1 | `WBUF` | N² = 64 | one weight tile. Byte order within the tile is set by the memory map. |
| 2 | `ABUF` | N = 8 | one activation vector, element 0 first |
| 3 | `ACC` | 4N = 32 | N × INT32, column 0 first, each little-endian |
| 4 | `PTAB` | 8 | `bias` i32, `M` u16, `s` u8 (bits 7:6 must be 0), 1 pad byte (must be 0) |

Addresses and counts are in units, never bytes. Each memory's depth is reported by `INFO`.

## Commands

| Code | Command | Request payload | ACK payload |
|---|---|---|---|
| `0x01` | `INFO` | none | [Info block](#info-block) (16 B) |
| `0x02` | `WRITE` | `mem` u8, `0` u8, `addr` u16, `count` u16, then `count × unit` bytes | none |
| `0x03` | `READ` | `mem` u8, `0` u8, `addr` u16, `count` u16 | `count × unit` bytes |
| `0x04` | `RUN` | `start_pc` u16 | [Status block](#status-block), sent **when the program halts** |
| `0x05` | `STATUS` | none | Status block |
| `0x06` | `ABORT` | none | Status block |
| `0x07` | `RESET` | none | Status block (state = idle) |
| `0x08` | `INFER` | `in_addr` u16, `start_pc` u16, `out_mem` u8, `0` u8, `out_addr` u16, `out_count` u16, then input bytes | Status block, then `out_count × unit` bytes of output |

**`LEN` must match exactly:**
- `WRITE`: 6 + `count × unit`
- `INFER`: 10 + input bytes, which must be a whole number of `ABUF` vectors, written from `in_addr` upward

Anything else gets NAK `0x03`.

**While a program is running,** only `INFO`, `STATUS` and `ABORT` are accepted. Every other command gets NAK `0x06` (busy). `RUN` and `INFER` reply only once the program halts, and the receiver keeps listening meanwhile, so `ABORT` can still be sent.

`INFER` replies with the status block even if the program halted with an error. In that case its output bytes are whatever the output region holds.

### Info block

| Offset | Size | Field |
|---|---|---|
| 0 | 1 | protocol version (currently 1) |
| 1 | 1 | N |
| 2 | 2 | `CLKS_PER_BIT` |
| 4 | 2 | `IMEM` depth |
| 6 | 2 | `WBUF` depth |
| 8 | 2 | `ABUF` depth |
| 10 | 2 | `ACC` depth |
| 12 | 2 | `PTAB` depth |
| 14 | 2 | maximum payload bytes (4096) |

### Status block

| Offset | Size | Field | Values |
|---|---|---|---|
| 0 | 1 | state | 0 idle, 1 running, 2 halted OK, 3 halted error |
| 1 | 1 | error | 0 none, or an [ISA error code](isa.md#errors) 1–6 |
| 2 | 2 | PC | PC of the `HALT`, of the faulting instruction, or of the current instruction |
| 4 | 2 | halt code | from the `HALT` instruction |
| 6 | 2 | reserved | 0 |

## Reply codes

| Code | Meaning | Side effects |
|---|---|---|
| `0x00` | ACK | the command was executed |
| `0x01` | bad CRC | `WRITE`/`INFER` data may have been written (see below); nothing executed |
| `0x02` | unknown command | none |
| `0x03` | bad length | none |
| `0x04` | payload too long (`LEN` > 4096) | none |
| `0x05` | bad argument: unknown memory, `addr + count` beyond depth, count 0, or a non-zero reserved byte | none |
| `0x06` | busy | none |

A request that times out partway through gets **no reply** (see [Timeouts](#timeouts-and-recovery)).

Reply codes describe the **packet**. Errors from the **program** (ISA error codes 1–6) are reported in the status block's error field, inside an ACK.

## Robustness rules

### Write-through

`WRITE` and `INFER` stream payload bytes straight into the target memory as they arrive. There is no packet buffer, which keeps BRAM free for weights.

- The CRC is checked after the last byte. If it fails, the FPGA replies NAK `0x01`, and the written range may hold bad data. The host must resend; `WRITE` is safe to repeat.
- **Nothing executes before its CRC passes.** `RUN` and `INFER` start the sequencer only after a good CRC.
- Arguments are checked from the header and fixed fields **before** any payload byte is written. A bad argument writes nothing: the remaining payload is read and discarded, and the FPGA replies NAK `0x05`.

### Duplicate requests (`SEQ`)

The FPGA remembers the `SEQ` of the last request it executed. If the next request is identical in `CMD` and `SEQ`, and its CRC is good, the FPGA **does not execute it again**. It rebuilds the reply instead:

| Command | Reply to a duplicate |
|---|---|
| `WRITE` | ACK; the data is **not** rewritten |
| `RUN` | current status block |
| `INFER` | current status block + output re-read from memory |
| `READ`, `INFO`, `STATUS` | executed normally; they have no side effects |
| `ABORT`, `RESET` | executed normally; they are idempotent |

This covers the case where a request arrived but its reply was lost: the host's resend must not, for example, run an accumulating program twice. The host must use a new `SEQ` (incrementing, wrapping at 255) for every new command, and the same `SEQ` only for a resend.

### Timeouts and recovery

| Side | Rule |
|---|---|
| FPGA | While idle, ignore every byte except `0xA5`. Inside a packet, a gap of more than **5 ms** between bytes discards the packet and returns to idle, with no reply. |
| Host | Send each packet with a single `write()` call, so the OS does not split it. |
| Host | Reply timeout: 100 ms by default; set per call for `RUN` and `INFER`, which can take longer. |
| Host | Before any resend, wait **10 ms with no traffic**. That's longer than the FPGA's 5 ms timeout, so the receiver is guaranteed to be idle and synchronized again. |
| Host | After 3 failed resends, send `RESET` and report an error to the caller. |

After a framing error or a bad CRC, a data byte that happens to equal `0xA5` can start a false packet. Its header or CRC then fails, and the 10 ms silence rule brings both sides back into sync.

## Example: one inference

Using the [ISA example](isa.md#example-a-196-64-10-network): write 25 input vectors to `ABUF` 0, run from PC 0, and read 2 vectors from `ABUF` 40.

```text
request:  A5 08 2A  D2 00                     CMD=INFER, SEQ=0x2A, LEN=210 (10 + 200)
          00 00  00 00  02 00  28 00  02 00   in_addr=0, start_pc=0, out_mem=ABUF, 0, out_addr=40, out_count=2
          <200 input bytes>
          <CRC hi> <CRC lo>

reply:    5A 00 2A  18 00                     ACK, SEQ=0x2A, LEN=24 (8 + 16)
          02 00  04 00  01 00  00 00          halted OK, error 0, PC=4, halt code=1, reserved
          <16 output bytes: 10 INT8 logits + 6 zero pad>
          <CRC hi> <CRC lo>
```

That is 248 bytes, about 2.5 ms on the wire at 1 Mbaud, plus one round trip of host latency.
