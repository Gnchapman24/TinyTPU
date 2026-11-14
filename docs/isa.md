# Instruction set architecture

This page is the single source of truth for the TinyTPU ISA. The RTL (`rtl/isa_pkg.sv`), the host compiler and the testbenches all follow it. Changing an opcode, a field position or a meaning requires agreement first (AGENTS.md §8).

Two related pages fill in the rest:

- [UART protocol](uart_protocol.md): byte layout of host commands
- [Memory map](memory_map.md): buffer depths, byte layouts, and how many bits of each address field are implemented

## Execution model

TinyTPU runs a **stored program**.

1. **The host uploads**, while the sequencer is halted:
   - a program into the instruction memory
   - the model's weights into the weight buffer
   - its per-channel parameters into the parameter table
2. **For each inference, the host:**
   - writes the input into the activation buffer
   - starts the program (`RUN`, or `INFER` to do all of this in one round trip)
   - reads the output back once the program halts

The sequencer:

- Fetches one 128-bit instruction per step. The program counter (PC) only ever increments. There are no branches, loops or scalar registers.
- Executes instructions **one at a time, in order**. Each one finishes completely before the next is fetched, so instructions never conflict with each other.
- Stops at `HALT`, or at the first error (see [Errors](#errors)).

Instructions are **macro operations**: one `MATMUL` computes a whole layer. A hardware address generator runs the tile loops.

## Memories and units

All addresses are in each memory's natural unit, never in bytes. N is the array size (8 in the build).

| Memory | Host ID name | Unit (one address) | Size at N = 8 |
|---|---|---|---|
| Instruction memory | `IMEM` | one instruction (128 bits) | 16 B |
| Weight buffer | `WBUF` | one N×N INT8 weight tile | 64 B |
| Activation buffer | `ABUF` | one vector of N 8-bit activations | 8 B |
| Accumulator buffer | `ACC` | one row of N INT32 accumulators | 32 B |
| Parameter table | `PTAB` | one output channel: `{bias: INT32, M: u16, s: u6}` | 54 bits |

The depth of each memory is set in the [memory map](memory_map.md). The numeric memory IDs used on the wire are set in the [UART protocol](uart_protocol.md#memories-and-units).

## Data layout (canonical)

A layer computes `Y[M × Nout] = X[M × K] · W[K × Nout]`, in tiles of N. **KT = K / N** is the number of tiles along the inner dimension, and **NT = Nout / N** the number along the output. Data must be stored in these layouts:

| Data | Address of tile/vector/row |
|---|---|
| Weight tile `(k, n)` | `w_base + n·KT + k` |
| Activation vector `(m, k)` | `a_base + m·KT + k` |
| Accumulator row `(m, n)` | `acc_base + m·NT + n` |
| Parameters for output channel `c = n·N + col` | `p_base + c` |

- **Padding:** the host zero-pads K and Nout up to multiples of N, in the weights and in the parameter table. M is not padded.
  - Padded output channels accumulate to 0 and requantize to 0, so the padding stays zero into the next layer.
- **Chaining:** `REQUANT` writes accumulator row `(m, n)` as one activation vector at `a_dst + m·NT + n`. That is exactly the canonical activation layout for a following layer whose KT equals this layer's NT. Layers chain without any data reshuffling.

Within a weight tile, row `i` holds the weights that multiply activation element `i`, for the tile's N output columns. The byte order inside a tile, vector, row and parameter entry is defined in the [memory map](memory_map.md#byte-layout-of-each-unit).

## Instruction format

Every instruction is 128 bits, split into eight 16-bit slots. On the wire it is 16 bytes, **least-significant byte first**.

| Bits | Slot | `HALT` | `NOP` | `MATMUL` | `REQUANT` |
|---|---|---|---|---|---|
| `[7:0]` | 0 | `0x00` | `0x01` | `0x10` | `0x20` |
| `[15:8]` | 0 | 0 | 0 | flags (below) | flags (below) |
| `[31:16]` | 1 | halt code | 0 | `w_base` | 0 |
| `[47:32]` | 2 | 0 | 0 | `a_base` | `a_dst` |
| `[63:48]` | 3 | 0 | 0 | `acc_base` | `acc_base` |
| `[79:64]` | 4 | 0 | 0 | `p_base` | `p_base` |
| `[95:80]` | 5 | 0 | 0 | `kt` | 0 |
| `[111:96]` | 6 | 0 | 0 | `nt` | `nt` |
| `[127:112]` | 7 | 0 | 0 | `m` | `m` |

**Flags (bits `[15:8]`):**

| Bit | `MATMUL` | `REQUANT` |
|---|---|---|
| 8 | `act_signed`: 1 = activations are INT8, 0 = UINT8 | `relu`: 1 = output UINT8 [0, 255], 0 = INT8 [−128, 127] |
| 10:9 | `acc_mode`: `00` overwrite, `01` accumulate, `10` bias, `11` illegal | 0 |
| others | 0 | 0 |

**Reserved bits** are every 0 in the tables above, plus any field bits beyond what the memory map implements for that buffer. They must be zero.

**Legal opcodes** are only `0x00`, `0x01`, `0x10` and `0x20`. All-zero memory therefore decodes as `HALT` with code 0, so a program that is too short stops safely.

## Instructions

### `HALT` (0x00)

Stops the sequencer with status **halted OK**, and records the 16-bit halt code from slot 1 for the host. The code is free for the program to use, for example to tell the host which program ran.

### `NOP` (0x01)

Does nothing; the PC advances.

### `MATMUL` (0x10)

One layer of matrix multiplication into the accumulator buffer:

```text
for n in 0 .. nt-1:                                   # output-column tile
  for k in 0 .. kt-1:                                 # inner-dimension tile
    W  = WBUF[w_base + n*kt + k]                      # N×N INT8, signed
    for m in 0 .. m-1:                                # activation rows
      x   = ABUF[a_base + m*kt + k]                   # N × 8-bit, INT8 if act_signed else UINT8
      p   = x · W                                     # N × products, exact
      r   = acc_base + m*nt + n
      if k == 0 and acc_mode == overwrite:   ACC[r] = p
      if k == 0 and acc_mode == bias:        ACC[r] = p + PTAB[p_base + n*N + col].bias   (per column)
      otherwise (k > 0, or accumulate):      ACC[r] = ACC[r] + p
```

- All accumulation is signed INT32 and **wraps** on overflow. The host keeps K within the no-overflow bound (AGENTS.md §2).
- `accumulate` mode lets a layer whose K is too large for one instruction be split across several `MATMUL`s.
- The loop order above defines the **result**. The hardware may reorder or overlap work, for example by prefetching the next weight tile, as long as the result is identical.

### `REQUANT` (0x20)

Requantizes accumulator rows to 8 bits and writes them into the activation buffer (write-back):

```text
for n in 0 .. nt-1:
  for m in 0 .. m-1:
    for col in 0 .. N-1:
      c   = n*N + col
      acc = ACC[acc_base + m*nt + n][col]                       # INT32
      M, s = PTAB[p_base + c].M, PTAB[p_base + c].s
      R   = 0 if s == 0 else 2**(s-1)                           # round half up
      y   = clamp((acc * M + R) >>> s, lo, hi)                  # full-width product, arithmetic shift
      ABUF[a_dst + m*nt + n][col] = y
```

`lo, hi` = `0, 255` when `relu = 1`, else `-128, 127`.

- To feed the result into the next layer, set that layer's `MATMUL` `act_signed` to `0` after a ReLU (UINT8), or `1` without one (INT8).
- Raw INT32 results are never written by an instruction. The host reads them directly from `ACC` while the sequencer is halted.

## Errors

The sequencer halts with status **halted error**, an error code, and the faulting PC when any of these happen. The host sees them in the status block's error field (see the [UART protocol](uart_protocol.md#status-block)).

| Code | Condition |
|---|---|
| 1 | Illegal opcode |
| 2 | Non-zero reserved bit, including address/count bits beyond the implemented width |
| 3 | `MATMUL` with `acc_mode = 11` |
| 4 | A count (`kt`, `nt`, `m`) of 0 |
| 5 | A generated address at or beyond its memory's depth |
| 6 | The PC ran past the end of instruction memory without a `HALT`. The reported PC is the IMEM depth, one past the last instruction. |

The executable reference for all of this is `tinytpu.model.Machine` ([API](api/model.md)), which the hardware is tested against.

Codes 1–4 are detected at decode, before any memory is changed. Code 5 is detected by the address generator as addresses are produced, so the instruction may have partly executed. Code 6 is detected at fetch.

## Host interaction

Host commands are not instructions. They are handled by the protocol engine around the sequencer. Byte layouts are in the [UART protocol](uart_protocol.md).

| Command | Effect | Allowed while running |
|---|---|---|
| `INFO` | Returns protocol version, N, `CLKS_PER_BIT`, and the depth of every memory | yes |
| `WRITE` | Writes `count` units to a memory starting at an address | no (NAK, busy) |
| `READ` | Returns `count` units from a memory starting at an address | no (NAK, busy) |
| `RUN` | Starts the sequencer at a PC. **The reply is sent when the program halts**, with the final status | no |
| `STATUS` | Returns state (idle / running / halted OK / halted error), PC, halt code, error code | yes |
| `ABORT` | Stops a running program and returns its status | yes |
| `RESET` | Returns the sequencer and protocol state to idle; memory contents are kept. The `btnC` button does the same | yes |
| `INFER` | Writes input data to `ABUF`, runs from a PC, then replies with status plus `count` units read from a memory: a full inference in one round trip | no |

## Example: a 196 → 64 → 10 network

Batch size 1 (M = 1), N = 8. Layer 1 has ReLU and the input pixels are unsigned. Layer 2 outputs signed logits, and the host takes the argmax.

**Buffer allocation:**

| Data | Memory | Addresses |
|---|---|---|
| Layer 1 weights (KT = 25, NT = 8 → 200 tiles) | `WBUF` | 0–199 |
| Layer 2 weights (KT = 8, NT = 2 → 16 tiles; Nout padded 10 → 16) | `WBUF` | 200–215 |
| Input (196 pixels padded to 200 → 25 vectors) | `ABUF` | 0–24 |
| Layer 1 output (64 channels → 8 vectors) | `ABUF` | 32–39 |
| Layer 2 output (16 channels, 10 used → 2 vectors) | `ABUF` | 40–41 |
| Layer 1 / layer 2 accumulators | `ACC` | 0–7 / 8–9 |
| Layer 1 / layer 2 parameters | `PTAB` | 0–63 / 64–79 |

**Program:**

| PC | Instruction | 128-bit value (slot 7 … slot 0) | Bytes on the wire (LSB first) |
|---|---|---|---|
| 0 | `MATMUL w=0 a=0 acc=0 p=0 kt=25 nt=8 m=1, act_signed=0, bias` | `0001 0008 0019 0000 0000 0000 0000 0410` | `10 04 00 00 00 00 00 00 00 00 19 00 08 00 01 00` |
| 1 | `REQUANT a_dst=32 acc=0 p=0 nt=8 m=1, relu=1` | `0001 0008 0000 0000 0000 0020 0000 0120` | `20 01 00 00 20 00 00 00 00 00 00 00 08 00 01 00` |
| 2 | `MATMUL w=200 a=32 acc=8 p=64 kt=8 nt=2 m=1, act_signed=0, bias` | `0001 0002 0008 0040 0008 0020 00c8 0410` | `10 04 c8 00 20 00 08 00 40 00 08 00 02 00 01 00` |
| 3 | `REQUANT a_dst=40 acc=8 p=64 nt=2 m=1, relu=0` | `0001 0002 0000 0040 0008 0028 0000 0020` | `20 00 00 00 28 00 08 00 40 00 00 00 02 00 01 00` |
| 4 | `HALT code=1` | `0000 0000 0000 0000 0000 0000 0001 0000` | `00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00` |

**Per inference:** one `INFER` writes 25 vectors (200 bytes) to `ABUF` 0, runs from PC 0, and replies with `ABUF` 40–41 (16 bytes). The host uses the first 10.
