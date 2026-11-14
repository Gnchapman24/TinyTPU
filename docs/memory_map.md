# Memory map

This page defines the size, implementation and byte layout of every TinyTPU memory. The [ISA](isa.md) defines how instructions address these memories. The [UART protocol](uart_protocol.md) defines how the host reads and writes them.

All numbers are the defaults for the build target (N = 8 on the xc7a35t). Every depth is an RTL parameter, and the hardware reports its actual depths in the protocol's `INFO` reply.

## Summary

| Memory | ID | Depth (units) | Unit | Data width | Built from | RAMB36 | Capacity |
|---|---|---|---|---|---|---|---|
| `IMEM` | 0 | 128 | instruction | 128 b | LUT RAM | 0 | 128 instructions |
| `WBUF` | 1 | 2048 | N×N weight tile | 64 b per row; N rows per tile | BRAM | 32 | 128 KB of weights |
| `ABUF` | 2 | 1024 | N-element vector | 64 b | BRAM | 2 | 8 KB of activations |
| `ACC` | 3 | 512 | row of N × INT32 | 256 b | BRAM | 4 | 512 rows |
| `PTAB` | 4 | 512 | output-channel parameters | 54 b | BRAM | 1 | 512 channels |
| **Total** | | | | | | **39 of 50 (78%)** | |

That stays within the BRAM budget of 80% (40 RAMB36). The remaining blocks are kept free for ILA debug cores.

## Why these choices

- **Weights stay on-chip,** so the weight buffer limits model size. It gets almost all the BRAM: 32 blocks of 4 KB each.
- **`IMEM` is LUT RAM** (about 300 LUTs). Macro-instruction programs are tiny, and this saves 2 RAMB36, which is 8 KB more weights.
- **`PTAB` stays in BRAM.** In LUT RAM it would cost about 450 LUTs to save a single block.
- **`WBUF` and `ABUF` are N bytes wide,** one weight-tile row or one vector per cycle. A weight tile loads into the array in N cycles, one row per cycle, which is how weights shift into a weight-stationary array anyway. A wider `WBUF` would load tiles faster, but compute is never the bottleneck: UART is.
- **All depths are powers of two,** so address decoding is simple and field widths come straight from `$clog2(depth)`.

## Model limits

The host compiler must reject models that exceed these limits. It reads the actual depths with `INFO`.

| Limit | Default | Comes from |
|---|---|---|
| Total weight tiles across all layers | 2048 (128 KB) | `WBUF` depth |
| Total output channels across all layers, padded to N | 512 | `PTAB` depth |
| Activation vectors in use at once (input + layer outputs) | 1024 (8 KB) | `ABUF` depth |
| Accumulator rows per layer, M × NT | 512 | `ACC` depth |
| Program length, including `HALT` | 128 instructions | `IMEM` depth |

**Examples:**
- 196 → 64 → 10 (13 KB of weights, 80 channels, 42 vectors) uses about 10% of each buffer.
- 784 → 128 → 10 (about 100 KB of weights: 1600 tiles, 144 channels) fits.
- 784 → 256 → 10 (about 200 KB) does not fit.

## Byte layout of each unit

### Weight tile (`WBUF`)

A tile holds `W[k·N + i][n·N + j]` for i, j in 0 … N−1: the weights connecting input element `k·N + i` to output channel `n·N + j`.

- Tile row `i` is stored at physical `WBUF` row `tile·N + i`, where `tile` is the ISA-level tile address. Row `i` holds the weights that multiply activation element `i`.
- Byte `j` of a row (bits `[8j+7 : 8j]`) is the weight for output column `j`.
- On the wire, a tile is its N rows in order, each row byte 0 first.

In NumPy, with `W` shaped `[K, Nout]` (padded) and dtype `int8`:

```python
tile_bytes = W[k*N:(k+1)*N, n*N:(n+1)*N].tobytes()   # row-major: exactly the wire and memory order
```

No transposes are needed anywhere.

### Activation vector (`ABUF`)

Byte `j` (bits `[8j+7 : 8j]`) is element `j`, i.e. input `k·N + j` of the current K-tile. It is INT8 or UINT8 depending on the instruction's mode bits; the memory doesn't care. On the wire: element 0 first.

```python
vec_bytes = x[k*N:(k+1)*N].astype(np.uint8 if unsigned else np.int8).tobytes()
```

### Accumulator row (`ACC`)

32-bit word `j` (bits `[32j+31 : 32j]`) is the signed INT32 accumulator for output column `j`. On the wire: column 0 first, each word little-endian, i.e. `row.astype('<i4').tobytes()`.

### Parameter entry (`PTAB`)

| Bits in memory | Field | Type |
|---|---|---|
| `[31:0]` | `bias` | INT32 |
| `[47:32]` | `M` | u16 |
| `[53:48]` | `s` | u6 |

On the wire, an entry is 8 bytes: `bias` (4, LE), `M` (2, LE), `s` (1, bits 7:6 = 0), pad (1, = 0).

```python
entry = struct.pack('<iHBx', bias, M, s)
```

### Instruction (`IMEM`)

128 bits as defined in the [ISA](isa.md#instruction-format). On the wire: 16 bytes, least-significant byte first.

## Implemented field widths

Only as many bits of each ISA and protocol field are implemented as the memory needs. Any higher bit must be zero: in an instruction, otherwise ISA error 2; in a protocol request, otherwise NAK `0x05`.

| Field | Implemented bits (default) | Rule |
|---|---|---|
| `w_base` | 11 | `$clog2(WBUF depth)` |
| `a_base`, `a_dst`, protocol `in_addr` | 10 | `$clog2(ABUF depth)` |
| `acc_base` | 9 | `$clog2(ACC depth)` |
| `p_base` | 9 | `$clog2(PTAB depth)` |
| `start_pc` | 7 | `$clog2(IMEM depth)` |
| `kt`, `nt`, `m` | 12 | fixed; a count can be up to the full depth of the memory it walks |
| Protocol `addr`, `out_addr` | per memory | `$clog2` of the addressed memory's depth |

A count within its width that still makes an access run past the end of a memory is caught when the address is generated (ISA error 5), or before any write in the protocol (NAK `0x05`).

## Ports and access rules

| Memory | Datapath side | Host side |
|---|---|---|
| `IMEM` | sequencer fetch (read) | read/write |
| `WBUF` | address generator, tile loads (read) | read/write |
| `ABUF` | `MATMUL` activations (read), `REQUANT` results (write) | read/write, including `INFER` input and output |
| `ACC` | `MATMUL` read-add-write, `REQUANT` read | read/write |
| `PTAB` | `MATMUL` bias and `REQUANT` M/s (read) | read/write |

- **The host accesses memories only while the sequencer is halted** (the protocol's busy rule). So each memory needs just a 2:1 mux between the datapath and the host, with no arbitration.
- **Within a memory,** `MATMUL` and `REQUANT` never run at the same time, because instructions execute one at a time.

## Timing and pipelining notes

- **`WBUF` read path.** 32 RAMB36 are spread across the device, so the read-data mux (64 bits × 32 ways, about 700 LUTs) is pipelined with 2–3 register stages. It is the main timing path to watch at 100 MHz. Its latency is fixed and is absorbed by the address generator's pipeline.
- **`ACC` read-add-write.** Each accumulation reads a row, adds the array output, and writes the row back over a few pipeline cycles. With the ISA loop order (`n`, then `k`, then `m`), the next update of the same row (`m, n`) comes after the next weight tile is loaded, at least N cycles later. That is longer than the pipeline, so no forwarding logic is needed.
  - The address-generator RTL must state this as an assumption.
  - Tests must include M = 1 (the tightest case) to check it.
