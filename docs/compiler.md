# Host compiler

!!! note "Design (task J7.1, approved 2026-11-16)"
    This page is the design for Phase 7 in `TASKS.md`; the modules are being built task by task. The choices made with it are listed under [Decisions](#decisions).

The compiler turns a small neural network, written as NumPy arrays, into everything the board needs to run it: a program, weight tiles and per-channel parameters. The runtime then loads those onto the board once and runs inferences with one `INFER` round trip per batch.

The board itself is unchanged: no new opcodes, fields or commands. Everything here is host-side Python in the `tinytpu` package.

## Scope

Decided 2026-11-16 (AGENTS.md §3, "Host compiler"):

- **Input:** the project's own layer API, with float weights and biases, plus calibration inputs. The compiler quantizes. Pre-quantized integer layers are accepted too.
- **Layers:** fully connected (`Dense`) with an optional ReLU. Nothing else: the ISA has no strides, so convolutions can't run on-chip.
- **Output:** the last layer's raw INT32 accumulators. The host converts them to float logits.
- **Demo:** MNIST, with an MLP trained in pure NumPy.
- **No new dependencies.** NumPy only, so everything runs on the Windows host environment too.

## Pipeline

```
nn.Model (float)  ──quantize──▶  quant.QModel (integers)  ──compile──▶  Compiled  ──load──▶  Runner
  + calibration inputs              forward_int(): the          program, WBUF tiles,     infer(x) on a
                                    bit-exact reference         PTAB entries, addresses  Device or Machine
```

There are four levels of results, and each is checked against the one before:

| Level | Computed by | Compared with the level before |
|---|---|---|
| 1. Float | `nn.Model.forward` | — |
| 2. Integer reference | `quant.QModel.forward_int`: `model.matmul` and `model.requant`, chained | Approximately (accuracy, error statistics) |
| 3. Compiled, on `Machine` | `Runner` with the `model.Machine` backend | **Bit-exact** |
| 4. Board, or simulated board | `Runner` with a `Device` | **Bit-exact** |

Level 2 is the ground truth for everything below it. Its integer math is entirely `tinytpu.model`'s, so it is never re-derived.

## Layer API (`tinytpu.nn`)

```python
from tinytpu import nn

net = nn.Model([
    nn.Dense(w1, b1, relu=True),    # w1: (784, 128) float, b1: (128,) float
    nn.Dense(w2, b2),               # w2: (128, 10);  the last layer gives the logits
])
y = net.forward(x)                  # x: (B, 784) float -> (B, 10) float
```

- **Weights are `(K, Nout)`**, so a layer computes `x @ w + b`, the same convention as `tinytpu.model.matmul`. A PyTorch `nn.Linear` weight is `(Nout, K)`, so transpose it.
- `b` is optional and defaults to zeros.
- Each layer's K must equal the previous layer's Nout. `nn.Model` checks this when it is built.
- **ReLU on the last layer** is applied on the host, to the float logits, because the last layer is never requantized.

## Quantization (`tinytpu.quant`)

```python
from tinytpu import quant

qnet = quant.quantize(net, calib_x)        # calib_x: (C, 784) representative inputs
acc = qnet.forward_int(quant_x)            # level 2: INT32 accumulators of the last layer
logits = qnet.dequantize(acc)              # float, same scale as net.forward
```

All quantization is symmetric, with no zero-points (AGENTS.md §2). Every scale is a positive float64.

### Weights: one scale per output channel

- `s_w[c] = max|W[:, c]| / 127`, so `q_w = round(W / s_w)` lies in −127…127.
- **An all-zero channel** gets `s_w[c] = 1 / 127`, so its bias can still be represented.

### Activations: one scale per layer input

Each layer's input `x` is represented as `x ≈ s_x · q_x`.

- **Network input:** from the calibration inputs.
  - Never negative: UINT8, `s_x = max(x) / 255`. MNIST pixels scaled to 0…1 therefore become exactly their original bytes.
  - Otherwise: INT8, `s_x = max|x| / 127`.
  - `quantize(..., input_scale=, input_signed=)` overrides both.
- **Hidden layer outputs:** from the float activations of the calibration inputs (level 1).
  - After a ReLU: UINT8, `s_y = max(y) / 255`. The next layer runs with `act_signed = 0`.
  - Without one: INT8, `s_y = max|y| / 127`. The next layer runs with `act_signed = 1`.
- The calibration statistic is the maximum. `calibration="percentile"` with `percentile=99.99` is an option, for activations with rare outliers.
- **A layer whose calibration activations are all zero** gets scale 1, so the arithmetic stays defined. A warning is printed.

### Biases

- `q_b[c] = round(b[c] / (s_x · s_w[c]))`, as INT32.
- The overflow rule of AGENTS.md §2 is checked per layer: `K · 32385 + max|q_b| < 2**31`. A layer that breaks it is rejected, with the layer index and the numbers, rather than clipped.

### Requantization: choosing `M` and `s`

A hidden layer's float output is `y[c] ≈ s_x · s_w[c] · acc[c]`, so its quantized output is `q_y[c] ≈ r[c] · acc[c]`, with the real multiplier `r[c] = s_x · s_w[c] / s_y`. The hardware computes `clamp((acc · M + R) >>> s)` with `M / 2**s ≈ r`, so:

- `s[c]` is the **largest** shift in 0…63 with `round(r[c] · 2**s) ≤ 65535`, and `M[c] = round(r[c] · 2**s[c])`. This keeps the most precision: unless `s` is 63, `M ≥ 32768`, a full 16 significant bits.
- **`r[c] > 65535.5`** can't be represented, even at `s = 0`, and is rejected. It would mean an output scale more than 65,535 times smaller than the input scale.
- **`r[c]` below `2**-64`** gives `M = 0`, and the channel's output is always 0. A warning is printed.
- `acc · M` fits the requantizer's 48-bit product: |acc| < 2³¹ and M < 2¹⁶.
- The clamp range comes from the ReLU bit (AGENTS.md §2): UINT8 with ReLU, INT8 without. So the hardware's clamp is the quantized ReLU.

### The last layer

The last layer is `MATMUL` only (bias mode). Its accumulators are the result:

- `logits[c] = acc[c] · s_x · s_w[c]`, in float64.
- **Argmax must use these logits, not `acc`:** each channel has its own scale.
- Its PTAB entries carry the bias. Their `M` and `s` are 0, because nothing reads them.

### Rounding

Every float-to-integer rounding in the quantizer (weights, biases, inputs, `M`) is **round to nearest, ties away from zero**, written out explicitly. `np.round` rounds ties to even, so it is not used. This only has to be consistent: the integer reference starts from the rounded values.

### Pre-quantized layers

`quant.QModel` can also be built directly from integer layers (`q_w`, `q_b`, `M`, `s`, ReLU, input signedness), with optional scales for dequantization. Tests use this to cover integer corner cases without going through floats.

## Code generation (`tinytpu.compiler`)

```python
from tinytpu import compiler

target = compiler.Target.from_info(dev.info())   # or compiler.Target.default(n=8)
prog = compiler.compile_model(qnet, target)      # batch size: the largest that fits (below)
prog.save("mnist.npz")
prog = compiler.Compiled.from_file("mnist.npz")   # e.g. on another PC
```

`Target` holds N and every memory depth. `Target.default(n)` uses the `docs/memory_map.md` depths, so a model can be compiled without a board and loaded later. Loading checks the target against the board's `INFO`.

### Program

For layers 1 … L, with `KT_l = ceil(K_l / N)` and `NT_l = ceil(Nout_l / N)`:

```
for each hidden layer l < L:
    MATMUL  w_base=W_l a_base=IN_l acc_base=0 p_base=P_l kt=KT_l nt=NT_l m=B act_signed=… bias
    REQUANT a_dst=OUT_l acc_base=0 p_base=P_l nt=NT_l m=B relu=relu_l
MATMUL  w_base=W_L a_base=IN_L acc_base=0 p_base=P_L kt=KT_L nt=NT_L m=B act_signed=… bias
HALT    tag
```

That is 2L instructions, so up to 64 layers fit in the default 128-entry `IMEM`.

### Memory allocation

| Memory | Allocation |
|---|---|
| `IMEM` | The program, from 0. |
| `WBUF` | Layers packed in order: `W_1 = 0`, `W_(l+1) = W_l + NT_l · KT_l`. |
| `PTAB` | Layers packed in order: `P_1 = 0`, `P_(l+1) = P_l + NT_l · N`. |
| `ACC` | **Every layer uses rows from 0.** Instructions run strictly in order, so `REQUANT` has read layer l's rows before layer l+1's `MATMUL` overwrites them. The last layer's `B · NT_L` rows are the output. |
| `ABUF` | **Two regions, used alternately.** Region A starts at 0 and holds the input (`B · KT_1` vectors). Layer 1 writes region B, layer 2 writes A, and so on. Each region is as large as the largest block placed in it, and B starts where A ends. A layer never reads and writes the same region. |

- Padding follows the ISA: K and Nout are zero-padded to multiples of N, in the weights and in PTAB (bias 0, `M = 0`, `s = 0`). Padded channels therefore stay 0 all the way through, and the next layer's padded weight rows are 0 as well.
- `INFER` writes the input at `in_addr = 0` and reads `B · NT_L` rows of `ACC` from 0.

### Batch size

The program's `m` field is fixed, so a compiled model has a fixed batch size B.

- By default, B is the largest batch that fits every limit below. In practice that is almost always INFER's 4096-byte payload. `compile_model(..., batch=B)` sets it explicitly.
- The runtime splits any number of inputs into batches of B and pads the last one with zeros. The padding costs UART time only on that last batch.
- Version 1 always uses `INFER`. A larger batch, sent with `WRITE` + `RUN` + `READ` (three round trips), could be added later; it wouldn't be faster on the wire.

### Limit checks

`compile_model` checks everything before generating anything, and raises `compiler.FitError` listing **every** limit that is exceeded, with the numbers:

| Limit | Condition | From |
|---|---|---|
| Program length | `2L ≤ IMEM` | `docs/memory_map.md` |
| Weight tiles | `Σ NT_l · KT_l ≤ WBUF` | `docs/memory_map.md` |
| Channels | `Σ NT_l · N ≤ PTAB` | `docs/memory_map.md` |
| Accumulator rows | `B · NT_l ≤ ACC`, every layer | `docs/memory_map.md` |
| Activation vectors | size of A + size of B `≤ ABUF` | `docs/memory_map.md` |
| Count fields | `KT_l`, `NT_l`, `B` ≤ 4095 | `docs/isa.md` (12-bit counts) |
| INFER request | `10 + B · KT_1 · N ≤ 4096` | `docs/uart_protocol.md` |
| INFER reply | `8 + B · NT_L · 4N ≤ 4096` | `docs/uart_protocol.md` |
| Accumulator overflow | per layer, checked by the quantizer | AGENTS.md §2 |

The generated program is then checked with `model.range_error` and `isa.decode_error`, so a compiler bug that would make the board halt with an error is caught at compile time instead.

### HALT tag

- The `HALT` code is a 16-bit tag: the low 16 bits of a hash of the program, tiles and parameters.
- After every `INFER`, the runtime checks that the board halted OK **with that tag**. This catches running against a different program, for example after `selftest` has overwritten the memories.
- It can't detect a changed `WBUF` under an unchanged program. `Runner.verify()` reads the program, the weights and the parameters back and compares them, for when that matters.

## Runtime

```python
runner = prog.load(dev)              # writes IMEM, WBUF and PTAB; checks N and depths against INFO
logits = runner.infer(x)             # (any number, 784) float -> (that number, 10) float
acc = runner.infer_int(qx)           # integers in, INT32 accumulators out (for bit-exact checks)

sim = prog.simulator()               # the same interface on a fresh model.Machine
```

- **Two backends, one interface:** a `Device` (the board, or the simulated board in cocotb) or a `model.Machine`. This is how levels 3 and 4 are compared.
- `infer` quantizes the input with the model's input scale (round, then clamp to the input type's range), packs it into canonical vectors, runs one `INFER` per batch, unpacks the `ACC` rows, drops the padding, and dequantizes.
- An `INFER` whose status is not HALTED_OK with the expected tag raises `compiler.RunError`, which carries the status.
- Loading refuses a backend whose N or memory depths differ from the target's, or whose packet limit is smaller.
- `Compiled.from_file` recomputes the tag from the image and refuses a file that doesn't match it.

### Compiled file

- A compiled model is one `.npz`, read with `allow_pickle=False`:
  - `program`: `(2L, 16)` uint8, the instruction words as sent on the wire
  - `tiles`: `(T, N, N)` int8
  - `ptab`: `(C, 8)` uint8, the PTAB entries as sent on the wire
  - `meta`: a JSON string with a format version, the target, B, the layer table (sizes, `act_signed`, ReLU), the addresses, the input scale and signedness, the per-channel output scales and the HALT tag
- So a model can be compiled in WSL and run from any Windows PC, which only needs NumPy and pyserial.

## MNIST demo (`python -m tinytpu.mnist`)

One command does every step, and caches each result:

1. **Download** the four MNIST files from a mirror, check their SHA-256 hashes (pinned in the code), and keep them in a cache outside the repo:
   - Windows: `%LOCALAPPDATA%\tinytpu\mnist`
   - Linux: `~/.cache/tinytpu/mnist`
   - `--data-dir` or `TINYTPU_DATA` overrides it.
2. **Train** an MLP in NumPy: He initialization, ReLU, softmax cross-entropy, Adam, fixed seed, a few epochs. This takes a few seconds per epoch on a CPU. The float weights are cached.
3. **Quantize** with 1,000 training images for calibration, and **compile** for the board's target.
4. **Run** the 10,000 test images (or `--count`) on the board.
5. **Report:**
   - float, integer-reference and board accuracy
   - how many board results equal `Machine`'s exactly (must be all)
   - ms per image and images per second
   - a few test digits drawn as text, with the board's prediction

Without `--port`, it stops after step 3 and reports the float and integer accuracy. The tests use a tiny synthetic dataset in the IDX file format and never touch the network.

### Model size

784-128-10. The figures are estimates for N = 8, at 1 Mbaud, with the latency timer at 1 ms:

| Model | Weight tiles | Channels | B | Per INFER on the wire | Per image | Expected accuracy |
|---|---|---|---|---|---|---|
| **784-128-10** | 1,600 / 2,048 | 144 / 512 | 5 | 3,937 B out, 335 B back: ~43 ms | ~9 ms | ~98% |
| 196-64-10 (14×14) | 216 / 2,048 | 80 / 512 | 20 | 4,017 B out, 1,295 B back: ~53 ms | ~2.7 ms | ~97% |

- Compute is negligible either way: under 0.3 ms per INFER for 784-128-10.
- 784-128-10 shows off weights staying on-chip: 78% of `WBUF`. `--size 14` would select the smaller one.

## Testing

| Test file | Covers |
|---|---|
| `tb/test_quant.py` | `M`/`s` at their edges (`s = 0`, `M` at 65535, `r` too large, `r` tiny), all-zero channels and activations, ties in rounding, bias overflow rejected, quantized vs float accuracy on synthetic data |
| `tb/test_compiler.py` | Random models with sizes that aren't multiples of N, signed and unsigned input, hidden layers with and without ReLU, at N = 2, 4 and 8: `Machine` must equal `forward_int` bit-exactly. Each limit just met and just exceeded. The save/load round trip. The HALT-tag check. |
| `tb/test_tinytpu_top.py` (new test) | A compiled model on the simulated board, through the real `Device` and `Link`: equal to `Machine` bit-exactly |
| `tb/test_mnist.py` | IDX parsing, the hash check and the cache, with synthetic files; training converges on a separable synthetic set |

As with the rest of the project, each test is checked against injected bugs (a wrong rounding, an off-by-one region size, a swapped tile order) before it counts as done.

## Decisions

All approved on 2026-11-16 (J7.1):

1. **MNIST model size:** 784-128-10. 196-64-10 on 14×14 images is an option (`--size 14`).
2. **Calibration statistic:** the maximum by default; a percentile is an option.
3. **Batches:** `INFER` only in version 1.
4. **Default batch size:** the largest that fits.
5. **HALT tag:** from a hash of the compiled model, checked after every `INFER`.
6. **Compiled file:** one `.npz` with a JSON metadata string.
7. **MNIST mirror:** torchvision's primary mirror, `https://ossci-datasets.s3.amazonaws.com/mnist/`, with SHA-256 hashes pinned when the downloader is written.
