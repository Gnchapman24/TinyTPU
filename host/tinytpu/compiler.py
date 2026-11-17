"""Code generation for the host compiler (task A7.3, ``docs/compiler.md``).

:func:`compile_model` turns a quantized :class:`tinytpu.quant.QModel` into a :class:`Compiled`
image for one :class:`Target` (N and the memory depths): the program, the weight tiles, the
parameter-table entries, and where the input goes and the output comes from.

* Each hidden layer is ``MATMUL`` (bias mode) + ``REQUANT``; the last layer is ``MATMUL`` only, and
  its INT32 accumulators are the output. The program ends with ``HALT tag``, where the tag is a
  16-bit hash of the whole image.
* Weights and parameters are packed layer after layer from address 0. Every layer uses ``ACC``
  from row 0. ``ABUF`` has two regions used alternately; the input is in the first, at 0.
* The batch size B (the program's ``m``) is fixed; by default it is the largest that fits.

Every limit is checked before anything is generated, and :class:`FitError` lists every one that
is exceeded. The generated words are then checked with :func:`tinytpu.isa.decode_error` and
:func:`tinytpu.model.range_error`, so the board can never halt with an error on them.

Example:
    >>> import numpy as np
    >>> from tinytpu import compiler, nn, quant
    >>> rng = np.random.default_rng(0)
    >>> net = nn.Model([nn.Dense(rng.normal(size=(784, 128)), relu=True), nn.Dense(rng.normal(size=(128, 10)))])
    >>> prog = compiler.compile_model(quant.quantize(net, rng.uniform(size=(10, 784))), compiler.Target.default(8))
    >>> prog.batch, len(prog.tiles), len(prog.ptab_bias), len(prog.program)
    (5, 1600, 144, 4)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tinytpu import isa, model
from tinytpu.isa import AccMode, Mem, SeqErr
from tinytpu.quant import QModel

if TYPE_CHECKING:
    from tinytpu.device import Info

MAX_PAYLOAD = 4096
"""Largest packet payload in bytes (``docs/uart_protocol.md``)."""
INFER_ARGS = 10
"""Bytes of an ``INFER`` request before the input data."""
STATUS_BYTES = 8
"""Bytes of an ``INFER`` reply before the output data (the status block)."""
MAX_COUNT = (1 << isa.COUNT_W) - 1
"""Largest ``kt``, ``nt`` or ``m`` (12-bit counts)."""


class FitError(ValueError):
    """A model does not fit the target.

    Attributes:
        problems: One message per limit that is exceeded, with the numbers.
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("the model does not fit the target:\n  - " + "\n  - ".join(self.problems))


def _pow2(v: int) -> bool:
    return v >= 1 and v & (v - 1) == 0


@dataclass(frozen=True)
class Target:
    """What a model is compiled for: the array size and every memory depth.

    Attributes:
        n: Array size N (a power of two, at least 2).
        depths: Depth of every memory, in its units (powers of two).
        max_payload: Largest packet payload in bytes.
    """

    n: int
    depths: dict[Mem, int] = field(default_factory=lambda: dict(isa.DEFAULT_DEPTHS))
    max_payload: int = MAX_PAYLOAD

    def __post_init__(self) -> None:
        if not (_pow2(self.n) and self.n >= 2):
            raise ValueError(f"N must be a power of two, at least 2; got {self.n}")
        if set(self.depths) != set(Mem):
            raise ValueError(f"depths must cover {[m.name for m in Mem]}")
        for mem, d in self.depths.items():
            if not (_pow2(d) and d <= 1 << isa.SLOT_W):
                raise ValueError(f"{Mem(mem).name} depth must be a power of two up to {1 << isa.SLOT_W}; got {d}")
        object.__setattr__(self, "depths", {Mem(m): int(d) for m, d in self.depths.items()})

    @classmethod
    def default(cls, n: int = 8) -> Target:
        """The ``docs/memory_map.md`` depths, for compiling without a board."""
        return cls(n)

    @classmethod
    def from_info(cls, info: Info) -> Target:
        """The target a board reports in its ``INFO`` block (:meth:`tinytpu.device.Device.info`)."""
        return cls(info.n, dict(info.depths), info.max_payload)


@dataclass(frozen=True)
class LayerPlan:
    """Where one layer lives on the board.

    Attributes:
        k: Input width.
        n_out: Output width.
        kt: Inner-dimension tiles.
        nt: Output-column tiles.
        w_base: First weight tile.
        p_base: First parameter-table entry.
        a_src: First input vector.
        a_dst: First output vector (``None`` for the last layer).
        relu: ReLU after the layer.
        act_signed: The input is INT8 (else UINT8).
    """

    k: int
    n_out: int
    kt: int
    nt: int
    w_base: int
    p_base: int
    a_src: int
    a_dst: int | None
    relu: bool
    act_signed: bool


@dataclass
class Compiled:
    """A model compiled for one target: everything the board needs, and how to talk to it.

    Attributes:
        target: The target it was compiled for.
        batch: Inputs per run (the program's ``m``).
        layers: Where each layer lives.
        program: Instruction words, ending with ``HALT tag``.
        tiles: Weight tiles ``(T, N, N)`` int8, for ``WBUF`` from 0.
        ptab_bias: Parameter-table biases, for ``PTAB`` from 0.
        ptab_mult: Parameter-table multipliers ``M``.
        ptab_shift: Parameter-table shifts ``s``.
        in_addr: First input vector in ``ABUF``.
        in_vectors: Input vectors per run (``batch * KT``).
        out_addr: First output row in ``ACC``.
        out_rows: Output rows per run (``batch * NT``).
        tag: The ``HALT`` code: 16 bits of a hash of the image.
        input_scale: Float value of one input step, or ``None`` (integer-only model).
        out_scale: Float value of one accumulator step per output channel, or ``None``.
    """

    target: Target
    batch: int
    layers: list[LayerPlan]
    program: list[int]
    tiles: np.ndarray
    ptab_bias: np.ndarray
    ptab_mult: np.ndarray
    ptab_shift: np.ndarray
    in_addr: int
    in_vectors: int
    out_addr: int
    out_rows: int
    tag: int
    input_scale: float | None = None
    out_scale: np.ndarray | None = None

    @property
    def k_in(self) -> int:
        """Input width."""
        return self.layers[0].k

    @property
    def n_out(self) -> int:
        """Output width."""
        return self.layers[-1].n_out

    @property
    def input_signed(self) -> bool:
        """Whether the input is INT8 (else UINT8)."""
        return self.layers[0].act_signed

    @property
    def last_relu(self) -> bool:
        """Whether a ReLU is applied to the output (on the host)."""
        return self.layers[-1].relu

    def ptab_bytes(self) -> bytes:
        """The parameter-table entries in wire form (8 bytes each)."""
        return b"".join(isa.ptab_entry(int(b), int(m), int(s))
                        for b, m, s in zip(self.ptab_bias, self.ptab_mult, self.ptab_shift))


# ------------------------------------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------------------------------------

def _shape(q: QModel, n: int) -> tuple[list[int], list[int]]:
    return ([model.ceil_div(layer.k, n) for layer in q.layers],
            [model.ceil_div(layer.n_out, n) for layer in q.layers])


def _regions(kts: list[int], nts: list[int], batch: int) -> tuple[int, int]:
    """Sizes of ABUF regions A (input, then odd hidden layers' outputs) and B (even ones')."""
    a, b = batch * kts[0], 0
    for i in range(len(nts) - 1):             # hidden layer i writes B if i is even, else A
        if i % 2 == 0:
            b = max(b, batch * nts[i])
        else:
            a = max(a, batch * nts[i])
    return a, b


def _problems(q: QModel, t: Target, batch: int) -> list[str]:
    n, d = t.n, t.depths
    kts, nts = _shape(q, n)
    layers = len(q.layers)
    out: list[str] = []
    if 2 * layers > d[Mem.IMEM]:
        out.append(f"program: {2 * layers} instructions for {layers} layers, but IMEM holds {d[Mem.IMEM]}")
    tiles = sum(kt * nt for kt, nt in zip(kts, nts))
    if tiles > d[Mem.WBUF]:
        out.append(f"weights: {tiles} tiles, but WBUF holds {d[Mem.WBUF]}")
    chans = sum(nt * n for nt in nts)
    if chans > d[Mem.PTAB]:
        out.append(f"channels: {chans} (padded to N), but PTAB holds {d[Mem.PTAB]}")
    for i, (kt, nt) in enumerate(zip(kts, nts)):
        if kt > MAX_COUNT:
            out.append(f"layer {i}: kt = {kt} exceeds the {isa.COUNT_W}-bit count limit {MAX_COUNT}")
        if nt > MAX_COUNT:
            out.append(f"layer {i}: nt = {nt} exceeds the {isa.COUNT_W}-bit count limit {MAX_COUNT}")
        if batch * nt > d[Mem.ACC]:
            out.append(f"layer {i}: {batch} x {nt} = {batch * nt} accumulator rows, but ACC holds {d[Mem.ACC]}")
    if batch < 1:
        out.append(f"batch size must be at least 1, got {batch}")
    if batch > MAX_COUNT:
        out.append(f"batch {batch} exceeds the {isa.COUNT_W}-bit count limit {MAX_COUNT}")
    ra, rb = _regions(kts, nts, batch)
    if ra + rb > d[Mem.ABUF]:
        out.append(f"activations: {ra} + {rb} = {ra + rb} vectors in two regions, but ABUF holds {d[Mem.ABUF]}")
    req = INFER_ARGS + batch * kts[0] * n
    if req > t.max_payload:
        out.append(f"INFER request: {req} bytes for a batch of {batch}, but the limit is {t.max_payload}")
    rep = STATUS_BYTES + batch * nts[-1] * 4 * n
    if rep > t.max_payload:
        out.append(f"INFER reply: {rep} bytes for a batch of {batch}, but the limit is {t.max_payload}")
    return out


def check_fit(q: QModel, target: Target, batch: int) -> None:
    """Raise :class:`FitError` listing every limit ``q`` exceeds at batch size ``batch``."""
    problems = _problems(q, target, batch)
    if problems:
        raise FitError(problems)


def max_batch(q: QModel, target: Target) -> int:
    """The largest batch size that fits every limit.

    Raises:
        FitError: If not even a batch of 1 fits.
    """
    check_fit(q, target, 1)
    lo, hi = 1, MAX_COUNT                      # every limit only gets tighter as the batch grows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _problems(q, target, mid):
            hi = mid - 1
        else:
            lo = mid
    return lo


# ------------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------------

def _tag(words: list[int], tiles: np.ndarray, ptab: bytes) -> int:
    h = hashlib.sha256()
    for w in words:
        h.update(isa.to_bytes(w))
    h.update(tiles.tobytes())
    h.update(ptab)
    return int.from_bytes(h.digest()[:2], "little")


def _pad(v: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    out[: len(v)] = v
    return out


def compile_model(q: QModel, target: Target, batch: int | None = None) -> Compiled:
    """Compile a quantized model for a target.

    Args:
        q: The quantized model.
        target: N and the memory depths (:meth:`Target.default` or :meth:`Target.from_info`).
        batch: Inputs per run. By default, the largest that fits (:func:`max_batch`).

    Returns:
        The compiled image.

    Raises:
        FitError: If the model does not fit, listing every limit that is exceeded.
    """
    if batch is None:
        batch = max_batch(q, target)
    check_fit(q, target, batch)
    n, d = target.n, target.depths
    kts, nts = _shape(q, n)
    region_a, _ = _regions(kts, nts, batch)
    last = len(q.layers) - 1

    plans: list[LayerPlan] = []
    words: list[int] = []
    w_base = p_base = 0
    a_src = 0                                   # region A
    for i, layer in enumerate(q.layers):
        a_dst = None if i == last else (region_a if i % 2 == 0 else 0)
        plans.append(LayerPlan(layer.k, layer.n_out, kts[i], nts[i], w_base, p_base, a_src, a_dst,
                               layer.relu, layer.act_signed))
        words.append(isa.matmul(w_base=w_base, a_base=a_src, acc_base=0, p_base=p_base, kt=kts[i], nt=nts[i],
                                m=batch, act_signed=layer.act_signed, acc_mode=AccMode.BIAS))
        if a_dst is not None:
            words.append(isa.requant(a_dst=a_dst, acc_base=0, p_base=p_base, nt=nts[i], m=batch, relu=layer.relu))
            a_src = a_dst
        w_base += kts[i] * nts[i]
        p_base += nts[i] * n

    tiles = np.concatenate([model.pack_weights(layer.w, n) for layer in q.layers]).astype(np.int8)
    bias = np.concatenate([_pad(layer.bias, nt * n) for layer, nt in zip(q.layers, nts)])
    mult = np.concatenate([_pad(layer.mult if i < last else np.zeros(0), nt * n)
                           for i, (layer, nt) in enumerate(zip(q.layers, nts))])
    shift = np.concatenate([_pad(layer.shift if i < last else np.zeros(0), nt * n)
                            for i, (layer, nt) in enumerate(zip(q.layers, nts))])
    ptab = b"".join(isa.ptab_entry(int(b), int(m), int(s)) for b, m, s in zip(bias, mult, shift))
    tag = _tag(words, tiles, ptab)
    words.append(isa.halt(tag))

    for pc, w in enumerate(words):              # a compiler bug must never reach the board
        err = isa.decode_error(w, d)
        if err is not SeqErr.NONE or model.range_error(w, n, d):
            raise AssertionError(f"compiler bug: instruction {pc} would halt the board ({err.name})")

    return Compiled(
        target=target, batch=batch, layers=plans, program=words, tiles=tiles,
        ptab_bias=bias, ptab_mult=mult, ptab_shift=shift,
        in_addr=0, in_vectors=batch * kts[0], out_addr=0, out_rows=batch * nts[-1], tag=tag,
        input_scale=q.input_scale, out_scale=None if q.out_scale is None else np.array(q.out_scale))

