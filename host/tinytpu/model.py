"""Bit-exact NumPy reference model of TinyTPU (task A3.1).

Two levels, both following ``docs/isa.md`` and AGENTS.md §2 exactly:

* **Math level:** :func:`matmul` (INT8 weights x INT8/UINT8 activations, INT32 accumulation that
  wraps on overflow) and :func:`requant` (``clamp((acc * M + R) >>> s)`` with round-half-up).
  The RTL datapath is tested against these.
* **Machine level:** :class:`Machine` holds the five memories and runs ISA programs
  instruction by instruction: canonical layouts, tile loops, bias/accumulate modes, write-back,
  and every error halt. The sequencer and the end-to-end tests are checked against it.

Layout helpers (:func:`pack_weights`, :func:`pack_activations`, ...) convert between ordinary
matrices and the canonical memory layouts, padding with zeros as the host compiler must.

Example:
    >>> import numpy as np
    >>> from tinytpu import model
    >>> x = np.array([[200, 3]]); w = np.array([[1, -2], [3, 4]])
    >>> model.matmul(x, w, act_signed=False)
    array([[ 209, -388]], dtype=int32)
    >>> model.requant(np.array([[-5, 5]]), mult=np.array([1, 1]), shift=np.array([1, 1]), relu=False)
    array([[-2,  3]])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tinytpu import isa
from tinytpu.isa import AccMode, Fields, Mem, Opcode, SeqErr, SeqState

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
MAX_PRODUCT = 127 * 255
"""Largest product magnitude: weights are limited to -127..127 and activations to 0..255 or -128..127."""


# ------------------------------------------------------------------------------------------------
# Math level
# ------------------------------------------------------------------------------------------------

def wrap32(x: np.ndarray | int) -> np.ndarray:
    """Wrap integers to signed 32 bits (two's complement), as the INT32 accumulators do."""
    a = np.asarray(x, dtype=np.int64)
    return ((a + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)


def as_activations(x: np.ndarray, act_signed: bool) -> np.ndarray:
    """Interpret raw activation bytes as INT8 (``act_signed``) or UINT8, returned as int64."""
    raw = np.asarray(x, dtype=np.int64) & 0xFF
    return np.where(raw >= 128, raw - 256, raw) if act_signed else raw


def matmul(x: np.ndarray, w: np.ndarray, act_signed: bool, acc: np.ndarray | None = None,
           bias: np.ndarray | None = None) -> np.ndarray:
    """INT32 result of ``x @ w`` with TinyTPU semantics.

    Args:
        x: Activations, shape ``(M, K)``. Values are taken modulo 256 and read as INT8 or UINT8
            according to ``act_signed``, exactly as the activation buffer bytes are.
        w: Weights, shape ``(K, Nout)``, signed INT8 (-128..127; the host keeps them in -127..127).
        act_signed: Activations are INT8 (``True``) or UINT8 (``False``).
        acc: Existing accumulators ``(M, Nout)`` to add to (accumulate mode), or ``None``.
        bias: Per-column INT32 bias ``(Nout,)`` added once (bias mode), or ``None``.

    Returns:
        ``(M, Nout)`` int32, wrapped on overflow like the hardware.
    """
    w64 = np.asarray(w, dtype=np.int64)
    if w64.min(initial=0) < -128 or w64.max(initial=0) > 127:
        raise ValueError("weights must be INT8")
    total = as_activations(x, act_signed) @ w64
    if bias is not None:
        total = total + np.asarray(bias, dtype=np.int64)[None, :]
    if acc is not None:
        total = total + np.asarray(acc, dtype=np.int64)
    return wrap32(total)


def requant(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray, relu: bool) -> np.ndarray:
    """Requantize INT32 accumulators to 8-bit values, per output column.

    ``y = clamp((acc * M + R) >>> s, lo, hi)`` with ``R = 0`` if ``s == 0`` else ``2**(s-1)``
    (round half up), a full-width product, and an arithmetic shift. The clamp range is
    [0, 255] with ``relu`` and [-128, 127] without.

    Args:
        acc: Accumulators ``(M, Nout)``, INT32.
        mult: Multipliers ``M`` per column ``(Nout,)``, 0..65535.
        shift: Shifts ``s`` per column ``(Nout,)``, 0..63.
        relu: Clamp to UINT8 (``True``) or INT8 (``False``).

    Returns:
        ``(M, Nout)`` int64 values in the clamp range.
    """
    a = np.asarray(acc, dtype=np.int64)
    m = np.asarray(mult, dtype=np.int64)[None, :]
    s = np.asarray(shift, dtype=np.int64)[None, :]
    if (m < 0).any() or (m > 0xFFFF).any() or (s < 0).any() or (s > 63).any():
        raise ValueError("M must be u16 and s must be 0..63")
    r = np.where(s == 0, 0, np.left_shift(1, np.maximum(s - 1, 0)))
    y = np.right_shift(a * m + r, s)            # numpy >> on int64 is arithmetic
    lo, hi = (0, 255) if relu else (-128, 127)
    return np.clip(y, lo, hi)


def k_fits(k: int, max_abs_bias: int = 0) -> bool:
    """True if an inner dimension ``k`` can never wrap the INT32 accumulators (AGENTS.md §2)."""
    return k * MAX_PRODUCT + max_abs_bias < (1 << 31)


# ------------------------------------------------------------------------------------------------
# Layout helpers (host compiler side)
# ------------------------------------------------------------------------------------------------

def _pad_to(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=a.dtype)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def pack_weights(w: np.ndarray, n: int) -> np.ndarray:
    """Return weight tiles in canonical order: tile ``n_t * KT + k_t`` is ``W[k_t, n_t]``.

    Args:
        w: ``(K, Nout)`` INT8 weights; padded with zeros to multiples of ``n``.
        n: Array size N.

    Returns:
        ``(NT * KT, N, N)`` int8, row ``i`` of a tile = weights for activation element ``i``.
    """
    k_dim, out_dim = w.shape
    kt, nt = ceil_div(k_dim, n), ceil_div(out_dim, n)
    wp = _pad_to(np.asarray(w, dtype=np.int8), kt * n, nt * n)
    return np.stack([wp[k * n:(k + 1) * n, j * n:(j + 1) * n] for j in range(nt) for k in range(kt)])


def pack_activations(x: np.ndarray, n: int) -> np.ndarray:
    """Return activation vectors in canonical order: vector ``m * KT + k_t``.

    Args:
        x: ``(M, K)`` activations (any integer type; stored as raw bytes, i.e. modulo 256).
        n: Array size N.

    Returns:
        ``(M * KT, N)`` uint8 raw bytes, K padded with zeros.
    """
    m_dim, k_dim = x.shape
    kt = ceil_div(k_dim, n)
    xp = _pad_to(np.asarray(x, dtype=np.int64) & 0xFF, m_dim, kt * n).astype(np.uint8)
    return xp.reshape(m_dim * kt, n)


def unpack_vectors(vecs: np.ndarray, m_dim: int, cols: int, signed: bool) -> np.ndarray:
    """Inverse of :func:`pack_activations` (also reads REQUANT outputs): ``(M, cols)`` int64."""
    n = vecs.shape[1]
    per_row = ceil_div(cols, n)
    flat = vecs.reshape(m_dim, per_row * n)[:, :cols]
    return as_activations(flat, signed)


def unpack_acc(rows: np.ndarray, m_dim: int, cols: int) -> np.ndarray:
    """Turn canonical accumulator rows ``(M * NT, N)`` into an ``(M, cols)`` int32 matrix."""
    n = rows.shape[1]
    nt = ceil_div(cols, n)
    return np.asarray(rows, dtype=np.int32).reshape(m_dim, nt * n)[:, :cols]


# ------------------------------------------------------------------------------------------------
# Machine level
# ------------------------------------------------------------------------------------------------

@dataclass
class Status:
    """Sequencer status, as in the protocol's status block.

    Attributes:
        state: Idle, running, halted OK, or halted with an error.
        err: Error code (``SeqErr.NONE`` unless halted with an error).
        pc: PC of the HALT or faulting instruction. For PC overrun it is the IMEM depth.
        halt_code: Code from the HALT instruction.
    """

    state: SeqState = SeqState.IDLE
    err: SeqErr = SeqErr.NONE
    pc: int = 0
    halt_code: int = 0


def range_error(word: int, n: int, depths: dict[Mem, int] | None = None) -> bool:
    """True if a decode-legal instruction would use an address beyond a memory's depth (ISA error 5).

    This is the pre-execution check of ``docs/isa.md``: the highest address the instruction uses
    in each memory it touches, so a faulting instruction writes nothing. It is the reference for
    the hardware range checker.

    Args:
        word: 128-bit instruction, already free of decode errors (:func:`tinytpu.isa.decode_error`).
        n: Array size N.
        depths: Memory depths. Defaults to :data:`tinytpu.isa.DEFAULT_DEPTHS`.
    """
    d = isa.DEFAULT_DEPTHS if depths is None else depths
    f = Fields.decode(word)
    if f.opcode == Opcode.MATMUL:
        last = {Mem.WBUF: f.w_base + f.nt * f.kt - 1,
                Mem.ABUF: f.a_base + f.m * f.kt - 1,
                Mem.ACC: f.acc_base + f.m * f.nt - 1}
        if AccMode((f.flags >> isa.FLAG_ACC_MODE_LSB) & 0b11) == AccMode.BIAS:
            last[Mem.PTAB] = f.p_base + f.nt * n - 1
    elif f.opcode == Opcode.REQUANT:
        last = {Mem.ACC: f.acc_base + f.m * f.nt - 1,
                Mem.ABUF: f.a_base + f.m * f.nt - 1,
                Mem.PTAB: f.p_base + f.nt * n - 1}
    else:
        return False
    return any(addr >= d[mem] for mem, addr in last.items())


class _Fault(Exception):
    def __init__(self, err: SeqErr):
        self.err = err


@dataclass
class Machine:
    """Instruction-level model: the five memories plus the sequencer.

    Attributes:
        n: Array size N.
        depths: Memory depths in units (defaults from ``docs/memory_map.md``).
    """

    n: int = 8
    depths: dict[Mem, int] = field(default_factory=lambda: dict(isa.DEFAULT_DEPTHS))
    status: Status = field(default_factory=Status)

    def __post_init__(self) -> None:
        d, n = self.depths, self.n
        self.imem = np.zeros(d[Mem.IMEM], dtype=object)              # 128-bit ints; 0 = HALT 0
        self.wbuf = np.zeros((d[Mem.WBUF], n, n), dtype=np.int8)
        self.abuf = np.zeros((d[Mem.ABUF], n), dtype=np.uint8)
        self.acc = np.zeros((d[Mem.ACC], n), dtype=np.int32)
        self.ptab_bias = np.zeros(d[Mem.PTAB], dtype=np.int64)
        self.ptab_mult = np.zeros(d[Mem.PTAB], dtype=np.int64)
        self.ptab_shift = np.zeros(d[Mem.PTAB], dtype=np.int64)

    # Loading ----------------------------------------------------------------------------------

    def load_program(self, words: list[int], at: int = 0) -> None:
        for i, w in enumerate(words):
            self.imem[at + i] = int(w)

    def load_weights(self, tiles: np.ndarray, at: int = 0) -> None:
        self.wbuf[at:at + len(tiles)] = tiles

    def load_vectors(self, vecs: np.ndarray, at: int = 0) -> None:
        self.abuf[at:at + len(vecs)] = np.asarray(vecs, dtype=np.int64) & 0xFF

    def load_params(self, bias: np.ndarray, mult: np.ndarray, shift: np.ndarray, at: int = 0) -> None:
        k = len(bias)
        self.ptab_bias[at:at + k] = bias
        self.ptab_mult[at:at + k] = mult
        self.ptab_shift[at:at + k] = shift

    # Execution --------------------------------------------------------------------------------

    def _check(self, mem: Mem, addr: int) -> int:
        # range_error() has already ruled this out; kept as a guard on the model itself.
        if not 0 <= addr < self.depths[mem]:
            raise _Fault(SeqErr.ADDR_RANGE)
        return addr

    def run(self, start_pc: int = 0) -> Status:
        """Run from ``start_pc`` until HALT or an error, like the RUN command; return the status."""
        pc = start_pc
        depth = self.depths[Mem.IMEM]
        while True:
            if pc >= depth:
                self.status = Status(SeqState.HALTED_ERROR, SeqErr.PC_OVERRUN, depth)
                return self.status
            word = int(self.imem[pc])
            err = isa.decode_error(word, self.depths)
            if err is not SeqErr.NONE:
                self.status = Status(SeqState.HALTED_ERROR, err, pc)
                return self.status
            if range_error(word, self.n, self.depths):
                self.status = Status(SeqState.HALTED_ERROR, SeqErr.ADDR_RANGE, pc)
                return self.status
            f = Fields.decode(word)
            try:
                if f.opcode == Opcode.HALT:
                    self.status = Status(SeqState.HALTED_OK, SeqErr.NONE, pc, f.w_base)
                    return self.status
                if f.opcode == Opcode.MATMUL:
                    self._matmul(f)
                elif f.opcode == Opcode.REQUANT:
                    self._requant(f)
            except _Fault as e:
                self.status = Status(SeqState.HALTED_ERROR, e.err, pc)
                return self.status
            pc += 1

    def _matmul(self, f: Fields) -> None:
        n = self.n
        act_signed = bool((f.flags >> isa.FLAG_ACT_SIGNED) & 1)
        mode = AccMode((f.flags >> isa.FLAG_ACC_MODE_LSB) & 0b11)
        for nt_i in range(f.nt):
            for k in range(f.kt):
                w = self.wbuf[self._check(Mem.WBUF, f.w_base + nt_i * f.kt + k)].astype(np.int64)
                for m in range(f.m):
                    x = as_activations(self.abuf[self._check(Mem.ABUF, f.a_base + m * f.kt + k)], act_signed)
                    r = self._check(Mem.ACC, f.acc_base + m * f.nt + nt_i)
                    p = x @ w
                    if k == 0 and mode == AccMode.OVERWRITE:
                        total = p
                    elif k == 0 and mode == AccMode.BIAS:
                        p0 = f.p_base + nt_i * n
                        self._check(Mem.PTAB, p0 + n - 1)
                        total = p + self.ptab_bias[p0:p0 + n]
                    else:
                        total = self.acc[r].astype(np.int64) + p
                    self.acc[r] = wrap32(total)

    def _requant(self, f: Fields) -> None:
        n = self.n
        relu = bool((f.flags >> isa.FLAG_RELU) & 1)
        for nt_i in range(f.nt):
            p0 = f.p_base + nt_i * n
            self._check(Mem.PTAB, p0 + n - 1)
            mult, shift = self.ptab_mult[p0:p0 + n], self.ptab_shift[p0:p0 + n]
            for m in range(f.m):
                r = self._check(Mem.ACC, f.acc_base + m * f.nt + nt_i)
                dst = self._check(Mem.ABUF, f.a_base + m * f.nt + nt_i)
                y = requant(self.acc[r][None, :], mult, shift, relu)[0]
                self.abuf[dst] = (y & 0xFF).astype(np.uint8)
