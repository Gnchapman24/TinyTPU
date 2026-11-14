"""TinyTPU ISA definitions and instruction encoding (Python mirror of ``rtl/isa_pkg.sv``).

``docs/isa.md`` is the source of truth. ``tb/test_isa.py`` checks that the constants here match
``rtl/isa_pkg.sv`` exactly, and that this module reproduces the worked example in ``docs/isa.md``.

Instructions are plain ``int`` values of 128 bits. Use :func:`to_bytes` / :func:`from_bytes` for the
16-byte wire and memory form (least-significant byte first).

Example:
    >>> from tinytpu import isa
    >>> word = isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=25, nt=8, m=1,
    ...                   act_signed=False, acc_mode=isa.AccMode.BIAS)
    >>> f"{word:032x}"
    '00010008001900000000000000000410'
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

INSTR_W = 128
"""Instruction width in bits."""
SLOT_W = 16
"""Width of each of the eight instruction slots, in bits."""
INSTR_BYTES = INSTR_W // 8
"""Bytes per instruction in IMEM and on the wire."""


class Opcode(IntEnum):
    """Instruction opcodes (bits [7:0])."""

    HALT = 0x00
    NOP = 0x01
    MATMUL = 0x10
    REQUANT = 0x20


class AccMode(IntEnum):
    """MATMUL accumulator mode (flags bits [2:1], instruction bits [10:9])."""

    OVERWRITE = 0b00
    ACCUMULATE = 0b01
    BIAS = 0b10
    ILLEGAL = 0b11


FLAG_ACT_SIGNED = 0
"""MATMUL flag bit: activations are INT8 (else UINT8)."""
FLAG_ACC_MODE_LSB = 1
"""MATMUL flag bits [2:1] hold :class:`AccMode`."""
FLAG_RELU = 0
"""REQUANT flag bit: clamp to [0, 255] (else [-128, 127])."""

FLAGS_MASK = {Opcode.HALT: 0x00, Opcode.NOP: 0x00, Opcode.MATMUL: 0x07, Opcode.REQUANT: 0x01}
"""Flag bits each opcode may set; all others are reserved and must be zero."""


class SeqState(IntEnum):
    """Sequencer state, as reported in the protocol status block."""

    IDLE = 0
    RUNNING = 1
    HALTED_OK = 2
    HALTED_ERROR = 3


class SeqErr(IntEnum):
    """Sequencer error codes (docs/isa.md, Errors)."""

    NONE = 0
    ILLEGAL_OPCODE = 1
    RESERVED_BIT = 2
    ACC_MODE = 3
    ZERO_COUNT = 4
    ADDR_RANGE = 5
    PC_OVERRUN = 6


class Mem(IntEnum):
    """Memory IDs used by the host protocol."""

    IMEM = 0
    WBUF = 1
    ABUF = 2
    ACC = 3
    PTAB = 4


DEFAULT_IMEM_DEPTH = 128
DEFAULT_WBUF_DEPTH = 2048
DEFAULT_ABUF_DEPTH = 1024
DEFAULT_ACC_DEPTH = 512
DEFAULT_PTAB_DEPTH = 512
COUNT_W = 12
"""Implemented bits of the ``kt``, ``nt`` and ``m`` counts."""
PTAB_ENTRY_W = 54
"""Bits per parameter-table entry in memory."""

DEFAULT_DEPTHS = {
    Mem.IMEM: DEFAULT_IMEM_DEPTH,
    Mem.WBUF: DEFAULT_WBUF_DEPTH,
    Mem.ABUF: DEFAULT_ABUF_DEPTH,
    Mem.ACC: DEFAULT_ACC_DEPTH,
    Mem.PTAB: DEFAULT_PTAB_DEPTH,
}
"""Default memory depths (docs/memory_map.md). Real depths come from the protocol's INFO reply."""


@dataclass(frozen=True)
class Fields:
    """All fields of one instruction, as raw slot values.

    Attributes:
        opcode: Bits [7:0], kept as ``int`` so illegal opcodes can be represented.
        flags: Bits [15:8].
        w_base: Slot 1. MATMUL weight base, or the HALT code.
        a_base: Slot 2. MATMUL activation base, or the REQUANT destination ``a_dst``.
        acc_base: Slot 3.
        p_base: Slot 4.
        kt: Slot 5.
        nt: Slot 6.
        m: Slot 7.
    """

    opcode: int
    flags: int = 0
    w_base: int = 0
    a_base: int = 0
    acc_base: int = 0
    p_base: int = 0
    kt: int = 0
    nt: int = 0
    m: int = 0

    def encode(self) -> int:
        """Pack the fields into a 128-bit instruction word.

        Raises:
            ValueError: If any field does not fit its slot.
        """
        for name, value, width in (("opcode", self.opcode, 8), ("flags", self.flags, 8)):
            if not 0 <= value < 1 << width:
                raise ValueError(f"{name}={value} does not fit {width} bits")
        slots = [self.opcode | (self.flags << 8), self.w_base, self.a_base, self.acc_base,
                 self.p_base, self.kt, self.nt, self.m]
        word = 0
        for i, s in enumerate(slots):
            if not 0 <= s < 1 << SLOT_W:
                raise ValueError(f"slot {i} value {s} does not fit {SLOT_W} bits")
            word |= s << (SLOT_W * i)
        return word

    @classmethod
    def decode(cls, word: int) -> Fields:
        """Split a 128-bit instruction word into its fields (no legality checks)."""
        if not 0 <= word < 1 << INSTR_W:
            raise ValueError("instruction word must be a 128-bit unsigned integer")
        s = [(word >> (SLOT_W * i)) & 0xFFFF for i in range(8)]
        return cls(opcode=s[0] & 0xFF, flags=s[0] >> 8, w_base=s[1], a_base=s[2],
                   acc_base=s[3], p_base=s[4], kt=s[5], nt=s[6], m=s[7])


def halt(code: int = 0) -> int:
    """Encode ``HALT``.

    Args:
        code: 16-bit halt code reported to the host.
    """
    return Fields(Opcode.HALT, w_base=code).encode()


def nop() -> int:
    """Encode ``NOP``."""
    return Fields(Opcode.NOP).encode()


def matmul(*, w_base: int, a_base: int, acc_base: int, p_base: int = 0, kt: int, nt: int, m: int,
           act_signed: bool, acc_mode: AccMode) -> int:
    """Encode ``MATMUL``: one layer, ``ACC (=, +=, = bias +) X · W`` over kt × nt tiles and m rows.

    Args:
        w_base: Weight base, in tiles.
        a_base: Activation source base, in vectors.
        acc_base: Accumulator base, in rows.
        p_base: Parameter-table base (used in bias mode).
        kt: Inner-dimension tiles (K / N).
        nt: Output-column tiles (Nout / N).
        m: Activation rows.
        act_signed: Activations are INT8 (``True``) or UINT8 (``False``).
        acc_mode: Overwrite, accumulate, or bias.
    """
    flags = (int(act_signed) << FLAG_ACT_SIGNED) | (int(acc_mode) << FLAG_ACC_MODE_LSB)
    return Fields(Opcode.MATMUL, flags, w_base, a_base, acc_base, p_base, kt, nt, m).encode()


def requant(*, a_dst: int, acc_base: int, p_base: int, nt: int, m: int, relu: bool) -> int:
    """Encode ``REQUANT``: requantize accumulator rows into the activation buffer.

    Args:
        a_dst: Activation destination base, in vectors.
        acc_base: Accumulator base, in rows.
        p_base: Parameter-table base.
        nt: Output-column tiles.
        m: Rows.
        relu: Clamp to UINT8 [0, 255] (``True``) or INT8 [-128, 127] (``False``).
    """
    flags = int(relu) << FLAG_RELU
    return Fields(Opcode.REQUANT, flags, 0, a_dst, acc_base, p_base, 0, nt, m).encode()


def to_bytes(word: int) -> bytes:
    """Return the 16-byte wire/memory form of an instruction (least-significant byte first)."""
    return word.to_bytes(INSTR_BYTES, "little")


def from_bytes(data: bytes) -> int:
    """Inverse of :func:`to_bytes`."""
    if len(data) != INSTR_BYTES:
        raise ValueError(f"an instruction is {INSTR_BYTES} bytes, got {len(data)}")
    return int.from_bytes(data, "little")


def _clog2(n: int) -> int:
    return max(1, (n - 1).bit_length())


def decode_error(word: int, depths: dict[Mem, int] | None = None) -> SeqErr:
    """Return the decode-time error the sequencer would raise for an instruction.

    This is the reference model for the hardware decoder. It covers ISA errors 1-4, which are
    detected before any memory is touched. Address-range errors (5) depend on execution, and
    PC overrun (6) on the program, so they are not checked here.

    Args:
        word: 128-bit instruction.
        depths: Memory depths that set the implemented field widths. Defaults to
            :data:`DEFAULT_DEPTHS`.

    Returns:
        :attr:`SeqErr.NONE` if the instruction is legal, else the first error found, in the order
        illegal opcode, reserved bit, acc_mode, zero count.
    """
    d = DEFAULT_DEPTHS if depths is None else depths
    f = Fields.decode(word)
    try:
        op = Opcode(f.opcode)
    except ValueError:
        return SeqErr.ILLEGAL_OPCODE

    count_ok = (1 << COUNT_W) - 1
    addr = {
        "w_base": (1 << _clog2(d[Mem.WBUF])) - 1,
        "a_base": (1 << _clog2(d[Mem.ABUF])) - 1,
        "acc_base": (1 << _clog2(d[Mem.ACC])) - 1,
        "p_base": (1 << _clog2(d[Mem.PTAB])) - 1,
    }
    # Allowed bits per slot field, per opcode (0 = the whole field is reserved).
    allowed = {
        Opcode.HALT: {"w_base": 0xFFFF},
        Opcode.NOP: {},
        Opcode.MATMUL: {**addr, "kt": count_ok, "nt": count_ok, "m": count_ok},
        Opcode.REQUANT: {"a_base": addr["a_base"], "acc_base": addr["acc_base"],
                         "p_base": addr["p_base"], "nt": count_ok, "m": count_ok},
    }[op]
    if f.flags & ~FLAGS_MASK[op] & 0xFF:
        return SeqErr.RESERVED_BIT
    for name in ("w_base", "a_base", "acc_base", "p_base", "kt", "nt", "m"):
        if getattr(f, name) & ~allowed.get(name, 0) & 0xFFFF:
            return SeqErr.RESERVED_BIT
    if op is Opcode.MATMUL and (f.flags >> FLAG_ACC_MODE_LSB) & 0b11 == AccMode.ILLEGAL:
        return SeqErr.ACC_MODE
    counts = {Opcode.MATMUL: ("kt", "nt", "m"), Opcode.REQUANT: ("nt", "m")}.get(op, ())
    if any(getattr(f, c) == 0 for c in counts):
        return SeqErr.ZERO_COUNT
    return SeqErr.NONE


def ptab_entry(bias: int, mult: int, shift: int) -> bytes:
    """Return the 8-byte wire form of one parameter-table entry (docs/memory_map.md).

    Args:
        bias: Signed INT32 bias.
        mult: Requantization multiplier M (u16).
        shift: Requantization shift s (0-63).
    """
    if not -(1 << 31) <= bias < 1 << 31:
        raise ValueError("bias must fit INT32")
    if not 0 <= mult < 1 << 16:
        raise ValueError("M must fit u16")
    if not 0 <= shift < 64:
        raise ValueError("s must be 0-63")
    return struct.pack("<iHBx", bias, mult, shift)
