"""Tests for the reference model tinytpu.model (task A3.1).

The model is checked against deliberately naive pure-Python references (integer loops, explicit
two's-complement wrapping) that share no code with it, at array sizes 2, 4 and 8. The machine
model runs real ISA programs, including the worked example from docs/isa.md, and must match the
math level exactly for any layer shape, including shapes that are not multiples of N.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import pytest

from tinytpu import isa, model
from tinytpu.isa import AccMode, Mem, SeqErr, SeqState

ROOT = Path(__file__).resolve().parents[1]
NS = [2, 4, 8]


# ------------------------------------------------------------------------------------------------
# Naive references
# ------------------------------------------------------------------------------------------------

def ref_wrap(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def ref_act(b: int, signed: bool) -> int:
    b &= 0xFF
    return b - 256 if signed and b >= 128 else b


def ref_matmul(x, w, signed, acc=None, bias=None):
    m_dim, k_dim, out_dim = len(x), len(w), len(w[0])
    out = []
    for i in range(m_dim):
        row = []
        for j in range(out_dim):
            s = sum(ref_act(x[i][k], signed) * w[k][j] for k in range(k_dim))
            if bias is not None:
                s += bias[j]
            if acc is not None:
                s += acc[i][j]
            row.append(ref_wrap(s))
        out.append(row)
    return out


def ref_requant_one(a: int, m: int, s: int, relu: bool) -> int:
    t = a * m + (0 if s == 0 else 1 << (s - 1))
    y = t >> s                                 # Python >> on ints is arithmetic (floor)
    lo, hi = (0, 255) if relu else (-128, 127)
    return min(max(y, lo), hi)


# ------------------------------------------------------------------------------------------------
# Math level
# ------------------------------------------------------------------------------------------------

def test_wrap32_matches_twos_complement():
    vals = [0, 1, -1, (1 << 31) - 1, 1 << 31, -(1 << 31), -(1 << 31) - 1, (1 << 32) + 5, -(1 << 40) + 7]
    assert model.wrap32(np.array(vals)).tolist() == [ref_wrap(v) for v in vals]


@pytest.mark.parametrize("signed", [False, True])
def test_matmul_matches_reference(signed):
    rng = np.random.default_rng(1)
    for _ in range(30):
        m, k, n = rng.integers(1, 6), rng.integers(1, 40), rng.integers(1, 12)
        x = rng.integers(0, 256, (m, k))
        w = rng.integers(-127, 128, (k, n))
        acc = rng.integers(-(1 << 31), 1 << 31, (m, n))
        bias = rng.integers(-(1 << 31), 1 << 31, n)
        assert model.matmul(x, w, signed).tolist() == ref_matmul(x.tolist(), w.tolist(), signed)
        assert model.matmul(x, w, signed, acc=acc).tolist() == ref_matmul(x.tolist(), w.tolist(), signed, acc=acc.tolist())
        assert model.matmul(x, w, signed, bias=bias).tolist() == ref_matmul(x.tolist(), w.tolist(), signed, bias=bias.tolist())


def test_matmul_activation_modes_differ_above_127():
    x, w = np.array([[200]]), np.array([[1]])
    assert model.matmul(x, w, act_signed=False)[0, 0] == 200
    assert model.matmul(x, w, act_signed=True)[0, 0] == -56


def test_matmul_extreme_products_and_wrap():
    x = np.full((1, 3), 255)
    w = np.full((3, 1), -127)
    assert model.matmul(x, w, act_signed=False)[0, 0] == -3 * 32385
    acc = np.array([[(1 << 31) - 1]])
    assert model.matmul(np.array([[1]]), np.array([[1]]), False, acc=acc)[0, 0] == -(1 << 31)  # wraps


def test_matmul_rejects_non_int8_weights():
    with pytest.raises(ValueError):
        model.matmul(np.array([[1]]), np.array([[128]]), False)


def test_requant_edge_cases():
    cases = [
        # (acc, M, s, relu, expected)
        (5, 1, 1, False, 3),            # 2.5 rounds half up to 3
        (-5, 1, 1, False, -2),          # -2.5 rounds half up to -2
        (7, 1, 0, False, 7),            # s = 0: no rounding term
        (300, 1, 0, False, 127),        # clamp high (INT8)
        (-300, 1, 0, False, -128),      # clamp low (INT8)
        (300, 1, 0, True, 255),         # clamp high (UINT8)
        (-1, 1, 0, True, 0),            # ReLU
        (1000, 0, 5, False, 0),         # M = 0
        ((1 << 31) - 1, 0xFFFF, 63, False, 0),          # largest product, largest shift
        (-(1 << 31), 0xFFFF, 40, False, -128),       # (-2**47 + 2**31 + 2**39) >> 40 = floor(-127.498)
        (-(1 << 31), 0xFFFF, 63, False, 0),          # -1.5e-5 rounds half up to 0
        (3, 3, 2, False, 2),            # 9/4 = 2.25 -> 2
        (6, 3, 2, False, 5),            # 18/4 = 4.5 -> 5 (tie, up)
        (-6, 3, 2, False, -4),          # -4.5 -> -4 (tie, up)
    ]
    for acc, m, s, relu, want in cases:
        got = model.requant(np.array([[acc]]), np.array([m]), np.array([s]), relu)[0, 0]
        assert got == want == ref_requant_one(acc, m, s, relu), (acc, m, s, relu)


def test_requant_matches_reference_random():
    rng = random.Random(2)
    for _ in range(2000):
        acc = rng.randint(-(1 << 31), (1 << 31) - 1)
        m, s, relu = rng.randint(0, 0xFFFF), rng.randint(0, 63), rng.random() < 0.5
        got = model.requant(np.array([[acc]]), np.array([m]), np.array([s]), relu)[0, 0]
        assert got == ref_requant_one(acc, m, s, relu), (acc, m, s, relu)


def test_requant_rejects_bad_params():
    with pytest.raises(ValueError):
        model.requant(np.array([[1]]), np.array([1 << 16]), np.array([0]), False)
    with pytest.raises(ValueError):
        model.requant(np.array([[1]]), np.array([1]), np.array([64]), False)


def test_k_bound():
    assert model.k_fits(66311) and not model.k_fits(66312)
    assert not model.k_fits(66311, max_abs_bias=20000)


# ------------------------------------------------------------------------------------------------
# Layouts
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", NS)
def test_pack_weights_layout(n):
    rng = np.random.default_rng(3)
    w = rng.integers(-127, 128, (13, 10))
    tiles = model.pack_weights(w, n)
    kt, nt = -(-13 // n), -(-10 // n)
    assert tiles.shape == (kt * nt, n, n)
    for j in range(nt):
        for k in range(kt):
            for i in range(n):
                for c in range(n):
                    r, col = k * n + i, j * n + c
                    want = w[r, col] if r < 13 and col < 10 else 0
                    assert tiles[j * kt + k, i, c] == want


@pytest.mark.parametrize("n", NS)
def test_pack_unpack_activations(n):
    x = np.random.default_rng(4).integers(0, 256, (3, 11))
    v = model.pack_activations(x, n)
    assert v.shape == (3 * -(-11 // n), n)
    assert (model.unpack_vectors(v, 3, 11, signed=False) == x).all()


# ------------------------------------------------------------------------------------------------
# Machine level
# ------------------------------------------------------------------------------------------------

def run_layer(n, x, w, signed, mode, bias=None, acc_init=None, depths=None):
    """Load one layer into a Machine, run MATMUL + HALT, return the accumulator matrix."""
    mach = model.Machine(n=n, depths=depths or dict(isa.DEFAULT_DEPTHS))
    m_dim, k_dim = x.shape
    out_dim = w.shape[1]
    kt, nt = -(-k_dim // n), -(-out_dim // n)
    mach.load_weights(model.pack_weights(w, n), at=5)
    mach.load_vectors(model.pack_activations(x, n), at=3)
    if bias is not None:
        mach.load_params(np.pad(bias, (0, nt * n - out_dim)), np.zeros(nt * n), np.zeros(nt * n), at=7)
    if acc_init is not None:
        rows = np.zeros((m_dim, nt * n), dtype=np.int64)
        rows[:, :out_dim] = acc_init
        mach.acc[2:2 + m_dim * nt] = rows.reshape(m_dim * nt, n)
    mach.load_program([
        isa.matmul(w_base=5, a_base=3, acc_base=2, p_base=7, kt=kt, nt=nt, m=m_dim, act_signed=signed, acc_mode=mode),
        isa.halt(42),
    ])
    st = mach.run()
    assert (st.state, st.err, st.pc, st.halt_code) == (SeqState.HALTED_OK, SeqErr.NONE, 1, 42)
    return model.unpack_acc(mach.acc[2:2 + m_dim * nt], m_dim, out_dim)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("signed", [False, True])
def test_machine_matmul_all_modes_any_shape(n, signed):
    rng = np.random.default_rng(5 + n)
    for m_dim, k_dim, out_dim in [(1, 1, 1), (1, n, n), (3, 13, 10), (2, 3 * n + 1, 2 * n - 1)]:
        x = rng.integers(0, 256, (m_dim, k_dim))
        w = rng.integers(-127, 128, (k_dim, out_dim))
        bias = rng.integers(-1000000, 1000000, out_dim)
        acc0 = rng.integers(-1000000, 1000000, (m_dim, out_dim))
        xl, wl = x.tolist(), w.tolist()
        assert run_layer(n, x, w, signed, AccMode.OVERWRITE).tolist() == ref_matmul(xl, wl, signed)
        assert run_layer(n, x, w, signed, AccMode.BIAS, bias=bias).tolist() == ref_matmul(xl, wl, signed, bias=bias.tolist())
        assert run_layer(n, x, w, signed, AccMode.ACCUMULATE, acc_init=acc0).tolist() == \
            ref_matmul(xl, wl, signed, acc=acc0.tolist())


@pytest.mark.parametrize("n", NS)
def test_machine_split_k_with_accumulate(n):
    """A layer split into two MATMULs (bias, then accumulate) equals one MATMUL over all of K."""
    rng = np.random.default_rng(6)
    x, w = rng.integers(0, 256, (2, 4 * n)), rng.integers(-127, 128, (4 * n, n))
    bias = rng.integers(-5000, 5000, n)
    mach = model.Machine(n=n)
    mach.load_weights(model.pack_weights(w[: 2 * n], n), at=0)
    mach.load_weights(model.pack_weights(w[2 * n:], n), at=10)
    mach.load_vectors(model.pack_activations(x[:, : 2 * n], n), at=0)
    mach.load_vectors(model.pack_activations(x[:, 2 * n:], n), at=20)
    mach.load_params(bias, np.zeros(n), np.zeros(n))
    mach.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=2, nt=1, m=2, act_signed=False, acc_mode=AccMode.BIAS),
        isa.matmul(w_base=10, a_base=20, acc_base=0, kt=2, nt=1, m=2, act_signed=False, acc_mode=AccMode.ACCUMULATE),
        isa.halt(),
    ])
    assert mach.run().state == SeqState.HALTED_OK
    got = model.unpack_acc(mach.acc[:2], 2, n)
    assert got.tolist() == ref_matmul(x.tolist(), w.tolist(), False, bias=bias.tolist())


@pytest.mark.parametrize("n", NS)
def test_machine_requant_writes_back(n):
    rng = np.random.default_rng(7)
    acc = rng.integers(-(1 << 20), 1 << 20, (3, 2 * n))
    mult, shift = rng.integers(0, 1 << 16, 2 * n), rng.integers(0, 24, 2 * n)
    for relu in (False, True):
        mach = model.Machine(n=n)
        mach.acc[4:4 + 6] = acc.reshape(6, n)
        mach.load_params(np.zeros(2 * n), mult, shift, at=9)
        mach.load_program([isa.requant(a_dst=30, acc_base=4, p_base=9, nt=2, m=3, relu=relu), isa.halt()])
        assert mach.run().state == SeqState.HALTED_OK
        got = model.unpack_vectors(mach.abuf[30:36], 3, 2 * n, signed=not relu)
        want = [[ref_requant_one(int(acc[i, j]), int(mult[j]), int(shift[j]), relu) for j in range(2 * n)] for i in range(3)]
        assert got.tolist() == want


def test_machine_runs_the_isa_doc_example():
    """The 196-64-10 program in docs/isa.md produces the same logits as the math-level model."""
    text = (ROOT / "docs" / "isa.md").read_text()
    rows = re.findall(r"^\| (\d+) \| `[^`]*` \| `([0-9a-f ]{39})` \|", text, re.M)
    program = [int(h.replace(" ", ""), 16) for _, h in rows]
    assert len(program) == 5
    n = 8
    rng = np.random.default_rng(8)
    x = rng.integers(0, 256, (1, 196))
    w1, w2 = rng.integers(-127, 128, (196, 64)), rng.integers(-127, 128, (64, 10))
    b1, b2 = rng.integers(-20000, 20000, 64), rng.integers(-20000, 20000, 10)
    m1, s1 = rng.integers(1, 1 << 16, 64), rng.integers(16, 24, 64)
    m2, s2 = rng.integers(1, 1 << 16, 10), rng.integers(16, 24, 10)

    mach = model.Machine(n=n)
    mach.load_weights(model.pack_weights(w1, n), at=0)            # 200 tiles
    mach.load_weights(model.pack_weights(w2, n), at=200)          # 16 tiles
    mach.load_vectors(model.pack_activations(x, n), at=0)         # 25 vectors
    mach.load_params(b1, m1, s1, at=0)
    mach.load_params(np.pad(b2, (0, 6)), np.pad(m2, (0, 6)), np.pad(s2, (0, 6)), at=64)
    mach.load_program(program)
    st = mach.run()
    assert (st.state, st.pc, st.halt_code) == (SeqState.HALTED_OK, 4, 1)

    h = model.requant(model.matmul(x, w1, False, bias=b1), m1, s1, relu=True)
    logits = model.requant(model.matmul(h, w2, False, bias=b2), m2, s2, relu=False)
    got = model.unpack_vectors(mach.abuf[40:42], 1, 10, signed=True)
    assert got.tolist() == logits.tolist()
    assert (model.unpack_vectors(mach.abuf[40:42], 1, 16, signed=True)[0, 10:] == 0).all()  # padding stays 0


@pytest.mark.parametrize("word, err", [
    (isa.Fields(0x11).encode(), SeqErr.ILLEGAL_OPCODE),
    (isa.nop() | 1 << 20, SeqErr.RESERVED_BIT),
    (isa.Fields(isa.Opcode.MATMUL, flags=0b110, kt=1, nt=1, m=1).encode(), SeqErr.ACC_MODE),
    (isa.requant(a_dst=0, acc_base=0, p_base=0, nt=1, m=0, relu=False), SeqErr.ZERO_COUNT),
])
def test_machine_decode_errors(word, err):
    mach = model.Machine(n=4)
    mach.load_program([isa.nop(), isa.nop(), word, isa.halt()])
    st = mach.run()
    assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, err, 2)


def test_machine_address_range_error():
    mach = model.Machine(n=4)
    depth = mach.depths[Mem.WBUF]
    mach.load_program([
        isa.matmul(w_base=depth - 1, a_base=0, acc_base=0, kt=2, nt=1, m=1, act_signed=False,
                   acc_mode=AccMode.OVERWRITE),
        isa.halt(),
    ])
    st = mach.run()
    assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, SeqErr.ADDR_RANGE, 0)


def test_machine_address_range_error_writes_nothing():
    """Error 5 is found before the instruction starts: ACC is untouched although the first
    tiles' addresses were in range."""
    mach = model.Machine(n=4, depths={**isa.DEFAULT_DEPTHS, Mem.ACC: 8})
    mach.acc[:] = 7
    mach.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, kt=1, nt=3, m=3, act_signed=False,
                   acc_mode=AccMode.OVERWRITE),    # ACC rows 0..8: only the last is out of range
        isa.halt(),
    ])
    st = mach.run()
    assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, SeqErr.ADDR_RANGE, 0)
    assert (mach.acc == 7).all()


def _addresses(word: int, n: int) -> dict:
    """Every address the docs/isa.md loops touch, per memory (naive enumeration)."""
    f = isa.Fields.decode(word)
    used: dict = {mem: set() for mem in Mem}
    if f.opcode == isa.Opcode.MATMUL:
        bias = (f.flags >> isa.FLAG_ACC_MODE_LSB) & 3 == AccMode.BIAS
        for nt_i in range(f.nt):
            for k in range(f.kt):
                used[Mem.WBUF].add(f.w_base + nt_i * f.kt + k)
                for m in range(f.m):
                    used[Mem.ABUF].add(f.a_base + m * f.kt + k)
                    used[Mem.ACC].add(f.acc_base + m * f.nt + nt_i)
                    if k == 0 and bias:
                        used[Mem.PTAB].update(f.p_base + nt_i * n + c for c in range(n))
    elif f.opcode == isa.Opcode.REQUANT:
        for nt_i in range(f.nt):
            for m in range(f.m):
                used[Mem.ACC].add(f.acc_base + m * f.nt + nt_i)
                used[Mem.ABUF].add(f.a_base + m * f.nt + nt_i)
                used[Mem.PTAB].update(f.p_base + nt_i * n + c for c in range(n))
    return used


def test_range_error_matches_enumeration():
    """range_error() is exactly "some address the loops generate is out of range"."""
    rng = random.Random(7)
    n = 4
    depths = {Mem.IMEM: 16, Mem.WBUF: 32, Mem.ABUF: 16, Mem.ACC: 16, Mem.PTAB: 32}
    seen = {True: 0, False: 0}
    for _ in range(3000):
        kt, nt, m = rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6)
        if rng.random() < 0.5:
            word = isa.matmul(w_base=rng.randrange(32), a_base=rng.randrange(16), acc_base=rng.randrange(16),
                              p_base=rng.randrange(32), kt=kt, nt=nt, m=m, act_signed=False,
                              acc_mode=rng.choice(list(AccMode)[:3]))
        else:
            word = isa.requant(a_dst=rng.randrange(16), acc_base=rng.randrange(16), p_base=rng.randrange(32),
                               nt=nt, m=m, relu=False)
        assert isa.decode_error(word, depths) is SeqErr.NONE
        want = any(a >= depths[mem] for mem, addrs in _addresses(word, n).items() for a in addrs)
        assert model.range_error(word, n, depths) == want
        seen[want] += 1
    assert min(seen.values()) > 500, seen
    assert not model.range_error(isa.halt(), n, depths) and not model.range_error(isa.nop(), n, depths)


def test_machine_pc_overrun():
    mach = model.Machine(n=4, depths={**isa.DEFAULT_DEPTHS, Mem.IMEM: 4})
    mach.load_program([isa.nop()] * 4)
    st = mach.run()
    assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, SeqErr.PC_OVERRUN, 4)


def test_machine_empty_memory_halts_immediately():
    st = model.Machine(n=4).run()
    assert (st.state, st.pc, st.halt_code) == (SeqState.HALTED_OK, 0, 0)


def test_tinytpu_n_env_is_respected():
    # The make targets set TINYTPU_N; the machine tests above cover all sizes regardless.
    assert int(os.environ.get("TINYTPU_N", "4")) in (2, 4, 8)
