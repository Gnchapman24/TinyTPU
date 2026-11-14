"""cocotb tests for rtl/range_check.sv (task A4.2): ISA error 5 before execution.

Expected values come from tinytpu.model.range_error at N = TINYTPU_N, for two sets of depths
(the defaults and small ones). Cases: every memory's exact boundary (highest address = depth - 1
passes, = depth fails), maximum counts, random legal instructions of every opcode, and back-to-
back checks; every check must finish exactly LATENCY cycles after start.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb_tools.runner import get_runner

from tinytpu import isa, model
from tinytpu.isa import AccMode, Mem

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
LATENCY = isa.COUNT_W + 2
CMAX = (1 << isa.COUNT_W) - 1

DEPTH_SETS = {
    "default": dict(isa.DEFAULT_DEPTHS),
    "small": {**isa.DEFAULT_DEPTHS, Mem.WBUF: 64, Mem.ABUF: 32, Mem.ACC: 16, Mem.PTAB: 32},
    # Full 16-bit address fields: base + product can exceed 2**24 (the adder needs its carry).
    "large": {**isa.DEFAULT_DEPTHS, Mem.WBUF: 1 << 16, Mem.ABUF: 1 << 16, Mem.ACC: 1 << 16, Mem.PTAB: 1 << 16},
}
DEPTHS = DEPTH_SETS[os.environ.get("RC_DEPTHS", "default")]
MODES = [AccMode.OVERWRITE, AccMode.ACCUMULATE, AccMode.BIAS]


def matmul(w=0, a=0, acc=0, p=0, kt=1, nt=1, m=1, mode=AccMode.OVERWRITE) -> int:
    return isa.matmul(w_base=w, a_base=a, acc_base=acc, p_base=p, kt=kt, nt=nt, m=m, act_signed=False, acc_mode=mode)


def requant(a=0, acc=0, p=0, nt=1, m=1) -> int:
    return isa.requant(a_dst=a, acc_base=acc, p_base=p, nt=nt, m=m, relu=False)


def boundary_words() -> list[int]:
    """For each memory, an instruction whose last address is depth - 1 and one with depth."""
    d = DEPTHS
    words = []
    for extra in (0, 1):
        # MATMUL WBUF: w_base + nt*kt - 1
        kt, nt = 3, 2
        words.append(matmul(w=d[Mem.WBUF] - kt * nt + extra, kt=kt, nt=nt))
        # MATMUL ABUF: a_base + m*kt - 1
        words.append(matmul(a=d[Mem.ABUF] - 2 * 3 + extra, kt=3, m=2))
        # MATMUL ACC: acc_base + m*nt - 1
        words.append(matmul(acc=d[Mem.ACC] - 2 * 2 + extra, nt=2, m=2))
        # MATMUL PTAB (bias mode only): p_base + nt*N - 1
        nt = min(CMAX, max(1, d[Mem.PTAB] // N // 2))
        for mode in MODES:
            words.append(matmul(p=d[Mem.PTAB] - nt * N + extra, nt=nt, mode=mode))
        # REQUANT ACC / ABUF / PTAB
        words.append(requant(acc=d[Mem.ACC] - 3 * 2 + extra, nt=2, m=3))
        words.append(requant(a=d[Mem.ABUF] - 3 * 2 + extra, nt=2, m=3))
        words.append(requant(p=d[Mem.PTAB] - 2 * N + extra, nt=2))
    return words


def random_word(rng: random.Random) -> int:
    d = DEPTHS

    def cnt() -> int:
        return rng.choice([1, rng.randint(1, 8), rng.randint(1, CMAX)])

    kind = rng.random()
    if kind < 0.05:
        return isa.halt(rng.randrange(1 << 16))
    if kind < 0.1:
        return isa.nop()
    if kind < 0.6:
        return matmul(rng.randrange(d[Mem.WBUF]), rng.randrange(d[Mem.ABUF]), rng.randrange(d[Mem.ACC]),
                      rng.randrange(d[Mem.PTAB]), cnt(), cnt(), cnt(), rng.choice(MODES))
    return requant(rng.randrange(d[Mem.ABUF]), rng.randrange(d[Mem.ACC]), rng.randrange(d[Mem.PTAB]), cnt(), cnt())


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.seen = {True: 0, False: 0}

    async def check(self, word: int, gap: int = 0) -> None:
        d = self.dut
        assert isa.decode_error(word, DEPTHS) is isa.SeqErr.NONE
        await FallingEdge(d.clk)
        d.start.value, d.instr.value = 1, word
        await FallingEdge(d.clk)
        d.start.value = 0
        d.instr.value = random.getrandbits(128)        # the input is captured at start only
        for cycle in range(1, LATENCY + 1):
            assert d.busy.value == 1
            if cycle < LATENCY:
                assert d.done.value == 0, f"done early, cycle {cycle}"
                await FallingEdge(d.clk)
        assert d.done.value == 1, "done not on time"
        want = model.range_error(word, N, DEPTHS)
        assert int(d.err.value) == want, f"{isa.Fields.decode(word)}: got {int(d.err.value)}, want {want}"
        self.seen[want] += 1
        await FallingEdge(d.clk)
        assert d.done.value == 0 and d.busy.value == 0
        for _ in range(gap):
            await FallingEdge(d.clk)


async def setup(dut) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.start.value = 0
    dut.instr.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return Bench(dut)


@cocotb.test()
async def boundaries(dut):
    b = await setup(dut)
    for w in boundary_words():
        await b.check(w)
    # extra = 1 fails for 4 MATMUL cases (PTAB only in bias mode) and 3 REQUANT cases
    assert b.seen == {True: 7, False: 11}, b.seen


@cocotb.test()
async def maximum_counts(dut):
    b = await setup(dut)
    d = DEPTHS
    top = {mem: d[mem] - 1 for mem in Mem}
    words = [
        matmul(kt=CMAX, nt=CMAX, m=CMAX), matmul(kt=CMAX, nt=1, m=1), matmul(kt=1, nt=CMAX, m=1, mode=AccMode.BIAS),
        requant(nt=CMAX, m=CMAX), requant(nt=1, m=CMAX), matmul(kt=1, nt=1, m=1),
        # Products that are multiples of 2**16: a truncated product would look small.
        matmul(kt=256, nt=256, m=256), requant(nt=256, m=256), matmul(kt=4096 - 1, nt=16 * 17, m=1),
        # Highest bases with the largest products: base + product passes 2**24.
        matmul(w=top[Mem.WBUF], a=top[Mem.ABUF], acc=top[Mem.ACC], p=top[Mem.PTAB], kt=CMAX, nt=CMAX, m=CMAX,
               mode=AccMode.BIAS),
        requant(a=top[Mem.ABUF], acc=top[Mem.ACC], p=top[Mem.PTAB], nt=CMAX, m=CMAX),
        # Only the WBUF sum is large (every other memory in range): its carry alone decides.
        matmul(w=top[Mem.WBUF], kt=CMAX, nt=CMAX, m=1, mode=AccMode.OVERWRITE),
    ]
    for w in words:
        await b.check(w)
    assert b.seen[True] >= 2, b.seen          # which ones fail depends on the depths


@cocotb.test()
async def random_instructions(dut):
    b = await setup(dut)
    rng = random.Random(1)
    for _ in range(600):
        await b.check(random_word(rng), gap=rng.choice([0, 0, 1, 3]))
    assert min(b.seen.values()) > 50, b.seen


@cocotb.test()
async def start_while_busy_is_ignored(dut):
    b = await setup(dut)
    d = dut
    bad, good = matmul(kt=CMAX, nt=CMAX), matmul()
    await FallingEdge(d.clk)
    d.start.value, d.instr.value = 1, bad
    await FallingEdge(d.clk)
    d.instr.value = good                                # start held high: must not restart
    for _ in range(LATENCY - 1):
        await FallingEdge(d.clk)
    assert d.done.value == 1 and d.err.value == model.range_error(bad, N, DEPTHS)
    d.start.value = 0
    await FallingEdge(d.clk)
    await b.check(good)


@cocotb.test()
async def reset_abandons_a_check(dut):
    b = await setup(dut)
    d = dut
    await FallingEdge(d.clk)
    d.start.value, d.instr.value = 1, matmul()
    await FallingEdge(d.clk)
    d.start.value = 0
    await ClockCycles(d.clk, 4)
    await FallingEdge(d.clk)
    d.rst.value = 1
    await FallingEdge(d.clk)
    d.rst.value = 0
    for _ in range(LATENCY + 2):
        assert d.done.value == 0 and d.busy.value == 0
        await FallingEdge(d.clk)
    await b.check(requant(nt=2, m=2))


@pytest.mark.parametrize("depths", list(DEPTH_SETS))
def test_range_check(depths):
    d = DEPTH_SETS[depths]
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"range_check_{depths}_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "isa_pkg.sv", ROOT / "rtl" / "range_check.sv"],
        hdl_toplevel="range_check",
        parameters={"N": N, "WBUF_DEPTH": d[Mem.WBUF], "ABUF_DEPTH": d[Mem.ABUF], "ACC_DEPTH": d[Mem.ACC],
                    "PTAB_DEPTH": d[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="range_check", test_module="test_range_check", test_dir=Path(__file__).parent,
                extra_env={"RC_DEPTHS": depths}, results_xml=str(build_dir / "results.xml"))
