"""cocotb tests for rtl/systolic_array.sv (task A3.3), at N = TINYTPU_N.

Each test builds a cycle-by-cycle schedule of weight-row loads and activation vectors that
follows the array's timing rules (loads overlapping the stream; a new tile N cycles after the
previous tile's last vector), runs it, and checks every result against tinytpu.model.matmul for
that vector's tile: values, order, and the exact 2N+1-cycle latency. out_valid must be low
whenever no result is due.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb_tools.runner import get_runner

from tinytpu import model

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
LAT = 2 * N + 1


@dataclass
class Cycle:
    vec: list[int] | None = None          # activation vector entering this cycle
    wrow: tuple[int, list[int]] | None = None   # (row, weights) loaded this cycle


@dataclass
class Schedule:
    cycles: list[Cycle] = field(default_factory=list)
    expected: dict[int, list[int]] = field(default_factory=dict)   # cycle -> result vector

    def add_tile(self, w: np.ndarray, vectors: list[np.ndarray], act_signed: bool,
                 rng: random.Random, max_gap: int = 0, spacing: int = N) -> None:
        """Load tile w (rows 0..N-1 on consecutive cycles) and stream vectors from its first
        load cycle, with random gaps; then leave ``spacing`` cycles (the rule says N) between
        this tile's last vector and the next tile's first load."""
        start = len(self.cycles)
        t = start
        for v in vectors:
            t += rng.randint(0, max_gap)
            self._at(t).vec = [int(x) for x in v]
            self.expected[t + LAT] = model.matmul(v[None, :], w, act_signed)[0].tolist()
            t += 1
        for r in range(N):
            self._at(start + r).wrow = (r, [int(x) for x in w[r]])
        last_vec = max((c for c in range(start, len(self.cycles)) if self.cycles[c].vec is not None),
                       default=start)
        self._at(max(last_vec + spacing, start + N) - 1)   # next tile's row 0 at last + spacing

    def _at(self, t: int) -> Cycle:
        while len(self.cycles) <= t:
            self.cycles.append(Cycle())
        return self.cycles[t]


def pack(vals: list[int], bits: int) -> int:
    mask = (1 << bits) - 1
    return sum((int(v) & mask) << (bits * i) for i, v in enumerate(vals))  # Python ints: no overflow


def unpack_signed(word: int, bits: int, count: int) -> list[int]:
    out = []
    for i in range(count):
        v = (word >> (bits * i)) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out


async def setup(dut, act_signed: bool) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.act_signed.value = int(act_signed)
    dut.a_valid.value = 0
    dut.w_load.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    dut.w_row.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def run(dut, sched: Schedule, act_signed: bool) -> int:
    """Drive the schedule; check every cycle's out_valid and out_data. Returns results checked."""
    await setup(dut, act_signed)
    total = len(sched.cycles) + LAT + 2
    checked = 0
    for t in range(total):
        await FallingEdge(dut.clk)
        if t > 0:
            valid = int(dut.out_valid.value)
            want = sched.expected.get(t)
            assert valid == (want is not None), f"cycle {t}: out_valid={valid}, expected {want is not None}"
            if want is not None:
                got = unpack_signed(int(dut.out_data.value), 32, N)
                assert got == want, f"cycle {t}: {got} != {want}"
                checked += 1
        c = sched.cycles[t] if t < len(sched.cycles) else Cycle()
        dut.a_valid.value = int(c.vec is not None)
        dut.a_in.value = pack(c.vec, 8) if c.vec is not None else 0
        dut.w_load.value = int(c.wrow is not None)
        if c.wrow is not None:
            dut.w_row.value = c.wrow[0]
            dut.w_in.value = pack(c.wrow[1], 8)
    assert checked == len(sched.expected)
    return checked


def rand_w(rng: np.random.Generator, lo: int = -127, hi: int = 127) -> np.ndarray:
    return rng.integers(lo, hi + 1, (N, N))


@cocotb.test()
async def one_tile_back_to_back_unsigned(dut):
    rng = np.random.default_rng(1)
    s = Schedule()
    s.add_tile(rand_w(rng), [rng.integers(0, 256, N) for _ in range(40)], False, random.Random(1))
    assert await run(dut, s, act_signed=False) == 40


@cocotb.test()
async def one_tile_signed(dut):
    rng = np.random.default_rng(2)
    s = Schedule()
    s.add_tile(rand_w(rng), [rng.integers(-128, 128, N) for _ in range(30)], True, random.Random(2))
    await run(dut, s, act_signed=True)


@cocotb.test()
async def many_tiles_min_spacing_with_gaps(dut):
    rng, prng = np.random.default_rng(3), random.Random(3)
    s = Schedule()
    for _ in range(8):
        vecs = [rng.integers(0, 256, N) for _ in range(prng.randint(1, 3 * N))]
        s.add_tile(rand_w(rng), vecs, False, prng, max_gap=prng.choice([0, 0, 2]))
    await run(dut, s, act_signed=False)


@cocotb.test()
async def extreme_values(dut):
    s = Schedule()
    prng = random.Random(4)
    w_neg = np.full((N, N), -128)
    w_pos = np.full((N, N), 127)
    s.add_tile(w_neg, [np.full(N, 255), np.full(N, 0), np.full(N, 128)], False, prng)
    s.add_tile(w_pos, [np.full(N, 255), np.full(N, 1)], False, prng)
    await run(dut, s, act_signed=False)


@cocotb.test()
async def extreme_values_signed(dut):
    s = Schedule()
    prng = random.Random(5)
    s.add_tile(np.full((N, N), -128), [np.full(N, -128), np.full(N, 127), np.full(N, -1)], True, prng)
    await run(dut, s, act_signed=True)


@cocotb.test()
async def identity_tile_passes_vectors_through(dut):
    rng = np.random.default_rng(6)
    s = Schedule()
    vecs = [rng.integers(0, 256, N) for _ in range(10)]
    s.add_tile(np.eye(N, dtype=int), vecs, False, random.Random(6))
    await run(dut, s, act_signed=False)
    for v, (_, want) in zip(vecs, sorted(s.expected.items())):
        assert want == [int(x) for x in v]


@cocotb.test()
async def reload_spacing_rule_is_tight(dut):
    """Loading the next tile one cycle earlier than the rule allows must corrupt the previous
    tile's last result, which shows the documented N-cycle spacing is the minimum."""
    rng = np.random.default_rng(8)
    s = Schedule()
    # N+1 vectors, so the last vector (not the tile's own N load cycles) sets the next start.
    first = [rng.integers(1, 256, N) for _ in range(N + 1)]
    s.add_tile(np.full((N, N), 3), first, False, random.Random(8), spacing=N - 1)
    assert len(s.cycles) == N + (N - 1), "schedule must really start the next tile one cycle early"
    s.add_tile(np.full((N, N), -5), [rng.integers(1, 256, N)], False, random.Random(9))
    try:
        await run(dut, s, act_signed=False)
    except AssertionError:
        return                                  # expected: the early reload broke a result
    raise AssertionError("loading one cycle early did not affect the result; the rule is not tight")


@cocotb.test()
async def reset_clears_results_in_flight(dut):
    rng = np.random.default_rng(7)
    await setup(dut, False)
    w = rand_w(rng)
    for r in range(N):                       # load a tile, then send vectors
        await FallingEdge(dut.clk)
        dut.w_load.value, dut.w_row.value, dut.w_in.value = 1, r, pack(list(w[r]), 8)
        dut.a_valid.value, dut.a_in.value = 1, pack(list(rng.integers(0, 256, N)), 8)
    await FallingEdge(dut.clk)
    dut.w_load.value = 0
    dut.a_valid.value = 0
    dut.rst.value = 1                        # reset with N results still in flight
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(LAT + 2):
        await FallingEdge(dut.clk)
        assert dut.out_valid.value == 0, "results in flight at reset must be dropped"


def test_systolic_array():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"systolic_array_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "pe.sv", ROOT / "rtl" / "systolic_array.sv"],
        hdl_toplevel="systolic_array",
        parameters={"N": N},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="systolic_array", test_module="test_systolic_array",
                test_dir=Path(__file__).parent, results_xml=str(build_dir / "results.xml"))
