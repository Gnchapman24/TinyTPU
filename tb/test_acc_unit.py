"""cocotb tests for rtl/acc_unit.sv (task A3.4), at N = TINYTPU_N with a 64-row buffer.

Every update is mirrored in a Python model (rows of N INT32 values, updated with
tinytpu.model.wrap32). Rows are first written through the host port, so the buffer's initial
contents never matter; results are read back through the read port. Update streams respect the
documented rule that the same row is not updated twice within 3 cycles.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

from tinytpu import model
from tinytpu.isa import AccMode

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
DEPTH = 64


def pack(vals) -> int:
    return sum((int(v) & 0xFFFFFFFF) << (32 * i) for i, v in enumerate(vals))


def unpack(word: int) -> list[int]:
    return [int(model.wrap32((word >> (32 * i)) & 0xFFFFFFFF)) for i in range(N)]


def rand_vec(rng: random.Random, lo: int = -(1 << 20), hi: int = 1 << 20) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(N)]


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.rows: dict[int, list[int]] = {}

    def idle_inputs(self) -> None:
        d = self.dut
        d.in_valid.value = 0
        d.rd_en.value = 0
        d.wr_en.value = 0

    async def host_write(self, addr: int, vals: list[int]) -> None:
        await FallingEdge(self.dut.clk)
        self.idle_inputs()
        self.dut.wr_en.value, self.dut.wr_addr.value, self.dut.wr_data.value = 1, addr, pack(vals)
        await RisingEdge(self.dut.clk)
        self.rows[addr] = [int(model.wrap32(v)) for v in vals]
        await FallingEdge(self.dut.clk)
        self.dut.wr_en.value = 0

    async def host_read(self, addr: int) -> list[int]:
        await FallingEdge(self.dut.clk)
        self.idle_inputs()
        self.dut.rd_en.value, self.dut.rd_addr.value = 1, addr
        await FallingEdge(self.dut.clk)
        self.dut.rd_en.value = 0
        return unpack(int(self.dut.rd_data.value))

    async def update(self, addr: int, mode: AccMode, data: list[int], bias: list[int] | None = None) -> None:
        """Drive one in_valid cycle and apply the same update to the model."""
        d = self.dut
        await FallingEdge(d.clk)
        d.rd_en.value = d.wr_en.value = 0
        d.in_valid.value, d.in_addr.value, d.in_mode.value = 1, addr, int(mode)
        d.in_data.value = pack(data)
        d.in_bias.value = pack(bias or [0] * N)
        await RisingEdge(d.clk)
        old = self.rows[addr]
        if mode == AccMode.OVERWRITE:
            new = data
        elif mode == AccMode.BIAS:
            new = [b + x for b, x in zip(bias, data)]
        else:
            new = [o + x for o, x in zip(old, data)]
        self.rows[addr] = [int(model.wrap32(v)) for v in new]

    async def idle(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            await FallingEdge(self.dut.clk)
            self.idle_inputs()

    async def drain_and_check(self) -> None:
        await self.idle(3)
        assert self.dut.busy.value == 0
        for addr, want in sorted(self.rows.items()):
            assert await self.host_read(addr) == want, f"row {addr}"


async def setup(dut) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut)
    dut.rst.value = 1
    b.idle_inputs()
    dut.in_mode.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return b


async def init_rows(b: Bench, rng: random.Random) -> None:
    for addr in range(DEPTH):
        await b.host_write(addr, rand_vec(rng))


@cocotb.test()
async def host_write_read_roundtrip(dut):
    b = await setup(dut)
    rng = random.Random(1)
    await init_rows(b, rng)
    for addr in rng.sample(range(DEPTH), 10):
        assert await b.host_read(addr) == b.rows[addr]


@cocotb.test()
async def each_mode(dut):
    b = await setup(dut)
    rng = random.Random(2)
    await init_rows(b, rng)
    await b.update(5, AccMode.OVERWRITE, rand_vec(rng))
    await b.update(6, AccMode.BIAS, rand_vec(rng), rand_vec(rng))
    await b.update(7, AccMode.ACCUMULATE, rand_vec(rng))
    await b.drain_and_check()


@cocotb.test()
async def matmul_shaped_stream(dut):
    """Rows m*NT + n, first k-tile in bias mode, later k-tiles accumulating, M vectors per tile."""
    b = await setup(dut)
    rng = random.Random(3)
    await init_rows(b, rng)
    m_dim, nt, kt = 5, 3, 4
    for n in range(nt):
        bias = rand_vec(rng)
        for k in range(kt):
            for m in range(m_dim):
                mode = AccMode.BIAS if k == 0 else AccMode.ACCUMULATE
                await b.update(m * nt + n, mode, rand_vec(rng, -50000, 50000), bias)
    await b.drain_and_check()


@cocotb.test()
async def one_row_updated_every_third_cycle(dut):
    """The tightest legal spacing for the same row: accumulate every 3rd cycle."""
    b = await setup(dut)
    rng = random.Random(4)
    await init_rows(b, rng)
    for i in range(30):
        await b.update(9, AccMode.ACCUMULATE if i else AccMode.OVERWRITE, rand_vec(rng))
        await b.idle(2)
    await b.drain_and_check()


@cocotb.test()
async def random_updates_respecting_spacing(dut):
    b = await setup(dut)
    rng = random.Random(5)
    await init_rows(b, rng)
    recent: list[int | None] = [None, None]
    for _ in range(600):
        if rng.random() < 0.2:
            await b.idle()
            recent = [recent[1], None]
            continue
        addr = rng.choice([a for a in range(DEPTH) if a not in recent])
        mode = rng.choice([AccMode.OVERWRITE, AccMode.BIAS, AccMode.ACCUMULATE])
        await b.update(addr, mode, rand_vec(rng, -(1 << 31), (1 << 31) - 1), rand_vec(rng, -(1 << 31), (1 << 31) - 1))
        recent = [recent[1], addr]
    await b.drain_and_check()


@cocotb.test()
async def int32_wraparound(dut):
    b = await setup(dut)
    await b.host_write(1, [(1 << 31) - 1] * N)
    await b.host_write(2, [-(1 << 31)] * N)
    await b.host_write(3, [0] * N)
    await b.update(1, AccMode.ACCUMULATE, [1] * N)
    await b.update(2, AccMode.ACCUMULATE, [-1] * N)
    await b.update(3, AccMode.BIAS, [(1 << 31) - 1] * N, [10] * N)
    await b.drain_and_check()
    assert b.rows[1] == [-(1 << 31)] * N and b.rows[2] == [(1 << 31) - 1] * N


@cocotb.test()
async def busy_flag_covers_the_pipeline(dut):
    b = await setup(dut)
    await b.host_write(4, [0] * N)
    await b.update(4, AccMode.OVERWRITE, [1] * N)
    seen = []
    for _ in range(4):
        await FallingEdge(dut.clk)
        b.idle_inputs()
        seen.append(int(dut.busy.value))
    assert seen == [1, 1, 0, 0], seen


@cocotb.test()
async def reset_drops_update_in_flight(dut):
    b = await setup(dut)
    await b.host_write(8, [123] * N)
    await FallingEdge(dut.clk)
    dut.in_valid.value, dut.in_addr.value, dut.in_mode.value = 1, 8, int(AccMode.OVERWRITE)
    dut.in_data.value = pack([999] * N)
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.rst.value = 1                       # reset while the update is in stage 1
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await b.idle(3)
    assert await b.host_read(8) == [123] * N


def test_acc_unit():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"acc_unit_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "isa_pkg.sv", ROOT / "rtl" / "acc_unit.sv"],
        hdl_toplevel="acc_unit",
        parameters={"N": N, "DEPTH": DEPTH},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="acc_unit", test_module="test_acc_unit", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
