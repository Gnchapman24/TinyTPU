"""cocotb tests for rtl/sync_fifo.sv (task A2.2).

Every cycle, the DUT is compared against a Python deque model: count, in_ready, out_valid and the
first-word-fall-through out_data. Runs at DEPTH 2, 4 and 16 (WIDTH 8), with the module's own
assertions active.
"""

from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))


def depth() -> int:
    return int(os.environ["FIFO_DEPTH"])


class Model:
    """Cycle-accurate reference: drive inputs, check outputs, then advance one clock."""

    def __init__(self, dut):
        self.dut = dut
        self.q: deque[int] = deque()
        self.pushes = 0
        self.pops = 0

    def check(self) -> None:
        d, q = self.dut, self.q
        assert int(d.count.value) == len(q), f"count {int(d.count.value)} != {len(q)}"
        assert int(d.in_ready.value) == (len(q) < depth()), "in_ready"
        assert int(d.out_valid.value) == (len(q) > 0), "out_valid"
        if q:
            assert int(d.out_data.value) == q[0], f"out_data {int(d.out_data.value):#x} != {q[0]:#x}"

    async def settle(self) -> None:
        """Wait for the falling edge (registered outputs updated), then check."""
        await FallingEdge(self.dut.clk)
        self.check()

    async def step(self, in_valid: bool, in_data: int, out_ready: bool) -> None:
        d = self.dut
        await FallingEdge(d.clk)
        d.in_valid.value = int(in_valid)
        d.in_data.value = in_data
        d.out_ready.value = int(out_ready)
        self.check()  # outputs depend only on registered state, so they are stable here
        push = in_valid and len(self.q) < depth()
        pop = out_ready and len(self.q) > 0
        await RisingEdge(d.clk)
        if pop:
            self.q.popleft()
            self.pops += 1
        if push:
            self.q.append(in_data)
            self.pushes += 1


async def setup(dut) -> Model:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.out_ready.value = 0
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return Model(dut)


@cocotb.test()
async def empty_after_reset(dut):
    m = await setup(dut)
    for _ in range(5):
        await m.step(False, 0, True)   # popping an empty FIFO does nothing
    assert m.pops == 0


@cocotb.test()
async def fill_then_drain_in_order(dut):
    m = await setup(dut)
    data = [(0xA0 + i) & 0xFF for i in range(depth() + 3)]
    for b in data:                     # the last 3 writes must be refused (full)
        await m.step(True, b, False)
    assert m.pushes == depth()
    for _ in range(depth() + 2):
        await m.step(False, 0, True)
    assert m.pops == depth()
    await m.settle()


@cocotb.test()
async def full_refuses_write_even_while_reading(dut):
    m = await setup(dut)
    for i in range(depth()):
        await m.step(True, i, False)
    # Full: write and read in the same cycle -> only the read happens (in_ready is low).
    await m.step(True, 0x77, True)
    assert m.pushes == depth() and m.pops == 1
    await m.step(True, 0x77, False)    # now there is space
    assert m.pushes == depth() + 1
    await m.settle()


@cocotb.test()
async def simultaneous_read_write_keeps_count(dut):
    m = await setup(dut)
    for i in range(depth() // 2):
        await m.step(True, i, False)
    for i in range(50):
        await m.step(True, 0x40 + i, True)
    await m.settle()
    assert int(dut.count.value) == depth() // 2


@cocotb.test()
async def random_traffic(dut):
    m = await setup(dut)
    rng = random.Random(7)
    # Phases bias the FIFO towards full, empty, and the middle.
    for p_write, p_read in [(0.9, 0.2), (0.2, 0.9), (0.5, 0.5), (0.8, 0.8), (0.95, 0.05), (0.05, 0.95)]:
        for _ in range(400):
            await m.step(rng.random() < p_write, rng.getrandbits(8), rng.random() < p_read)
    assert m.pushes > 300 and m.pops > 300  # sanity: real traffic flowed in every phase mix


@cocotb.test()
async def reset_empties(dut):
    m = await setup(dut)
    for i in range(depth()):
        await m.step(True, i, False)
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    m.q.clear()
    m.check()
    await m.step(True, 0x5A, False)
    await m.step(False, 0, True)
    assert m.pops == 1


@pytest.mark.parametrize("fifo_depth", [2, 4, 16])
def test_sync_fifo(fifo_depth):
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"sync_fifo_d{fifo_depth}_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "sync_fifo.sv"],
        hdl_toplevel="sync_fifo",
        parameters={"WIDTH": 8, "DEPTH": fifo_depth},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        hdl_toplevel="sync_fifo",
        test_module="test_sync_fifo",
        test_dir=Path(__file__).parent,
        extra_env={"FIFO_DEPTH": str(fifo_depth)},
        results_xml=str(build_dir / "results.xml"),
    )
