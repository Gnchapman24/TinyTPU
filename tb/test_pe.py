"""cocotb tests for rtl/pe.sv (task A3.2).

The PE is compared every cycle against a cycle-accurate Python model of its four registers
(a_q, w_q, m_q, p_q): a_out, psum_out, the stationary weight, the 3-cycle product latency, and
48-bit wraparound of the partial sum.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
AW, WW, PW = 9, 9, 48


def wrap(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class PeModel:
    """Registers updated exactly as the RTL's always_ff block."""

    def __init__(self):
        self.a_q = self.w_q = self.m_q = self.p_q = 0

    def edge(self, a_in: int, w_in: int, w_load: bool, psum_in: int) -> None:
        a_q, w_q, m_q = self.a_q, self.w_q, self.m_q
        self.a_q = a_in
        if w_load:
            self.w_q = w_in
        self.m_q = a_q * w_q
        self.p_q = wrap(psum_in + m_q, PW)


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.model = PeModel()
        self.checked = 0

    async def cycle(self, a_in: int, w_in: int, w_load: bool, psum_in: int, check: bool = True):
        d = self.dut
        await FallingEdge(d.clk)
        if check:
            assert d.a_out.value.to_signed() == self.model.a_q, "a_out"
            assert d.psum_out.value.to_signed() == self.model.p_q, \
                f"psum_out {d.psum_out.value.to_signed()} != {self.model.p_q}"
            self.checked += 1
        d.a_in.value = a_in
        d.w_in.value = w_in
        d.w_load.value = int(w_load)
        d.psum_in.value = psum_in
        if check:
            # Outputs are registered: driving new inputs must not change them before the edge.
            await Timer(1, unit="ns")
            assert d.a_out.value.to_signed() == self.model.a_q, "a_out changed without a clock edge"
            assert d.psum_out.value.to_signed() == self.model.p_q, "psum_out changed without a clock edge"
        await RisingEdge(d.clk)
        self.model.edge(a_in, w_in, w_load, psum_in)


async def setup(dut) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut)
    # Warm-up: load a known weight and push known values through every register.
    await b.cycle(0, 0, True, 0, check=False)
    for _ in range(3):
        await b.cycle(0, 0, False, 0, check=False)
    return b


@cocotb.test()
async def product_latency_is_three_cycles(dut):
    """Measured directly: a_in sampled at edge t appears (times w) in psum_out after edge t+2."""
    b = await setup(dut)
    await b.cycle(0, 5, True, 100)           # weight = 5
    await b.cycle(7, 0, False, 100)          # activation 7 sampled at this edge (call it t)
    seen = []
    for _ in range(4):                       # one read per cycle; psum_in held at 100
        await FallingEdge(dut.clk)
        seen.append(dut.psum_out.value.to_signed())
        dut.a_in.value = 0
        dut.w_load.value = 0
        dut.psum_in.value = 100
        await RisingEdge(dut.clk)
    # Edges t, t+1, t+2: a_q = 7, then m_q = 35, then p_q = 100 + 35. So the product is in
    # psum_out after the third edge that sees it, as the header states.
    assert seen == [100, 100, 135, 100], seen


@cocotb.test()
async def weight_is_stationary(dut):
    b = await setup(dut)
    await b.cycle(0, -3, True, 0)
    rng = random.Random(1)
    for _ in range(50):                     # w_in changes but w_load is low
        await b.cycle(rng.randint(-256, 255), rng.randint(-256, 255), False, rng.randint(-1000, 1000))


@cocotb.test()
async def extreme_operands(dut):
    b = await setup(dut)
    for a, w in [(255, -128), (-128, -128), (-256, -256), (255, 127), (-256, 255), (0, -256)]:
        await b.cycle(0, w, True, 0)
        await b.cycle(a, 0, False, 0)
        for _ in range(3):
            await b.cycle(0, 0, False, 0)


@cocotb.test()
async def partial_sum_wraps_at_48_bits(dut):
    b = await setup(dut)
    await b.cycle(0, 255, True, 0)
    await b.cycle(255, 0, False, 0)
    big = (1 << 47) - 10
    for _ in range(4):
        await b.cycle(0, 0, False, big)      # big + 65025 wraps negative


@cocotb.test()
async def random_streams(dut):
    b = await setup(dut)
    rng = random.Random(2)
    for _ in range(3000):
        await b.cycle(
            rng.randint(-256, 255),
            rng.randint(-256, 255),
            rng.random() < 0.1,
            rng.randint(-(1 << 47), (1 << 47) - 1),
        )


def test_pe():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"pe_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "pe.sv"],
        hdl_toplevel="pe",
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="pe", test_module="test_pe", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
