"""cocotb tests for rtl/requant.sv (task A3.5).

Expected values always come from tinytpu.model.requant (low byte, as REQUANT writes it into the
activation buffer). The module has no N parameter; TINYTPU_N only shapes the REQUANT-like stream
(N columns, each with its own M and s) and names the build directory.

A scoreboard drives at most one value per cycle (optionally with gaps) and checks every result,
its tag and the order, plus the exact latency.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

from tinytpu import model

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
LATENCY = 6
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def expect(acc: int, mult: int, shift: int, relu: bool) -> int:
    y = model.requant(np.array([[acc]]), np.array([mult]), np.array([shift]), relu)[0, 0]
    return int(y) & 0xFF


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.cycle = 0
        self.sent: list[tuple[int, int, int, int, bool, int]] = []   # (cycle, acc, M, s, relu, tag)
        self.got: list[tuple[int, int, int]] = []                     # (cycle, data, tag)
        cocotb.start_soon(self._monitor())

    async def _monitor(self) -> None:
        while True:
            await RisingEdge(self.dut.clk)
            self.cycle += 1
            await FallingEdge(self.dut.clk)       # outputs are registered: stable here
            if self.dut.out_valid.value == 1:
                self.got.append((self.cycle, int(self.dut.out_data.value), int(self.dut.out_tag.value)))

    def idle_inputs(self) -> None:
        d = self.dut
        d.in_valid.value = 0
        # Garbage on the data inputs while idle: it must never show up as a result.
        d.in_acc.value = random.getrandbits(32)
        d.in_mult.value = random.getrandbits(16)
        d.in_shift.value = random.getrandbits(6)
        d.in_relu.value = random.getrandbits(1)
        d.in_tag.value = random.getrandbits(8)

    async def send(self, acc: int, mult: int, shift: int, relu: bool, tag: int = 0) -> None:
        d = self.dut
        await FallingEdge(d.clk)
        d.in_valid.value = 1
        d.in_acc.value = acc & 0xFFFFFFFF
        d.in_mult.value, d.in_shift.value, d.in_relu.value, d.in_tag.value = mult, shift, int(relu), tag
        self.sent.append((self.cycle, acc, mult, shift, relu, tag))

    async def idle(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            await FallingEdge(self.dut.clk)
            self.idle_inputs()

    async def drain_and_check(self) -> None:
        await self.idle(LATENCY + 2)
        assert len(self.got) == len(self.sent), f"sent {len(self.sent)}, got {len(self.got)}"
        for (c_in, acc, mult, shift, relu, tag), (c_out, data, otag) in zip(self.sent, self.got):
            what = f"acc={acc} M={mult} s={shift} relu={relu}"
            assert c_out - c_in == LATENCY, f"{what}: latency {c_out - c_in}"
            assert otag == tag, f"{what}: tag {otag} != {tag}"
            assert data == expect(acc, mult, shift, relu), f"{what}: got {data:#04x}, want {expect(acc, mult, shift, relu):#04x}"
        self.sent.clear()
        self.got.clear()


async def setup(dut) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut)
    dut.rst.value = 1
    b.idle_inputs()
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    b.got.clear()
    return b


def tie_cases() -> list[tuple[int, int, int]]:
    """(acc, M, s) whose exact quotient is k + 1/2: round half up must go towards +inf."""
    cases = []
    for s in (1, 2, 5, 16, 31, 40, 46, 47):
        for k in (-3, -2, -1, 0, 1, 2, 127, -128, 255):
            p = (2 * k + 1) << (s - 1)          # P = (k + 1/2) * 2**s
            for mult in (1, 3, 65535):
                if p % mult == 0 and INT32_MIN <= p // mult <= INT32_MAX:
                    cases.append((p // mult, mult, s))
    return cases


def edge_cases() -> list[tuple[int, int, int]]:
    cases = []
    # s = 0: no rounding, plain clamp. Values around both clamp edges of both ranges.
    for acc in (-130, -129, -128, -127, -1, 0, 1, 126, 127, 128, 254, 255, 256, 257, INT32_MIN, INT32_MAX):
        cases.append((acc, 1, 0))
    # Clamp edges reached through the shift: y exactly at / one past each edge.
    for y in (-129, -128, -127, -1, 0, 1, 127, 128, 255, 256):
        cases.append((y * 256, 1, 8))
        cases.append((y * 256 - 128, 1, 8))      # tie just below: rounds up to y
        cases.append((y * 256 - 129, 1, 8))      # just below the tie: rounds to y - 1
    # Extremes of every operand.
    for acc in (INT32_MIN, INT32_MIN + 1, -1, 0, 1, INT32_MAX):
        for mult in (0, 1, 2, 32768, 65535):
            for s in (0, 1, 7, 8, 15, 16, 31, 32, 39, 40, 46, 47, 48, 49, 62, 63):
                cases.append((acc, mult, s))
    return cases + tie_cases()


@cocotb.test()
async def latency_and_registered_output(dut):
    """One value: out_valid exactly LATENCY cycles later, never earlier, and only for one cycle."""
    b = await setup(dut)
    await b.idle(LATENCY + 1)
    await b.send(-1000, 300, 10, False, tag=0x5A)
    for i in range(LATENCY + 3):
        await FallingEdge(dut.clk)
        b.idle_inputs()                          # new inputs mid-cycle must not reach the outputs
        valid = int(dut.out_valid.value)
        assert valid == (i == LATENCY - 1), f"cycle {i + 1}: out_valid={valid}"
        if valid:
            assert int(dut.out_tag.value) == 0x5A
            assert int(dut.out_data.value) == expect(-1000, 300, 10, False)
    b.sent.clear()
    b.got.clear()


@cocotb.test()
async def directed_edge_cases(dut):
    b = await setup(dut)
    for i, (acc, mult, s) in enumerate(edge_cases()):
        for relu in (False, True):
            await b.send(acc, mult, s, relu, tag=i & 0xFF)
    await b.drain_and_check()


@cocotb.test()
async def ties_round_half_up(dut):
    """Check the expected values of the tie cases explicitly, not only against the model."""
    b = await setup(dut)
    for acc, mult, s in tie_cases():
        k2 = (acc * mult) >> (s - 1)            # = 2k + 1
        assert k2 % 2 == 1
        want = max(-128, min(127, (k2 + 1) // 2))
        assert expect(acc, mult, s, False) == want & 0xFF
        await b.send(acc, mult, s, False)
    await b.drain_and_check()


def rand_value(rng: random.Random) -> tuple[int, int, int]:
    """Random operands, mostly with a shift that lands y near the 8-bit range."""
    acc = rng.choice([
        rng.randint(INT32_MIN, INT32_MAX),
        rng.randint(-(1 << 20), 1 << 20),
        rng.randint(-512, 512),
    ])
    mult = rng.choice([rng.randint(0, 65535), rng.randint(1, 255), 1 << rng.randint(0, 15)])
    mag = abs(acc * mult).bit_length()
    s = rng.choice([rng.randint(0, 63), max(0, mag - rng.randint(5, 10))])
    return acc, mult, s


@cocotb.test()
async def random_back_to_back(dut):
    b = await setup(dut)
    rng = random.Random(1)
    for i in range(4000):
        acc, mult, s = rand_value(rng)
        await b.send(acc, mult, s, rng.random() < 0.5, tag=i & 0xFF)
    await b.drain_and_check()


@cocotb.test()
async def requant_shaped_stream(dut):
    """M rows x N columns, per-column M/s, tag = column, with random idle gaps."""
    b = await setup(dut)
    rng = random.Random(2)
    for relu in (True, False):
        mults = [rng.randint(1, 65535) for _ in range(N)]
        shifts = [rng.randint(16, 30) for _ in range(N)]
        acc = [[rng.randint(-(1 << 22), 1 << 22) for _ in range(N)] for _ in range(20)]
        for row in acc:
            for col in range(N):
                if rng.random() < 0.2:
                    await b.idle(rng.randint(1, 3))
                await b.send(row[col], mults[col], shifts[col], relu, tag=col)
        await b.drain_and_check()


@cocotb.test()
async def reset_drops_values_in_flight(dut):
    b = await setup(dut)
    for i in range(LATENCY - 2):                # the oldest is one stage from the output
        await b.send(i * 1000, 1, 0, True, tag=i)
    await FallingEdge(dut.clk)
    b.idle_inputs()
    dut.rst.value = 1
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await b.idle(LATENCY + 2)
    assert b.got == [], b.got
    b.sent.clear()
    # And the pipeline works normally afterwards.
    await b.send(300, 1, 1, True, tag=7)
    await b.drain_and_check()


def test_requant():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"requant_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "requant.sv"],
        hdl_toplevel="requant",
        parameters={"TAG_W": 8},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="requant", test_module="test_requant", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
