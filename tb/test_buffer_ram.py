"""cocotb tests for rtl/buffer_ram.sv (task A3.6).

The same tests run on several configurations shaped like the real buffers at N = TINYTPU_N
(small depths keep the simulation fast):

- ``wbuf``: one weight-tile row per entry, RD_PIPE = 2, tag = {w_load, w_row}
- ``abuf``: one activation vector per entry, byte lanes (LANES = N), RD_PIPE = 1
- ``ptab``: 54-bit parameter entries, RD_PIPE = 1
- ``bare``: no read pipeline (RD_PIPE = 0), two lanes

A Python model mirrors every write (per lane). Every read checks data, tag and exact latency.
The side that host_sel does not select is driven with random garbage (enables low) throughout,
so a wrong mux selection shows up as wrong data.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))

CONFIGS = {
    "wbuf": {"WIDTH": 8 * N, "DEPTH": 16 * N, "LANES": 1, "RD_PIPE": 2, "TAG_W": 1 + max(1, (N - 1).bit_length())},
    "abuf": {"WIDTH": 8 * N, "DEPTH": 64, "LANES": N, "RD_PIPE": 1, "TAG_W": 1},
    "ptab": {"WIDTH": 54, "DEPTH": 32, "LANES": 1, "RD_PIPE": 1, "TAG_W": 4},
    "bare": {"WIDTH": 16, "DEPTH": 16, "LANES": 2, "RD_PIPE": 0, "TAG_W": 3},
}

# Set by test_buffer_ram() for the simulation run.
CFG = CONFIGS[os.environ.get("BUF_CFG", "abuf")]
W, DEPTH, LANES, RD_PIPE, TAG_W = (CFG[k] for k in ("WIDTH", "DEPTH", "LANES", "RD_PIPE", "TAG_W"))
LW = W // LANES
LAT = 1 + RD_PIPE


class Bench:
    def __init__(self, dut, rng: random.Random):
        self.dut = dut
        self.rng = rng
        self.cycle = 0
        self.rows: list[int] = [0] * DEPTH
        self.pending: list[tuple[int, int, int]] = []   # (issue cycle, expected data, expected tag)
        self.reads_checked = 0
        cocotb.start_soon(self._monitor())

    async def _monitor(self) -> None:
        while True:
            await RisingEdge(self.dut.clk)
            self.cycle += 1
            await FallingEdge(self.dut.clk)
            if self.dut.rd_valid.value == 0:
                assert int(self.dut.rd_tag.value) == 0, f"cycle {self.cycle}: rd_tag without a datapath read"
            else:
                assert self.pending, f"cycle {self.cycle}: rd_valid without a read"
                c_in, want, tag = self.pending.pop(0)
                assert self.cycle - c_in == LAT, f"latency {self.cycle - c_in}, want {LAT}"
                got = int(self.dut.rd_data.value)
                assert got == want, f"read issued in cycle {c_in}: got {got:#x}, want {want:#x}"
                assert int(self.dut.rd_tag.value) == tag, "tag (0 for host reads)"
                self.reads_checked += 1

    def garbage(self) -> None:
        """Idle both sides; random values on every non-enable input."""
        d, r = self.dut, self.rng
        d.dp_rd_en.value = d.dp_wr_en.value = d.host_rd_en.value = d.host_wr_en.value = 0
        d.dp_rd_addr.value = r.randrange(DEPTH)
        d.dp_rd_tag.value = r.getrandbits(TAG_W)
        d.dp_wr_addr.value = r.randrange(DEPTH)
        d.dp_wr_data.value = r.getrandbits(W)
        d.dp_wr_be.value = r.getrandbits(LANES)
        d.host_rd_addr.value = r.randrange(DEPTH)
        d.host_wr_addr.value = r.randrange(DEPTH)
        d.host_wr_data.value = r.getrandbits(W)

    async def cycle_(self, host: bool, rd: int | None = None, tag: int = 0,
                     wr: tuple[int, int, int] | None = None) -> None:
        """One clock cycle: optionally a read of row `rd` and a write `wr` = (row, data, lane mask)."""
        d = self.dut
        await FallingEdge(d.clk)
        self.garbage()
        d.host_sel.value = int(host)
        if rd is not None:
            if host:
                d.host_rd_en.value, d.host_rd_addr.value = 1, rd
            else:
                d.dp_rd_en.value, d.dp_rd_addr.value, d.dp_rd_tag.value = 1, rd, tag
            # The cycle that ends at the next edge is self.cycle (the monitor counts edges).
            self.pending.append((self.cycle, self.rows[rd], 0 if host else tag))
        if wr is not None:
            addr, data, be = wr
            if host:
                d.host_wr_en.value, d.host_wr_addr.value, d.host_wr_data.value = 1, addr, data
                be = (1 << LANES) - 1
            else:
                d.dp_wr_en.value, d.dp_wr_addr.value, d.dp_wr_data.value, d.dp_wr_be.value = 1, addr, data, be
            for lane in range(LANES):
                if (be >> lane) & 1:
                    mask = ((1 << LW) - 1) << (lane * LW)
                    self.rows[addr] = (self.rows[addr] & ~mask) | (data & mask)

    async def idle(self, cycles: int = 1, host: bool = False) -> None:
        for _ in range(cycles):
            await self.cycle_(host)

    async def drain(self) -> None:
        await self.idle(LAT + 2)
        assert not self.pending, f"{len(self.pending)} reads never returned"

    async def host_fill(self) -> None:
        for addr in range(DEPTH):
            await self.cycle_(True, wr=(addr, self.rng.getrandbits(W), 0))


async def setup(dut, seed: int) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut, random.Random(seed))
    dut.rst.value = 1
    dut.host_sel.value = 1
    b.garbage()
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return b


@cocotb.test()
async def host_fill_and_read_back(dut):
    b = await setup(dut, 1)
    await b.host_fill()
    for addr in range(DEPTH):
        await b.cycle_(True, rd=addr)
    await b.drain()
    assert b.reads_checked == DEPTH


@cocotb.test()
async def latency_and_registered_output(dut):
    """One read: rd_valid exactly LAT cycles later, for one cycle; outputs never change mid-cycle."""
    b = await setup(dut, 2)
    await b.host_fill()
    await b.idle(LAT + 1)
    await b.cycle_(False, rd=3, tag=1)
    seen = []
    for _ in range(LAT + 3):
        await FallingEdge(dut.clk)
        before = (int(dut.rd_valid.value), int(dut.rd_data.value), int(dut.rd_tag.value))
        b.garbage()
        dut.dp_rd_en.value, dut.dp_rd_addr.value = 1, (3 + 1) % DEPTH   # a new read, mid-cycle
        await cocotb.triggers.Timer(1, unit="ns")
        after = (int(dut.rd_valid.value), int(dut.rd_data.value), int(dut.rd_tag.value))
        assert before == after, "an input reached the outputs without a clock edge"
        dut.dp_rd_en.value = 0
        seen.append(before[0])
    assert seen[:LAT + 1] == [0] * (LAT - 1) + [1, 0], seen
    b.pending.clear()


@cocotb.test()
async def datapath_stream(dut):
    """Back-to-back datapath reads (with tags) and lane writes, random gaps, every row."""
    b = await setup(dut, 3)
    await b.host_fill()
    r = b.rng
    for _ in range(40 * DEPTH):
        rd = r.randrange(DEPTH) if r.random() < 0.8 else None
        wr = None
        if r.random() < 0.5:
            addr = r.choice([a for a in range(DEPTH) if a != rd])
            wr = (addr, r.getrandbits(W), r.getrandbits(LANES))
        await b.cycle_(False, rd=rd, tag=r.getrandbits(TAG_W), wr=wr)
    await b.drain()


@cocotb.test()
async def each_lane_alone(dut):
    b = await setup(dut, 4)
    await b.host_fill()
    for lane in range(LANES):
        addr = lane % DEPTH
        await b.cycle_(False, wr=(addr, (1 << W) - 1 if lane % 2 else 0, 1 << lane))
        await b.cycle_(False, rd=addr)
    await b.cycle_(False, wr=(5, b.rng.getrandbits(W), 0))      # no lanes: no change
    await b.cycle_(False, rd=5)
    await b.drain()


@cocotb.test()
async def switching_between_host_and_datapath(dut):
    """host_sel may change between any two cycles, even with reads in flight."""
    b = await setup(dut, 5)
    await b.host_fill()
    r = b.rng
    for _ in range(20 * DEPTH):
        host = r.random() < 0.5
        rd = r.randrange(DEPTH) if r.random() < 0.7 else None
        wr = None
        if r.random() < 0.5:
            addr = r.choice([a for a in range(DEPTH) if a != rd])
            wr = (addr, r.getrandbits(W), r.getrandbits(LANES))
        await b.cycle_(host, rd=rd, tag=r.getrandbits(TAG_W), wr=wr)
    await b.drain()


@cocotb.test()
async def reset_drops_reads_in_flight(dut):
    b = await setup(dut, 6)
    await b.host_fill()
    # LAT reads, the last one in the same cycle as reset: none of them may come out.
    for addr in range(LAT):
        await b.cycle_(False, rd=addr)
    dut.rst.value = 1
    b.pending.clear()
    await FallingEdge(dut.clk)
    b.garbage()
    dut.rst.value = 0
    await b.idle(LAT + 2)
    assert b.reads_checked == 0
    await b.cycle_(False, rd=7)                 # and reads work again afterwards
    await b.drain()


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_buffer_ram(cfg):
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"buffer_ram_{cfg}_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "buffer_ram.sv"],
        hdl_toplevel="buffer_ram",
        parameters=CONFIGS[cfg],
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="buffer_ram", test_module="test_buffer_ram", test_dir=Path(__file__).parent,
                extra_env={"BUF_CFG": cfg}, results_xml=str(build_dir / "results.xml"))
