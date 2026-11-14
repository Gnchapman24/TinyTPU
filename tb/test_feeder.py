"""cocotb tests for the feeder path (task A3.6): WBUF and ABUF read paths driving the systolic array.

tb/hdl/feeder_tb.sv connects two buffer_ram instances to systolic_array with nothing in between:
the array's w_load / w_row / a_valid come only from the buffers' read tags. These tests act as
the address generator (A4.2): they load the canonical memory images (tinytpu.model.pack_*)
through the host ports, then issue datapath reads in the MATMUL loop order (n, k, m) with the
array's timing rules shifted by the buffers' read latencies:

- the N rows of a tile are read on consecutive cycles;
- WBUF's latency is longer than ABUF's, so vector reads start W_LAT - A_LAT cycles after the
  tile's first row read: vector 0 then enters the array with row 0;
- the next tile's first row read is at least N cycles after the previous tile's last vector
  entered the array.

Each result vector is checked against tinytpu.model.matmul for its (n, k, m).
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
WROWS, AROWS = 32 * N, 64
W_PIPE, A_PIPE = 2, 1
W_LAT, A_LAT = 1 + W_PIPE, 1 + A_PIPE
ARRAY_LAT = 2 * N + 1


def row_word(row: np.ndarray) -> int:
    """Byte j of the word = element j (memory map)."""
    return int.from_bytes(np.asarray(row).astype(np.uint8).tobytes(), "little")


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.cycle = 0
        self.results: list[tuple[int, list[int]]] = []
        cocotb.start_soon(self._monitor())

    async def _monitor(self) -> None:
        while True:
            await RisingEdge(self.dut.clk)
            self.cycle += 1
            await FallingEdge(self.dut.clk)
            if self.dut.out_valid.value == 1:
                word = int(self.dut.out_data.value)
                vals = [int(model.wrap32((word >> (32 * j)) & 0xFFFFFFFF)) for j in range(N)]
                self.results.append((self.cycle, vals))

    def idle_inputs(self) -> None:
        d = self.dut
        d.w_host_wr_en.value = d.a_host_wr_en.value = 0
        d.w_rd_en.value = d.a_rd_en.value = 0

    async def load(self, tiles: np.ndarray, vecs: np.ndarray) -> None:
        d = self.dut
        d.host_sel.value = 1
        for t, tile in enumerate(tiles):
            for i in range(N):
                await FallingEdge(d.clk)
                self.idle_inputs()
                d.w_host_wr_en.value, d.w_host_wr_addr.value = 1, t * N + i
                d.w_host_wr_data.value = row_word(tile[i])
        for a, vec in enumerate(vecs):
            await FallingEdge(d.clk)
            self.idle_inputs()
            d.a_host_wr_en.value, d.a_host_wr_addr.value, d.a_host_wr_data.value = 1, a, row_word(vec)
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.host_sel.value = 0

    async def run_schedule(self, sched: dict[int, dict]) -> None:
        """Drive datapath reads: sched[cycle offset] = {"w": (addr, row)} and/or {"a": addr}."""
        d = self.dut
        for c in range(max(sched) + 1):
            await FallingEdge(d.clk)
            self.idle_inputs()
            ops = sched.get(c, {})
            if "w" in ops:
                d.w_rd_en.value, d.w_rd_addr.value, d.w_rd_row.value = 1, *ops["w"]
            if "a" in ops:
                d.a_rd_en.value, d.a_rd_addr.value = 1, ops["a"]
        for _ in range(W_LAT + ARRAY_LAT + 2):
            await FallingEdge(d.clk)
            self.idle_inputs()


async def setup(dut) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut)
    dut.rst.value = 1
    dut.host_sel.value = 1
    dut.act_signed.value = 0
    b.idle_inputs()
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return b


def schedule(kt: int, nt: int, m_dim: int, rng: random.Random | None):
    """Read schedule in MATMUL loop order, at the tightest legal spacing (plus random gaps if rng)."""
    sched: dict[int, dict] = {}
    order: list[tuple[int, int, int]] = []
    t0 = 0
    for n in range(nt):
        for k in range(kt):
            tile = n * kt + k
            for i in range(N):
                sched.setdefault(t0 + i, {})["w"] = (tile * N + i, i)
            ta = t0 + (W_LAT - A_LAT)
            for m in range(m_dim):
                if rng and rng.random() < 0.2:
                    ta += rng.randint(1, 3)       # gaps between vectors are allowed
                sched.setdefault(ta, {})["a"] = m * kt + k
                order.append((n, k, m))
                ta += 1
            last_entry = ta - 1 + A_LAT           # array cycle of the tile's last vector
            t0 = max(t0 + N, last_entry + N - W_LAT)
            if rng and rng.random() < 0.3:
                t0 += rng.randint(1, 4)
    return sched, order


async def run_matmul(b: Bench, rng: random.Random, m_dim: int, k_dim: int, out_dim: int,
                     act_signed: bool, gaps: bool) -> None:
    w = np.array([[rng.randint(-127, 127) for _ in range(out_dim)] for _ in range(k_dim)])
    lo, hi = (-128, 127) if act_signed else (0, 255)
    x = np.array([[rng.randint(lo, hi) for _ in range(k_dim)] for _ in range(m_dim)])
    tiles = model.pack_weights(w, N)
    vecs = model.pack_activations(x, N)
    kt, nt = model.ceil_div(k_dim, N), model.ceil_div(out_dim, N)
    assert len(tiles) * N <= WROWS and len(vecs) <= AROWS

    b.dut.act_signed.value = int(act_signed)
    await b.load(tiles, vecs)
    sched, order = schedule(kt, nt, m_dim, rng if gaps else None)
    b.results.clear()
    await b.run_schedule(sched)

    assert len(b.results) == len(order), f"{len(b.results)} results, want {len(order)}"
    for (n, k, m), (_, got) in zip(order, b.results):
        tile = tiles[n * kt + k].astype(np.int64)
        want = model.matmul(vecs[m * kt + k][None, :], tile, act_signed)[0]
        assert got == [int(v) for v in want], f"n={n} k={k} m={m}: {got} != {list(want)}"
    # Summed over k, the partial results are the full matmul (padding contributes zeros).
    full = np.zeros((m_dim, nt * N), dtype=np.int64)
    for (n, k, m), (_, got) in zip(order, b.results):
        full[m, n * N:(n + 1) * N] += got
    assert (full[:, :out_dim] == model.matmul(x, w, act_signed)).all()


@cocotb.test()
async def single_vector_per_tile_tightest(dut):
    """M = 1: tiles back to back, every next tile at the minimum spacing."""
    b = await setup(dut)
    rng = random.Random(1)
    await run_matmul(b, rng, 1, 3 * N, 2 * N, act_signed=True, gaps=False)
    await run_matmul(b, rng, 1, 3 * N, 2 * N, act_signed=False, gaps=False)


@cocotb.test()
async def several_vectors_odd_sizes(dut):
    """Dimensions that are not multiples of N, both activation modes, minimum spacing."""
    b = await setup(dut)
    rng = random.Random(2)
    for signed in (True, False):
        await run_matmul(b, rng, 5, 2 * N + 1, N + 3, act_signed=signed, gaps=False)


@cocotb.test()
async def random_gaps(dut):
    b = await setup(dut)
    rng = random.Random(3)
    for i in range(4):
        await run_matmul(b, rng, rng.randint(1, 6), rng.randint(1, 3 * N), rng.randint(1, 3 * N),
                         act_signed=bool(i % 2), gaps=True)


@cocotb.test()
async def extreme_values(dut):
    """All -128 / 127 activations against all -127 / 127 weights (the largest products)."""
    b = await setup(dut)
    for act_signed, a_val in ((True, -128), (True, 127), (False, 255)):
        for w_val in (-127, 127):
            w = np.full((2 * N, N), w_val)
            x = np.full((2, 2 * N), a_val)
            tiles, vecs = model.pack_weights(w, N), model.pack_activations(x, N)
            b.dut.act_signed.value = int(act_signed)
            await b.load(tiles, vecs)
            sched, order = schedule(2, 1, 2, None)
            b.results.clear()
            await b.run_schedule(sched)
            full = np.zeros((2, N), dtype=np.int64)
            for (n, k, m), (_, got) in zip(order, b.results):
                full[m] += got
            assert (full == model.matmul(x, w, act_signed)).all(), (act_signed, a_val, w_val)


def test_feeder():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"feeder_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "pe.sv", ROOT / "rtl" / "systolic_array.sv", ROOT / "rtl" / "buffer_ram.sv",
                 ROOT / "tb" / "hdl" / "feeder_tb.sv"],
        hdl_toplevel="feeder_tb",
        parameters={"N": N, "WROWS": WROWS, "AROWS": AROWS, "W_PIPE": W_PIPE, "A_PIPE": A_PIPE},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="feeder_tb", test_module="test_feeder", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
