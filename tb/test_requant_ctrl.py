"""cocotb tests for rtl/requant_ctrl.sv (task A4.2): REQUANT on the real memories vs tinytpu.model.

tb/hdl/requant_tb.sv wires requant_ctrl to acc_unit (ACC), PTAB and ABUF (buffer_ram) and the
requantizer. Each test fills ACC, PTAB and ABUF with random contents, mirrored in a
tinytpu.model.Machine with the same depths, runs REQUANT instructions on both, and compares
the whole activation buffer bit-exactly (so bytes outside the destination must be untouched).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb_tools.runner import get_runner

from tinytpu import isa, model
from tinytpu.isa import Mem, SeqState

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 2, Mem.ABUF: 64, Mem.ACC: 32, Mem.PTAB: 64}


def row_word(row) -> int:
    return int.from_bytes(np.asarray(row).astype(np.uint8).tobytes(), "little")


def acc_word(row) -> int:
    return sum((int(v) & 0xFFFFFFFF) << (32 * j) for j, v in enumerate(row))


class Bench:
    def __init__(self, dut, seed: int):
        self.dut = dut
        self.rng = random.Random(seed)
        self.mach = model.Machine(n=N, depths=dict(DEPTHS))

    def idle_inputs(self) -> None:
        d = self.dut
        for s in (d.start, d.abort_req, d.c_wr_en, d.p_wr_en, d.a_wr_en, d.a_rd_en):
            s.value = 0

    async def write(self, en, addr_sig, data_sig, addr: int, data: int) -> None:
        await FallingEdge(self.dut.clk)
        self.idle_inputs()
        en.value, addr_sig.value, data_sig.value = 1, addr, data

    async def load(self, acc_scale: int = 1 << 20) -> None:
        d, r, mach = self.dut, self.rng, self.mach
        mach.acc[:] = np.array([[r.choice([r.randint(-acc_scale, acc_scale), r.randint(-(1 << 31), (1 << 31) - 1)])
                                 for _ in range(N)] for _ in range(DEPTHS[Mem.ACC])], dtype=np.int32)
        mach.abuf[:] = np.array([[r.randrange(256) for _ in range(N)] for _ in range(DEPTHS[Mem.ABUF])], dtype=np.uint8)
        mach.ptab_bias[:] = [r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_mult[:] = [r.choice([r.randrange(1 << 16), r.randrange(1, 256)]) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_shift[:] = [r.choice([r.randrange(64), r.randint(8, 24)]) for _ in range(DEPTHS[Mem.PTAB])]
        d.host_sel.value = 1
        for c in range(DEPTHS[Mem.ACC]):
            await self.write(d.c_wr_en, d.c_wr_addr, d.c_wr_data, c, acc_word(mach.acc[c]))
        for p in range(DEPTHS[Mem.PTAB]):
            entry = (int(mach.ptab_shift[p]) << 48) | (int(mach.ptab_mult[p]) << 32) | (int(mach.ptab_bias[p]) & 0xFFFFFFFF)
            await self.write(d.p_wr_en, d.p_wr_addr, d.p_wr_data, p, entry)
        for a in range(DEPTHS[Mem.ABUF]):
            await self.write(d.a_wr_en, d.a_wr_addr, d.a_wr_data, a, row_word(mach.abuf[a]))
        await FallingEdge(d.clk)
        self.idle_inputs()

    async def read_abuf(self) -> np.ndarray:
        d = self.dut
        rows = []
        for a in range(DEPTHS[Mem.ABUF]):
            await FallingEdge(d.clk)
            self.idle_inputs()
            d.a_rd_en.value, d.a_rd_addr.value = 1, a
            await FallingEdge(d.clk)
            d.a_rd_en.value = 0
            await FallingEdge(d.clk)                 # ABUF read latency 2
            word = int(d.a_rd_data.value)
            rows.append([(word >> (8 * j)) & 0xFF for j in range(N)])
        return np.array(rows, dtype=np.uint8)

    async def start(self, word: int) -> None:
        d = self.dut
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.host_sel.value = 0
        d.instr.value = word
        d.start.value = 1
        await FallingEdge(d.clk)
        d.start.value = 0

    async def run(self, word: int) -> int:
        d = self.dut
        assert isa.decode_error(word, DEPTHS) is isa.SeqErr.NONE and not model.range_error(word, N, DEPTHS)
        await self.start(word)
        for cycles in range(1, 100000):
            if d.done.value == 1:
                break
            await FallingEdge(d.clk)
        else:
            raise AssertionError("REQUANT did not finish")
        await FallingEdge(d.clk)
        assert d.idle.value == 1
        d.host_sel.value = 1
        self.mach.load_program([word, isa.halt()])
        assert self.mach.run().state == SeqState.HALTED_OK
        got = await self.read_abuf()
        diff = np.argwhere(got != self.mach.abuf)
        assert diff.size == 0, f"{isa.Fields.decode(word)}: ABUF differs at (vector, byte) {diff[:4].tolist()}"
        return cycles

    def random_requant(self, nt: int | None = None, m: int | None = None, relu: bool | None = None) -> int:
        r = self.rng
        while True:
            word = isa.requant(a_dst=r.randrange(DEPTHS[Mem.ABUF]), acc_base=r.randrange(DEPTHS[Mem.ACC]),
                               p_base=r.randrange(DEPTHS[Mem.PTAB]), nt=nt or r.randint(1, 3), m=m or r.randint(1, 4),
                               relu=r.random() < 0.5 if relu is None else relu)
            if not model.range_error(word, N, DEPTHS):
                return word


async def setup(dut, seed: int) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut, seed)
    dut.rst.value = 1
    dut.host_sel.value = 1
    dut.instr.value = 0
    b.idle_inputs()
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await b.load()
    return b


@cocotb.test()
async def relu_and_int8(dut):
    b = await setup(dut, 1)
    for relu in (True, False):
        cycles = await b.run(b.random_requant(nt=2, m=3, relu=relu))
        assert cycles <= 2 * 3 * N + 12, cycles               # one value per cycle, then the pipeline
        dut._log.info(f"nt=2 m=3: {cycles} cycles")


@cocotb.test()
async def single_value_rows(dut):
    b = await setup(dut, 2)
    for _ in range(6):
        await b.run(b.random_requant(nt=1, m=1))


@cocotb.test()
async def chained_layer_layout(dut):
    """The docs/isa.md pattern: rows (m, n) at acc_base + m*nt + n -> a_dst + m*nt + n."""
    b = await setup(dut, 3)
    await b.run(isa.requant(a_dst=8, acc_base=0, p_base=0, nt=4, m=2, relu=True))
    await b.run(isa.requant(a_dst=40, acc_base=8, p_base=4 * N, nt=2, m=3, relu=False))


@cocotb.test()
async def random_instructions(dut):
    b = await setup(dut, 4)
    for i in range(40):
        await b.run(b.random_requant())
        if i % 10 == 9:
            await b.load()


@cocotb.test()
async def abort_then_run_again(dut):
    b = await setup(dut, 5)
    d = dut
    for wait in (0, 1, 2, 5, 9, 20):
        await b.start(b.random_requant(nt=3, m=4))
        for _ in range(wait):
            await FallingEdge(d.clk)
        d.abort_req.value = 1
        for _ in range(40):
            await FallingEdge(d.clk)
            assert d.done.value == 0, "done after abort"
            if d.idle.value == 1:
                break
        else:
            raise AssertionError("abort did not reach idle")
        d.abort_req.value = 0
        for _ in range(12):                      # idle means nothing issued is still in flight
            await FallingEdge(d.clk)
            assert d.u_ctrl.a_wr_en.value == 0, "ABUF write after the unit reported idle"
        d.host_sel.value = 1
        await b.load()                           # ABUF holds a partly executed REQUANT
        await b.run(b.random_requant())


def test_requant_ctrl():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"requant_ctrl_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / f for f in ("isa_pkg.sv", "buffer_ram.sv", "acc_unit.sv", "requant.sv", "requant_ctrl.sv")]
        + [ROOT / "tb" / "hdl" / "requant_tb.sv"],
        hdl_toplevel="requant_tb",
        parameters={"N": N, "ABUF_DEPTH": DEPTHS[Mem.ABUF], "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="requant_tb", test_module="test_requant_ctrl", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
