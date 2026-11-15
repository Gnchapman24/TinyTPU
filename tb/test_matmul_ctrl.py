"""cocotb tests for rtl/matmul_ctrl.sv (task A4.2): MATMUL on the real datapath vs tinytpu.model.

tb/hdl/matmul_tb.sv wires matmul_ctrl to WBUF / ABUF / PTAB (buffer_ram), systolic_array and
acc_unit. Each test fills the memories with random contents, mirrored in a
tinytpu.model.Machine with the same depths, runs MATMUL instructions on both, and compares the
whole accumulator buffer bit-exactly. acc_unit's own assertion checks that no ACC row is updated
twice within 3 cycles (the binding case is N = 2, M = 1).
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
from tinytpu.isa import AccMode, Mem, SeqState

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 64, Mem.ABUF: 64, Mem.ACC: 32, Mem.PTAB: 64}
MODES = [AccMode.OVERWRITE, AccMode.ACCUMULATE, AccMode.BIAS]


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
        for s in (d.start, d.abort_req, d.w_wr_en, d.a_wr_en, d.p_wr_en, d.c_wr_en, d.c_rd_en):
            s.value = 0

    async def write(self, en, addr_sig, data_sig, addr: int, data: int) -> None:
        await FallingEdge(self.dut.clk)
        self.idle_inputs()
        en.value, addr_sig.value, data_sig.value = 1, addr, data

    async def randomize_and_load(self) -> None:
        """Random contents in every memory, written to the hardware and to the model."""
        d, r, mach = self.dut, self.rng, self.mach
        mach.wbuf[:] = np.array([[[r.randint(-127, 127) for _ in range(N)] for _ in range(N)]
                                 for _ in range(DEPTHS[Mem.WBUF])], dtype=np.int8)
        mach.abuf[:] = np.array([[r.randrange(256) for _ in range(N)] for _ in range(DEPTHS[Mem.ABUF])], dtype=np.uint8)
        mach.ptab_bias[:] = [r.randint(-(1 << 24), 1 << 24) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_mult[:] = [r.randrange(1 << 16) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_shift[:] = [r.randrange(64) for _ in range(DEPTHS[Mem.PTAB])]
        d.host_sel.value = 1
        for t in range(DEPTHS[Mem.WBUF]):
            for i in range(N):
                await self.write(d.w_wr_en, d.w_wr_addr, d.w_wr_data, t * N + i, row_word(mach.wbuf[t][i]))
        for a in range(DEPTHS[Mem.ABUF]):
            await self.write(d.a_wr_en, d.a_wr_addr, d.a_wr_data, a, row_word(mach.abuf[a]))
        for p in range(DEPTHS[Mem.PTAB]):
            entry = (int(mach.ptab_shift[p]) << 48) | (int(mach.ptab_mult[p]) << 32) | (int(mach.ptab_bias[p]) & 0xFFFFFFFF)
            await self.write(d.p_wr_en, d.p_wr_addr, d.p_wr_data, p, entry)
        await self.randomize_acc()

    async def randomize_acc(self) -> None:
        d, r = self.dut, self.rng
        self.mach.acc[:] = np.array([[r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(N)]
                                     for _ in range(DEPTHS[Mem.ACC])], dtype=np.int32)
        for c in range(DEPTHS[Mem.ACC]):
            await self.write(d.c_wr_en, d.c_wr_addr, d.c_wr_data, c, acc_word(self.mach.acc[c]))
        await FallingEdge(d.clk)
        self.idle_inputs()

    async def read_acc(self) -> np.ndarray:
        d = self.dut
        rows = []
        for c in range(DEPTHS[Mem.ACC]):
            await FallingEdge(d.clk)
            self.idle_inputs()
            d.c_rd_en.value, d.c_rd_addr.value = 1, c
            await FallingEdge(d.clk)
            d.c_rd_en.value = 0
            word = int(d.c_rd_data.value)
            rows.append([int(model.wrap32((word >> (32 * j)) & 0xFFFFFFFF)) for j in range(N)])
        return np.array(rows, dtype=np.int64)

    async def run(self, word: int, timeout: int = 200000) -> int:
        """Execute one MATMUL on the hardware and the model; compare ACC; return the cycle count."""
        d = self.dut
        assert isa.decode_error(word, DEPTHS) is isa.SeqErr.NONE
        assert not model.range_error(word, N, DEPTHS)
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.host_sel.value = 0
        d.instr.value = word
        d.start.value = 1
        await FallingEdge(d.clk)
        d.start.value = 0
        for cycles in range(1, timeout):
            if d.done.value == 1:
                break
            await FallingEdge(d.clk)
        else:
            raise AssertionError("MATMUL did not finish")
        await FallingEdge(d.clk)
        assert d.idle.value == 1 and d.acc_busy.value == 0
        d.host_sel.value = 1

        self.mach.load_program([word, isa.halt()])
        st = self.mach.run()
        assert st.state == SeqState.HALTED_OK, st
        got = await self.read_acc()
        diff = np.argwhere(got != self.mach.acc.astype(np.int64))
        assert diff.size == 0, f"{isa.Fields.decode(word)}: ACC differs at (row, col) {diff[:4].tolist()}"
        return cycles

    def random_matmul(self, mode: AccMode | None = None, signed: bool | None = None,
                      kt: int | None = None, nt: int | None = None, m: int | None = None) -> int:
        r = self.rng
        while True:
            word = isa.matmul(
                w_base=r.randrange(DEPTHS[Mem.WBUF]), a_base=r.randrange(DEPTHS[Mem.ABUF]),
                acc_base=r.randrange(DEPTHS[Mem.ACC]), p_base=r.randrange(DEPTHS[Mem.PTAB]),
                kt=kt or r.randint(1, 4), nt=nt or r.randint(1, 3), m=m or r.randint(1, 5),
                act_signed=r.random() < 0.5 if signed is None else signed,
                acc_mode=mode if mode is not None else r.choice(MODES))
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
    await b.randomize_and_load()
    return b


@cocotb.test()
async def each_mode_both_activation_types(dut):
    b = await setup(dut, 1)
    for mode in MODES:
        for signed in (False, True):
            await b.run(b.random_matmul(mode, signed, kt=2, nt=2, m=3))


@cocotb.test()
async def single_row_tightest_spacing(dut):
    """M = 1: tiles back to back at T = max(3, N); the same ACC row every tile of a column."""
    b = await setup(dut, 2)
    for mode in MODES:
        cycles = await b.run(b.random_matmul(mode, kt=4, nt=2, m=1))
        dut._log.info(f"M=1 kt=4 nt=2 {mode.name}: {cycles} cycles")


@cocotb.test()
async def split_k_overwrite_then_accumulate(dut):
    b = await setup(dut, 3)
    first = isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=2, nt=2, m=3, act_signed=True, acc_mode=AccMode.BIAS)
    second = isa.matmul(w_base=4, a_base=6, acc_base=0, p_base=0, kt=2, nt=2, m=3, act_signed=True,
                        acc_mode=AccMode.ACCUMULATE)
    await b.run(first)
    await b.run(second)


@cocotb.test()
async def random_instructions(dut):
    b = await setup(dut, 4)
    for i in range(40):
        await b.run(b.random_matmul())
        if i % 10 == 9:
            await b.randomize_acc()


@cocotb.test()
async def extreme_values(dut):
    """All weights -127/127 and all activations -128/127/255 into one ACC row per column tile."""
    b = await setup(dut, 5)
    d, mach = dut, b.mach
    for w_val, a_val, signed in ((127, 127, True), (-127, -128, True), (-127, 255, False), (127, 255, False)):
        mach.wbuf[:] = w_val
        mach.abuf[:] = a_val & 0xFF
        d.host_sel.value = 1
        for t in range(DEPTHS[Mem.WBUF]):
            for i in range(N):
                await b.write(d.w_wr_en, d.w_wr_addr, d.w_wr_data, t * N + i, row_word(mach.wbuf[t][i]))
        for a in range(DEPTHS[Mem.ABUF]):
            await b.write(d.a_wr_en, d.a_wr_addr, d.a_wr_data, a, row_word(mach.abuf[a]))
        await b.run(isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=8, nt=2, m=4, act_signed=signed,
                               acc_mode=AccMode.BIAS))


@cocotb.test()
async def abort_then_run_again(dut):
    b = await setup(dut, 6)
    d = dut
    for wait in (0, 1, 3, 7, 15, 40, 90):
        word = b.random_matmul(kt=4, nt=3, m=5)
        await FallingEdge(d.clk)
        b.idle_inputs()
        d.host_sel.value = 0
        d.instr.value = word
        d.start.value = 1
        await FallingEdge(d.clk)
        d.start.value = 0
        for _ in range(wait):
            await FallingEdge(d.clk)
        d.abort_req.value = 1
        for cycles in range(300):
            await FallingEdge(d.clk)
            assert d.done.value == 0 or cycles == 0 and wait > 0, "done after abort"
            if d.idle.value == 1 and d.acc_busy.value == 0:
                break
        else:
            raise AssertionError("abort did not reach idle")
        d.abort_req.value = 0
        d.host_sel.value = 1
        await b.randomize_acc()                  # ACC holds a partly executed MATMUL
        await b.run(b.random_matmul())


def test_matmul_ctrl():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"matmul_ctrl_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / f for f in ("isa_pkg.sv", "sync_fifo.sv", "buffer_ram.sv", "pe.sv",
                                            "systolic_array.sv", "acc_unit.sv", "matmul_ctrl.sv")]
        + [ROOT / "tb" / "hdl" / "matmul_tb.sv"],
        hdl_toplevel="matmul_tb",
        parameters={"N": N, "WBUF_DEPTH": DEPTHS[Mem.WBUF], "ABUF_DEPTH": DEPTHS[Mem.ABUF],
                    "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="matmul_tb", test_module="test_matmul_ctrl", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
