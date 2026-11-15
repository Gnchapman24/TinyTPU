"""End-to-end cocotb tests for rtl/tpu_core.sv (task A4.2): stored programs vs tinytpu.model.Machine.

The host port loads every memory with the same contents as a Machine of the same depths; a
program runs on both; then the status (state, error, PC, halt code) and the complete ACC and
ABUF contents must be equal, bit for bit. Because address-range errors are found before an
instruction starts, memories must match even when a program halts with an error.
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
from tinytpu.isa import AccMode, Mem, SeqErr, SeqState

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 256, Mem.ABUF: 64, Mem.ACC: 32, Mem.PTAB: 128}


def vec_word(row) -> int:
    return int.from_bytes(np.asarray(row).astype(np.uint8).tobytes(), "little")


def acc_word(row) -> int:
    return sum((int(v) & 0xFFFFFFFF) << (32 * j) for j, v in enumerate(row))


def ptab_word(mach: model.Machine, p: int) -> int:
    return (int(mach.ptab_shift[p]) << 48) | (int(mach.ptab_mult[p]) << 32) | (int(mach.ptab_bias[p]) & 0xFFFFFFFF)


class Bench:
    def __init__(self, dut, seed: int):
        self.dut = dut
        self.rng = random.Random(seed)
        self.mach = model.Machine(n=N, depths=dict(DEPTHS))

    def idle_inputs(self) -> None:
        d = self.dut
        d.start.value = d.abort_req.value = d.host_rd_en.value = d.host_wr_en.value = 0

    async def write(self, mem: Mem, addr: int, data: int) -> None:
        d = self.dut
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.host_wr_en.value, d.host_mem.value, d.host_addr.value, d.host_wr_data.value = 1, int(mem), addr, data

    async def read(self, mem: Mem, addr: int) -> int:
        d = self.dut
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.host_rd_en.value, d.host_mem.value, d.host_addr.value = 1, int(mem), addr
        await FallingEdge(d.clk)
        d.host_rd_en.value = 0
        for _ in range(6):
            if d.host_rd_valid.value == 1:
                return int(d.host_rd_data.value)
            await FallingEdge(d.clk)
        raise AssertionError(f"no read data from {mem.name}")

    # Loading ----------------------------------------------------------------------------------

    async def load_all(self) -> None:
        """Random contents everywhere (IMEM all HALT 0), on both sides."""
        r, mach = self.rng, self.mach
        mach.imem[:] = 0
        mach.wbuf[:] = np.array([[[r.randint(-127, 127) for _ in range(N)] for _ in range(N)]
                                 for _ in range(DEPTHS[Mem.WBUF])], dtype=np.int8)
        mach.abuf[:] = np.array([[r.randrange(256) for _ in range(N)] for _ in range(DEPTHS[Mem.ABUF])], dtype=np.uint8)
        mach.ptab_bias[:] = [r.randint(-(1 << 20), 1 << 20) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_mult[:] = [r.randrange(1, 1 << 16) for _ in range(DEPTHS[Mem.PTAB])]
        mach.ptab_shift[:] = [r.randint(10, 24) for _ in range(DEPTHS[Mem.PTAB])]
        mach.acc[:] = np.array([[r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(N)]
                                for _ in range(DEPTHS[Mem.ACC])], dtype=np.int32)
        for i in range(DEPTHS[Mem.IMEM]):
            await self.write(Mem.IMEM, i, 0)
        for t in range(DEPTHS[Mem.WBUF]):
            for i in range(N):
                await self.write(Mem.WBUF, t * N + i, vec_word(mach.wbuf[t][i]))
        await self.sync_abuf_acc()
        for p in range(DEPTHS[Mem.PTAB]):
            await self.write(Mem.PTAB, p, ptab_word(mach, p))

    async def sync_abuf_acc(self) -> None:
        for a in range(DEPTHS[Mem.ABUF]):
            await self.write(Mem.ABUF, a, vec_word(self.mach.abuf[a]))
        for c in range(DEPTHS[Mem.ACC]):
            await self.write(Mem.ACC, c, acc_word(self.mach.acc[c]))

    async def load_program(self, words: list[int]) -> None:
        words = list(words) + [0] * (DEPTHS[Mem.IMEM] - len(words))
        self.mach.imem[:] = 0
        self.mach.load_program(words)
        for i, w in enumerate(words):
            await self.write(Mem.IMEM, i, w)

    # Running and checking ---------------------------------------------------------------------

    def status(self) -> tuple[int, int, int, int]:
        d = self.dut
        return (int(d.state.value), int(d.err.value), int(d.pc.value), int(d.halt_code.value))

    async def run(self, start_pc: int = 0, timeout: int = 400000) -> int:
        d = self.dut
        await FallingEdge(d.clk)
        self.idle_inputs()
        d.start.value, d.start_pc.value = 1, start_pc
        await FallingEdge(d.clk)
        d.start.value = 0
        for cycles in range(timeout):
            if int(d.state.value) != SeqState.RUNNING:
                return cycles
            await FallingEdge(d.clk)
        raise AssertionError("program did not stop")

    async def check(self, start_pc: int = 0) -> model.Status:
        """Run the loaded program on both sides and compare status, ACC and ABUF."""
        cycles = await self.run(start_pc)
        st = self.mach.run(start_pc)
        want = (int(st.state), int(st.err), st.pc, st.halt_code)
        assert self.status() == want, f"status {self.status()}, model {want}"
        for c in range(DEPTHS[Mem.ACC]):
            word = await self.read(Mem.ACC, c)
            got = [int(model.wrap32((word >> (32 * j)) & 0xFFFFFFFF)) for j in range(N)]
            assert got == [int(v) for v in self.mach.acc[c]], f"ACC row {c}"
        for a in range(DEPTHS[Mem.ABUF]):
            word = await self.read(Mem.ABUF, a)
            assert word == vec_word(self.mach.abuf[a]), f"ABUF vector {a}"
        self.dut._log.debug(f"{cycles} cycles")
        return st


async def setup(dut, seed: int) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut, seed)
    dut.rst.value = 1
    dut.start_pc.value = 0
    dut.host_mem.value = 0
    dut.host_addr.value = 0
    dut.host_wr_data.value = 0
    b.idle_inputs()
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await b.load_all()
    return b


# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------

@cocotb.test()
async def host_port_reads_back_every_memory(dut):
    b = await setup(dut, 1)
    for i, w in enumerate([b.rng.getrandbits(128) for _ in range(DEPTHS[Mem.IMEM])]):
        await b.write(Mem.IMEM, i, w)
        assert await b.read(Mem.IMEM, i) == w
    for _ in range(20):
        t, i = b.rng.randrange(DEPTHS[Mem.WBUF]), b.rng.randrange(N)
        assert await b.read(Mem.WBUF, t * N + i) == vec_word(b.mach.wbuf[t][i])
        p = b.rng.randrange(DEPTHS[Mem.PTAB])
        assert await b.read(Mem.PTAB, p) == ptab_word(b.mach, p)


@cocotb.test()
async def two_layer_network(dut):
    """A small MLP as in docs/isa.md: MATMUL(bias) + REQUANT(relu), then MATMUL(bias) + REQUANT."""
    b = await setup(dut, 2)
    r, mach = b.rng, b.mach
    k1, h, o = 5 * N + 3, 2 * N + 1, N + 2                    # sizes that are not multiples of N
    kt1, nt1, nt2 = model.ceil_div(k1, N), model.ceil_div(h, N), model.ceil_div(o, N)
    w1 = np.array([[r.randint(-127, 127) for _ in range(h)] for _ in range(k1)])
    w2 = np.array([[r.randint(-127, 127) for _ in range(o)] for _ in range(h)])
    x = np.array([[r.randrange(256) for _ in range(k1)]])
    t1, t2 = model.pack_weights(w1, N), model.pack_weights(w2, N)
    mach.load_weights(t1, 0)
    mach.load_weights(t2, len(t1))
    mach.load_vectors(model.pack_activations(x, N), 0)
    pad = lambda v, n: np.concatenate([v, np.zeros(n - len(v), dtype=np.int64)])
    b1, b2 = pad(np.array([r.randint(-5000, 5000) for _ in range(h)]), nt1 * N), pad(np.array([r.randint(-5000, 5000) for _ in range(o)]), nt2 * N)
    m1, m2 = pad(np.array([r.randint(1000, 30000) for _ in range(h)]), nt1 * N), pad(np.array([r.randint(1000, 30000) for _ in range(o)]), nt2 * N)
    s1, s2 = pad(np.full(h, 22), nt1 * N), pad(np.full(o, 22), nt2 * N)
    mach.load_params(b1, m1, s1, 0)
    mach.load_params(b2, m2, s2, nt1 * N)
    for t in range(len(t1) + len(t2)):
        for i in range(N):
            await b.write(Mem.WBUF, t * N + i, vec_word(mach.wbuf[t][i]))
    for p in range((nt1 + nt2) * N):
        await b.write(Mem.PTAB, p, ptab_word(mach, p))
    await b.sync_abuf_acc()

    a_hidden, a_out = 32, 48
    await b.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=kt1, nt=nt1, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=a_hidden, acc_base=0, p_base=0, nt=nt1, m=1, relu=True),
        isa.matmul(w_base=len(t1), a_base=a_hidden, acc_base=8, p_base=nt1 * N, kt=nt1, nt=nt2, m=1, act_signed=False,
                   acc_mode=AccMode.BIAS),
        isa.requant(a_dst=a_out, acc_base=8, p_base=nt1 * N, nt=nt2, m=1, relu=False),
        isa.halt(1),
    ])
    st = await b.check()
    assert st.state == SeqState.HALTED_OK and st.pc == 4 and st.halt_code == 1
    # And the result is the network computed directly.
    hid = model.requant(model.matmul(x, w1, False, bias=b1[:h]), m1[:h], s1[:h], relu=True)
    out = model.requant(model.matmul(hid, w2, False, bias=b2[:o]), m2[:o], s2[:o], relu=False)
    got = model.unpack_vectors(mach.abuf[a_out:a_out + nt2], 1, o, signed=True)
    assert got.tolist() == out.tolist()


@cocotb.test()
async def docs_example_program(dut):
    """The exact docs/isa.md program (N = 8 only: its fields assume N = 8)."""
    if N != 8:
        return
    b = await setup(dut, 3)
    await b.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=25, nt=8, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=32, acc_base=0, p_base=0, nt=8, m=1, relu=True),
        isa.matmul(w_base=200, a_base=32, acc_base=8, p_base=64, kt=8, nt=2, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=40, acc_base=8, p_base=64, nt=2, m=1, relu=False),
        isa.halt(1),
    ])
    st = await b.check()
    assert st.state == SeqState.HALTED_OK and st.pc == 4


def random_instruction(r: random.Random) -> int:
    d = DEPTHS
    kind = r.random()
    if kind < 0.45:
        return isa.matmul(w_base=r.randrange(d[Mem.WBUF]), a_base=r.randrange(d[Mem.ABUF] - 8),
                          acc_base=r.randrange(d[Mem.ACC] - 6), p_base=r.randrange(d[Mem.PTAB] - 3 * N),
                          kt=r.randint(1, 3), nt=r.randint(1, 3), m=r.randint(1, 3), act_signed=r.random() < 0.5,
                          acc_mode=r.choice([AccMode.OVERWRITE, AccMode.ACCUMULATE, AccMode.BIAS]))
    if kind < 0.85:
        return isa.requant(a_dst=r.randrange(d[Mem.ABUF] - 8), acc_base=r.randrange(d[Mem.ACC] - 6),
                           p_base=r.randrange(d[Mem.PTAB] - 3 * N), nt=r.randint(1, 3), m=r.randint(1, 3),
                           relu=r.random() < 0.5)
    if kind < 0.95:
        return isa.nop()
    return r.choice([isa.Fields(0x33).encode(), isa.nop() | 1 << 50,
                     isa.matmul(w_base=d[Mem.WBUF] - 1, a_base=0, acc_base=0, kt=3, nt=1, m=1, act_signed=False,
                                acc_mode=AccMode.OVERWRITE)])


@cocotb.test()
async def random_programs(dut):
    b = await setup(dut, 4)
    stops = {}
    for i in range(12):
        prog = [random_instruction(b.rng) for _ in range(b.rng.randint(1, DEPTHS[Mem.IMEM] - 1))]
        if b.rng.random() < 0.8:
            prog.append(isa.halt(i))
        await b.load_program(prog)
        st = await b.check()
        stops[(st.state, st.err)] = stops.get((st.state, st.err), 0) + 1
    dut._log.info(f"stops: {stops}")
    assert stops.get((SeqState.HALTED_OK, SeqErr.NONE), 0) > 0


@cocotb.test()
async def abort_then_run_again(dut):
    b = await setup(dut, 5)
    d = dut
    prog = [random_instruction(b.rng) for _ in range(6)] + [isa.halt(7)]
    for wait in (3, 20, 60, 150, 35, 45, 75, 90, 110, 130):
        await b.load_program(prog)
        await FallingEdge(d.clk)
        b.idle_inputs()
        d.start.value = 1
        await FallingEdge(d.clk)
        d.start.value = 0
        for _ in range(wait):
            await FallingEdge(d.clk)
        was_running = int(d.state.value) == SeqState.RUNNING
        d.abort_req.value = 1
        await FallingEdge(d.clk)
        d.abort_req.value = 0
        for _ in range(200):
            if int(d.state.value) != SeqState.RUNNING:
                break
            await FallingEdge(d.clk)
        else:
            raise AssertionError("ABORT did not stop the program")
        if was_running:
            assert b.status()[:2] == (SeqState.IDLE, 0)
            # Idle after ABORT means drained: no unit may still be working.
            assert d.u_mm.idle.value == 1 and d.u_rq.idle.value == 1 and d.u_acc.busy.value == 0
        # Memories may hold a partly executed instruction: resynchronize, then run normally.
        await b.sync_abuf_acc()
        await b.check()


def test_tpu_core():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"tpu_core_N{N}"
    rtl = ["isa_pkg.sv", "sync_fifo.sv", "buffer_ram.sv", "pe.sv", "systolic_array.sv", "acc_unit.sv", "requant.sv",
           "instr_decode.sv", "range_check.sv", "sequencer.sv", "matmul_ctrl.sv", "requant_ctrl.sv", "tpu_core.sv"]
    runner.build(
        sources=[ROOT / "rtl" / f for f in rtl],
        hdl_toplevel="tpu_core",
        parameters={"N": N, "IMEM_DEPTH": DEPTHS[Mem.IMEM], "WBUF_DEPTH": DEPTHS[Mem.WBUF],
                    "ABUF_DEPTH": DEPTHS[Mem.ABUF], "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="tpu_core", test_module="test_tpu_core", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
