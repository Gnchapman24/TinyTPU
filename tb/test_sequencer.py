"""cocotb tests for rtl/sequencer.sv (task A4.2): fetch, checks, dispatch, HALT, errors, ABORT.

The testbench models the sequencer's surroundings cycle by cycle:
- IMEM: a Python list, answering each read after IMEM_LAT cycles;
- the execution units: on exec_start they take a random number of cycles, then pulse exec_done;
  on exec_abort they drain for a few cycles and report exec_idle. exec_instr must stay stable
  while an instruction executes.

The expected behaviour comes from tinytpu.model.Machine: the sequence of MATMUL / REQUANT
instructions it executes (as (pc, word)) and its final status (state, error, PC, halt code).
Machine runs with small depths and small counts, so it is quick.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer
from cocotb_tools.runner import get_runner

from tinytpu import isa, model
from tinytpu.isa import AccMode, Mem, SeqErr, SeqState

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
IMEM_LAT = int(os.environ.get("SEQ_IMEM_LAT", "1"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 64, Mem.ABUF: 32, Mem.ACC: 16, Mem.PTAB: 64}


class TracingMachine(model.Machine):
    """Machine that records each MATMUL / REQUANT it executes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.trace: list[int] = []

    def _matmul(self, f):
        self.trace.append(f.encode())
        super()._matmul(f)

    def _requant(self, f):
        self.trace.append(f.encode())
        super()._requant(f)


def expected(program: list[int], start_pc: int) -> tuple[list[int], model.Status]:
    mach = TracingMachine(n=N, depths=dict(DEPTHS))
    mach.load_program(program)
    st = mach.run(start_pc)
    return mach.trace, st


class Env:
    """IMEM and execution-unit models, driven at every falling edge."""

    def __init__(self, dut, rng: random.Random):
        self.dut = dut
        self.rng = rng
        self.imem = [0] * DEPTHS[Mem.IMEM]
        self.reads: list[tuple[int, int]] = []          # (cycles left, addr)
        self.exec_left: int | None = None               # cycles until exec_done; None = idle
        self.drain_left: int | None = None              # cycles of draining after exec_abort
        self.exec_instr: int | None = None
        self.executed: list[int] = []
        self.exec_lat = (0, 6)
        self.started_after_abort = False
        self.cycle = 0
        cocotb.start_soon(self._run())

    async def _run(self) -> None:
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            self.cycle += 1
            # IMEM response.
            d.imem_rd_valid.value = 0
            d.imem_rd_data.value = self.rng.getrandbits(128)   # garbage when not valid
            for i, (left, addr) in enumerate(self.reads):
                self.reads[i] = (left - 1, addr)
            if self.reads and self.reads[0][0] == 0:
                _, addr = self.reads.pop(0)
                d.imem_rd_valid.value = 1
                d.imem_rd_data.value = self.imem[addr]
            # Execution unit.
            d.exec_done.value = 0
            if self.drain_left is not None:
                self.drain_left -= 1
                if self.drain_left <= 0:
                    self.drain_left = None
            elif self.exec_left is not None:
                if self.exec_left == 0:
                    d.exec_done.value = 1
                    self.exec_left = None
                else:
                    self.exec_left -= 1
            d.exec_idle.value = int(self.exec_left is None and self.drain_left is None)
            # Outputs the DUT acts on at the next edge.
            await Timer(1, unit="ns")
            if d.imem_rd_en.value == 1:
                self.reads.append((IMEM_LAT, int(d.imem_rd_addr.value)))
            if self.exec_left is not None and d.exec_done.value == 0:
                assert int(d.exec_instr.value) == self.exec_instr, "exec_instr changed during execution"
            if d.exec_start.value == 1:
                assert self.exec_left is None and self.drain_left is None, "exec_start while busy"
                assert d.abort_req.value == 0, "exec_start in the cycle of an ABORT request"
                self.exec_instr = int(d.exec_instr.value)
                self.executed.append(self.exec_instr)
                self.exec_left = self.rng.randint(*self.exec_lat)
            if d.exec_abort.value == 1 and self.exec_left is not None:
                self.exec_left = None
                self.drain_left = self.rng.randint(0, 5)


async def setup(dut, seed: int) -> Env:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.start.value = 0
    dut.start_pc.value = 0
    dut.abort_req.value = 0
    dut.imem_rd_valid.value = 0
    dut.imem_rd_data.value = 0
    dut.exec_done.value = 0
    dut.exec_idle.value = 1
    env = Env(dut, random.Random(seed))
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return env


def status(dut) -> tuple[int, int, int, int]:
    return (int(dut.state.value), int(dut.err.value), int(dut.pc.value), int(dut.halt_code.value))


async def run_program(env: Env, program: list[int], start_pc: int = 0, timeout: int = 20000) -> None:
    """Load, start, wait for the sequencer to stop, compare with the model."""
    d = env.dut
    env.imem = list(program) + [0] * (DEPTHS[Mem.IMEM] - len(program))
    env.executed = []
    await FallingEdge(d.clk)
    d.start.value, d.start_pc.value = 1, start_pc
    await FallingEdge(d.clk)
    d.start.value = 0
    for _ in range(timeout):
        if int(d.state.value) != SeqState.RUNNING:
            break
        await FallingEdge(d.clk)
    else:
        raise AssertionError("program did not stop")
    trace, st = expected(env.imem, start_pc)
    assert env.executed == trace, f"executed {len(env.executed)} instructions, model {len(trace)}"
    want = (int(st.state), int(st.err), st.pc, st.halt_code)
    assert status(d) == want, f"status {status(d)}, want {want}"


def mm(rng: random.Random, big: bool = False) -> int:
    """MATMUL; with small counts and bases it is always in range, with `big` often not."""
    hi, room = (6, 0) if big else (2, 8)
    return isa.matmul(w_base=rng.randrange(DEPTHS[Mem.WBUF] - room), a_base=rng.randrange(DEPTHS[Mem.ABUF] - room),
                      acc_base=rng.randrange(DEPTHS[Mem.ACC] - room), p_base=rng.randrange(DEPTHS[Mem.PTAB] - 2 * N),
                      kt=rng.randint(1, hi), nt=rng.randint(1, hi), m=rng.randint(1, hi),
                      act_signed=rng.random() < 0.5,
                      acc_mode=rng.choice([AccMode.OVERWRITE, AccMode.ACCUMULATE, AccMode.BIAS]))


def rq(rng: random.Random, big: bool = False) -> int:
    hi, room = (6, 0) if big else (2, 8)
    return isa.requant(a_dst=rng.randrange(DEPTHS[Mem.ABUF] - room), acc_base=rng.randrange(DEPTHS[Mem.ACC] - room),
                       p_base=rng.randrange(DEPTHS[Mem.PTAB] - 2 * N), nt=rng.randint(1, hi), m=rng.randint(1, hi),
                       relu=rng.random() < 0.5)


def fault(rng: random.Random) -> int:
    """An instruction that halts with some error (1-5), or occasionally happens to be legal."""
    kind = rng.randrange(6)
    if kind == 0:
        return isa.Fields(rng.choice([0x02, 0x11, 0x21, 0xFF])).encode()                 # 1
    if kind == 1:
        return rng.choice([mm, rq])(rng) ^ (1 << rng.randrange(16, 128))                 # mostly 2
    if kind == 2:
        return (mm(rng) & ~(0b11 << 9)) | (0b11 << 9)                                    # 3
    if kind == 3:
        return mm(rng) & ~(0xFFFF << (16 * rng.choice([5, 6, 7])))                       # 4
    if kind == 4:
        return rng.choice([mm, rq])(rng, big=True)                                       # often 5
    return rng.getrandbits(128)


def random_program(rng: random.Random) -> list[int]:
    """Mostly legal instructions; some programs fill IMEM without a HALT (error 6)."""
    no_halt = rng.random() < 0.25
    length = DEPTHS[Mem.IMEM] if no_halt else rng.randint(1, DEPTHS[Mem.IMEM])
    prog = []
    for _ in range(length):
        r = rng.random()
        if r < 0.4:
            w = mm(rng)
        elif r < 0.7:
            w = rq(rng)
        elif r < 0.9:
            w = isa.nop()
        elif r < 0.93 and not no_halt:
            w = isa.halt(rng.randrange(1 << 16))
        elif r < 0.96 and not no_halt:
            w = fault(rng)
        else:
            w = isa.nop()
        prog.append(w)
    return prog


@cocotb.test()
async def empty_memory_halts_at_once(dut):
    env = await setup(dut, 1)
    await run_program(env, [], start_pc=0)
    assert status(dut) == (SeqState.HALTED_OK, 0, 0, 0)


@cocotb.test()
async def docs_style_program(dut):
    env = await setup(dut, 2)
    prog = [
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=3, nt=2, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=8, acc_base=0, p_base=0, nt=2, m=1, relu=True),
        isa.nop(),
        isa.matmul(w_base=6, a_base=8, acc_base=2, p_base=2 * N, kt=2, nt=1, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=10, acc_base=2, p_base=2 * N, nt=1, m=1, relu=False),
        isa.halt(1),
    ]
    await run_program(env, prog)
    assert status(dut) == (SeqState.HALTED_OK, 0, 5, 1)
    assert len(env.executed) == 4


@cocotb.test()
async def each_error_code(dut):
    env = await setup(dut, 3)
    rng = random.Random(3)
    ok = mm(rng)
    cases = {
        SeqErr.ILLEGAL_OPCODE: isa.Fields(0x11).encode(),
        SeqErr.RESERVED_BIT: isa.nop() | 1 << 40,
        SeqErr.ACC_MODE: (ok & ~(0b11 << 9)) | (0b11 << 9),
        SeqErr.ZERO_COUNT: ok & ~(0xFFFF << 96),
        SeqErr.ADDR_RANGE: isa.matmul(w_base=DEPTHS[Mem.WBUF] - 1, a_base=0, acc_base=0, kt=2, nt=1, m=1,
                                      act_signed=False, acc_mode=AccMode.OVERWRITE),
    }
    for err, word in cases.items():
        await run_program(env, [ok, isa.nop(), word, isa.halt()])
        assert status(dut)[:3] == (SeqState.HALTED_ERROR, err, 2)
    # PC overrun: no HALT anywhere; the reported PC is the IMEM depth.
    await run_program(env, [isa.nop()] * DEPTHS[Mem.IMEM], start_pc=3)
    assert status(dut)[:3] == (SeqState.HALTED_ERROR, SeqErr.PC_OVERRUN, DEPTHS[Mem.IMEM])
    # A new run clears the error.
    await run_program(env, [isa.halt(9)])
    assert status(dut) == (SeqState.HALTED_OK, 0, 0, 9)


@cocotb.test()
async def random_programs(dut):
    env = await setup(dut, 4)
    rng = random.Random(4)
    stops = {}
    executed = 0
    for _ in range(150):
        env.exec_lat = rng.choice([(0, 0), (0, 3), (5, 20)])
        prog = random_program(rng)
        await run_program(env, prog, start_pc=rng.randrange(min(4, len(prog))))
        key = (int(dut.state.value), int(dut.err.value))
        stops[key] = stops.get(key, 0) + 1
        executed += len(env.executed)
    dut._log.info(f"random programs: {executed} instructions executed; stops {stops}")
    assert len(stops) == 7, stops          # halted OK and every error code 1-6
    assert executed >= 500, executed


async def wait_cycles(dut, n: int) -> None:
    for _ in range(n):
        await FallingEdge(dut.clk)


async def abort_after(env: Env, program: list[int], cycles: int) -> tuple[int, int, int, int]:
    """Start, assert abort_req after `cycles`, wait for the stop; return the status."""
    d = env.dut
    env.imem = list(program) + [0] * (DEPTHS[Mem.IMEM] - len(program))
    env.executed = []
    await FallingEdge(d.clk)
    d.start.value, d.start_pc.value = 1, 0
    await FallingEdge(d.clk)
    d.start.value = 0
    await wait_cycles(d, cycles)
    pc_at_abort = int(d.pc.value)
    was_running = int(d.state.value) == SeqState.RUNNING
    d.abort_req.value = 1
    await FallingEdge(d.clk)
    d.abort_req.value = 0
    n_exec = len(env.executed)
    for _ in range(100):
        if int(d.state.value) != SeqState.RUNNING:
            break
        assert d.exec_start.value == 0
        await FallingEdge(d.clk)
    else:
        raise AssertionError("abort did not stop the sequencer")
    assert len(env.executed) == n_exec, "an instruction started after ABORT"
    if was_running:
        assert status(d)[:3] == (SeqState.IDLE, 0, pc_at_abort), (status(d), pc_at_abort)
        assert env.exec_left is None and env.drain_left is None and not env.reads
    return status(d)


@cocotb.test()
async def abort_at_every_point(dut):
    """ABORT in every cycle of a program's run: fetch, wait, decode, check, execute, drain."""
    env = await setup(dut, 5)
    rng = random.Random(5)
    env.exec_lat = (8, 8)
    prog = [isa.nop(), mm(rng), rq(rng), isa.nop(), mm(rng), isa.halt(3)]
    _, st = expected(prog + [0] * (DEPTHS[Mem.IMEM] - len(prog)), 0)
    assert st.state == SeqState.HALTED_OK
    for cycles in range(0, 90):
        await abort_after(env, prog, cycles)
        await run_program(env, prog)                      # the next run is unaffected


@cocotb.test()
async def abort_while_stopped_changes_nothing(dut):
    env = await setup(dut, 6)
    await run_program(env, [isa.nop(), isa.halt(7)])
    before = status(dut)
    dut.abort_req.value = 1
    await wait_cycles(dut, 3)
    dut.abort_req.value = 0
    await wait_cycles(dut, 2)
    assert status(dut) == before


@cocotb.test()
async def start_while_running_is_ignored(dut):
    env = await setup(dut, 7)
    rng = random.Random(7)
    env.exec_lat = (10, 10)
    prog = [mm(rng), mm(rng), isa.halt(2)]
    env.imem = prog + [0] * (DEPTHS[Mem.IMEM] - len(prog))
    env.executed = []
    await FallingEdge(dut.clk)
    dut.start.value, dut.start_pc.value = 1, 0
    await FallingEdge(dut.clk)
    dut.start_pc.value = 2                               # keep start high with another PC
    for _ in range(80):
        await FallingEdge(dut.clk)
        if int(dut.state.value) != SeqState.RUNNING:
            break
    dut.start.value = 0
    assert env.executed == prog[:2]
    # It halted, and start was still high in that cycle: a new run from PC 2 may have begun. Let it finish.
    await wait_cycles(dut, 40)
    assert status(dut) == (SeqState.HALTED_OK, 0, 2, 2)


@cocotb.test()
async def reset_mid_run(dut):
    env = await setup(dut, 8)
    rng = random.Random(8)
    env.exec_lat = (20, 20)
    env.imem = [mm(rng), isa.halt(1)] + [0] * (DEPTHS[Mem.IMEM] - 2)
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    await wait_cycles(dut, 25)                           # inside the MATMUL
    dut.rst.value = 1
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    assert status(dut) == (SeqState.IDLE, 0, 0, 0)
    await wait_cycles(dut, 30)                           # the environment drains
    await run_program(env, [isa.halt(4)])


@pytest.mark.parametrize("imem_lat", [1, 3])
def test_sequencer(imem_lat):
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"sequencer_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / f for f in ("isa_pkg.sv", "instr_decode.sv", "range_check.sv", "sequencer.sv")],
        hdl_toplevel="sequencer",
        parameters={"N": N, "IMEM_DEPTH": DEPTHS[Mem.IMEM], "WBUF_DEPTH": DEPTHS[Mem.WBUF],
                    "ABUF_DEPTH": DEPTHS[Mem.ABUF], "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="sequencer", test_module="test_sequencer", test_dir=Path(__file__).parent,
                extra_env={"SEQ_IMEM_LAT": str(imem_lat)},
                results_xml=str(build_dir / f"results_lat{imem_lat}.xml"))
