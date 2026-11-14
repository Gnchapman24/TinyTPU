"""cocotb tests for rtl/instr_decode.sv (task A4.2): decode errors 1-4 against tinytpu.isa.decode_error.

The decoder is combinational. Every expected value comes from the reference, at two sets of
buffer depths (the defaults and small ones, which move the implemented address widths):

- the docs/isa.md example program;
- every single-bit flip of legal words of each opcode (each reserved bit, each field, each flag);
- every opcode value with otherwise legal fields;
- random words, and random legal words with zeroed counts.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from tinytpu import isa
from tinytpu.isa import AccMode, Mem

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))

DEPTH_SETS = {
    "default": dict(isa.DEFAULT_DEPTHS),
    "small": {**isa.DEFAULT_DEPTHS, Mem.WBUF: 64, Mem.ABUF: 32, Mem.ACC: 16, Mem.PTAB: 8},
}
DEPTHS = DEPTH_SETS[os.environ.get("DEC_DEPTHS", "default")]


def legal_words(rng: random.Random) -> list[int]:
    """Legal instructions of every opcode with random in-range fields."""
    def addr(mem: Mem) -> int:
        return rng.randrange(DEPTHS[mem])

    def count() -> int:
        return rng.choice([1, rng.randint(1, (1 << isa.COUNT_W) - 1)])

    words = [isa.halt(rng.randrange(1 << 16)), isa.nop()]
    for _ in range(4):
        words.append(isa.matmul(w_base=addr(Mem.WBUF), a_base=addr(Mem.ABUF), acc_base=addr(Mem.ACC),
                                p_base=addr(Mem.PTAB), kt=count(), nt=count(), m=count(),
                                act_signed=rng.random() < 0.5,
                                acc_mode=rng.choice([AccMode.OVERWRITE, AccMode.ACCUMULATE, AccMode.BIAS])))
        words.append(isa.requant(a_dst=addr(Mem.ABUF), acc_base=addr(Mem.ACC), p_base=addr(Mem.PTAB),
                                 nt=count(), m=count(), relu=rng.random() < 0.5))
    return words


class Checker:
    def __init__(self, dut):
        self.dut = dut
        self.counts: dict[isa.SeqErr, int] = {}

    async def check(self, word: int) -> None:
        self.dut.instr.value = word
        await Timer(1, unit="ns")
        want = isa.decode_error(word, DEPTHS)
        got = int(self.dut.err.value)
        assert got == want, f"word {word:032x}: got {got}, want {int(want)} ({want.name})"
        self.counts[want] = self.counts.get(want, 0) + 1


@cocotb.test()
async def docs_example_program(dut):
    c = Checker(dut)
    words = [
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=25, nt=8, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=32, acc_base=0, p_base=0, nt=8, m=1, relu=True),
        isa.matmul(w_base=200, a_base=32, acc_base=8, p_base=64, kt=8, nt=2, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=40, acc_base=8, p_base=64, nt=2, m=1, relu=False),
        isa.halt(1),
        0,                                       # all-zero memory: HALT 0
    ]
    for w in words:
        await c.check(w)


@cocotb.test()
async def every_single_bit_flip(dut):
    c = Checker(dut)
    rng = random.Random(1)
    for word in legal_words(rng):
        await c.check(word)
        for bit in range(isa.INSTR_W):
            await c.check(word ^ (1 << bit))
    for err in (isa.SeqErr.NONE, isa.SeqErr.ILLEGAL_OPCODE, isa.SeqErr.RESERVED_BIT,
                isa.SeqErr.ACC_MODE, isa.SeqErr.ZERO_COUNT):
        assert c.counts.get(err, 0) > 0, f"no case produced {err.name}: {c.counts}"


@cocotb.test()
async def every_opcode_value(dut):
    c = Checker(dut)
    rng = random.Random(2)
    for word in legal_words(rng):
        for op in range(256):
            await c.check((word & ~0xFF) | op)


@cocotb.test()
async def zero_counts_and_acc_modes(dut):
    c = Checker(dut)
    rng = random.Random(3)
    for word in legal_words(rng):
        for slot in (5, 6, 7):                   # kt, nt, m
            await c.check(word & ~(0xFFFF << (16 * slot)))
        for mode in range(4):
            await c.check((word & ~(0b11 << 9)) | (mode << 9))
    assert c.counts.get(isa.SeqErr.ZERO_COUNT, 0) > 0 and c.counts.get(isa.SeqErr.ACC_MODE, 0) > 0


@cocotb.test()
async def random_words(dut):
    c = Checker(dut)
    rng = random.Random(4)
    ops = [0x00, 0x01, 0x10, 0x20]
    for _ in range(3000):
        word = rng.getrandbits(128)
        if rng.random() < 0.7:                   # mostly legal opcodes, so the deeper checks run
            word = (word & ~0xFF) | rng.choice(ops)
        if rng.random() < 0.5:                   # sparse words reach the zero-count / mode checks
            word &= rng.getrandbits(128) & rng.getrandbits(128) & rng.getrandbits(128)
        await c.check(word)


@pytest.mark.parametrize("depths", list(DEPTH_SETS))
def test_instr_decode(depths):
    d = DEPTH_SETS[depths]
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"instr_decode_{depths}_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "isa_pkg.sv", ROOT / "rtl" / "instr_decode.sv"],
        hdl_toplevel="instr_decode",
        parameters={"WBUF_DEPTH": d[Mem.WBUF], "ABUF_DEPTH": d[Mem.ABUF], "ACC_DEPTH": d[Mem.ACC],
                    "PTAB_DEPTH": d[Mem.PTAB]},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="instr_decode", test_module="test_instr_decode", test_dir=Path(__file__).parent,
                extra_env={"DEC_DEPTHS": depths}, results_xml=str(build_dir / "results.xml"))
