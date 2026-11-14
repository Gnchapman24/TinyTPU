"""Consistency tests for the ISA definitions (task A1.5).

Checks that the three copies of the ISA agree:

* ``rtl/isa_pkg.sv``: constants, enums and packed-struct layouts, read with pyslang
* ``host/tinytpu/isa.py``: the Python mirror and encoder
* ``docs/isa.md``: the source of truth, via its worked example program

It also runs the package through Verilator, to check that the packed-struct layout really decodes
Python-encoded instructions bit for bit.
"""

from __future__ import annotations

import os
import random
import re
import struct
from pathlib import Path

import cocotb
import pyslang
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from tinytpu import isa

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))


# ------------------------------------------------------------------------------------------------
# Reading rtl/isa_pkg.sv with pyslang
# ------------------------------------------------------------------------------------------------

def _const(cv) -> int:
    """Convert a slang ConstantValue such as ``8'd16``, ``2'b10`` or ``128`` to ``int``."""
    s = str(cv)
    if "'" not in s:
        return int(s)
    digits = s.split("'", 1)[1].lstrip("sS")
    base = {"b": 2, "o": 8, "d": 10, "h": 16}[digits[0].lower()]
    return int(digits[1:].replace("_", ""), base)


@pytest.fixture(scope="module")
def sv():
    """Parameters, enums and struct layouts from isa_pkg, as plain Python dicts."""
    sm = pyslang.SourceManager()
    tree = pyslang.syntax.SyntaxTree.fromFile(str(ROOT / "rtl" / "isa_pkg.sv"), sm)
    comp = pyslang.ast.Compilation()
    comp.addSyntaxTree(tree)
    errors = [d for d in comp.getAllDiagnostics() if d.isError()]
    assert not errors, "rtl/isa_pkg.sv does not compile"
    params, enums, structs = {}, {}, {}
    SK = pyslang.ast.SymbolKind
    for member in comp.getPackage("isa_pkg"):
        if member.kind == SK.Parameter:
            params[member.name] = _const(member.value)
        elif member.kind == SK.TypeAlias:
            t = member.targetType.type
            if t.kind == SK.EnumType:
                enums[member.name] = {v.name: _const(v.value) for v in t}
            elif t.kind == SK.PackedStructType:
                structs[member.name] = {f.name: (f.bitOffset, f.type.bitWidth) for f in t}
    return {"params": params, "enums": enums, "structs": structs}


def test_parameters_match(sv):
    p = sv["params"]
    assert p["INSTR_W"] == isa.INSTR_W
    assert p["SLOT_W"] == isa.SLOT_W
    assert p["FLAG_ACT_SIGNED"] == isa.FLAG_ACT_SIGNED
    assert p["FLAG_ACC_MODE_LSB"] == isa.FLAG_ACC_MODE_LSB
    assert p["FLAG_RELU"] == isa.FLAG_RELU
    for op in isa.Opcode:
        assert p[f"FLAGS_MASK_{op.name}"] == isa.FLAGS_MASK[op], op.name
    for mem in isa.Mem:
        assert p[f"DEFAULT_{mem.name}_DEPTH"] == isa.DEFAULT_DEPTHS[mem], mem.name
    assert p["COUNT_W"] == isa.COUNT_W
    assert p["PTAB_ENTRY_W"] == isa.PTAB_ENTRY_W


@pytest.mark.parametrize(
    "sv_enum, py_enum, prefix",
    [
        ("opcode_t", isa.Opcode, "OP_"),
        ("acc_mode_t", isa.AccMode, "ACC_"),
        ("seq_state_t", isa.SeqState, "ST_"),
        ("seq_err_t", isa.SeqErr, "ERR_"),
        ("mem_id_t", isa.Mem, "MEM_"),
    ],
)
def test_enums_match(sv, sv_enum, py_enum, prefix):
    sv_values = {name.removeprefix(prefix): v for name, v in sv["enums"][sv_enum].items()}
    py_values = {e.name: int(e) for e in py_enum}
    assert sv_values == py_values


def test_instr_layout_matches(sv):
    # (bit offset, width) of each field, as encoded by isa.Fields.
    expected = {
        "opcode": (0, 8), "flags": (8, 8), "w_base": (16, 16), "a_base": (32, 16),
        "acc_base": (48, 16), "p_base": (64, 16), "kt": (80, 16), "nt": (96, 16), "m": (112, 16),
    }
    assert sv["structs"]["instr_t"] == expected
    # And the Python encoder agrees with that layout field by field.
    for name, (offset, width) in expected.items():
        value = (1 << width) - 1
        word = isa.Fields(**{"opcode": 0, name: value}).encode()
        assert word == value << offset, name


def test_ptab_layout_matches(sv):
    assert sv["structs"]["ptab_entry_t"] == {"bias": (0, 32), "mult": (32, 16), "s": (48, 6)}
    raw = isa.ptab_entry(-5, 0xBEEF, 63)
    bias, mult, s = struct.unpack("<iHBx", raw)
    assert (bias, mult, s) == (-5, 0xBEEF, 63)


# ------------------------------------------------------------------------------------------------
# docs/isa.md worked example
# ------------------------------------------------------------------------------------------------

def _example_program() -> list[int]:
    B = isa.AccMode.BIAS
    return [
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=25, nt=8, m=1, act_signed=False, acc_mode=B),
        isa.requant(a_dst=32, acc_base=0, p_base=0, nt=8, m=1, relu=True),
        isa.matmul(w_base=200, a_base=32, acc_base=8, p_base=64, kt=8, nt=2, m=1, act_signed=False, acc_mode=B),
        isa.requant(a_dst=40, acc_base=8, p_base=64, nt=2, m=1, relu=False),
        isa.halt(1),
    ]


def test_docs_example_program():
    text = (ROOT / "docs" / "isa.md").read_text()
    rows = re.findall(r"^\| (\d+) \| `[^`]*` \| `([0-9a-f ]{39})` \| `([0-9a-f ]{47})` \|$", text, re.M)
    assert len(rows) == 5, "could not find the 5 example rows in docs/isa.md"
    for (pc, hex_slots, wire), word in zip(rows, _example_program()):
        assert int(hex_slots.replace(" ", ""), 16) == word, f"PC {pc}: value"
        assert bytes.fromhex(wire) == isa.to_bytes(word), f"PC {pc}: wire bytes"
        assert isa.decode_error(word) is isa.SeqErr.NONE, f"PC {pc}: should be legal"


# ------------------------------------------------------------------------------------------------
# Decode-time legality (reference model for the hardware decoder)
# ------------------------------------------------------------------------------------------------

def test_decode_errors():
    E = isa.SeqErr
    ok = isa.matmul(w_base=1, a_base=2, acc_base=3, p_base=4, kt=1, nt=1, m=1, act_signed=True,
                    acc_mode=isa.AccMode.ACCUMULATE)
    assert isa.decode_error(ok) is E.NONE
    assert isa.decode_error(0) is E.NONE                         # all-zero memory = HALT 0
    assert isa.decode_error(isa.Fields(0x11).encode()) is E.ILLEGAL_OPCODE
    assert isa.decode_error(ok | 1 << 15) is E.RESERVED_BIT      # flag bit 7
    assert isa.decode_error(isa.Fields(isa.Opcode.NOP, w_base=1).encode()) is E.RESERVED_BIT
    assert isa.decode_error(ok | 1 << (16 + 11)) is E.RESERVED_BIT  # w_base bit 11 (depth 2048)
    assert isa.decode_error(ok | 1 << (80 + 12)) is E.RESERVED_BIT  # kt bit 12 (COUNT_W)
    bad_mode = isa.Fields.decode(ok)
    bad = isa.Fields(**{**bad_mode.__dict__, "flags": bad_mode.flags | 0b110}).encode()
    assert isa.decode_error(bad) is E.ACC_MODE
    assert isa.decode_error(isa.requant(a_dst=0, acc_base=0, p_base=0, nt=0, m=1, relu=False)) is E.ZERO_COUNT
    # Smaller depths shrink the implemented widths.
    small = {**isa.DEFAULT_DEPTHS, isa.Mem.WBUF: 1024}
    assert isa.decode_error(ok | 1 << (16 + 10), small) is E.RESERVED_BIT


def test_encoder_rejects_overflow():
    with pytest.raises(ValueError):
        isa.matmul(w_base=1 << 16, a_base=0, acc_base=0, kt=1, nt=1, m=1, act_signed=False,
                   acc_mode=isa.AccMode.OVERWRITE)
    with pytest.raises(ValueError):
        isa.ptab_entry(0, 0, 64)


# ------------------------------------------------------------------------------------------------
# Struct layout in Verilator (cocotb)
# ------------------------------------------------------------------------------------------------

@cocotb.test()
async def struct_layout_in_verilator(dut):
    rng = random.Random(1234)
    words = [isa.halt(0xABCD), isa.nop(), *_example_program()]
    words += [rng.getrandbits(128) for _ in range(200)]
    for word in words:
        dut.instr_raw.value = word
        await Timer(1, unit="ns")
        f = isa.Fields.decode(word)
        for name in ("opcode", "flags", "w_base", "a_base", "acc_base", "p_base", "kt", "nt", "m"):
            assert int(getattr(dut, name).value) == getattr(f, name), f"{name} of {word:032x}"
        assert int(dut.act_signed.value) == (f.flags >> isa.FLAG_ACT_SIGNED) & 1
        assert int(dut.acc_mode.value) == (f.flags >> isa.FLAG_ACC_MODE_LSB) & 3
        assert int(dut.relu.value) == (f.flags >> isa.FLAG_RELU) & 1

    for _ in range(100):
        bias, mult, s = rng.randint(-(1 << 31), (1 << 31) - 1), rng.getrandbits(16), rng.getrandbits(6)
        raw = int.from_bytes(isa.ptab_entry(bias, mult, s)[:7], "little")  # 54 bits used
        dut.ptab_raw.value = raw
        await Timer(1, unit="ns")
        assert int(dut.bias.value.to_signed()) == bias
        assert int(dut.mult.value) == mult
        assert int(dut.shift.value) == s


def test_struct_layout_in_verilator():
    runner = get_runner("verilator")
    runner.build(
        sources=[ROOT / "rtl" / "isa_pkg.sv", ROOT / "tb" / "hdl" / "isa_pkg_wrap.sv"],
        hdl_toplevel="isa_pkg_wrap",
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=ROOT / "sim_build" / f"isa_pkg_N{N}",
        always=True,
    )
    runner.test(hdl_toplevel="isa_pkg_wrap", test_module="test_isa",
                test_dir=Path(__file__).parent, testcase="struct_layout_in_verilator")
