"""Tests for rtl/crc16_ccitt.sv and its Python reference tinytpu.crc (task A2.3).

The Python reference is first checked against the standard check value and against Python's own
``binascii.crc_hqx`` (an independent CRC-CCITT implementation). The RTL is then checked against
the reference: random messages with idle gaps, clear-with-data, back-to-back bytes, and the
zero-residue check the packet engine relies on, including bit-flip and byte-order errors.
"""

from __future__ import annotations

import binascii
import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

from tinytpu.crc import CHECK_VALUE, INIT, check, crc16, crc_bytes

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))


# ------------------------------------------------------------------------------------------------
# Python reference
# ------------------------------------------------------------------------------------------------

def test_reference_check_value():
    assert crc16(b"123456789") == CHECK_VALUE == 0x29B1
    assert crc16(b"") == INIT


def test_reference_matches_binascii():
    rng = random.Random(11)
    for n in list(range(40)) + [255, 256, 1000, 4096]:
        data = rng.randbytes(n)
        assert crc16(data) == binascii.crc_hqx(data, 0xFFFF), n


def test_reference_incremental():
    data = random.Random(12).randbytes(300)
    assert crc16(data[100:], crc16(data[:100])) == crc16(data)


def test_reference_residue_and_errors():
    rng = random.Random(13)
    for _ in range(200):
        msg = rng.randbytes(rng.randint(1, 64))
        frame = msg + crc_bytes(msg)
        assert check(frame)
        bit = rng.randrange(len(frame) * 8)                    # any single-bit error is caught
        bad = bytearray(frame)
        bad[bit // 8] ^= 1 << (bit % 8)
        assert not check(bytes(bad))
        if crc16(msg) >> 8 != crc16(msg) & 0xFF:                # wrong CRC byte order is caught
            assert not check(msg + crc16(msg).to_bytes(2, "little"))


# ------------------------------------------------------------------------------------------------
# RTL
# ------------------------------------------------------------------------------------------------

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.clear.value = 0
    dut.valid.value = 0
    dut.data.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def feed(dut, data: bytes, clear_first: bool = True, gaps: random.Random | None = None) -> None:
    """Send bytes, optionally with random idle cycles; the first byte rides on the clear."""
    for i, b in enumerate(data):
        if gaps is not None:
            for _ in range(gaps.choice([0, 0, 0, 1, 3])):
                dut.valid.value = 0
                dut.clear.value = 0
                await FallingEdge(dut.clk)
        dut.clear.value = int(clear_first and i == 0)
        dut.data.value = b
        dut.valid.value = 1
        await FallingEdge(dut.clk)
    dut.valid.value = 0
    dut.clear.value = 0
    if clear_first and not data:
        dut.clear.value = 1
        await FallingEdge(dut.clk)
        dut.clear.value = 0


def crc_out(dut) -> int:
    return int(dut.crc.value)


@cocotb.test()
async def reset_value(dut):
    await setup(dut)
    assert crc_out(dut) == INIT and dut.zero.value == 0


@cocotb.test()
async def check_value(dut):
    await setup(dut)
    await feed(dut, b"123456789")
    assert crc_out(dut) == 0x29B1


@cocotb.test()
async def random_messages_with_gaps(dut):
    await setup(dut)
    rng = random.Random(14)
    for _ in range(60):
        msg = rng.randbytes(rng.randint(0, 300))
        await feed(dut, msg, gaps=rng)
        await ClockCycles(dut.clk, rng.randint(0, 3), rising=False)   # idle cycles hold the value
        assert crc_out(dut) == crc16(msg), f"len {len(msg)}"
        assert int(dut.zero.value) == (crc16(msg) == 0)


@cocotb.test()
async def zero_flag_needs_all_16_bits(dut):
    """CRCs with exactly one zero byte (0x00nn or 0xnn00) must not set zero."""
    await setup(dut)
    rng = random.Random(16)
    found = {"low": [], "high": []}
    while len(found["low"]) < 3 or len(found["high"]) < 3:
        msg = rng.randbytes(4)
        c = crc16(msg)
        if c & 0xFF == 0 and c >> 8:
            found["low"].append(msg)
        elif c >> 8 == 0 and c & 0xFF:
            found["high"].append(msg)
    for msg in found["low"][:3] + found["high"][:3]:
        await feed(dut, msg)
        assert crc_out(dut) == crc16(msg)
        assert dut.zero.value == 0, f"crc {crc16(msg):#06x} is not zero"


@cocotb.test()
async def clear_without_data_restarts(dut):
    await setup(dut)
    await feed(dut, b"garbage")
    dut.clear.value = 1
    await FallingEdge(dut.clk)
    dut.clear.value = 0
    assert crc_out(dut) == INIT
    await feed(dut, b"123456789", clear_first=False)
    assert crc_out(dut) == 0x29B1


@cocotb.test()
async def zero_residue_and_error_detection(dut):
    await setup(dut)
    rng = random.Random(15)
    for _ in range(40):
        msg = rng.randbytes(rng.randint(1, 80))
        frame = msg + crc_bytes(msg)
        await feed(dut, frame)
        assert dut.zero.value == 1, "intact frame must leave zero"
        bit = rng.randrange(len(frame) * 8)
        bad = bytearray(frame)
        bad[bit // 8] ^= 1 << (bit % 8)
        await feed(dut, bytes(bad))
        assert dut.zero.value == 0, f"flipped bit {bit} not detected"
        c = crc16(msg)
        if c >> 8 != c & 0xFF:
            await feed(dut, msg + c.to_bytes(2, "little"))
            assert dut.zero.value == 0, "low-byte-first CRC must not check"


def test_crc16_rtl():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"crc16_ccitt_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "crc16_ccitt.sv"],
        hdl_toplevel="crc16_ccitt",
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        hdl_toplevel="crc16_ccitt",
        test_module="test_crc16",
        test_dir=Path(__file__).parent,
        results_xml=str(build_dir / "results.xml"),
    )
