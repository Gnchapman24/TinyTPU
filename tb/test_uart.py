"""cocotb tests for rtl/uart_rx.sv and rtl/uart_tx.sv (task A2.1).

Each test runs twice: at CLKS_PER_BIT = 8 for speed, and at the real 100 (1 Mbaud at 100 MHz).
The modules' own assertions (frame waveform, single-cycle pulses) are active, because the build
defines ASSERTIONS and uses --assert.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb.utils import get_sim_time
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
CLK_NS = 10


def clks() -> int:
    return int(os.environ["UART_CLKS"])


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

async def setup(dut, loopback: bool = False) -> None:
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.rst.value = 1
    dut.loopback.value = int(loopback)
    dut.rx_pin.value = 1
    dut.tx_valid.value = 0
    dut.tx_data.value = 0
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


class RxMonitor:
    """Records every rx_valid / rx_frame_err pulse as ("byte", value, cycle) or ("ferr", None, cycle)."""

    def __init__(self, dut):
        self.dut = dut
        self.events: list[tuple[str, int | None, int]] = []
        self.cycle = 0
        cocotb.start_soon(self._run())

    async def _run(self):
        while True:
            await RisingEdge(self.dut.clk)
            self.cycle += 1
            if self.dut.rx_valid.value == 1:
                self.events.append(("byte", int(self.dut.rx_data.value), self.cycle))
            if self.dut.rx_frame_err.value == 1:
                self.events.append(("ferr", None, self.cycle))

    def bytes(self) -> list[int]:
        return [v for kind, v, _ in self.events if kind == "byte"]


async def drive_bit(dut, level: int, cycles: int) -> None:
    dut.rx_pin.value = level
    if cycles > 0:
        await ClockCycles(dut.clk, cycles, rising=False)


async def send_frame(dut, byte: int, bit_cycles: int, stop: int = 1) -> None:
    """Drive one 8N1 frame on rx_pin, changing levels on falling clock edges."""
    await drive_bit(dut, 0, bit_cycles)
    for i in range(8):
        await drive_bit(dut, (byte >> i) & 1, bit_cycles)
    await drive_bit(dut, stop, bit_cycles)
    dut.rx_pin.value = 1


async def wait_events(mon: RxMonitor, count: int, limit_cycles: int) -> None:
    for _ in range(limit_cycles):
        if len(mon.events) >= count:
            return
        await RisingEdge(mon.dut.clk)
    raise AssertionError(f"expected {count} receiver events, got {mon.events}")


async def capture_tx(dut, nbytes: int, c: int) -> list[int]:
    """Decode nbytes frames from tx_line, checking every cycle of every bit, not just mid-bit."""
    out = []
    for _ in range(nbytes):
        while dut.tx_line.value == 1:
            await RisingEdge(dut.clk)
        byte = 0
        for bit in range(10):
            levels = []
            for _ in range(c):
                levels.append(int(dut.tx_line.value))
                await RisingEdge(dut.clk)
            assert len(set(levels)) == 1, f"tx_line changed inside bit {bit}: {levels}"
            level = levels[0]
            if bit == 0:
                assert level == 0, "start bit must be low"
            elif bit <= 8:
                byte |= level << (bit - 1)
            else:
                assert level == 1, "stop bit must be high"
        out.append(byte)
    return out


async def send_tx(dut, data: list[int], hold: bool = True) -> None:
    """Hand bytes to uart_tx through the valid/ready handshake."""
    for b in data:
        await FallingEdge(dut.clk)
        dut.tx_data.value = b
        dut.tx_valid.value = 1
        while True:
            await RisingEdge(dut.clk)
            if dut.tx_ready.value == 1:
                break
        if not hold:
            await FallingEdge(dut.clk)
            dut.tx_valid.value = 0
    await FallingEdge(dut.clk)
    dut.tx_valid.value = 0


# ------------------------------------------------------------------------------------------------
# Receiver tests
# ------------------------------------------------------------------------------------------------

@cocotb.test()
async def rx_after_reset_is_quiet(dut):
    await setup(dut)
    mon = RxMonitor(dut)
    await ClockCycles(dut.clk, 20 * clks())
    assert mon.events == []
    assert dut.tx_line.value == 1 and dut.tx_ready.value == 1


@cocotb.test()
async def rx_random_bytes(dut):
    c = clks()
    await setup(dut)
    mon = RxMonitor(dut)
    rng = random.Random(1)
    data = [0x00, 0xFF, 0x55, 0xAA, 0xA5, 0x5A] + [rng.getrandbits(8) for _ in range(40 if c <= 8 else 8)]
    for b in data:
        await send_frame(dut, b, c)
        await drive_bit(dut, 1, rng.choice([0, 0, 1, c, 3 * c]))  # idle gap
    await wait_events(mon, len(data), 20 * c)
    assert mon.bytes() == data
    assert all(kind == "byte" for kind, _, _ in mon.events)


@cocotb.test()
async def rx_latency(dut):
    """Measure the cycles from the start bit's falling edge to valid, as stated in the header."""
    c = clks()
    await setup(dut)
    mon = RxMonitor(dut)
    await ClockCycles(dut.clk, 5, rising=False)
    start = mon.cycle
    await send_frame(dut, 0x3C, c)
    await wait_events(mon, 1, 20 * c)
    # The monitor samples at rising edges, so it sees a registered pulse one edge after it rises.
    latency = mon.events[0][2] - start - 1
    assert latency == 3 + c // 2 + 9 * c, f"latency {latency}"


@cocotb.test()
async def rx_frame_error(dut):
    c = clks()
    await setup(dut)
    mon = RxMonitor(dut)
    await send_frame(dut, 0x81, c, stop=0)       # stop bit low
    await drive_bit(dut, 0, 2 * c)                # line stays low (break) for a while
    await drive_bit(dut, 1, c)                    # line returns high: receiver resynchronizes
    await send_frame(dut, 0x42, c)
    await wait_events(mon, 2, 20 * c)
    kinds = [kind for kind, _, _ in mon.events]
    assert kinds == ["ferr", "byte"], mon.events
    assert mon.bytes() == [0x42]


@cocotb.test()
async def rx_glitch_rejected(dut):
    c = clks()
    await setup(dut)
    mon = RxMonitor(dut)
    for width in (1, max(1, c // 4)):            # low pulses well under half a bit
        await drive_bit(dut, 0, width)
        await drive_bit(dut, 1, 2 * c)
    assert mon.events == []
    await send_frame(dut, 0x99, c)
    await wait_events(mon, 1, 20 * c)
    assert mon.bytes() == [0x99]


@cocotb.test()
async def rx_baud_tolerance(dut):
    """At 100 cycles per bit, frames 3% fast or slow must still be received."""
    c = clks()
    if c < 50:
        return  # integer bit lengths can't express a few-percent error at small CLKS_PER_BIT
    await setup(dut)
    mon = RxMonitor(dut)
    data = [0x00, 0xFF, 0x55, 0xAA, 0x0F]
    for bit_cycles in (c * 97 // 100, c * 103 // 100):
        for b in data:
            await send_frame(dut, b, bit_cycles)
    await wait_events(mon, 2 * len(data), 20 * c)
    assert mon.bytes() == data * 2


# ------------------------------------------------------------------------------------------------
# Transmitter and loopback tests
# ------------------------------------------------------------------------------------------------

@cocotb.test()
async def tx_waveform_and_back_to_back(dut):
    c = clks()
    await setup(dut)
    rng = random.Random(2)
    data = [0x00, 0xFF, 0x01, 0x80] + [rng.getrandbits(8) for _ in range(12 if c <= 8 else 2)]
    cap = cocotb.start_soon(capture_tx(dut, len(data), c))
    t0 = get_sim_time(unit="ns")
    await send_tx(dut, data, hold=True)
    assert await cap == data
    # Held valid: frames follow back-to-back at exactly 10 bits each.
    elapsed_cycles = (get_sim_time(unit="ns") - t0) // CLK_NS
    assert elapsed_cycles <= 10 * c * len(data) + 3, f"{elapsed_cycles} cycles for {len(data)} bytes"


@cocotb.test()
async def tx_ready_handshake(dut):
    c = clks()
    await setup(dut)
    assert dut.tx_ready.value == 1
    await FallingEdge(dut.clk)
    dut.tx_data.value = 0x5A
    dut.tx_valid.value = 1
    await RisingEdge(dut.clk)              # accepted here
    await FallingEdge(dut.clk)
    dut.tx_valid.value = 0
    low = 0
    while dut.tx_ready.value == 0:
        low += 1
        await FallingEdge(dut.clk)
    assert low == 10 * c - 1, f"ready low for {low} cycles"   # high again in the last stop cycle


@cocotb.test()
async def loopback_random(dut):
    c = clks()
    await setup(dut, loopback=True)
    mon = RxMonitor(dut)
    rng = random.Random(3)
    data = [rng.getrandbits(8) for _ in range(100 if c <= 8 else 10)]
    await send_tx(dut, data, hold=True)
    await wait_events(mon, len(data), 20 * c)
    assert mon.bytes() == data


# ------------------------------------------------------------------------------------------------
# pytest entry: build and run at both bit lengths
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("clks_per_bit", [8, 100])
def test_uart(clks_per_bit):
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"uart_c{clks_per_bit}_N{N}"
    runner.build(
        sources=[ROOT / "rtl" / "uart_tx.sv", ROOT / "rtl" / "uart_rx.sv", ROOT / "tb" / "hdl" / "uart_tb.sv"],
        hdl_toplevel="uart_tb",
        parameters={"CLKS_PER_BIT": clks_per_bit},
        build_args=["--assert", "-DASSERTIONS", "-Wall", "-Wno-DECLFILENAME"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        hdl_toplevel="uart_tb",
        test_module="test_uart",
        test_dir=Path(__file__).parent,
        extra_env={"UART_CLKS": str(clks_per_bit)},
        results_xml=str(build_dir / "results.xml"),
    )
