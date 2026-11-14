"""End-to-end tests for the bring-up design rtl/loopback_top.sv (task A2.4).

The DUT is driven only through its board pins (RsRx, RsTx, btnC, sw, led), with the UART models
in uart_bfm.py and the host-side packet code in tinytpu.packet, so these tests exercise the same
path as the board test. Parameters are scaled down for speed: 8 clocks per bit, a 2000-cycle
timeout, short LED timers.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb_tools.runner import get_runner

from tinytpu.packet import SYNC_REQ, Cmd, Code, build_request, parse_reply
from uart_bfm import UartSink, UartSource

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))

C = 8                 # CLKS_PER_BIT
TIMEOUT = 2000        # TIMEOUT_CYCLES
LED_STRETCH_BITS = 4
BLINK_BIT = 6
REPLY_WAIT = 40 * 10 * C   # generous wait for the first reply byte


class Bench:
    def __init__(self, dut):
        self.dut = dut
        self.src = UartSource(dut.clk, dut.RsRx, C)
        self.sink = UartSink(dut.clk, dut.RsTx, C)

    def led(self, i: int) -> int:
        return (int(self.dut.led.value) >> i) & 1

    def led_hi(self) -> int:
        return int(self.dut.led.value) >> 8

    async def transact(self, packet: bytes):
        await self.src.send(packet)
        return parse_reply(await self.sink.read_packet(REPLY_WAIT))


async def start(dut, sw: int = 0, power_up: bool = False) -> Bench:
    """Start the clock. All tests share one simulation, so every test except the power-up one
    (which runs first) presses the reset button to begin from a clean state."""
    dut.btnC.value = 0
    dut.sw.value = sw
    dut.RsRx.value = 1
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b = Bench(dut)
    await ClockCycles(dut.clk, 20)
    if not power_up:
        await press_reset(dut)
    return b


async def press_reset(dut) -> None:
    await FallingEdge(dut.clk)
    dut.btnC.value = 1
    await ClockCycles(dut.clk, 5, rising=False)
    dut.btnC.value = 0
    await ClockCycles(dut.clk, 20, rising=False)


@cocotb.test()
async def resets_itself_after_power_up(dut):
    b = await start(dut, power_up=True)  # no button press: the design must reset itself
    assert dut.RsTx.value == 1
    assert b.led_hi() == 0 and all(b.led(i) == 0 for i in (4, 5, 6, 7))
    r = await b.transact(build_request(Cmd.ECHO, 1, b"up"))
    assert r.code == Code.ACK and r.payload == b"up"


@cocotb.test()
async def echo_many_sizes(dut):
    b = await start(dut)
    rng = random.Random(21)
    sizes = [0, 1, 2, 7, 64, 255, 256] + [rng.randint(0, 256) for _ in range(5)]
    for seq, size in enumerate(sizes):
        payload = rng.randbytes(size)
        r = await b.transact(build_request(Cmd.ECHO, seq, payload))
        assert r.code == Code.ACK, size
        assert r.seq == seq
        assert r.payload == payload, size
    assert b.led_hi() == len(sizes)          # good-packet counter
    assert b.sink.framing_errors == 0


@cocotb.test()
async def bad_crc_is_nakked(dut):
    b = await start(dut)
    pkt = bytearray(build_request(Cmd.ECHO, 9, b"payload"))
    pkt[-1] ^= 0x01
    r = await b.transact(bytes(pkt))
    assert (r.code, r.seq, r.payload) == (Code.BAD_CRC, 9, b"")
    await ClockCycles(dut.clk, 2)
    assert b.led(4) == 1
    r = await b.transact(build_request(Cmd.ECHO, 10, b"ok"))  # recovers
    assert r.code == Code.ACK and r.payload == b"ok"


@cocotb.test()
async def unknown_command_is_nakked(dut):
    b = await start(dut)
    r = await b.transact(build_request(Cmd.INFO, 3, b""))
    assert (r.code, r.payload) == (Code.UNKNOWN_CMD, b"")
    await ClockCycles(dut.clk, 2)
    assert b.led(7) == 1


@cocotb.test()
async def too_long_is_nakked(dut):
    b = await start(dut)
    r = await b.transact(build_request(Cmd.ECHO, 4, bytes(300)))
    assert (r.code, r.payload) == (Code.TOO_LONG, b"")
    r = await b.transact(build_request(Cmd.ECHO, 5, b"after"))
    assert r.code == Code.ACK and r.payload == b"after"


@cocotb.test()
async def partial_packet_times_out(dut):
    b = await start(dut)
    pkt = build_request(Cmd.ECHO, 6, b"abcdef")
    await b.src.send(pkt[:7])                       # stop mid-payload
    await b.src.idle(TIMEOUT + 50)
    assert len(b.sink.data) == 0, "no reply to a timed-out request"
    assert b.led(6) == 1
    r = await b.transact(build_request(Cmd.ECHO, 7, b"next"))
    assert r.code == Code.ACK and r.payload == b"next"


@cocotb.test()
async def garbage_before_sync_is_ignored(dut):
    b = await start(dut)
    rng = random.Random(22)
    junk = bytes(x for x in rng.randbytes(40) if x != SYNC_REQ)
    await b.src.send(junk)
    r = await b.transact(build_request(Cmd.ECHO, 8, b"clean"))
    assert r.code == Code.ACK and r.payload == b"clean"


@cocotb.test()
async def framing_error_lights_led(dut):
    b = await start(dut)
    await FallingEdge(dut.clk)
    await b.src.send_byte(0x33, stop=0)
    await b.src.idle(3 * C)
    assert b.led(5) == 1
    r = await b.transact(build_request(Cmd.ECHO, 11, b"x"))
    assert r.code == Code.ACK


@cocotb.test()
async def raw_mode_echoes_bytes(dut):
    b = await start(dut, sw=0b01)
    await ClockCycles(dut.clk, 5)
    assert b.led(3) == 1
    data = random.Random(23).randbytes(100)
    await b.src.send(data)
    assert await b.sink.read(len(data), REPLY_WAIT) == data


@cocotb.test()
async def error_counter_and_reset_button(dut):
    b = await start(dut)
    pkt = bytearray(build_request(Cmd.ECHO, 1, b"z"))
    pkt[-2] ^= 0xFF
    for _ in range(3):
        await b.transact(bytes(pkt))
    await b.transact(build_request(Cmd.ECHO, 2, b"z"))
    await ClockCycles(dut.clk, 2)
    assert b.led_hi() == 1                          # good count
    dut.sw.value = 0b10
    await ClockCycles(dut.clk, 5)
    assert b.led_hi() == 3                          # error count
    assert b.led(4) == 1
    await press_reset(dut)
    assert b.led_hi() == 0 and b.led(4) == 0


@cocotb.test()
async def leds_heartbeat_and_activity(dut):
    b = await start(dut)
    seen = set()
    for _ in range(2 ** (BLINK_BIT + 2)):
        await FallingEdge(dut.clk)
        seen.add(b.led(0))
    assert seen == {0, 1}, "heartbeat LED must toggle"
    task = cocotb.start_soon(b.src.send(build_request(Cmd.ECHO, 1, b"act")))
    rx_on = tx_on = False
    for _ in range(40 * 10 * C):
        await FallingEdge(dut.clk)
        rx_on |= bool(b.led(1))
        tx_on |= bool(b.led(2))
    await task
    assert rx_on and tx_on
    assert b.led(1) == 0 and b.led(2) == 0, "activity LEDs go off again"


def test_loopback_top():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"loopback_top_N{N}"
    rtl = ["reset_gen", "uart_rx", "uart_tx", "sync_fifo", "crc16_ccitt", "loopback_echo", "loopback_top"]
    runner.build(
        sources=[ROOT / "rtl" / f"{m}.sv" for m in rtl],
        hdl_toplevel="loopback_top",
        parameters={"CLKS_PER_BIT": C, "TIMEOUT_CYCLES": TIMEOUT,
                    "LED_STRETCH_BITS": LED_STRETCH_BITS, "BLINK_BIT": BLINK_BIT},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        hdl_toplevel="loopback_top",
        test_module="test_loopback",
        test_dir=Path(__file__).parent,
        results_xml=str(build_dir / "results.xml"),
    )

