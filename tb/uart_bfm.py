"""UART bus-functional models for cocotb testbenches: drive and decode 8N1 serial lines.

Timing is counted in clock cycles of the DUT clock, so the models work at any CLKS_PER_BIT.
"""

from __future__ import annotations

import cocotb
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge


class UartSource:
    """Drives a serial line (e.g. the DUT's RsRx), changing levels on falling clock edges."""

    def __init__(self, clk, line, bit_cycles: int):
        self.clk, self.line, self.bit_cycles = clk, line, bit_cycles
        self.line.value = 1

    async def _bit(self, level: int, cycles: int | None = None) -> None:
        self.line.value = level
        await ClockCycles(self.clk, cycles or self.bit_cycles, rising=False)

    async def send_byte(self, byte: int, stop: int = 1) -> None:
        """Send one frame. ``stop=0`` forces a framing error."""
        await self._bit(0)
        for i in range(8):
            await self._bit((byte >> i) & 1)
        await self._bit(stop)
        self.line.value = 1

    async def send(self, data: bytes) -> None:
        """Send bytes back to back."""
        await FallingEdge(self.clk)
        for b in data:
            await self.send_byte(b)

    async def idle(self, cycles: int) -> None:
        self.line.value = 1
        await ClockCycles(self.clk, cycles, rising=False)


class UartSink:
    """Decodes a serial line (e.g. the DUT's RsTx) into bytes, sampling at mid-bit."""

    def __init__(self, clk, line, bit_cycles: int):
        self.clk, self.line, self.bit_cycles = clk, line, bit_cycles
        self.data = bytearray()
        self.framing_errors = 0
        cocotb.start_soon(self._run())

    async def _run(self) -> None:
        c = self.bit_cycles
        while True:
            await RisingEdge(self.clk)
            if self.line.value != 0:
                continue
            await ClockCycles(self.clk, c // 2)
            if self.line.value != 0:
                continue  # glitch
            byte = 0
            for i in range(8):
                await ClockCycles(self.clk, c)
                byte |= int(self.line.value) << i
            await ClockCycles(self.clk, c)
            if self.line.value != 1:
                self.framing_errors += 1
            self.data.append(byte)

    async def read(self, n: int, timeout_cycles: int) -> bytes:
        """Wait for and remove the first ``n`` bytes; raise if they don't arrive in time."""
        for _ in range(timeout_cycles):
            if len(self.data) >= n:
                break
            await RisingEdge(self.clk)
        if len(self.data) < n:
            raise TimeoutError(f"expected {n} bytes, got {len(self.data)}: {bytes(self.data).hex(' ')}")
        out = bytes(self.data[:n])
        del self.data[:n]
        return out

    async def read_packet(self, timeout_cycles: int) -> bytes:
        """Wait for one complete packet (header, payload, CRC) and return its bytes."""
        header = await self.read(5, timeout_cycles)
        length = int.from_bytes(header[3:5], "little")
        return header + await self.read(length + 2, timeout_cycles + (length + 2) * 10 * self.bit_cycles)
