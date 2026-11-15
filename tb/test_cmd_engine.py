"""cocotb tests for rtl/cmd_engine.sv (task A4.1): the host protocol on the real tpu_core.

tb/hdl/engine_tb.sv connects cmd_engine to tpu_core without the UART, so requests are fed as
bytes and replies collected as bytes, at full speed. Requests are built and replies parsed with
the host's own tinytpu.packet; memory contents use the host's wire formats (tinytpu.isa
ptab_entry, tile bytes, ...); programs are checked against tinytpu.model.Machine. Every rule of
docs/uart_protocol.md is exercised: each command, each NAK code and their priority, write-through
with the PTAB exception, duplicates, pending RUN / INFER replies with STATUS, ABORT and RESET
while running, the inter-byte timeout, and noise between packets.
"""

from __future__ import annotations

import os
import random
import struct
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

from tinytpu import isa, model
from tinytpu.isa import AccMode, Mem, SeqState
from tinytpu.packet import Cmd, Code, build_request, packet_length, parse_reply

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 64, Mem.ABUF: 64, Mem.ACC: 32, Mem.PTAB: 64}
TIMEOUT = 300
UNIT = {Mem.IMEM: 16, Mem.WBUF: N * N, Mem.ABUF: N, Mem.ACC: 4 * N, Mem.PTAB: 8}


def mem_args(mem: Mem, addr: int, count: int) -> bytes:
    return struct.pack("<BBHH", int(mem), 0, addr, count)


class Dev:
    """Byte-level host side: sends requests, collects replies."""

    def __init__(self, dut, seed: int):
        self.dut = dut
        self.rng = random.Random(seed)
        self.rx: bytearray = bytearray()
        self.seq = 0
        self.gap = (0, 3)                       # idle cycles between request bytes
        cocotb.start_soon(self._collect())

    async def _collect(self) -> None:
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            d.tx_ready.value = int(self.rng.random() < 0.6)
            await RisingEdge(d.clk)
            if d.tx_valid.value == 1 and d.tx_ready.value == 1:
                self.rx.append(int(d.tx_data.value))

    async def send(self, data: bytes) -> None:
        d = self.dut
        for b in data:
            await FallingEdge(d.clk)
            d.rx_valid.value, d.rx_data.value = 1, b
            await FallingEdge(d.clk)
            d.rx_valid.value = 0
            for _ in range(self.rng.randint(*self.gap)):
                await FallingEdge(d.clk)

    def next_seq(self) -> int:
        self.seq = (self.seq + 1) & 0xFF
        return self.seq

    async def reply(self, timeout: int = 200000):
        """Wait for one complete reply; return it parsed."""
        for _ in range(timeout):
            if len(self.rx) >= 5 and len(self.rx) >= packet_length(bytes(self.rx[:5])):
                n = packet_length(bytes(self.rx[:5]))
                frame = bytes(self.rx[:n])
                del self.rx[:n]
                return parse_reply(frame)
            await FallingEdge(self.dut.clk)
        raise AssertionError(f"no reply (got {bytes(self.rx).hex(' ')})")

    async def no_reply(self, cycles: int = 2000) -> None:
        for _ in range(cycles):
            await FallingEdge(self.dut.clk)
        assert not self.rx, f"unexpected bytes {bytes(self.rx).hex(' ')}"

    async def cmd(self, cmd: Cmd, payload: bytes = b"", seq: int | None = None, expect: Code = Code.ACK):
        seq = self.next_seq() if seq is None else seq
        await self.send(build_request(cmd, seq, payload))
        r = await self.reply()
        assert r.seq == seq, f"reply SEQ {r.seq} != {seq}"
        assert r.code == expect, f"{cmd.name}: reply {r.code!r}, want {expect!r}"
        return r

    # Memory helpers in wire units ------------------------------------------------------------

    async def write(self, mem: Mem, addr: int, data: bytes, expect: Code = Code.ACK, seq: int | None = None):
        assert len(data) % UNIT[mem] == 0
        return await self.cmd(Cmd.WRITE, mem_args(mem, addr, len(data) // UNIT[mem]) + data, seq=seq, expect=expect)

    async def read(self, mem: Mem, addr: int, count: int) -> bytes:
        r = await self.cmd(Cmd.READ, mem_args(mem, addr, count))
        assert len(r.payload) == count * UNIT[mem]
        return r.payload

    async def status(self) -> tuple[int, int, int, int]:
        r = await self.cmd(Cmd.STATUS)
        return struct.unpack("<BBHH", r.payload[:6])


def unpack_status(payload: bytes) -> tuple[int, int, int, int]:
    assert len(payload) >= 8 and payload[6:8] == b"\x00\x00"
    return struct.unpack("<BBHH", payload[:6])


async def setup(dut, seed: int) -> Dev:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.rx_valid.value = 0
    dut.rx_data.value = 0
    dut.tx_ready.value = 0
    dev = Dev(dut, seed)
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    return dev


# Mirror of the device in the model, loaded through WRITE commands.

def wire(mach: model.Machine, mem: Mem, unit: int) -> bytes:
    if mem == Mem.IMEM:
        return isa.to_bytes(int(mach.imem[unit]))
    if mem == Mem.WBUF:
        return mach.wbuf[unit].astype(np.int8).tobytes()
    if mem == Mem.ABUF:
        return mach.abuf[unit].astype(np.uint8).tobytes()
    if mem == Mem.ACC:
        return mach.acc[unit].astype("<i4").tobytes()
    return isa.ptab_entry(int(mach.ptab_bias[unit]), int(mach.ptab_mult[unit]), int(mach.ptab_shift[unit]))


async def upload(dev: Dev, mach: model.Machine, mem: Mem, chunk: int = 9) -> None:
    for a in range(0, DEPTHS[mem], chunk):
        n = min(chunk, DEPTHS[mem] - a)
        await dev.write(mem, a, b"".join(wire(mach, mem, a + i) for i in range(n)))


def random_machine(rng: random.Random) -> model.Machine:
    mach = model.Machine(n=N, depths=dict(DEPTHS))
    mach.wbuf[:] = np.array([[[rng.randint(-127, 127) for _ in range(N)] for _ in range(N)]
                             for _ in range(DEPTHS[Mem.WBUF])], dtype=np.int8)
    mach.abuf[:] = np.array([[rng.randrange(256) for _ in range(N)] for _ in range(DEPTHS[Mem.ABUF])], dtype=np.uint8)
    mach.acc[:] = np.array([[rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(N)]
                            for _ in range(DEPTHS[Mem.ACC])], dtype=np.int32)
    mach.ptab_bias[:] = [rng.randint(-(1 << 20), 1 << 20) for _ in range(DEPTHS[Mem.PTAB])]
    mach.ptab_mult[:] = [rng.randrange(1, 1 << 16) for _ in range(DEPTHS[Mem.PTAB])]
    mach.ptab_shift[:] = [rng.randint(10, 24) for _ in range(DEPTHS[Mem.PTAB])]
    return mach


async def upload_all(dev: Dev, mach: model.Machine) -> None:
    for mem in (Mem.IMEM, Mem.WBUF, Mem.ABUF, Mem.ACC, Mem.PTAB):
        await upload(dev, mach, mem)


async def check_all(dev: Dev, mach: model.Machine) -> None:
    for mem in (Mem.IMEM, Mem.WBUF, Mem.ABUF, Mem.ACC, Mem.PTAB):
        per = max(1, 4096 // UNIT[mem])
        for a in range(0, DEPTHS[mem], per):
            n = min(per, DEPTHS[mem] - a)
            got = await dev.read(mem, a, n)
            want = b"".join(wire(mach, mem, a + i) for i in range(n))
            assert got == want, f"{mem.name} {a}..{a + n - 1} differs"


def long_program() -> list[int]:
    """A program that runs for a while (many accumulating MATMULs) and halts with code 9."""
    mm = isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=4, nt=4, m=4, act_signed=True,
                    acc_mode=AccMode.ACCUMULATE)
    return [mm] * 8 + [isa.halt(9)]


# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------

@cocotb.test()
async def info_block(dut):
    dev = await setup(dut, 1)
    r = await dev.cmd(Cmd.INFO)
    fields = struct.unpack("<BBHHHHHHH", r.payload)
    assert fields == (1, N, 100, DEPTHS[Mem.IMEM], DEPTHS[Mem.WBUF], DEPTHS[Mem.ABUF], DEPTHS[Mem.ACC],
                      DEPTHS[Mem.PTAB], 4096), fields
    assert unpack_status((await dev.cmd(Cmd.STATUS)).payload) == (SeqState.IDLE, 0, 0, 0)


@cocotb.test()
async def write_read_every_memory(dut):
    dev = await setup(dut, 2)
    mach = random_machine(dev.rng)
    for i in range(DEPTHS[Mem.IMEM]):
        mach.imem[i] = dev.rng.getrandbits(128)
    await upload_all(dev, mach)
    await check_all(dev, mach)


@cocotb.test()
async def nak_codes_and_priority(dut):
    dev = await setup(dut, 3)
    mach = random_machine(dev.rng)
    await upload(dev, mach, Mem.ABUF)
    good = mem_args(Mem.ABUF, 0, 1) + bytes(N)
    # 0x01 bad CRC (and nothing else matters).
    pkt = bytearray(build_request(Cmd.STATUS, 7))
    pkt[-1] ^= 0x10
    await dev.send(bytes(pkt))
    assert (await dev.reply()).code == Code.BAD_CRC
    # 0x02 unknown command (also with a payload, and the old bring-up ECHO).
    await dev.cmd(0x09, expect=Code.UNKNOWN_CMD)
    await dev.cmd(Cmd.ECHO, b"hello", expect=Code.UNKNOWN_CMD)
    # 0x04 payload too long beats bad length.
    await dev.cmd(Cmd.STATUS, bytes(4097), expect=Code.TOO_LONG)
    # 0x03 bad length.
    await dev.cmd(Cmd.STATUS, b"\x00", expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.RUN, b"\x00", expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.READ, mem_args(Mem.ABUF, 0, 1) + b"\x00", expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.WRITE, good + b"\x00", expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.WRITE, good[:-1], expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.WRITE, b"\x02\x00\x00", expect=Code.BAD_LENGTH)
    await dev.cmd(Cmd.INFER, struct.pack("<HHBBHH", 0, 0, 2, 0, 0, 1) + bytes(N + 1), expect=Code.BAD_LENGTH)
    # 0x05 bad argument, and it writes nothing.
    junk = bytes([0xEE] * N)
    for args in (mem_args(5, 0, 1), struct.pack("<BBHH", 2, 1, 0, 1), mem_args(Mem.ABUF, 0, 0),
                 mem_args(Mem.ABUF, DEPTHS[Mem.ABUF], 1), mem_args(Mem.ABUF, DEPTHS[Mem.ABUF] - 1, 2)):
        cnt = struct.unpack("<H", args[4:6])[0]
        await dev.cmd(Cmd.WRITE, args + junk * cnt, expect=Code.BAD_ARG)
    assert await dev.read(Mem.ABUF, 0, DEPTHS[Mem.ABUF]) == b"".join(wire(mach, Mem.ABUF, a) for a in range(DEPTHS[Mem.ABUF]))
    await dev.cmd(Cmd.READ, mem_args(Mem.ACC, 0, DEPTHS[Mem.ACC] + 1), expect=Code.BAD_ARG)
    big = 4096 // UNIT[Mem.WBUF] + 1                       # reply over 4096 bytes
    if big <= DEPTHS[Mem.WBUF]:
        await dev.cmd(Cmd.READ, mem_args(Mem.WBUF, 0, big), expect=Code.BAD_ARG)
    await dev.cmd(Cmd.RUN, struct.pack("<H", DEPTHS[Mem.IMEM]), expect=Code.BAD_ARG)
    await dev.cmd(Cmd.INFER, struct.pack("<HHBBHH", 0, 0, 7, 0, 0, 1), expect=Code.BAD_ARG)
    await dev.cmd(Cmd.INFER, struct.pack("<HHBBHH", DEPTHS[Mem.ABUF], 0, 2, 0, 0, 1) + bytes(N), expect=Code.BAD_ARG)
    # Nothing ran.
    assert unpack_status((await dev.cmd(Cmd.STATUS)).payload)[0] == SeqState.IDLE


@cocotb.test()
async def ptab_reserved_bits(dut):
    dev = await setup(dut, 4)
    mach = random_machine(dev.rng)
    await upload(dev, mach, Mem.PTAB)
    entries = [isa.ptab_entry(100 + i, 200 + i, i) for i in range(4)]
    bad = bytearray(entries[1])
    bad[6] |= 0x40                                          # reserved shift bit
    bad2 = bytearray(entries[3])
    bad2[7] = 1                                             # pad byte
    await dev.write(Mem.PTAB, 10, entries[0] + bytes(bad) + entries[2] + bytes(bad2), expect=Code.BAD_ARG)
    got = await dev.read(Mem.PTAB, 10, 4)
    assert got[0:8] == entries[0] and got[16:24] == entries[2]            # good entries written
    assert got[8:16] == wire(mach, Mem.PTAB, 11) and got[24:32] == wire(mach, Mem.PTAB, 13)  # bad ones not


@cocotb.test()
async def run_program_and_infer(dut):
    """A two-layer network uploaded with WRITE, run with RUN, then again with INFER."""
    dev = await setup(dut, 5)
    r = dev.rng
    mach = random_machine(r)
    k1, h, o = 3 * N + 1, 2 * N - 1, N + 1
    kt1, nt1, nt2 = model.ceil_div(k1, N), model.ceil_div(h, N), model.ceil_div(o, N)
    w1 = np.array([[r.randint(-127, 127) for _ in range(h)] for _ in range(k1)])
    w2 = np.array([[r.randint(-127, 127) for _ in range(o)] for _ in range(h)])
    t1, t2 = model.pack_weights(w1, N), model.pack_weights(w2, N)
    mach.load_weights(t1, 0)
    mach.load_weights(t2, len(t1))
    prog = [
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=kt1, nt=nt1, m=1, act_signed=False, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=20, acc_base=0, p_base=0, nt=nt1, m=1, relu=True),
        isa.matmul(w_base=len(t1), a_base=20, acc_base=8, p_base=nt1 * N, kt=nt1, nt=nt2, m=1, act_signed=False,
                   acc_mode=AccMode.BIAS),
        isa.requant(a_dst=30, acc_base=8, p_base=nt1 * N, nt=nt2, m=1, relu=False),
        isa.halt(3),
    ]
    mach.load_program(prog)
    await upload_all(dev, mach)

    # RUN: the reply comes when the program halts.
    x = np.array([[r.randrange(256) for _ in range(k1)]])
    xv = model.pack_activations(x, N)
    await dev.write(Mem.ABUF, 0, xv.tobytes())
    mach.load_vectors(xv, 0)
    rr = await dev.cmd(Cmd.RUN, struct.pack("<H", 0))
    st = mach.run(0)
    assert unpack_status(rr.payload) == (st.state, st.err, st.pc, st.halt_code) == (SeqState.HALTED_OK, 0, 4, 3)
    await check_all(dev, mach)

    # INFER: input, run and output in one round trip.
    x2 = np.array([[r.randrange(256) for _ in range(k1)]])
    xv2 = model.pack_activations(x2, N)
    mach.load_vectors(xv2, 0)
    st = mach.run(0)
    ri = await dev.cmd(Cmd.INFER, struct.pack("<HHBBHH", 0, 0, int(Mem.ABUF), 0, 30, nt2) + xv2.tobytes())
    assert unpack_status(ri.payload) == (st.state, st.err, st.pc, st.halt_code)
    assert ri.payload[8:] == mach.abuf[30:30 + nt2].tobytes()
    await check_all(dev, mach)


@cocotb.test()
async def duplicates(dut):
    dev = await setup(dut, 6)
    mach = random_machine(dev.rng)
    await upload_all(dev, mach)
    # WRITE: the resend is ACKed but not written.
    a, b = bytes([1] * N), bytes([2] * N)
    await dev.write(Mem.ABUF, 5, a, seq=40)
    await dev.write(Mem.ABUF, 5, b, seq=40)
    assert await dev.read(Mem.ABUF, 5, 1) == a
    # RUN of an accumulating program: the resend after it halted only reports the status.
    prog = [isa.matmul(w_base=0, a_base=0, acc_base=0, kt=1, nt=1, m=1, act_signed=True, acc_mode=AccMode.ACCUMULATE),
            isa.halt(5)]
    await dev.write(Mem.IMEM, 0, b"".join(isa.to_bytes(w) for w in prog))
    r1 = await dev.cmd(Cmd.RUN, b"\x00\x00", seq=41)
    acc_after = await dev.read(Mem.ACC, 0, 1)
    r2 = await dev.cmd(Cmd.RUN, b"\x00\x00", seq=41)
    assert r2.payload == r1.payload and await dev.read(Mem.ACC, 0, 1) == acc_after
    # A different SEQ runs it again.
    await dev.cmd(Cmd.RUN, b"\x00\x00", seq=42)
    assert await dev.read(Mem.ACC, 0, 1) != acc_after
    # Polling STATUS between does not hide the duplicate: the resend does not run it again.
    acc_42 = await dev.read(Mem.ACC, 0, 1)
    await dev.cmd(Cmd.STATUS)
    await dev.cmd(Cmd.RUN, b"\x00\x00", seq=42)
    assert await dev.read(Mem.ACC, 0, 1) == acc_42
    # RUN resent while still running: no extra reply; exactly one reply when it halts.
    await dev.write(Mem.IMEM, 0, b"".join(isa.to_bytes(w) for w in long_program()))
    await dev.send(build_request(Cmd.RUN, 43, b"\x00\x00"))
    await dev.send(build_request(Cmd.RUN, 43, b"\x00\x00"))
    r = await dev.reply()
    assert r.seq == 43 and unpack_status(r.payload)[0] == SeqState.HALTED_OK
    await dev.no_reply()


@cocotb.test()
async def infer_duplicate_answers_from_the_original(dut):
    """A duplicate INFER (same CMD and SEQ) is answered from the original's output region, even
    if its own payload names another, or an invalid, region (found by the formal proof)."""
    dev = await setup(dut, 9)
    mach = random_machine(dev.rng)
    mach.load_program([isa.halt(1)])
    await upload_all(dev, mach)
    args = struct.pack("<HHBBHH", 0, 0, int(Mem.ABUF), 0, 3, 2) + bytes(range(N))
    first = await dev.cmd(Cmd.INFER, args, seq=90)
    assert len(first.payload) == 8 + 2 * N
    for other in (struct.pack("<HHBBHH", 0, 0, 7, 0, 0, 0xFFFF),               # invalid region
                  struct.pack("<HHBBHH", 0, 0, int(Mem.ACC), 0, 0, 4),         # another valid one
                  struct.pack("<HHBBHH", 0, 0, int(Mem.ABUF), 0, 3, 2)):       # identical
        again = await dev.cmd(Cmd.INFER, other + bytes(range(N)), seq=90)
        assert again.payload == first.payload


@cocotb.test()
async def busy_status_abort_reset(dut):
    dev = await setup(dut, 7)
    mach = random_machine(dev.rng)
    mach.load_program(long_program())
    await upload_all(dev, mach)
    # STATUS and INFO while running get their own replies; busy commands are refused.
    await dev.send(build_request(Cmd.RUN, 50, b"\x00\x00"))
    for _ in range(50):
        await FallingEdge(dut.clk)
    st = unpack_status((await dev.cmd(Cmd.STATUS, seq=51)).payload)
    assert st[0] == SeqState.RUNNING
    await dev.cmd(Cmd.INFO, seq=52)
    await dev.cmd(Cmd.WRITE, mem_args(Mem.ABUF, 0, 1) + bytes(N), seq=53, expect=Code.BUSY)
    await dev.cmd(Cmd.READ, mem_args(Mem.ABUF, 0, 1), seq=54, expect=Code.BUSY)
    await dev.cmd(Cmd.RUN, b"\x00\x00", seq=55, expect=Code.BUSY)
    r = await dev.reply()                                   # then the RUN reply
    assert r.seq == 50 and unpack_status(r.payload)[0] == SeqState.HALTED_OK

    # ABORT while running: its reply replaces the RUN reply.
    await dev.send(build_request(Cmd.RUN, 60, b"\x00\x00"))
    for _ in range(80):
        await FallingEdge(dut.clk)
    ra = await dev.cmd(Cmd.ABORT, seq=61)
    s = unpack_status(ra.payload)
    assert s[0] == SeqState.IDLE and s[1] == 0
    await dev.no_reply(4000)
    assert unpack_status((await dev.cmd(Cmd.ABORT)).payload)[0] == SeqState.IDLE   # idempotent

    # RESET while running: likewise, and memory is kept.
    await dev.send(build_request(Cmd.RUN, 70, b"\x00\x00"))
    for _ in range(80):
        await FallingEdge(dut.clk)
    rr = await dev.cmd(Cmd.RESET, seq=71)
    assert unpack_status(rr.payload) == (SeqState.IDLE, 0, 0, 0)
    await dev.no_reply(4000)
    assert await dev.read(Mem.IMEM, 0, len(long_program())) == b"".join(isa.to_bytes(w) for w in long_program())


@cocotb.test()
async def timeout_and_noise(dut):
    dev = await setup(dut, 8)
    # Noise before a packet is ignored.
    await dev.send(bytes([0x00, 0x5A, 0xFF, 0x13]))
    await dev.cmd(Cmd.STATUS)
    # A partial request is dropped after the timeout, without a reply.
    pkt = build_request(Cmd.WRITE, 80, mem_args(Mem.ABUF, 0, 1) + bytes(N))
    await dev.send(pkt[:7])
    await dev.no_reply(TIMEOUT + 200)
    await dev.cmd(Cmd.STATUS, seq=81)
    # Back-to-back bytes (no gaps) work too.
    dev.gap = (0, 0)
    await dev.cmd(Cmd.INFO)
    await dev.write(Mem.ABUF, 3, bytes(range(N)))
    assert await dev.read(Mem.ABUF, 3, 1) == bytes(range(N))


def test_cmd_engine():
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"cmd_engine_N{N}"
    rtl = ["isa_pkg.sv", "sync_fifo.sv", "crc16_ccitt.sv", "buffer_ram.sv", "pe.sv", "systolic_array.sv",
           "acc_unit.sv", "requant.sv", "instr_decode.sv", "range_check.sv", "sequencer.sv", "matmul_ctrl.sv",
           "requant_ctrl.sv", "tpu_core.sv", "cmd_engine.sv"]
    runner.build(
        sources=[ROOT / "rtl" / f for f in rtl] + [ROOT / "tb" / "hdl" / "engine_tb.sv"],
        hdl_toplevel="engine_tb",
        parameters={"N": N, "IMEM_DEPTH": DEPTHS[Mem.IMEM], "WBUF_DEPTH": DEPTHS[Mem.WBUF], "ABUF_DEPTH": DEPTHS[Mem.ABUF],
                    "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB], "TIMEOUT_CYCLES": TIMEOUT},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="engine_tb", test_module="test_cmd_engine", test_dir=Path(__file__).parent,
                results_xml=str(build_dir / "results.xml"))
