"""End-to-end tests of rtl/tinytpu_top.sv (task A5.1), through the board pins only.

The host side is the real host code: tinytpu.device.Device on a tinytpu.link.Link, run as
ordinary blocking Python in a cocotb bridge thread (cocotb.task.bridge). The link's serial port
is SimSerial, which drives RsRx and decodes RsTx with the UART models of tb/uart_bfm.py, and its
clock is the simulation time, so the link's timeouts, quiet times and resends behave exactly as
on the board, only at simulated speed. Expected results come from tinytpu.model (the plain
matmul / requant math and Machine).

Tests: INFO; a two-layer network with sizes that are not multiples of N, in both activation
modes, uploaded with WRITE, run with RUN and with INFER; a model compiled with tinytpu.compiler
and run through its runtime; the MNIST demo's board half on synthetic digits; random matrices;
the board self-test
script (tinytpu.selftest) itself; RESET while running; and the link's recovery through the real
board (a corrupted request, a lost reply). They run with
CLKS_PER_BIT = 8 for speed, and a shorter set at the real CLKS_PER_BIT = 100.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.simtime import get_sim_time
from cocotb.task import bridge, resume
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb_tools.runner import get_runner

from tinytpu import compiler, example, isa, mnist, model, nn, quant, selftest
from tinytpu.device import Device
from tinytpu.isa import AccMode, Mem, SeqErr, SeqState
from tinytpu.link import Link
from uart_bfm import UartSink, UartSource
from test_mnist import quadrant_images

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("TINYTPU_N", "4"))
CLKS = int(os.environ.get("TOP_CLKS", "8"))
DEPTHS = {Mem.IMEM: 16, Mem.WBUF: 64, Mem.ABUF: 64, Mem.ACC: 32, Mem.PTAB: 64}
PERIOD_NS = 10
BYTE_S = 10 * CLKS * PERIOD_NS * 1e-9              # one byte on the line, in seconds
TIMEOUT_CYCLES = 40 * 10 * CLKS                    # engine inter-byte timeout: 40 byte times


class SimSerial:
    """pyserial-like port on the simulated board pins (blocking; call from a bridge thread).

    ``drop_reply`` / ``corrupt_request`` inject one fault each when set (for recovery tests).
    """

    def __init__(self, dut):
        self.dut = dut
        self.src = UartSource(dut.clk, dut.RsRx, CLKS)
        self.sink = UartSink(dut.clk, dut.RsTx, CLKS)
        self.timeout: float | None = 0.0
        self.writes = 0
        self.corrupt_request = False
        self.drop_reply = False

    @resume
    async def _send(self, data: bytes) -> None:
        await self.src.send(data)

    @resume
    async def _wait(self, size: int, seconds: float) -> bytes:
        cycles = max(1, int(seconds / (PERIOD_NS * 1e-9)))
        step = 5 * CLKS
        waited = 0
        while len(self.sink.data) < size and waited < cycles:
            await ClockCycles(self.dut.clk, step)
            waited += step
        out = bytes(self.sink.data[:size])
        del self.sink.data[:size]
        return out

    def write(self, data: bytes) -> int:
        self.writes += 1
        data = bytearray(data)
        if self.corrupt_request:
            self.corrupt_request = False
            data[len(data) // 2] ^= 0x04
        if self.drop_reply:                       # this request's reply will be lost
            self.drop_reply = False
            self._send(bytes(data))
            self._wait(1 << 16, 300 * BYTE_S)    # swallow whatever comes back
            return len(data)
        self._send(bytes(data))
        return len(data)

    def read(self, size: int = 1) -> bytes:
        return self._wait(size, self.timeout or 0.0)

    def close(self) -> None:
        pass


def sim_seconds() -> float:
    return get_sim_time("sec")


def make_device(dut) -> tuple[Device, SimSerial]:
    port = SimSerial(dut)
    # Timeouts in simulated seconds, scaled to the bit rate: room for the protocol's largest reply
    # (4096-byte payload + 7 framing bytes; the self-test's WBUF read is 1.5 KB at N = 8), and a
    # quiet time longer than the board's inter-byte timeout.
    link = Link(port, timeout=5000 * BYTE_S, silence=2 * TIMEOUT_CYCLES * PERIOD_NS * 1e-9, clock=sim_seconds)
    return Device(link, run_timeout=20000 * BYTE_S), port


async def setup(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, PERIOD_NS, unit="ns").start())
    dut.btnC.value = 1
    dut.RsRx.value = 1
    dut.sw.value = 0
    await ClockCycles(dut.clk, 5)
    await FallingEdge(dut.clk)
    dut.btnC.value = 0
    await ClockCycles(dut.clk, 20)


# ------------------------------------------------------------------------------------------------
# Host-side scenarios (blocking code, as a user would write it)
# ------------------------------------------------------------------------------------------------

def host_network(dev: Device, seed: int, act_signed: bool) -> None:
    rng = random.Random(seed)
    info = dev.info()
    assert info.n == N and info.depths == DEPTHS and info.clks_per_bit == CLKS
    k1, o, nt2, tiles, params, prog, reference = selftest.two_layer_network(rng, N, act_signed)
    dev.load_weights(tiles)
    dev.load_params(*params)
    dev.load_program(prog)
    lo, hi = (-128, 127) if act_signed else (0, 255)
    for _ in range(2):
        x = np.array([[rng.randint(lo, hi) for _ in range(k1)]])
        # RUN: input with WRITE, output with READ.
        dev.load_vectors(model.pack_activations(x, N), 0)
        st = dev.run()
        assert (st.state, st.err, st.pc, st.halt_code) == (SeqState.HALTED_OK, SeqErr.NONE, 4, 7), st
        got = model.unpack_vectors(dev.read_vectors(30, nt2), 1, o, signed=True)
        assert got.tolist() == reference(x).tolist()
        # INFER: all of it in one round trip.
        x2 = np.array([[rng.randint(lo, hi) for _ in range(k1)]])
        st, out = dev.infer(model.pack_activations(x2, N).tobytes(), out_addr=30, out_count=nt2)
        assert st.state == SeqState.HALTED_OK
        got2 = model.unpack_vectors(np.frombuffer(out, dtype=np.uint8).reshape(nt2, N), 1, o, signed=True)
        assert got2.tolist() == reference(x2).tolist()


def host_compiled_model(dev: Device, seed: int) -> None:
    """A float model, quantized and compiled for the board's INFO, loaded and run in batches
    through the compiler's runtime: bit-exact against the same image on model.Machine."""
    rng = np.random.default_rng(seed)
    sizes, relus = [2 * N + 3, N + 5, 7], [True, False]
    net = nn.Model([nn.Dense(rng.normal(0, 0.4, (k, m)), rng.normal(0, 0.1, m), relu=r)
                    for k, m, r in zip(sizes, sizes[1:], relus)])
    q = quant.quantize(net, rng.uniform(0, 1, (200, sizes[0])))
    target = compiler.Target.from_info(dev.info())
    assert compiler.compile_model(q, target).batch == compiler.max_batch(q, target)
    c = compiler.compile_model(q, target, batch=2)
    runner = c.load(dev)
    assert runner.verify()
    x = rng.uniform(0, 1, (5, sizes[0]))                      # three INFERs, the last one padded
    qx = q.quantize_input(x)
    assert np.array_equal(runner.infer_int(qx), c.simulator().infer_int(qx))
    assert np.array_equal(runner.infer_int(qx), q.forward_int(qx))
    assert np.array_equal(runner.infer(x), q.forward(x))


def host_mnist_demo(dev: Device, seed: int) -> None:
    """tinytpu.mnist's board half: train on synthetic 8x8 'digits', compile for INFO, classify."""
    rng = np.random.default_rng(seed)
    side, hidden = (4, 6) if N == 2 else (8, 12)       # at N = 2, 64-12-10 needs 192 of WBUF's 64 tiles
    train_x, train_y = quadrant_images(rng, 400, side)
    test_x, test_y = quadrant_images(rng, 20, side)
    net = mnist.train(mnist.prepare(train_x), train_y, hidden=hidden, epochs=80, out=lambda _: None)
    q = quant.quantize(net, mnist.prepare(train_x[:100]))
    c = compiler.compile_model(q, compiler.Target.from_info(dev.info()))
    runner = c.load(dev)
    assert runner.verify()
    x = mnist.prepare(test_x)
    res = mnist.run_on_board(runner, q, x, progress=c.batch, out=lambda _: None)
    assert res.exact == len(x) and res.machine_ok
    # Exactness is the point; the accuracy shows the model learned (80 epochs: 100% on 5 seeds offline).
    assert mnist.accuracy(q.dequantize(res.acc), test_y) == mnist.accuracy(q.forward(x), test_y) >= 0.9


def host_random_matmuls(dev: Device, seed: int) -> None:
    """Random matrices (any shape that fits), both activation modes, raw INT32 results vs the model."""
    rng = random.Random(seed)
    for i in range(3):
        m_dim, k_dim, n_dim = rng.randint(1, 3), rng.randint(1, 3 * N), rng.randint(1, 2 * N)
        signed = bool(i % 2)
        w = np.array([[rng.randint(-127, 127) for _ in range(n_dim)] for _ in range(k_dim)])
        lo, hi = (-128, 127) if signed else (0, 255)
        x = np.array([[rng.randint(lo, hi) for _ in range(k_dim)] for _ in range(m_dim)])
        kt, nt = model.ceil_div(k_dim, N), model.ceil_div(n_dim, N)
        dev.load_weights(model.pack_weights(w, N))
        dev.load_vectors(model.pack_activations(x, N))
        dev.load_program([isa.matmul(w_base=0, a_base=0, acc_base=0, kt=kt, nt=nt, m=m_dim, act_signed=signed,
                                     acc_mode=AccMode.OVERWRITE), isa.halt()])
        assert dev.run().state == SeqState.HALTED_OK
        got = model.unpack_acc(dev.read_acc(0, m_dim * nt), m_dim, n_dim)
        assert (got == model.matmul(x, w, signed)).all()


def host_recovery(dev: Device, port: SimSerial) -> None:
    """The link's resend rules against the real engine: a corrupted request and a lost reply."""
    dev.info()
    dev.write(Mem.ABUF, 0, bytes(range(N)))
    # Corrupted request: the board answers NAK BAD_CRC and the link resends.
    port.corrupt_request = True
    dev.write(Mem.ABUF, 1, bytes([0x11] * N))
    assert dev.link.stats.nak_bad_crc == 1 and dev.read(Mem.ABUF, 1, 1) == bytes([0x11] * N)
    # Lost reply to a WRITE: the link times out and resends with the same SEQ; the board treats
    # it as a duplicate and ACKs it.
    port.drop_reply = True
    dev.write(Mem.ABUF, 2, bytes([0x22] * N))
    assert dev.link.stats.timeouts == 1 and dev.read(Mem.ABUF, 2, 1) == bytes([0x22] * N)
    # Lost reply to a RUN of an accumulating program: the resend must not run it twice.
    acc0 = [0] * N
    dev.write(Mem.ACC, 0, np.array(acc0, dtype="<i4").tobytes())
    dev.load_weights(np.ones((1, N, N), dtype=np.int8))
    dev.load_vectors(np.ones((1, N), dtype=np.uint8))
    dev.load_program([isa.matmul(w_base=0, a_base=0, acc_base=0, kt=1, nt=1, m=1, act_signed=False,
                                 acc_mode=AccMode.ACCUMULATE), isa.halt()])
    port.drop_reply = True
    assert dev.run(timeout=1000 * BYTE_S).state == SeqState.HALTED_OK   # the program halts at once
    assert dev.link.stats.timeouts == 2
    assert dev.read_acc(0, 1).tolist() == [[N] * N]       # ran once, not twice
    assert dev.status().state == SeqState.HALTED_OK


def host_reset_while_running(dev: Device) -> None:
    """RESET (the command, not btnC) stops a running program and resets the core; memory is kept."""
    from tinytpu.packet import Cmd, build_request
    long = [isa.matmul(w_base=0, a_base=0, acc_base=0, kt=4, nt=4, m=4, act_signed=True,
                       acc_mode=AccMode.ACCUMULATE)] * 12 + [isa.halt(9)]
    dev.load_program(long)
    dev.link.send_raw(build_request(Cmd.RUN, dev.link.next_seq(), b"\x00\x00"))   # no wait for its reply
    assert dev.status().state == SeqState.RUNNING
    st = dev.reset()
    assert (st.state, st.err, st.pc) == (SeqState.IDLE, SeqErr.NONE, 0), st
    assert dev.status().state == SeqState.IDLE                 # the core really stopped
    assert dev.read(Mem.IMEM, 0, len(long)) == b"".join(isa.to_bytes(w) for w in long)
    assert dev.run(start_pc=len(long) - 1).halt_code == 9      # and it runs again normally


# ------------------------------------------------------------------------------------------------
# cocotb tests
# ------------------------------------------------------------------------------------------------

FULL = CLKS <= 16          # at the real CLKS_PER_BIT only INFO and one network run (time)


@cocotb.test()
async def info_and_status(dut):
    await setup(dut)
    dev, _ = make_device(dut)

    @bridge
    def host():
        info = dev.info()
        assert (info.version, info.n, info.clks_per_bit, info.max_payload) == (1, N, CLKS, 4096)
        st = dev.status()
        assert st.state == SeqState.IDLE
        dev.write(Mem.ABUF, 5, bytes(range(N)))
        assert dev.read(Mem.ABUF, 5, 1) == bytes(range(N))

    await host()


@cocotb.test()
async def network_unsigned_input(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_network)(dev, 1, False)


@cocotb.test()
async def compiled_model(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_compiled_model)(dev, 4)


@cocotb.test(skip=not FULL)
async def mnist_demo(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_mnist_demo)(dev, 5)


@cocotb.test(skip=not FULL)
async def network_signed_input(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_network)(dev, 2, True)


@cocotb.test(skip=not FULL)
async def random_matrices(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_random_matmuls)(dev, 3)


@cocotb.test(skip=not FULL)
async def selftest_script(dut):
    """The board self-test (python -m tinytpu.selftest) passes against the simulated board and
    reports a wrong result as FAIL; the example script's matmul (python -m tinytpu.example) is
    right too. The manual checks run with the testbench reading the LEDs and pressing btnC."""
    await setup(dut)
    dev, _ = make_device(dut)
    lines: list[str] = []

    async def push_button() -> None:
        await FallingEdge(dut.clk)
        dut.btnC.value = 1
        await ClockCycles(dut.clk, 5)
        await FallingEdge(dut.clk)
        dut.btnC.value = 0
        await ClockCycles(dut.clk, 20)

    asked: list[str] = []

    async def read_leds() -> int:
        return int(dut.led.value)

    def ask(question: str) -> bool:
        """Answer like a person reading the board, from the LED pins."""
        asked.append(question)
        led = resume(read_leds)()
        code = (led >> 12) & 0xF
        if "led[5] on" in question:
            return bool(led & 0x20) and not led & 0x10 and f"= {code:04b} " in question
        return bool(led & 0x10) and not led & 0x20 and code == 0

    @bridge
    def host() -> None:
        ok = selftest.run_all(dev, seed=5, count=2, rounds=2, out=lines.append, ask=ask,
                              press=resume(push_button))
        assert ok, "\n".join(lines)
        assert sum(line.startswith("PASS") for line in lines) == 8 and len(asked) == 4
        # A wrong answer must fail the LED check.
        failed: list[str] = []
        checks = selftest.Checks(dev, seed=8, out=failed.append)
        assert not checks.run("leds", lambda: checks.leds(lambda q: "led[4] on" in q))
        assert "error 1, error 4, error 6" in failed[0], failed
        # Without a button press the check must fail (the halted program is still reported).
        failed: list[str] = []
        checks = selftest.Checks(dev, seed=7, out=failed.append)
        assert not checks.run("button", lambda: checks.button(lambda: None))
        assert "expected IDLE" in failed[0], failed
        # The example script's function (python -m tinytpu.example) against plain NumPy.
        r = np.random.default_rng(1)
        for signed in (False, True):
            w = r.integers(-127, 128, size=(2 * N + 3, N + 1))
            x = r.integers(-128, 128, size=(3, 2 * N + 3)) if signed else r.integers(0, 256, size=(3, 2 * N + 3))
            assert np.array_equal(example.matmul_on_board(dev, x, w, signed), x @ w)
        # A board giving a wrong result must be reported: pretend the model expects otherwise.
        real = model.matmul
        try:
            model.matmul = lambda *a, **k: real(*a, **k) + 1
            failed: list[str] = []
            checks = selftest.Checks(dev, seed=6, out=failed.append)
            assert not checks.run("matmul", lambda: checks.matmul(1))
            assert failed[0].startswith("FAIL") and "wrong results" in failed[0], failed
        finally:
            model.matmul = real

    await host()
    dut._log.info("\n".join(lines))


@cocotb.test(skip=not FULL)
async def reset_while_running(dut):
    await setup(dut)
    dev, _ = make_device(dut)
    await bridge(host_reset_while_running)(dev)


@cocotb.test(skip=not FULL)
async def link_recovery(dut):
    await setup(dut)
    dev, port = make_device(dut)
    await bridge(host_recovery)(dev, port)


@pytest.mark.parametrize("clks", [8, 100])
def test_tinytpu_top(clks):
    runner = get_runner("verilator")
    build_dir = ROOT / "sim_build" / f"tinytpu_top_c{clks}_N{N}"
    rtl = ["isa_pkg.sv", "reset_gen.sv", "uart_rx.sv", "uart_tx.sv", "sync_fifo.sv", "crc16_ccitt.sv", "buffer_ram.sv",
           "pe.sv", "systolic_array.sv", "acc_unit.sv", "requant.sv", "instr_decode.sv", "range_check.sv",
           "sequencer.sv", "matmul_ctrl.sv", "requant_ctrl.sv", "tpu_core.sv", "cmd_engine.sv", "tinytpu_top.sv"]
    runner.build(
        sources=[ROOT / "rtl" / f for f in rtl],
        hdl_toplevel="tinytpu_top",
        parameters={"N": N, "IMEM_DEPTH": DEPTHS[Mem.IMEM], "WBUF_DEPTH": DEPTHS[Mem.WBUF], "ABUF_DEPTH": DEPTHS[Mem.ABUF],
                    "ACC_DEPTH": DEPTHS[Mem.ACC], "PTAB_DEPTH": DEPTHS[Mem.PTAB], "CLKS_PER_BIT": clks,
                    "TIMEOUT_CYCLES": 40 * 10 * clks, "LED_STRETCH_BITS": 4, "BLINK_BIT": 6},
        build_args=["--assert", "-DASSERTIONS", "-Wall"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(hdl_toplevel="tinytpu_top", test_module="test_tinytpu_top", test_dir=Path(__file__).parent,
                extra_env={"TOP_CLKS": str(clks)}, results_xml=str(build_dir / "results.xml"))
