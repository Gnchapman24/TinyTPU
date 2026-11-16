"""Board self-test for the full TinyTPU design ``tinytpu_top`` (tasks A6.2 and H5.5).

Runs a sequence of checks through :class:`tinytpu.device.Device` and prints PASS/FAIL for each.
Every result is compared bit-exactly with :mod:`tinytpu.model`:

1. ``info``: the INFO block (protocol version, N, ``CLKS_PER_BIT``, memory depths).
2. ``memories``: random data written to every memory and read back.
3. ``matmul``: random matrices, both activation modes, sizes that are not multiples of N
   (raw INT32 results from ACC).
4. ``network``: a two-layer network (MATMUL + REQUANT twice) run with RUN, then with INFER.
5. ``errors``: programs that must halt with ISA errors (illegal opcode, address range, PC
   overrun), and a request the board must refuse (NAK).
6. ``speed``: INFER round trips; reports the time per inference.

With ``--manual``, two interactive checks follow, which need someone at the board:

7. ``leds``: programs halt with ISA errors 1, 4 and 6, then OK; for each, the user confirms that
   ``led[5]`` / ``led[4]`` and the error code on ``led[15:12]`` show it.
8. ``button``: the user presses ``btnC``, which must return the core to idle and keep every
   memory's contents.

**The test overwrites every memory on the board.** Board setup: ``tinytpu_top.bit`` loaded,
freshly reset with ``btnC``.

Windows usage (host venv, see tools/setup.md)::

    uv run --no-default-groups python -m tinytpu.selftest --probe
    uv run --no-default-groups python -m tinytpu.selftest --port COM7
    uv run --no-default-groups python -m tinytpu.selftest --port COM7 --manual
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from collections.abc import Callable

import numpy as np

from tinytpu import isa, model
from tinytpu.device import Device, DeviceError
from tinytpu.isa import AccMode, Mem, SeqErr, SeqState
from tinytpu.link import BAUD, Link, LinkError, find_ports, scaled_timeout
from tinytpu.packet import Cmd, Code


def two_layer_network(rng: random.Random, n: int, act_signed: bool):
    """A small MLP with sizes that are not multiples of ``n``, laid out for the board.

    Returns ``(k_in, n_out, out_vecs, tiles, params, program, reference)``: the input width, the
    output width, the number of output vectors (at ABUF 30), the weight tiles (WBUF 0), the
    per-channel parameters (PTAB 0), the program (IMEM 0), and a function computing the
    expected output for an input matrix with :mod:`tinytpu.model`.
    """
    k1, h, o = 3 * n + 1, 2 * n + 1, n + 1
    w1 = np.array([[rng.randint(-127, 127) for _ in range(h)] for _ in range(k1)])
    w2 = np.array([[rng.randint(-127, 127) for _ in range(o)] for _ in range(h)])
    b1 = np.array([rng.randint(-3000, 3000) for _ in range(h)])
    b2 = np.array([rng.randint(-3000, 3000) for _ in range(o)])
    m1 = np.array([rng.randint(1000, 20000) for _ in range(h)])
    m2 = np.array([rng.randint(1000, 20000) for _ in range(o)])
    s1, s2 = np.full(h, 21), np.full(o, 21)
    kt1, nt1, nt2 = model.ceil_div(k1, n), model.ceil_div(h, n), model.ceil_div(o, n)
    t1, t2 = model.pack_weights(w1, n), model.pack_weights(w2, n)
    relu1 = not act_signed                     # a signed network keeps an INT8 hidden layer
    program = [
        isa.matmul(w_base=0, a_base=0, acc_base=0, p_base=0, kt=kt1, nt=nt1, m=1, act_signed=act_signed,
                   acc_mode=AccMode.BIAS),
        isa.requant(a_dst=20, acc_base=0, p_base=0, nt=nt1, m=1, relu=relu1),
        isa.matmul(w_base=len(t1), a_base=20, acc_base=8, p_base=nt1 * n, kt=nt1, nt=nt2, m=1,
                   act_signed=not relu1, acc_mode=AccMode.BIAS),
        isa.requant(a_dst=30, acc_base=8, p_base=nt1 * n, nt=nt2, m=1, relu=False),
        isa.halt(7),
    ]

    def pad(v, size):
        return np.concatenate([v, np.zeros(size - len(v), dtype=np.int64)])

    params = (np.concatenate([pad(b1, nt1 * n), pad(b2, nt2 * n)]),
              np.concatenate([pad(m1, nt1 * n), pad(m2, nt2 * n)]),
              np.concatenate([pad(s1, nt1 * n), pad(s2, nt2 * n)]))

    def reference(x):
        hid = model.requant(model.matmul(x, w1, act_signed, bias=b1), m1, s1, relu=relu1)
        return model.requant(model.matmul(hid, w2, not relu1, bias=b2), m2, s2, relu=False)

    return k1, o, nt2, np.concatenate([t1, t2]), params, program, reference


class Checks:
    """Runs named checks against one board, records results, and prints a line per check."""

    def __init__(self, dev: Device, seed: int = 0, out=print):
        self.dev = dev
        self.rng = random.Random(seed)
        self.out = out
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn: Callable[[], str]) -> bool:
        try:
            detail = fn() or ""
            ok = True
        except (LinkError, DeviceError, AssertionError, TimeoutError, ValueError) as e:
            detail, ok = f"{type(e).__name__}: {e}", False
        self.results.append((name, ok, detail))
        self.out(f"{'PASS' if ok else 'FAIL'}  {name:<9} {detail}")
        return ok

    @property
    def n(self) -> int:
        return self.dev.info().n

    # ---------------------------------------------------------------------------------------------

    def info(self) -> str:
        i = self.dev.info(refresh=True)
        assert i.version == 1, f"protocol version {i.version}, expected 1"
        assert i.max_payload == 4096, f"maximum payload {i.max_payload}, expected 4096"
        assert i.n >= 2 and i.n & (i.n - 1) == 0, f"N = {i.n} is not a power of two"
        st = self.dev.status()
        depths = ", ".join(f"{m.name} {d}" for m, d in i.depths.items())
        return f"N = {i.n}, CLKS_PER_BIT = {i.clks_per_bit}, {depths}; state {st.state.name}"

    def _pattern(self, mem: Mem, count: int) -> bytes:
        """Random contents for ``count`` units of ``mem`` (PTAB entries keep reserved bits 0)."""
        if mem == Mem.PTAB:
            return b"".join(isa.ptab_entry(self.rng.randint(-(1 << 31), (1 << 31) - 1),
                                           self.rng.randrange(1 << 16), self.rng.randrange(64))
                            for _ in range(count))
        return self.rng.randbytes(count * self.dev.unit_bytes(mem))

    def memories(self, units: int = 24) -> str:
        total = 0
        for mem in (Mem.IMEM, Mem.WBUF, Mem.ABUF, Mem.ACC, Mem.PTAB):
            depth = self.dev.info().depths[mem]
            count = min(units, depth)
            addr = self.rng.randrange(depth - count + 1)
            data = self._pattern(mem, count)
            self.dev.write(mem, addr, data)
            got = self.dev.read(mem, addr, count)
            assert got == data, f"{mem.name} {addr}..{addr + count - 1}: read back differs"
            total += len(data)
        return f"5 memories, {total} bytes written and read back"

    def matmul(self, count: int = 6) -> str:
        n = self.n
        for i in range(count):
            m_dim, k_dim, n_dim = self.rng.randint(1, 3), self.rng.randint(1, 3 * n), self.rng.randint(1, 2 * n)
            signed = bool(i % 2)
            w = np.array([[self.rng.randint(-127, 127) for _ in range(n_dim)] for _ in range(k_dim)])
            lo, hi = (-128, 127) if signed else (0, 255)
            x = np.array([[self.rng.randint(lo, hi) for _ in range(k_dim)] for _ in range(m_dim)])
            kt, nt = model.ceil_div(k_dim, n), model.ceil_div(n_dim, n)
            self.dev.load_weights(model.pack_weights(w, n))
            self.dev.load_vectors(model.pack_activations(x, n))
            self.dev.load_program([isa.matmul(w_base=0, a_base=0, acc_base=0, kt=kt, nt=nt, m=m_dim,
                                              act_signed=signed, acc_mode=AccMode.OVERWRITE), isa.halt()])
            st = self.dev.run()
            assert st.state == SeqState.HALTED_OK, f"program halted with {st}"
            got = model.unpack_acc(self.dev.read_acc(0, m_dim * nt), m_dim, n_dim)
            want = model.matmul(x, w, signed)
            bad = np.argwhere(got != want)
            assert bad.size == 0, (f"{m_dim}x{k_dim} @ {k_dim}x{n_dim} ({'INT8' if signed else 'UINT8'}): "
                                   f"{len(bad)} wrong results, first at {bad[0].tolist()}")
        return f"{count} random matmuls, both activation modes, all results exact"

    def network(self, inputs: int = 3) -> str:
        n = self.n
        for act_signed in (False, True):
            k1, o, nt2, tiles, params, program, reference = two_layer_network(self.rng, n, act_signed)
            self.dev.load_weights(tiles)
            self.dev.load_params(*params)
            self.dev.load_program(program)
            lo, hi = (-128, 127) if act_signed else (0, 255)
            for _ in range(inputs):
                x = np.array([[self.rng.randint(lo, hi) for _ in range(k1)]])
                self.dev.load_vectors(model.pack_activations(x, n))
                st = self.dev.run()
                assert (st.state, st.err, st.pc, st.halt_code) == (SeqState.HALTED_OK, SeqErr.NONE, 4, 7), st
                got = model.unpack_vectors(self.dev.read_vectors(30, nt2), 1, o, signed=True)
                assert got.tolist() == reference(x).tolist(), "RUN output differs from the model"
                x2 = np.array([[self.rng.randint(lo, hi) for _ in range(k1)]])
                st, out = self.dev.infer(model.pack_activations(x2, n).tobytes(), out_addr=30, out_count=nt2)
                assert st.state == SeqState.HALTED_OK, st
                got2 = model.unpack_vectors(np.frombuffer(out, dtype=np.uint8).reshape(nt2, n), 1, o, signed=True)
                assert got2.tolist() == reference(x2).tolist(), "INFER output differs from the model"
        return f"{2 * 2 * inputs} inferences (RUN and INFER, UINT8 and INT8 inputs), all outputs exact"

    def errors(self) -> str:
        d = self.dev.info().depths
        cases = [
            ([isa.nop(), isa.Fields(0x33).encode()], SeqErr.ILLEGAL_OPCODE, 1),
            ([isa.matmul(w_base=d[Mem.WBUF] - 1, a_base=0, acc_base=0, kt=2, nt=1, m=1, act_signed=False,
                         acc_mode=AccMode.OVERWRITE)], SeqErr.ADDR_RANGE, 0),
        ]
        for prog, err, pc in cases:
            self.dev.load_program(prog + [isa.halt()])
            st = self.dev.run()
            assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, err, pc), f"expected {err.name}, got {st}"
        # PC overrun: NOPs up to the end of IMEM, started near the end.
        last = d[Mem.IMEM] - 1
        self.dev.load_program([isa.nop()], at=last)
        st = self.dev.run(start_pc=last)
        assert (st.state, st.err, st.pc) == (SeqState.HALTED_ERROR, SeqErr.PC_OVERRUN, d[Mem.IMEM]), st
        # A request the board must refuse: a READ beyond the end of ACC.
        reply = self.dev.link.transact(Cmd.READ, struct.pack("<BBHH", int(Mem.ACC), 0, d[Mem.ACC], 1))
        assert reply.code == Code.BAD_ARG, f"READ beyond ACC: expected NAK BAD_ARG, got {reply.code!r}"
        return "illegal opcode, address range and PC overrun halt correctly; bad READ gets NAK (led[6] on)"

    def speed(self, rounds: int = 20) -> str:
        n = self.n
        k1, o, nt2, tiles, params, program, _ = two_layer_network(self.rng, n, False)
        self.dev.load_weights(tiles)
        self.dev.load_params(*params)
        self.dev.load_program(program)
        x = model.pack_activations(np.array([[self.rng.randrange(256) for _ in range(k1)]]), n).tobytes()
        start = time.perf_counter()
        for _ in range(rounds):
            st, _ = self.dev.infer(x, out_addr=30, out_count=nt2)
            assert st.state == SeqState.HALTED_OK, st
        ms = 1000 * (time.perf_counter() - start) / rounds
        return f"{rounds} INFERs of a {k1}-{2 * n + 1}-{o} network: {ms:.2f} ms each"

    def leds(self, ask: Callable[[str], bool]) -> str:
        """The status LEDs: halt with ISA errors 1, 4 and 6 and then OK, and ask (``ask`` returns
        True for yes) whether the LEDs show each state."""
        depth = self.dev.info().depths[Mem.IMEM]
        cases = [(0, 0xFF, SeqErr.ILLEGAL_OPCODE),         # opcode 0xFF is not defined
                 (0, 0x10, SeqErr.ZERO_COUNT),             # MATMUL with every count 0
                 (depth - 1, isa.nop(), SeqErr.PC_OVERRUN)]  # NOP in the last slot, then off the end
        wrong = []
        for pc, word, err in cases:
            self.dev.load_program([word], at=pc)
            st = self.dev.run(pc)
            assert st.state == SeqState.HALTED_ERROR and st.err == err, f"expected error {int(err)}, got {st}"
            if not ask(f"led[5] on, led[4] off, and led[15:12] = {int(err):04b} (error {int(err)})?"):
                wrong.append(f"error {int(err)}")
        self.dev.load_program([isa.halt(1)])
        assert self.dev.run().state == SeqState.HALTED_OK
        if not ask("led[4] on, led[5] off, and led[15:12] all off?"):
            wrong.append("halted OK")
        assert not wrong, "LEDs wrong for: " + ", ".join(wrong)
        return "led[4], led[5] and the error code on led[15:12] confirmed for errors 1, 4, 6 and OK"

    def button(self, press: Callable[[], None]) -> str:
        """``btnC`` resets the design but keeps memories: leave data in every memory and a halted
        program, call ``press`` (which waits until the user has pressed the button), then
        expect an idle core and unchanged memories."""
        depths = self.dev.info().depths
        saved = {}
        for mem in (Mem.WBUF, Mem.ABUF, Mem.ACC, Mem.PTAB):
            count = min(4, depths[mem])
            addr = self.rng.randrange(depths[mem] - count + 1)
            saved[mem] = (addr, count, self._pattern(mem, count))
            self.dev.write(mem, addr, saved[mem][2])
        pc = self.rng.randrange(depths[Mem.IMEM])
        word = isa.halt(0x5A)
        self.dev.load_program([word], at=pc)
        saved[Mem.IMEM] = (pc, 1, isa.to_bytes(word))
        st = self.dev.run(pc)
        assert st.state == SeqState.HALTED_OK, f"setup program: {st}"
        press()
        st = self.dev.status()
        assert st.state == SeqState.IDLE, (f"after the button the core is {st.state.name}, "
                                           "expected IDLE (was btnC pressed?)")
        for mem, (addr, count, data) in saved.items():
            assert self.dev.read(mem, addr, count) == data, f"{mem.name} {addr}..: changed by the reset"
        return "btnC returned the core to IDLE; every memory kept its contents"

    # ---------------------------------------------------------------------------------------------

    def summary(self) -> bool:
        failed = [name for name, ok, _ in self.results if not ok]
        s = self.dev.link.stats
        self.out(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed. "
                 f"Link: {s.requests} requests, {s.resends} resends, {s.timeouts} timeouts, "
                 f"{s.bad_replies} bad replies, {s.nak_bad_crc} NAK BAD_CRC.")
        if failed:
            self.out("FAILED: " + ", ".join(failed))
        return not failed


def run_all(dev: Device, seed: int = 0, count: int = 6, rounds: int = 20, out=print,
            ask: Callable[[str], bool] | None = None, press: Callable[[], None] | None = None) -> bool:
    """Run every check and return True if all passed. The manual checks run last: ``leds`` with
    ``ask`` (asks the user a yes/no question), ``button`` with ``press`` (returns once ``btnC``
    has been pressed)."""
    c = Checks(dev, seed, out)
    if not c.run("info", c.info):
        return c.summary()                    # nothing else can work without INFO
    c.run("memories", c.memories)
    c.run("matmul", lambda: c.matmul(count))
    c.run("network", c.network)
    c.run("errors", c.errors)
    c.run("speed", lambda: c.speed(rounds))
    if ask is not None:
        c.run("leds", lambda: c.leds(ask))
    if press is not None:
        c.run("button", lambda: c.button(press))
    return c.summary()


def probe(ports: list[str], out=print, baud: int = BAUD) -> list[str]:
    """Send INFO to each port; return the ports where the TinyTPU design answers.

    A port that answers INFO with NAK ``UNKNOWN_CMD`` runs the bring-up design instead. A port
    that is busy (for example open in Vivado's hardware manager) is reported and skipped.
    """
    found = []
    for port in ports:
        try:
            with Link.open(port, baud, timeout=scaled_timeout(0.05, baud), retries=0) as link:
                link.wait_for_silence()
                reply = link.transact(Cmd.INFO)
            if reply.code == Code.ACK and len(reply.payload) == 16:
                out(f"{port}: TinyTPU answered (N = {reply.payload[1]})")
                found.append(port)
            elif reply.code == Code.UNKNOWN_CMD:
                out(f"{port}: the bring-up design (loopback_top) answered; load tinytpu_top.bit")
            else:
                out(f"{port}: unexpected reply {reply.code!r}")
        except (LinkError, OSError) as e:
            out(f"{port}: no valid answer ({type(e).__name__})")
    if not found:
        out("No port answered. Is tinytpu_top.bit loaded?")
    return found


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(prog="python -m tinytpu.selftest", description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port, e.g. COM7 (default: the only FT2232H port found)")
    ap.add_argument("--list-ports", action="store_true", help="list FT2232H serial ports and exit")
    ap.add_argument("--probe", action="store_true", help="send INFO to each FT2232H port, report which answers, exit")
    ap.add_argument("--count", type=int, default=6, help="random matmuls in the matmul check")
    ap.add_argument("--rounds", type=int, default=20, help="INFER round trips in the speed check")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--manual", action="store_true",
                    help="finish with interactive checks at the board: confirm LEDs, press btnC")
    ap.add_argument("--baud", type=int, default=BAUD,
                    help="bit rate; must match the bitstream (default 1000000; the fallback builds use 115200)")
    args = ap.parse_args(argv)

    ports = find_ports()
    if args.list_ports:
        print("\n".join(ports) if ports else "no FT2232H serial port found")
        return 0
    if args.probe:
        return 0 if probe(ports, baud=args.baud) else 1
    port = args.port or (ports[0] if len(ports) == 1 else None)
    if port is None:
        print(f"FT2232H ports found: {ports or 'none'}. Use --port, or --probe to find the board's UART.",
              file=sys.stderr)
        return 2

    print(f"TinyTPU self-test on {port} at {args.baud} baud (overwrites the board's memories)\n")
    with Link.open(port, args.baud) as link:
        link.wait_for_silence()
        ask = press = None
        if args.manual:
            def ask(question: str) -> bool:
                return input(f"\n>>> On the board: {question} [y/n] ").strip().lower().startswith("y")

            def press() -> None:
                input("\n>>> Press and release btnC on the board, then press Enter here: ")
                link.wait_for_silence()
        ok = run_all(Device(link), args.seed, args.count, args.rounds, ask=ask, press=press)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
