"""Board test for the UART bring-up design ``loopback_top`` (tasks A2.5 and H2.6).

Runs a sequence of checks over the real serial link and prints PASS/FAIL for each:

1. ``ping``: short ECHO packets come back unchanged.
2. ``sizes``: payloads of every interesting length up to 256 bytes, plus random ones.
3. ``throughput``: back-to-back 256-byte echoes; reports round-trip time and data rate.
4. ``bad_crc``: a deliberately corrupted request gets NAK ``BAD_CRC``.
5. ``unknown_cmd``: a command the bring-up design lacks gets NAK ``UNKNOWN_CMD``.
6. ``too_long``: a 300-byte payload gets NAK ``TOO_LONG``.
7. ``timeout``: a request cut off part-way gets no reply, and the next one works.

Board setup: ``sw[0]`` off (packet mode), freshly reset with ``btnC``. ``--raw`` instead tests raw
byte echo, which needs ``sw[0]`` on.

Windows usage (host venv, see tools/setup.md)::

    uv run --no-default-groups python -m tinytpu.bringup --list-ports
    uv run --no-default-groups python -m tinytpu.bringup --port COM5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable

from tinytpu.link import BAUD, Link, LinkError, find_ports, scaled_timeout
from tinytpu.packet import Cmd, Code, build_request

LOOPBACK_MAX_PAYLOAD = 256
"""Largest payload the bring-up design echoes."""


class Checks:
    """Runs named checks, records results, and prints a line per check."""

    def __init__(self, link: Link, out=print):
        self.link = link
        self.out = out
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn: Callable[[], str]) -> bool:
        try:
            detail = fn() or ""
            ok = True
        except (LinkError, AssertionError, TimeoutError, ValueError) as e:
            detail, ok = f"{type(e).__name__}: {e}", False
        self.results.append((name, ok, detail))
        self.out(f"{'PASS' if ok else 'FAIL'}  {name:<12} {detail}")
        return ok

    # ---------------------------------------------------------------------------------------------

    def ping(self) -> str:
        for payload in (b"", b"\x00", b"hello", bytes(range(16))):
            self.link.echo(payload)
        return "4 echoes"

    def sizes(self, count: int, seed: int) -> str:
        rng = random.Random(seed)
        lengths = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]
        lengths += [rng.randint(0, LOOPBACK_MAX_PAYLOAD) for _ in range(count)]
        total = 0
        for n in lengths:
            self.link.echo(rng.randbytes(n))
            total += n
        return f"{len(lengths)} echoes, {total} payload bytes"

    def throughput(self, rounds: int) -> str:
        payload = random.Random(1).randbytes(LOOPBACK_MAX_PAYLOAD)
        start = time.perf_counter()
        for _ in range(rounds):
            self.link.echo(payload)
        elapsed = time.perf_counter() - start
        wire = rounds * 2 * (len(payload) + 7)  # request + reply bytes
        return (f"{rounds} x {len(payload)} B: {1000 * elapsed / rounds:.2f} ms per round trip, "
                f"{wire / elapsed / 1000:.1f} kB/s on the wire (line limit 100 kB/s)")

    def bad_crc(self) -> str:
        seq = self.link.next_seq()
        packet = bytearray(build_request(Cmd.ECHO, seq, b"corrupt me"))
        packet[-1] ^= 0x01
        self.link.send_raw(bytes(packet))
        reply = self.link.read_reply(seq)
        assert reply.code == Code.BAD_CRC, f"expected NAK BAD_CRC, got {reply.code!r}"
        assert reply.payload == b"", "NAK must have no payload"
        return "NAK BAD_CRC (led[4] should now be on)"

    def unknown_cmd(self) -> str:
        reply = self.link.transact(Cmd.INFO)
        assert reply.code == Code.UNKNOWN_CMD, f"expected NAK UNKNOWN_CMD, got {reply.code!r}"
        return "NAK UNKNOWN_CMD (led[7] should now be on)"

    def too_long(self) -> str:
        reply = self.link.transact(Cmd.ECHO, bytes(LOOPBACK_MAX_PAYLOAD + 44))
        assert reply.code == Code.TOO_LONG, f"expected NAK TOO_LONG, got {reply.code!r}"
        return "NAK TOO_LONG"

    def timeout(self) -> str:
        seq = self.link.next_seq()
        packet = build_request(Cmd.ECHO, seq, b"never finished")
        self.link.send_raw(packet[:8])                # stop part-way through the payload
        try:
            self.link.read_reply(seq, timeout=0.050)  # 10x the FPGA's 5 ms timeout
        except TimeoutError:
            pass
        else:
            raise AssertionError("a partial request must get no reply")
        self.link.echo(b"after timeout")
        return "partial request ignored, next echo OK (led[6] should now be on)"

    def raw(self, count: int, seed: int) -> str:
        data = random.Random(seed).randbytes(count)
        self.link.send_raw(data)
        got = bytearray()
        end = time.monotonic() + 1.0 + count * 2e-5
        while len(got) < count and time.monotonic() < end:
            self.link.transport.timeout = 0.05
            got += self.link.transport.read(count - len(got))
        assert bytes(got) == data, f"sent {count} bytes, got {len(got)} back" + (
            "" if len(got) != count else " (different bytes)")
        return f"{count} bytes echoed unchanged"

    # ---------------------------------------------------------------------------------------------

    def summary(self) -> bool:
        failed = [name for name, ok, _ in self.results if not ok]
        s = self.link.stats
        self.out(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed. "
                 f"Link: {s.requests} requests, {s.resends} resends, {s.timeouts} timeouts, "
                 f"{s.bad_replies} bad replies, {s.nak_bad_crc} NAK BAD_CRC.")
        if failed:
            self.out("FAILED: " + ", ".join(failed))
        return not failed


def run_all(link: Link, count: int = 100, rounds: int = 50, seed: int = 0, raw: bool = False,
            out=print) -> bool:
    """Run every check (or only the raw-echo check) and return True if all passed."""
    c = Checks(link, out)
    if raw:
        c.run("raw", lambda: c.raw(count * 10, seed))
    else:
        c.run("ping", c.ping)
        c.run("sizes", lambda: c.sizes(count, seed))
        c.run("throughput", lambda: c.throughput(rounds))
        c.run("bad_crc", c.bad_crc)
        c.run("unknown_cmd", c.unknown_cmd)
        c.run("too_long", c.too_long)
        c.run("timeout", c.timeout)
    return c.summary()


def probe(ports: list[str], out=print, baud: int = BAUD) -> list[str]:
    """Send one short ECHO to each port and return the ports that answer correctly.

    Needs the bring-up bitstream loaded, in packet mode (``sw[0]`` off). A port that is busy
    (for example open in Vivado's hardware manager) is reported and skipped.
    """
    found = []
    for port in ports:
        try:
            with Link.open(port, baud, timeout=scaled_timeout(0.05, baud), retries=0) as link:
                link.wait_for_silence()
                link.echo(b"probe")
            out(f"{port}: TinyTPU bring-up design answered")
            found.append(port)
        except (LinkError, OSError) as e:
            out(f"{port}: no valid answer ({type(e).__name__})")
    if not found:
        out("No port answered. Is the bring-up bitstream loaded and sw[0] off?")
    return found


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(prog="python -m tinytpu.bringup", description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port, e.g. COM5 (default: the only FT2232H port found)")
    ap.add_argument("--list-ports", action="store_true", help="list FT2232H serial ports and exit")
    ap.add_argument("--count", type=int, default=100, help="random payloads in the sizes check")
    ap.add_argument("--rounds", type=int, default=50, help="round trips in the throughput check")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--raw", action="store_true", help="test raw byte echo instead (sw[0] on)")
    ap.add_argument("--baud", type=int, default=BAUD,
                    help="bit rate; must match the bitstream (default 1000000; the fallback builds use 115200)")
    ap.add_argument("--probe", action="store_true",
                    help="send one ECHO to each FT2232H port and report which answers, then exit")
    args = ap.parse_args(argv)

    ports = find_ports()
    if args.list_ports:
        print("\n".join(ports) if ports else "no FT2232H serial port found")
        return 0
    if args.probe:
        return 0 if probe(ports, baud=args.baud) else 1
    port = args.port or (ports[0] if len(ports) == 1 else None)
    if port is None:
        print(f"FT2232H ports found: {ports or 'none'}. Use --port, or --probe to find the "
              "board's UART.", file=sys.stderr)
        return 2

    print(f"TinyTPU bring-up test on {port} at {args.baud} baud, packet mode {'off (raw)' if args.raw else 'on'}\n")
    with Link.open(port, args.baud) as link:
        link.wait_for_silence()
        ok = run_all(link, args.count, args.rounds, args.seed, args.raw)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
