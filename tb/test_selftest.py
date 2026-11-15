"""Tests for the command-line side of tinytpu.selftest (task A6.2), without hardware.

The checks themselves run against the simulated board in tb/test_tinytpu_top.py
(``selftest_script``). Here: port probing tells the TinyTPU design, the bring-up design and a
silent port apart, and main() refuses to guess between several ports.
"""

from __future__ import annotations

import struct

from fake_echo import FakeEchoBoard
from tinytpu import selftest
from tinytpu.link import Link
from tinytpu.packet import Cmd, Code, build_reply, packet_length


class FakeTinyTpu:
    """Answers INFO like tinytpu_top (N = 8, default depths); nothing else."""

    def __init__(self):
        self.timeout: float | None = 0.01
        self.out = bytearray()

    def write(self, data: bytes) -> int:
        if data[0] == 0xA5 and len(data) == packet_length(data[:5]) and data[1] == Cmd.INFO:
            info = struct.pack("<BBHHHHHHH", 1, 8, 100, 128, 2048, 1024, 512, 512, 4096)
            self.out += build_reply(Code.ACK, data[2], info)
        return len(data)

    def read(self, size: int = 1) -> bytes:
        got = bytes(self.out[:size])
        del self.out[:size]
        return got

    def close(self) -> None:
        pass


def test_probe_tells_the_designs_apart(monkeypatch):
    silent = FakeEchoBoard()
    silent.drop_replies = 100                          # e.g. the FT2232H's JTAG channel
    boards = {"COM5": silent, "COM6": FakeEchoBoard(), "COM7": FakeTinyTpu()}
    monkeypatch.setattr(selftest.Link, "open",
                        classmethod(lambda cls, port, **kw: Link(boards[port], silence=0.006, **kw)))
    lines: list[str] = []
    assert selftest.probe(["COM5", "COM6", "COM7"], out=lines.append) == ["COM7"]
    assert lines[0].startswith("COM5: no valid answer")
    assert "bring-up design" in lines[1]
    assert lines[2] == "COM7: TinyTPU answered (N = 8)"


def test_probe_reports_nothing_found(monkeypatch):
    monkeypatch.setattr(selftest.Link, "open",
                        classmethod(lambda cls, port, **kw: Link(FakeEchoBoard(), silence=0.006, **kw)))
    lines: list[str] = []
    assert selftest.probe(["COM6"], out=lines.append) == []
    assert lines[-1].startswith("No port answered")


def test_main_refuses_to_guess_between_ports(monkeypatch, capsys):
    monkeypatch.setattr(selftest, "find_ports", lambda: ["COM6", "COM7"])
    assert selftest.main([]) == 2
    assert "--probe" in capsys.readouterr().err


def test_main_lists_ports(monkeypatch, capsys):
    monkeypatch.setattr(selftest, "find_ports", lambda: ["COM6", "COM7"])
    assert selftest.main(["--list-ports"]) == 0
    assert capsys.readouterr().out.split() == ["COM6", "COM7"]
