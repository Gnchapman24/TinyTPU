"""Tests for tinytpu.link and tinytpu.bringup against a simulated board (tb/fake_echo.py), task A2.5.

Covers normal echoes, sequence numbers, every resend path of the protocol (lost reply, corrupted
reply, corrupted request, stale reply, junk on the line), giving up with RESET, NAKs that must
not be resent, port discovery, and the full bring-up check sequence, including that it detects a
broken board.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fake_echo import FakeEchoBoard
from tinytpu import link as link_mod
from tinytpu.bringup import run_all
from tinytpu.link import Link, LinkError
from tinytpu.packet import Cmd, Code


def make(board: FakeEchoBoard | None = None, **kw) -> tuple[Link, FakeEchoBoard]:
    board = board or FakeEchoBoard()
    return Link(board, timeout=kw.pop("timeout", 0.05), silence=kw.pop("silence", 0.006), **kw), board


def seqs(board: FakeEchoBoard) -> list[int]:
    return [p[2] for p in board.requests]


def test_echo_roundtrip():
    link, board = make()
    link.echo(b"hello")
    link.echo(b"")
    link.echo(bytes(range(256)))
    assert link.stats.resends == 0 and len(board.requests) == 3


def test_sequence_numbers_increment_and_wrap():
    link, board = make()
    for _ in range(300):
        link.echo(b"x")
    assert seqs(board) == [i & 0xFF for i in range(300)]


def test_lost_reply_is_resent_with_same_seq():
    link, board = make()
    board.drop_replies = 1
    link.echo(b"retry me")
    assert link.stats.timeouts == 1 and link.stats.resends == 1
    assert seqs(board) == [0, 0]


def test_corrupted_reply_is_resent():
    link, board = make()
    board.corrupt_replies = 1
    link.echo(b"retry me")
    assert link.stats.bad_replies == 1 and link.stats.resends == 1
    assert seqs(board) == [0, 0]


def test_nak_bad_crc_is_resent_with_same_seq():
    link, board = make()
    board.corrupt_requests = 1
    link.echo(b"line noise")
    assert link.stats.nak_bad_crc == 1 and link.stats.resends == 1
    assert board.requests[1][2] == 0


def test_stale_reply_is_skipped():
    link, board = make()
    link.echo(b"first")
    board.stale_first = 1
    link.echo(b"second")
    assert link.stats.stale_replies == 1 and link.stats.resends == 0


def test_junk_before_reply_is_skipped():
    link, board = make()
    board.junk_before = bytes([0x00, 0xFF, 0x5B, 0x12])
    link.echo(b"after junk")
    assert link.stats.resends == 0


def test_gives_up_after_retries_and_sends_reset():
    link, board = make(retries=3)
    board.drop_replies = 100
    with pytest.raises(LinkError, match="no reply after 3 resends"):
        link.echo(b"dead board")
    cmds = [p[1] for p in board.requests]
    assert cmds == [Cmd.ECHO] * 4 + [Cmd.RESET]
    assert seqs(board)[:4] == [0, 0, 0, 0]


@pytest.mark.parametrize("cmd, payload, code", [
    (Cmd.INFO, b"", Code.UNKNOWN_CMD),
    (Cmd.ECHO, bytes(300), Code.TOO_LONG),
])
def test_other_naks_are_returned_not_resent(cmd, payload, code):
    link, _ = make()
    reply = link.transact(cmd, payload)
    assert reply.code == code and reply.payload == b""
    assert link.stats.resends == 0


def test_waits_for_silence_before_every_resend(monkeypatch):
    link, board = make(retries=3)
    calls: list[int] = []
    real = link.wait_for_silence
    monkeypatch.setattr(link, "wait_for_silence", lambda *a, **k: (calls.append(len(board.requests)), real(*a, **k)))
    board.drop_replies = 2
    link.echo(b"x")
    assert calls == [1, 2], "silence before resend 1 and 2, never before the first attempt"


def test_wait_for_silence_discards_trailing_bytes():
    link, board = make()
    board.outbuf += b"\x5a late reply bytes"
    link.wait_for_silence()
    assert board.outbuf == b""
    link.echo(b"clean")


def test_payload_limit():
    link, _ = make()
    with pytest.raises(ValueError):
        link.transact(Cmd.ECHO, bytes(4097))


def test_find_ports_filters_ft2232h(monkeypatch):
    fake = [
        SimpleNamespace(device="COM3", vid=0x0403, pid=0x6001),   # another FTDI chip
        SimpleNamespace(device="COM5", vid=0x0403, pid=0x6010),   # Basys3
        SimpleNamespace(device="COM7", vid=None, pid=None),
    ]
    import serial.tools.list_ports
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: fake)
    assert link_mod.find_ports() == ["COM5"]


def test_bringup_passes_on_a_good_board():
    link, _ = make()
    lines: list[str] = []
    assert run_all(link, count=20, rounds=5, out=lines.append)
    assert sum(line.startswith("PASS") for line in lines) == 7, lines


def test_bringup_passes_with_line_noise():
    link, board = make()
    board.drop_replies, board.corrupt_replies, board.corrupt_requests = 1, 1, 1
    assert run_all(link, count=20, rounds=5, out=lambda _: None)
    # Every injected fault was hit and recovered by resending. (Faults can land on the same
    # packet, and a reply corrupted in its length field shows up as a timeout, so only check
    # that resends happened rather than an exact count.)
    assert board.drop_replies == board.corrupt_replies == board.corrupt_requests == 0
    assert link.stats.resends >= 2


def test_bringup_detects_a_broken_board():
    link, board = make()
    board.wrong_payload = True
    lines: list[str] = []
    assert not run_all(link, count=5, rounds=2, out=lines.append)
    assert any(line.startswith("FAIL  ping") for line in lines)


def test_probe_finds_the_answering_port(monkeypatch):
    from tinytpu import bringup

    jtag = FakeEchoBoard()
    jtag.drop_replies = 100            # the JTAG channel never answers
    boards = {"COM6": jtag, "COM7": FakeEchoBoard()}
    monkeypatch.setattr(bringup.Link, "open",
                        classmethod(lambda cls, port, **kw: Link(boards[port], silence=0.006, **kw)))
    lines: list[str] = []
    assert bringup.probe(["COM6", "COM7"], out=lines.append) == ["COM7"]
    assert lines[0].startswith("COM6: no valid answer")


def test_main_refuses_to_guess_between_ports(monkeypatch, capsys):
    from tinytpu import bringup

    monkeypatch.setattr(bringup, "find_ports", lambda: ["COM6", "COM7"])
    assert bringup.main([]) == 2
    assert "--probe" in capsys.readouterr().err


def test_bringup_raw_mode():
    link, _ = make(FakeEchoBoard(raw=True))
    assert run_all(link, count=10, raw=True, out=lambda _: None)
