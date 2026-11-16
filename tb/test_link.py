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
                        classmethod(lambda cls, port, *_, **kw: Link(boards[port], silence=0.006, **kw)))
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


class _TimeoutCountingBoard(FakeEchoBoard):
    """Counts assignments to ``timeout``: on Windows, pyserial reconfigures the port on each one,
    which corrupts bytes the FTDI chip is still sending (seen on the board)."""

    def __init__(self, timeout: float, **kw):
        super().__init__(**kw)
        self._timeout = timeout
        self.timeout_sets = 0

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @timeout.setter
    def timeout(self, value: float | None) -> None:
        self.timeout_sets = getattr(self, "timeout_sets", 0) + 1
        self._timeout = value


@pytest.mark.parametrize("raw", [False, True])
def test_link_never_reconfigures_an_open_port(raw):
    """A port opened with the link's poll timeout is never touched again, through every check
    (resends and RESET included); a port with another timeout is set once."""
    board = _TimeoutCountingBoard(link_mod.POLL, raw=raw)
    board.timeout_sets = 0
    board.drop_replies, board.corrupt_requests = 1, 1
    link, _ = make(board)
    assert run_all(link, count=10, rounds=3, raw=raw, out=lambda _: None)
    assert board.timeout_sets == 0

    board = _TimeoutCountingBoard(0.5)
    board.timeout_sets = 0
    link, _ = make(board)
    for _ in range(5):
        link.echo(b"x")
    assert board.timeout_sets == 1


class _SilentBoard:
    """Never answers; each read advances a fake clock by its timeout (no real waiting)."""

    def __init__(self, clock: list[float]):
        self.clock = clock
        self.timeout: float | None = 0.0
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def read(self, size: int = 1) -> bytes:
        self.clock[0] += self.timeout or 0.0
        return b""

    def close(self) -> None:
        pass


def test_injected_clock_drives_every_deadline():
    """With an injected clock, timeouts and quiet times follow it, not wall-clock time: a
    simulated transport can run the real resend rules at simulated speed."""
    import time as real_time

    now = [1000.0]
    board = _SilentBoard(now)
    link = Link(board, timeout=5.0, silence=1.0, clock=lambda: now[0])
    t0 = real_time.monotonic()
    with pytest.raises(LinkError):
        link.transact(Cmd.STATUS)
    assert real_time.monotonic() - t0 < 1.0               # 4 x 5 s of fake time, no real waiting
    assert now[0] - 1000.0 >= 4 * 5.0                     # every attempt waited its full timeout
    statuses = [p for p in board.writes if p[1] == Cmd.STATUS]
    assert len(statuses) == 4 and len({p[2] for p in statuses}) == 1   # same SEQ every time
    assert board.writes[-1][1] == Cmd.RESET


def test_open_scales_the_reply_timeout_to_the_baud_rate(monkeypatch):
    import serial

    opened = []

    class FakeSerial:
        def __init__(self, port, baud, **kw):
            opened.append((port, baud))

        def reset_input_buffer(self):
            pass

    monkeypatch.setattr(serial, "Serial", FakeSerial)
    assert Link.open("COM7").timeout == link_mod.DEFAULT_TIMEOUT                 # 1 Mbaud
    assert Link.open("COM7", 115_200).timeout == pytest.approx(0.100 * 1e6 / 115_200)
    assert Link.open("COM7", 115_200, timeout=0.3).timeout == 0.3              # explicit wins
    assert Link.open("COM7", 2_000_000).timeout == link_mod.DEFAULT_TIMEOUT    # never shorter
    assert opened == [("COM7", 1_000_000), ("COM7", 115_200), ("COM7", 115_200), ("COM7", 2_000_000)]
