"""Reliable request/reply link to the TinyTPU over the Basys3 USB-UART (``docs/uart_protocol.md``).

:class:`Link` sends one request at a time and waits for its reply. It implements the protocol's
host-side recovery rules:

* each packet goes out in a single ``write()``;
* on a timeout, a bad reply (wrong CRC or garbled), or a NAK ``BAD_CRC``, it waits for 10 ms of
  silence and resends the same packet with the **same** sequence number, so the FPGA can spot a
  duplicate;
* after 3 failed resends it sends ``RESET`` (best effort) and raises :class:`LinkError`.

Every other NAK is returned to the caller unchanged, because resending would not help.

The transport is anything with pyserial's ``read``/``write``/``timeout`` interface, so tests can
substitute a simulated device. The clock is injectable too: every deadline is measured with
``clock()`` (default :func:`time.monotonic`), so a simulated transport can supply simulated
time and the same resend rules run unchanged in a cocotb testbench.

Example (Windows, board on COM5):
    >>> from tinytpu.link import Link            # doctest: +SKIP
    >>> with Link.open("COM5") as link:          # doctest: +SKIP
    ...     link.echo(b"hello")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tinytpu.packet import (
    CRC_LEN,
    HEADER_LEN,
    MAX_PAYLOAD,
    SYNC_RSP,
    Cmd,
    Code,
    Reply,
    build_request,
    parse_reply,
)

BAUD = 1_000_000
"""Link rate (``CLKS_PER_BIT`` = 100 at 100 MHz)."""
SILENCE = 0.010
"""Quiet time before a resend: longer than the FPGA's 5 ms inter-byte timeout."""
DEFAULT_TIMEOUT = 0.100
"""Default time to wait for a complete reply at 1 Mbaud, in seconds (the largest reply, 4103
bytes, takes 41 ms on the wire). :meth:`Link.open` scales it for slower links."""
RETRIES = 3
"""Resends after the first attempt before giving up."""
POLL = 0.010
"""Read timeout of the port, in seconds: every read returns after at most this long, and
:class:`Link` checks its own deadlines between reads.

It is set once and never changed while a packet may be in flight. On Windows, pyserial applies
every assignment to ``timeout`` by reconfiguring the port (``SetCommState``), and on the Basys3's
FTDI chip that corrupts or drops bytes still being transmitted. Found on the board: changing the
timeout during each reply made about 1 in 2 long requests fail at 115,200 baud."""

FTDI_VID = 0x0403
"""USB vendor ID of the Basys3's FTDI chip."""
FT2232H_PID = 0x6010
"""USB product ID of the FT2232H (JTAG on one channel, UART on the other)."""


def scaled_timeout(seconds: float, baud: int) -> float:
    """Return ``seconds`` (a timeout chosen for 1 Mbaud) stretched for a slower ``baud``."""
    return seconds * max(1.0, BAUD / baud)


class Transport(Protocol):
    """The subset of ``serial.Serial`` that :class:`Link` uses."""

    timeout: float | None

    def write(self, data: bytes) -> int | None: ...
    def read(self, size: int = 1) -> bytes: ...
    def close(self) -> None: ...


class LinkError(Exception):
    """The board did not answer correctly after all resends."""


@dataclass
class Stats:
    """Counters kept by a :class:`Link`.

    Attributes:
        requests: Commands issued by the caller.
        resends: Packets sent again after a timeout, bad reply, or NAK BAD_CRC.
        timeouts: Replies that did not arrive in time.
        bad_replies: Replies with a wrong CRC or a garbled header.
        nak_bad_crc: NAK BAD_CRC replies (the board saw a corrupted request).
        stale_replies: Valid replies discarded because their sequence number did not match.
    """

    requests: int = 0
    resends: int = 0
    timeouts: int = 0
    bad_replies: int = 0
    nak_bad_crc: int = 0
    stale_replies: int = 0


class _BadReply(Exception):
    pass


class Link:
    """One request in flight at a time, with the protocol's resend rules.

    Args:
        transport: An open serial port (or a stand-in with the same interface).
        timeout: Default seconds to wait for a complete reply.
        retries: Resends after the first attempt before giving up.
        silence: Seconds of quiet line required before a resend.
        clock: Returns the current time in seconds; deadlines and quiet times are measured with
            it. Defaults to :func:`time.monotonic`; a simulation passes simulated time.
    """

    def __init__(self, transport: Transport, timeout: float = DEFAULT_TIMEOUT,
                 retries: int = RETRIES, silence: float = SILENCE,
                 clock: Callable[[], float] = time.monotonic):
        self.transport = transport
        self.clock = clock
        self.timeout = timeout
        self.retries = retries
        self.silence_time = silence
        self.stats = Stats()
        self._seq = 0

    # ---------------------------------------------------------------------------------------------
    # Opening and closing
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def open(cls, port: str, baud: int = BAUD, **kwargs: Any) -> Link:
        """Open a serial port at 8N1 with no flow control and return a :class:`Link` on it.

        Args:
            port: Port name, e.g. ``"COM5"`` on Windows or ``"/dev/ttyUSB1"`` on Linux.
            baud: Bit rate. Must match the FPGA's ``CLKS_PER_BIT`` (100 MHz / ``CLKS_PER_BIT``).
            **kwargs: Passed to :class:`Link`. Without ``timeout``, the default reply timeout is
                scaled to ``baud`` (:func:`scaled_timeout`).
        """
        import serial  # pyserial; imported here so the module also works without a port

        ser = serial.Serial(port, baud, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, timeout=POLL, write_timeout=1.0,
                            xonxoff=False, rtscts=False, dsrdtr=False)
        ser.reset_input_buffer()
        kwargs.setdefault("timeout", scaled_timeout(DEFAULT_TIMEOUT, baud))
        return cls(ser, **kwargs)

    def close(self) -> None:
        """Close the underlying port."""
        self.transport.close()

    def __enter__(self) -> Link:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------------------------------------
    # Requests
    # ---------------------------------------------------------------------------------------------

    def next_seq(self) -> int:
        """Return the sequence number for a new command (0-255, wrapping)."""
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return seq

    def transact(self, cmd: Cmd | int, payload: bytes = b"", timeout: float | None = None) -> Reply:
        """Send one command and return its reply, resending as the protocol requires.

        Args:
            cmd: Command code.
            payload: Command payload, at most :data:`~tinytpu.packet.MAX_PAYLOAD` bytes.
            timeout: Seconds to wait for each reply; defaults to the link's timeout. ``RUN`` and
                ``INFER`` may need longer.

        Returns:
            The reply. Any NAK other than ``BAD_CRC`` is returned as is.

        Raises:
            LinkError: If no good reply arrived after all resends.
        """
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD}")
        self.stats.requests += 1
        seq = self.next_seq()
        packet = build_request(cmd, seq, payload)
        wait = self.timeout if timeout is None else timeout
        problem = ""
        for attempt in range(self.retries + 1):
            if attempt:
                self.stats.resends += 1
                self.wait_for_silence()
            self.transport.write(packet)
            try:
                reply = self._read_reply(seq, self.clock() + wait)
            except TimeoutError:
                self.stats.timeouts += 1
                problem = "no reply"
                continue
            except _BadReply as e:
                self.stats.bad_replies += 1
                problem = f"bad reply ({e})"
                continue
            if reply.code == Code.BAD_CRC:
                self.stats.nak_bad_crc += 1
                problem = "board reported a CRC error"
                continue
            return reply
        self._best_effort_reset()
        raise LinkError(f"command {int(cmd):#04x} seq {seq}: {problem} after {self.retries} resends")

    def echo(self, payload: bytes, timeout: float | None = None) -> None:
        """Send a bring-up ``ECHO`` and check the payload comes back unchanged.

        Raises:
            LinkError: On a NAK, a different payload, or no reply.
        """
        reply = self.transact(Cmd.ECHO, payload, timeout)
        if reply.code != Code.ACK:
            raise LinkError(f"ECHO got NAK {reply.code!r}")
        if reply.payload != payload:
            raise LinkError(f"ECHO payload mismatch: sent {len(payload)} bytes, got {len(reply.payload)}")

    def send_raw(self, data: bytes) -> None:
        """Write bytes exactly as given (for fault-injection tests); no reply is awaited."""
        self.transport.write(data)

    def read_reply(self, seq: int, timeout: float | None = None) -> Reply:
        """Wait for one reply with sequence number ``seq`` (after :meth:`send_raw`).

        Raises:
            TimeoutError: If no matching reply arrived in time.
            ValueError: If a reply arrived but was garbled.
        """
        wait = self.timeout if timeout is None else timeout
        try:
            return self._read_reply(seq, self.clock() + wait)
        except _BadReply as e:
            raise ValueError(str(e)) from None

    def wait_for_silence(self, quiet: float | None = None, limit: float = 1.0) -> None:
        """Discard input until the line has been quiet for ``quiet`` seconds (default 10 ms).

        Args:
            quiet: Required quiet time.
            limit: Give up waiting after this many seconds in total.
        """
        quiet = self.silence_time if quiet is None else quiet
        end = self.clock() + limit
        last = self.clock()
        self._poll_timeout()
        while self.clock() < end:
            if self.transport.read(4096):
                last = self.clock()
            elif self.clock() - last >= quiet:
                return

    # ---------------------------------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------------------------------

    def _poll_timeout(self) -> None:
        """Give the port its fixed read timeout, touching it only if it differs (see :data:`POLL`)."""
        if self.transport.timeout != POLL:
            self.transport.timeout = POLL

    def _read_exact(self, n: int, deadline: float) -> bytes:
        buf = bytearray()
        self._poll_timeout()
        while len(buf) < n:
            if deadline - self.clock() <= 0:
                raise TimeoutError
            buf += self.transport.read(n - len(buf))
        return bytes(buf)

    def _read_reply(self, seq: int, deadline: float) -> Reply:
        """Read replies until one with ``seq`` arrives; skip junk before a sync byte."""
        while True:
            if self._read_exact(1, deadline)[0] != SYNC_RSP:
                continue  # resynchronize on the next reply sync byte
            header = bytes([SYNC_RSP]) + self._read_exact(HEADER_LEN - 1, deadline)
            length = int.from_bytes(header[3:5], "little")
            if length > MAX_PAYLOAD:
                raise _BadReply(f"length {length} in header {header.hex(' ')}")
            rest = self._read_exact(length + CRC_LEN, deadline)
            try:
                reply = parse_reply(header + rest)
            except ValueError as e:
                raise _BadReply(str(e)) from None
            if reply.seq != seq:
                self.stats.stale_replies += 1  # a late reply to an earlier attempt
                continue
            return reply

    def _best_effort_reset(self) -> None:
        try:
            self.wait_for_silence()
            self.transport.write(build_request(Cmd.RESET, self.next_seq()))
            self.wait_for_silence()
        except Exception:  # noqa: BLE001 - reset is advisory; the caller gets the LinkError
            pass


def find_ports() -> list[str]:
    """Return the serial ports that belong to an FTDI FT2232H (the Basys3's USB chip).

    The FT2232H has two channels (JTAG and UART), and Windows may give both a COM port with
    identical USB details. Other FT2232H boards (e.g. a Nandland Go Board) also match. So this
    can return several ports; ``python -m tinytpu.bringup --probe`` finds which one is the
    TinyTPU's UART.
    """
    from serial.tools import list_ports

    return [p.device for p in list_ports.comports()
            if p.vid == FTDI_VID and p.pid == FT2232H_PID]
