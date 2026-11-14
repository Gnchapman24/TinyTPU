"""UART packet framing for TinyTPU (``docs/uart_protocol.md``).

Requests are ``A5 CMD SEQ LEN payload CRC`` and replies are ``5A CODE SEQ LEN payload CRC``.
``LEN`` is a little-endian u16. The CRC-16/CCITT-FALSE covers every byte after the sync byte and
is sent high byte first.

Example:
    >>> from tinytpu.packet import Cmd, build_request, build_reply, parse_reply, Code
    >>> req = build_request(Cmd.ECHO, seq=7, payload=b"hi")
    >>> req.hex(" ")
    'a5 7e 07 02 00 68 69 3e 9f'
    >>> parse_reply(build_reply(Code.ACK, 7, b"hi"))
    Reply(code=<Code.ACK: 0>, seq=7, payload=b'hi')
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from tinytpu.crc import check, crc_bytes

SYNC_REQ = 0xA5
"""First byte of every request."""
SYNC_RSP = 0x5A
"""First byte of every reply."""
HEADER_LEN = 5
"""Sync, command or code, sequence number, and the 2-byte length."""
CRC_LEN = 2
MAX_PAYLOAD = 4096
"""Largest payload the protocol allows (the FPGA reports its own limit in ``INFO``)."""


class Cmd(IntEnum):
    """Request command codes."""

    INFO = 0x01
    WRITE = 0x02
    READ = 0x03
    RUN = 0x04
    STATUS = 0x05
    ABORT = 0x06
    RESET = 0x07
    INFER = 0x08
    ECHO = 0x7E
    """Bring-up only (``loopback_echo``): echo the payload back. Not part of the final protocol."""


class Code(IntEnum):
    """Reply codes: ``ACK`` or a NAK reason."""

    ACK = 0x00
    BAD_CRC = 0x01
    UNKNOWN_CMD = 0x02
    BAD_LENGTH = 0x03
    TOO_LONG = 0x04
    BAD_ARG = 0x05
    BUSY = 0x06


@dataclass(frozen=True)
class Reply:
    """A decoded reply packet.

    Attributes:
        code: ``Code.ACK`` or a NAK code (an ``int`` if the code is unknown).
        seq: Sequence number copied from the request.
        payload: Reply payload bytes.
    """

    code: Code | int
    seq: int
    payload: bytes


def _frame(sync: int, first: int, seq: int, payload: bytes) -> bytes:
    if not 0 <= seq <= 0xFF:
        raise ValueError("seq must be 0-255")
    if len(payload) > 0xFFFF:
        raise ValueError("payload longer than 65535 bytes")
    body = bytes([first, seq]) + len(payload).to_bytes(2, "little") + payload
    return bytes([sync]) + body + crc_bytes(body)


def build_request(cmd: Cmd | int, seq: int, payload: bytes = b"") -> bytes:
    """Return the bytes of one request packet.

    Args:
        cmd: Command code.
        seq: Sequence number, 0-255.
        payload: Command payload.
    """
    return _frame(SYNC_REQ, int(cmd), seq, payload)


def build_reply(code: Code | int, seq: int, payload: bytes = b"") -> bytes:
    """Return the bytes of one reply packet (the FPGA's side; used by tests and simulators)."""
    return _frame(SYNC_RSP, int(code), seq, payload)


def packet_length(header: bytes) -> int:
    """Return the total length of a packet, including CRC, from its first 5 bytes."""
    if len(header) < HEADER_LEN:
        raise ValueError("need the 5 header bytes")
    return HEADER_LEN + int.from_bytes(header[3:5], "little") + CRC_LEN


def parse_reply(data: bytes) -> Reply:
    """Decode and check one complete reply packet.

    Raises:
        ValueError: If the sync byte, length or CRC is wrong.
    """
    if len(data) < HEADER_LEN + CRC_LEN or data[0] != SYNC_RSP:
        raise ValueError(f"not a reply packet: {data[:8].hex(' ')}")
    if len(data) != packet_length(data):
        raise ValueError(f"reply length {len(data)} != {packet_length(data)} from its header")
    if not check(data[1:]):
        raise ValueError("reply CRC mismatch")
    try:
        code: Code | int = Code(data[1])
    except ValueError:
        code = data[1]
    return Reply(code=code, seq=data[2], payload=bytes(data[HEADER_LEN:-CRC_LEN]))
