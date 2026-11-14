"""A Python model of the bring-up board (rtl/loopback_echo.sv) behind a pyserial-like interface.

Used to test tinytpu.link and tinytpu.bringup without hardware. It follows the same rules as the
RTL: ECHO with up to 256 bytes gets an ACK and the payload back; bad CRC, too long, and unknown
commands get NAKs (checked in that order); a partial request is dropped after an idle gap.

Fault injection (each counter applies to that many upcoming packets):
    drop_replies:     replies that are never sent (lost)
    corrupt_replies:  replies with one bit flipped
    corrupt_requests: requests with one bit flipped on arrival (the board then NAKs BAD_CRC)
    junk_before:      bytes sent before the next reply
    stale_first:      send an extra reply with the wrong SEQ before the next real reply
    wrong_payload:    echo a different payload (a broken board)
"""

from __future__ import annotations

import time

from tinytpu.crc import check
from tinytpu.packet import SYNC_REQ, Cmd, Code, build_reply

MAX_PAYLOAD = 256


class FakeEchoBoard:
    def __init__(self, raw: bool = False, rx_timeout: float = 0.003):
        self.raw = raw
        self.rx_timeout = rx_timeout
        self.timeout: float | None = 0.01
        self.inbuf = bytearray()
        self.outbuf = bytearray()
        self.last_rx = 0.0
        self.requests: list[bytes] = []       # complete requests as received
        self.drop_replies = 0
        self.corrupt_replies = 0
        self.corrupt_requests = 0
        self.junk_before = b""
        self.stale_first = 0
        self.wrong_payload = False
        self.closed = False

    # pyserial-like interface ---------------------------------------------------------------------

    def write(self, data: bytes) -> int:
        now = time.monotonic()
        if self.raw:
            self.outbuf += data
            return len(data)
        if self.inbuf and now - self.last_rx > self.rx_timeout:
            self.inbuf.clear()                      # partial request timed out
        self.last_rx = now
        self.inbuf += data
        self._parse()
        return len(data)

    def read(self, size: int = 1) -> bytes:
        if not self.outbuf:
            time.sleep(self.timeout or 0)
            return b""
        out = bytes(self.outbuf[:size])
        del self.outbuf[:size]
        return out

    def close(self) -> None:
        self.closed = True

    # model ---------------------------------------------------------------------------------------

    def _parse(self) -> None:
        while True:
            while self.inbuf and self.inbuf[0] != SYNC_REQ:
                del self.inbuf[0]                   # junk before a sync byte is ignored
            if len(self.inbuf) < 5:
                return
            total = 5 + int.from_bytes(self.inbuf[3:5], "little") + 2
            if len(self.inbuf) < total:
                return
            packet = bytearray(self.inbuf[:total])
            del self.inbuf[:total]
            if self.corrupt_requests:
                self.corrupt_requests -= 1
                packet[len(packet) // 2] ^= 0x10
            self.requests.append(bytes(packet))
            self._reply(bytes(packet))

    def _reply(self, packet: bytes) -> None:
        cmd, seq, payload = packet[1], packet[2], packet[5:-2]
        if not check(packet[1:]):
            reply = build_reply(Code.BAD_CRC, seq)
        elif len(payload) > MAX_PAYLOAD:
            reply = build_reply(Code.TOO_LONG, seq)
        elif cmd != Cmd.ECHO:
            reply = build_reply(Code.UNKNOWN_CMD, seq)
        else:
            echoed = bytes(reversed(payload)) if self.wrong_payload and len(payload) > 1 else payload
            reply = build_reply(Code.ACK, seq, echoed)
        if self.drop_replies:
            self.drop_replies -= 1
            return
        if self.stale_first:
            self.stale_first -= 1
            self.outbuf += build_reply(Code.ACK, (seq - 1) & 0xFF, b"stale")
        if self.junk_before:
            self.outbuf += self.junk_before
            self.junk_before = b""
        if self.corrupt_replies:
            self.corrupt_replies -= 1
            reply = bytearray(reply)
            reply[-3] ^= 0x01
            reply = bytes(reply)
        self.outbuf += reply
