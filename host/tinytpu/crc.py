"""CRC-16/CCITT-FALSE, the checksum on every TinyTPU UART packet (Python reference for ``rtl/crc16_ccitt.sv``).

Polynomial 0x1021, initial value 0xFFFF, no reflection, final XOR 0x0000. On the wire the CRC is
sent **high byte first**, and it covers every packet byte after the sync byte
(``docs/uart_protocol.md``).

Example:
    >>> from tinytpu.crc import crc16, crc_bytes, check
    >>> hex(crc16(b"123456789"))
    '0x29b1'
    >>> frame = b"\\x01\\x2a\\x00\\x00"
    >>> check(frame + crc_bytes(frame))
    True
"""

from __future__ import annotations

POLY = 0x1021
INIT = 0xFFFF
CHECK_VALUE = 0x29B1
"""CRC of the ASCII bytes ``b"123456789"``."""


def crc16(data: bytes, crc: int = INIT) -> int:
    """Return the CRC-16/CCITT-FALSE of ``data``.

    Args:
        data: Message bytes.
        crc: Starting value, for continuing a CRC across several calls. Defaults to the initial
            value 0xFFFF.

    Returns:
        The 16-bit CRC.
    """
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ POLY) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def crc_bytes(data: bytes) -> bytes:
    """Return the 2 CRC bytes to append to ``data``, high byte first (wire order)."""
    return crc16(data).to_bytes(2, "big")


def check(data_with_crc: bytes) -> bool:
    """Return ``True`` if a message followed by its 2 wire-order CRC bytes is intact.

    Running the CRC over the message and its own CRC (high byte first) leaves zero exactly when
    they match, which is also how the hardware checks packets.
    """
    return len(data_with_crc) >= 2 and crc16(data_with_crc) == 0
