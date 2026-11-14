# `tinytpu.crc`

CRC-16/CCITT-FALSE for UART packets: the Python reference for `rtl/crc16_ccitt.sv`. `tb/test_crc16.py` checks it against the standard check value and Python's `binascii.crc_hqx`, and checks the RTL against it. See the [UART protocol](../uart_protocol.md) for where the CRC sits in a packet.

::: tinytpu.crc
