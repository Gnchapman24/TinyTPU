# `tinytpu.packet`

UART packet framing: building requests, building and checking replies. The byte format is defined in the [UART protocol](../uart_protocol.md). The bring-up command `ECHO` (0x7E) only exists in the [bring-up design](../bringup.md).

::: tinytpu.packet
