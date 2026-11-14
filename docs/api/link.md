# `tinytpu.link`

The host side of the serial link: one request in flight, with the [UART protocol](../uart_protocol.md)'s resend rules (same `SEQ` on resend, 10 ms of silence first, `RESET` after 3 failures). `tb/test_link.py` tests every recovery path against a simulated board.

::: tinytpu.link
