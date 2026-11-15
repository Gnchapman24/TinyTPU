# `tinytpu.device`

The commands of the [UART protocol](../uart_protocol.md) as methods of one board, on top of [`tinytpu.link`](link.md): `info`, `status`, `write` / `read` (in each memory's unit, split into packets automatically), `run`, `infer`, `abort`, `reset`, plus helpers that load programs, weight tiles, activation vectors and per-channel parameters in the canonical layouts of the [ISA](../isa.md). Every NAK raises `DeviceError`.

The same class drives the board and the end-to-end simulation (`tb/test_tinytpu_top.py`), where the link gets a simulated serial port on the top level's pins and the simulation time as its clock.

::: tinytpu.device
