"""High-level host API for a TinyTPU board: the commands of ``docs/uart_protocol.md`` as methods.

:class:`Device` sits on a :class:`tinytpu.link.Link` (which handles packets, CRCs, resends and
timeouts) and speaks in the protocol's units: instructions, weight tiles, activation vectors,
accumulator rows and parameter entries, in the canonical layouts of ``docs/isa.md``. Transfers
larger than one packet are split automatically. Every NAK raises :class:`DeviceError`.

The same class drives the real board and the cocotb end-to-end test (which gives the link a
simulated serial port and a simulated clock), so both exercise the same host code.

Example:
    >>> from tinytpu.link import Link                   # doctest: +SKIP
    >>> from tinytpu.device import Device               # doctest: +SKIP
    >>> with Link.open("COM5") as link:                 # doctest: +SKIP
    ...     dev = Device(link)
    ...     print(dev.info())
    ...     dev.load_program(words)
    ...     status = dev.run()
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from tinytpu import isa
from tinytpu.isa import Mem, SeqErr, SeqState
from tinytpu.link import Link
from tinytpu.model import Status
from tinytpu.packet import Cmd, Code, Reply

RUN_TIMEOUT = 10.0
"""Default seconds to wait for a RUN / INFER reply (the program's run time plus the transfer)."""


class DeviceError(Exception):
    """The board answered a command with a NAK.

    Attributes:
        cmd: The command.
        code: The NAK code.
    """

    def __init__(self, cmd: Cmd, code: Code | int):
        self.cmd = cmd
        self.code = code
        name = code.name if isinstance(code, Code) else f"{int(code):#04x}"
        super().__init__(f"{cmd.name} was refused: NAK {name}")


@dataclass(frozen=True)
class Info:
    """The board's INFO block.

    Attributes:
        version: Protocol version.
        n: Array size N.
        clks_per_bit: UART bit length in clock cycles.
        depths: Depth of every memory, in its units.
        max_payload: Largest packet payload in bytes.
    """

    version: int
    n: int
    clks_per_bit: int
    depths: dict[Mem, int]
    max_payload: int


def _status(payload: bytes) -> Status:
    state, err, pc, halt = struct.unpack("<BBHH", payload[:6])
    return Status(SeqState(state), SeqErr(err), pc, halt)


class Device:
    """One TinyTPU board.

    Args:
        link: An open link to the board.
        run_timeout: Default seconds to wait for a RUN / INFER reply.
    """

    def __init__(self, link: Link, run_timeout: float = RUN_TIMEOUT):
        self.link = link
        self.run_timeout = run_timeout
        self._info: Info | None = None

    # ---------------------------------------------------------------------------------------------
    # Plumbing
    # ---------------------------------------------------------------------------------------------

    def _cmd(self, cmd: Cmd, payload: bytes = b"", timeout: float | None = None) -> Reply:
        reply = self.link.transact(cmd, payload, timeout)
        if reply.code != Code.ACK:
            raise DeviceError(cmd, reply.code)
        return reply

    def unit_bytes(self, mem: Mem) -> int:
        """Bytes per unit of ``mem`` on the wire (depends on N)."""
        n = self.info().n
        return {Mem.IMEM: isa.INSTR_BYTES, Mem.WBUF: n * n, Mem.ABUF: n, Mem.ACC: 4 * n, Mem.PTAB: 8}[Mem(mem)]

    # ---------------------------------------------------------------------------------------------
    # Commands
    # ---------------------------------------------------------------------------------------------

    def info(self, refresh: bool = False) -> Info:
        """Return the INFO block (cached after the first call)."""
        if self._info is None or refresh:
            f = struct.unpack("<BBHHHHHHH", self._cmd(Cmd.INFO).payload)
            depths = dict(zip((Mem.IMEM, Mem.WBUF, Mem.ABUF, Mem.ACC, Mem.PTAB), f[3:8]))
            self._info = Info(version=f[0], n=f[1], clks_per_bit=f[2], depths=depths, max_payload=f[8])
        return self._info

    def status(self) -> Status:
        """Return the sequencer status."""
        return _status(self._cmd(Cmd.STATUS).payload)

    def write(self, mem: Mem, addr: int, data: bytes) -> None:
        """Write whole units of ``mem`` starting at unit ``addr`` (split into packets as needed)."""
        unit = self.unit_bytes(mem)
        if len(data) % unit:
            raise ValueError(f"{len(data)} bytes is not a whole number of {Mem(mem).name} units ({unit} B)")
        per = max(1, (self.info().max_payload - 6) // unit)
        for i in range(0, len(data) // unit, per):
            chunk = data[i * unit:(i + per) * unit]
            self._cmd(Cmd.WRITE, struct.pack("<BBHH", int(mem), 0, addr + i, len(chunk) // unit) + chunk)

    def read(self, mem: Mem, addr: int, count: int) -> bytes:
        """Read ``count`` units of ``mem`` starting at unit ``addr`` (split into packets as needed)."""
        unit = self.unit_bytes(mem)
        per = max(1, self.info().max_payload // unit)
        out = bytearray()
        for i in range(0, count, per):
            n = min(per, count - i)
            out += self._cmd(Cmd.READ, struct.pack("<BBHH", int(mem), 0, addr + i, n)).payload
        return bytes(out)

    def run(self, start_pc: int = 0, timeout: float | None = None) -> Status:
        """Run the stored program from ``start_pc``; return the status once it halts."""
        reply = self._cmd(Cmd.RUN, struct.pack("<H", start_pc), self.run_timeout if timeout is None else timeout)
        return _status(reply.payload)

    def infer(self, inputs: bytes, *, out_addr: int, out_count: int, in_addr: int = 0, start_pc: int = 0,
              out_mem: Mem = Mem.ABUF, timeout: float | None = None) -> tuple[Status, bytes]:
        """One inference in one round trip: write ``inputs`` (whole ABUF vectors) at ``in_addr``,
        run from ``start_pc``, and return the status and ``out_count`` units of ``out_mem``."""
        if len(inputs) % self.unit_bytes(Mem.ABUF):
            raise ValueError("INFER input must be whole activation vectors")
        args = struct.pack("<HHBBHH", in_addr, start_pc, int(out_mem), 0, out_addr, out_count)
        reply = self._cmd(Cmd.INFER, args + inputs, self.run_timeout if timeout is None else timeout)
        return _status(reply.payload), reply.payload[8:]

    def abort(self) -> Status:
        """Stop a running program; return the status (idle)."""
        return _status(self._cmd(Cmd.ABORT).payload)

    def reset(self) -> Status:
        """Return the sequencer and the protocol state to idle (memories are kept)."""
        return _status(self._cmd(Cmd.RESET).payload)

    # ---------------------------------------------------------------------------------------------
    # Canonical-layout helpers
    # ---------------------------------------------------------------------------------------------

    def load_program(self, words: list[int], at: int = 0) -> None:
        """Write 128-bit instruction words into IMEM."""
        self.write(Mem.IMEM, at, b"".join(isa.to_bytes(w) for w in words))

    def load_weights(self, tiles: np.ndarray, at: int = 0) -> None:
        """Write weight tiles ``(T, N, N)`` int8 (from :func:`tinytpu.model.pack_weights`)."""
        self.write(Mem.WBUF, at, np.asarray(tiles, dtype=np.int8).tobytes())

    def load_vectors(self, vecs: np.ndarray, at: int = 0) -> None:
        """Write activation vectors ``(V, N)`` (raw bytes; from :func:`tinytpu.model.pack_activations`)."""
        self.write(Mem.ABUF, at, (np.asarray(vecs, dtype=np.int64) & 0xFF).astype(np.uint8).tobytes())

    def load_params(self, bias, mult, shift, at: int = 0) -> None:
        """Write per-channel parameters (bias INT32, M u16, s 0..63) for consecutive channels."""
        self.write(Mem.PTAB, at, b"".join(isa.ptab_entry(int(b), int(m), int(s)) for b, m, s in zip(bias, mult, shift)))

    def read_vectors(self, addr: int, count: int) -> np.ndarray:
        """Read activation vectors as ``(count, N)`` uint8 (raw bytes)."""
        return np.frombuffer(self.read(Mem.ABUF, addr, count), dtype=np.uint8).reshape(count, self.info().n)

    def read_acc(self, addr: int, count: int) -> np.ndarray:
        """Read accumulator rows as ``(count, N)`` int32."""
        return np.frombuffer(self.read(Mem.ACC, addr, count), dtype="<i4").reshape(count, self.info().n).astype(np.int32)
