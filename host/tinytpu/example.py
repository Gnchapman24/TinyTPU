"""Example: multiply two matrices on a TinyTPU board and check the result (task A6.2).

A minimal, commented starting point for your own host code. It computes ``Y = X @ W`` for
random INT8 weights and UINT8 activations of any size that fits, the way any program uses the
board:

1. pad and tile the matrices into the canonical layouts (:mod:`tinytpu.model` helpers);
2. upload the weight tiles and activation vectors (:class:`tinytpu.device.Device`);
3. upload a two-instruction program: one ``MATMUL`` for the whole product, then ``HALT``;
4. ``RUN`` it, read the INT32 accumulators back, undo the tiling;
5. compare with NumPy.

Windows usage (host venv, see tools/setup.md)::

    uv run --no-default-groups python -m tinytpu.example --port COM7 --m 4 --k 50 --n 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from tinytpu import isa, model
from tinytpu.device import Device
from tinytpu.isa import AccMode, SeqState
from tinytpu.link import Link


def matmul_on_board(dev: Device, x: np.ndarray, w: np.ndarray, act_signed: bool) -> np.ndarray:
    """Return ``x @ w`` computed on the board, as INT32 (wrapping like the hardware).

    Args:
        dev: The board.
        x: Activations ``(M, K)``: UINT8 values (0..255), or INT8 (-128..127) with ``act_signed``.
        w: Weights ``(K, Nout)``, INT8 (the host quantizer keeps them in -127..127).
        act_signed: Whether ``x`` is INT8 (else UINT8).
    """
    n = dev.info().n
    m_dim, k_dim = x.shape
    out_dim = w.shape[1]
    kt, nt = model.ceil_div(k_dim, n), model.ceil_div(out_dim, n)
    if not model.k_fits(k_dim):
        raise ValueError(f"K = {k_dim} could overflow the INT32 accumulators")

    # 1 + 2: canonical layouts (weight tile n*KT + k, vector m*KT + k), zero-padded to N.
    dev.load_weights(model.pack_weights(w, n), at=0)
    dev.load_vectors(model.pack_activations(x, n), at=0)

    # 3: one MATMUL covers the whole product: ACC row m*NT + n gets X[m] . W[:, n-th tile].
    dev.load_program([
        isa.matmul(w_base=0, a_base=0, acc_base=0, kt=kt, nt=nt, m=m_dim, act_signed=act_signed,
                   acc_mode=AccMode.OVERWRITE),
        isa.halt(1),
    ])

    # 4: run, then read the M*NT accumulator rows and drop the padding.
    status = dev.run()
    if status.state != SeqState.HALTED_OK:
        raise RuntimeError(f"program stopped with {status}")
    return model.unpack_acc(dev.read_acc(0, m_dim * nt), m_dim, out_dim)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(prog="python -m tinytpu.example", description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="serial port, e.g. COM7 (python -m tinytpu.selftest --probe)")
    ap.add_argument("--m", type=int, default=4, help="rows of X")
    ap.add_argument("--k", type=int, default=50, help="columns of X = rows of W")
    ap.add_argument("--n", type=int, default=20, help="columns of W")
    ap.add_argument("--signed", action="store_true", help="INT8 activations instead of UINT8")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    w = rng.integers(-127, 128, size=(args.k, args.n))
    x = rng.integers(-128, 128, size=(args.m, args.k)) if args.signed else rng.integers(0, 256, size=(args.m, args.k))

    with Link.open(args.port) as link:
        dev = Device(link)
        print(f"Board: N = {dev.info().n}; computing ({args.m} x {args.k}) @ ({args.k} x {args.n})")
        y = matmul_on_board(dev, x, w, args.signed)

    want = x.astype(np.int64) @ w.astype(np.int64)        # plain NumPy, no TinyTPU code
    ok = np.array_equal(y, want)
    print(y if y.size <= 64 else f"{y.shape} result")
    print("matches NumPy" if ok else f"MISMATCH in {np.count_nonzero(y != want)} of {y.size} values")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
