"""MNIST on TinyTPU: train an MLP in NumPy, quantize, compile, and classify on the board (task A7.5).

One command does every step, and caches what it can::

    python -m tinytpu.mnist                 # download, train, quantize, compile; float and integer accuracy
    python -m tinytpu.mnist --port COM7     # ... then classify the test set on the board

1. **Download** the four MNIST files from torchvision's mirror (:data:`MIRROR`), check their SHA-256
   hashes (:data:`FILES`), and keep them in a cache outside the repo (:func:`data_dir`).
2. **Train** a 784-128-10 MLP (``--size 14``: 196-128-10 on 2x2-averaged images) with ReLU,
   softmax cross-entropy and Adam, from a fixed seed. The float weights are cached.
3. **Quantize** with 1,000 training images for calibration and **compile** for the board (or for
   the default N = 8 target without one).
4. **Run** the test images on the board, in batches of one ``INFER`` each.
5. **Report** float, integer and board accuracy; how many board results equal the integer
   reference exactly (all of them must: the first batch is also checked on ``model.Machine``);
   the speed; and a few digits drawn as text.

See ``docs/compiler.md`` (MNIST demo) and ``docs/host.md``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinytpu import compiler, nn, quant
from tinytpu.compiler import Compiled, Runner, Target

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
"""torchvision's primary MNIST mirror (the original files, gzipped)."""

FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
"""SHA-256 of each file. Pinned 2026-11-16; the MD5 sums matched those torchvision publishes."""

CALIBRATION = 1000
"""Training images used to set the activation scales."""


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------

def data_dir(override: str | Path | None = None) -> Path:
    """The cache folder: ``override``, else ``$TINYTPU_DATA``, else ``%LOCALAPPDATA%\\tinytpu\\mnist``
    on Windows or ``~/.cache/tinytpu/mnist`` elsewhere."""
    if override:
        return Path(override)
    if os.environ.get("TINYTPU_DATA"):
        return Path(os.environ["TINYTPU_DATA"])
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "tinytpu" / "mnist"
    return Path.home() / ".cache" / "tinytpu" / "mnist"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(directory: Path, url: str = MIRROR, files: dict[str, str] | None = None,
          out: Callable[[str], None] = print) -> None:
    """Make sure every file is in ``directory`` with the right hash, downloading what isn't.

    A file is downloaded to a temporary name and only renamed once its hash is right, so a broken
    download never leaves a file that looks valid.

    Raises:
        ValueError: If a downloaded file has the wrong hash.
        OSError: If a download fails.
    """
    files = FILES if files is None else files
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        path = directory / name
        if path.exists() and _sha256(path) == digest:
            continue
        out(f"downloading {url}{name}")
        tmp = path.with_suffix(path.suffix + ".part")
        req = urllib.request.Request(url + name, headers={"User-Agent": "tinytpu"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        if _sha256(tmp) != digest:
            tmp.unlink()
            raise ValueError(f"{name}: downloaded file has the wrong SHA-256; not using it")
        tmp.replace(path)


def read_idx(path: Path) -> np.ndarray:
    """Read a gzipped IDX file of unsigned bytes: images ``(n, rows, cols)`` or labels ``(n,)``.

    Raises:
        ValueError: If the header is not an unsigned-byte IDX header or the data is short.
    """
    data = gzip.decompress(Path(path).read_bytes())
    if len(data) < 4 or data[0:3] != b"\x00\x00\x08" or not 1 <= data[3] <= 3:
        raise ValueError(f"{path}: not an unsigned-byte IDX file")
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(data[3])]
    start, size = 4 + 4 * len(dims), int(np.prod(dims))
    if len(data) != start + size:
        raise ValueError(f"{path}: expected {size} data bytes, found {len(data) - start}")
    return np.frombuffer(data, dtype=np.uint8, offset=start).reshape(dims)


@dataclass
class Dataset:
    """MNIST (or a stand-in with the same file layout) as raw bytes.

    Attributes:
        train_images: ``(n, rows, cols)`` uint8.
        train_labels: ``(n,)`` uint8.
        test_images: ``(m, rows, cols)`` uint8.
        test_labels: ``(m,)`` uint8.
    """

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def load(directory: Path, download: bool = True, out: Callable[[str], None] = print) -> Dataset:
    """Load the dataset from ``directory``, downloading missing files if ``download``."""
    if download:
        fetch(directory, out=out)
    parts = [read_idx(directory / name) for name in FILES]
    ds = Dataset(*parts)
    if len(ds.train_images) != len(ds.train_labels) or len(ds.test_images) != len(ds.test_labels):
        raise ValueError("image and label counts differ")
    return ds


def prepare(images: np.ndarray, pool: int = 1) -> np.ndarray:
    """Pixels to model inputs ``(n, pixels)`` in 0..1, averaging ``pool`` x ``pool`` blocks."""
    x = np.asarray(images, dtype=np.float64) / 255.0
    if pool > 1:
        n, r, c = x.shape
        if r % pool or c % pool:
            raise ValueError(f"{r}x{c} images don't divide into {pool}x{pool} blocks")
        x = x.reshape(n, r // pool, pool, c // pool, pool).mean(axis=(2, 4))
    return x.reshape(len(x), -1)


# ------------------------------------------------------------------------------------------------
# Training (NumPy only)
# ------------------------------------------------------------------------------------------------

def loss_and_grads(params: list[np.ndarray], xb: np.ndarray, tb: np.ndarray) -> tuple[float, list[np.ndarray]]:
    """Mean softmax cross-entropy of the MLP ``[w1, b1, w2, b2]`` on a batch, and its gradients.

    Args:
        params: ``w1 (k, h)``, ``b1 (h,)``, ``w2 (h, c)``, ``b2 (c,)``.
        xb: Inputs ``(n, k)``.
        tb: One-hot targets ``(n, c)``.
    """
    w1, c1, w2, c2 = params
    h = np.maximum(xb @ w1 + c1, 0)
    z = h @ w2 + c2
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    loss = float(-(tb * np.log(p + 1e-12)).sum()) / len(xb)
    dz = (p - tb) / len(xb)
    dh = (dz @ w2.T) * (h > 0)
    return loss, [xb.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]


def train(x: np.ndarray, y: np.ndarray, hidden: int = 128, epochs: int = 8, seed: int = 0, classes: int = 10,
          lr: float = 1e-3, batch: int = 128, out: Callable[[str], None] = print) -> nn.Model:
    """Train an MLP ``pixels -> hidden (ReLU) -> classes`` with softmax cross-entropy and Adam.

    Args:
        x: Inputs ``(n, pixels)`` in 0..1.
        y: Labels ``(n,)`` in 0..classes-1.
        hidden: Hidden-layer width.
        epochs: Passes over the data.
        seed: Seed for the initial weights and the shuffling.
        classes: Number of classes.
        lr: Adam learning rate.
        batch: Minibatch size.
        out: Progress printer.
    """
    rng = np.random.default_rng(seed)
    xf, k = np.asarray(x, dtype=np.float32), x.shape[1]
    params = [rng.normal(0, np.sqrt(2 / k), (k, hidden)).astype(np.float32), np.zeros(hidden, np.float32),
              rng.normal(0, np.sqrt(2 / hidden), (hidden, classes)).astype(np.float32), np.zeros(classes, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    onehot = np.eye(classes, dtype=np.float32)[np.asarray(y)]
    for epoch in range(epochs):
        order = rng.permutation(len(xf))
        loss = 0.0
        for i in range(0, len(xf), batch):
            idx = order[i:i + batch]
            batch_loss, grads = loss_and_grads(params, xf[idx], onehot[idx])
            loss += batch_loss * len(idx)
            step += 1
            for j, g in enumerate(grads):
                m[j] = b1 * m[j] + (1 - b1) * g
                v[j] = b2 * v[j] + (1 - b2) * g * g
                params[j] -= lr * (m[j] / (1 - b1 ** step)) / (np.sqrt(v[j] / (1 - b2 ** step)) + eps)
        out(f"  epoch {epoch + 1}/{epochs}: loss {loss / len(xf):.4f}")
    w1, c1, w2, c2 = (p.astype(np.float64) for p in params)
    return nn.Model([nn.Dense(w1, c1, relu=True), nn.Dense(w2, c2)])


def trained_model(ds: Dataset, directory: Path, pool: int, hidden: int, epochs: int, seed: int,
                  retrain: bool = False, out: Callable[[str], None] = print) -> nn.Model:
    """Train, or load the float weights cached from an earlier run with the same settings."""
    pixels = ds.train_images.shape[1] * ds.train_images.shape[2] // (pool * pool)
    path = directory / f"mlp-{pixels}-{hidden}-e{epochs}-s{seed}.npz"
    if path.exists() and not retrain:
        with np.load(path, allow_pickle=False) as f:
            out(f"using the trained weights in {path}")
            return nn.Model([nn.Dense(f["w1"], f["b1"], relu=True), nn.Dense(f["w2"], f["b2"])])
    out(f"training {pixels}-{hidden}-10 for {epochs} epochs (seed {seed})")
    net = train(prepare(ds.train_images, pool), ds.train_labels, hidden, epochs, seed, out=out)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(path, w1=net.layers[0].w, b1=net.layers[0].b, w2=net.layers[1].w, b2=net.layers[1].b)
    return net


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows whose highest score is the label."""
    return float(np.mean(np.argmax(scores, axis=1) == np.asarray(labels))) if len(labels) else 0.0


# ------------------------------------------------------------------------------------------------
# Board
# ------------------------------------------------------------------------------------------------

@dataclass
class BoardResult:
    """What the board did with the test images.

    Attributes:
        acc: The board's INT32 outputs ``(count, 10)``.
        exact: How many rows equal the integer reference exactly.
        machine_ok: Whether the first batch also equals ``model.Machine``'s.
        seconds: Wall time of all the ``INFER`` round trips.
    """

    acc: np.ndarray
    exact: int
    machine_ok: bool
    seconds: float


def run_on_board(runner: Runner, q: quant.QModel, x: np.ndarray, progress: int = 0,
                 out: Callable[[str], None] = print) -> BoardResult:
    """Classify ``x`` (model inputs in 0..1) on the board and check every output.

    Args:
        runner: The compiled model, loaded on the board.
        q: The quantized model: the integer reference.
        x: Model inputs ``(count, pixels)``.
        progress: Print a line every this many images (0: never).
        out: Progress printer.
    """
    qx = q.quantize_input(x)
    c = runner.compiled
    step = progress if progress > 0 else max(len(qx), 1)
    step = max(c.batch, step - step % c.batch)               # whole batches per chunk
    parts = []
    start = time.perf_counter()
    for i in range(0, len(qx), step):
        parts.append(runner.infer_int(qx[i:i + step]))
        if progress and i + step < len(qx):
            out(f"  {i + step} / {len(qx)}")
    seconds = time.perf_counter() - start
    acc = np.concatenate(parts) if parts else np.zeros((0, c.n_out), dtype=np.int32)
    exact = int(np.all(acc == q.forward_int(qx), axis=1).sum())
    first = qx[:c.batch]
    machine_ok = bool(np.array_equal(c.simulator().infer_int(first), acc[:len(first)]))
    return BoardResult(acc, exact, machine_ok, seconds)


def show_digits(images: np.ndarray, labels: np.ndarray, predictions: np.ndarray, count: int = 3) -> str:
    """Draw the first ``count`` images as text, side by side, with label and prediction."""
    shades = " .:-=+*#%@"
    blocks = []
    for img, lab, pred in list(zip(images, labels, predictions))[:count]:
        a = np.asarray(img, dtype=np.float64)
        if len(a) % 2:
            a = np.vstack([a, np.zeros((1, a.shape[1]))])
        a = (a[0::2] + a[1::2]) / 2                           # two pixel rows per text row
        rows = ["".join(shades[min(int(v / 256 * len(shades)), len(shades) - 1)] for v in r) for r in a]
        mark = "ok" if lab == pred else "WRONG"
        rows.append(f"label {lab}  board {pred}  {mark}".ljust(a.shape[1]))
        blocks.append(rows)
    if not blocks:
        return ""
    return "\n".join("   ".join(b[i] for b in blocks) for i in range(len(blocks[0])))


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    from tinytpu.device import Device
    from tinytpu.link import BAUD, Link

    ap = argparse.ArgumentParser(prog="python -m tinytpu.mnist", description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port, e.g. COM7; without it, stop after compiling")
    ap.add_argument("--baud", type=int, default=BAUD,
                    help="bit rate; must match the bitstream (default 1000000; the fallback builds use 115200)")
    ap.add_argument("--size", type=int, choices=(28, 14), default=28, help="image side: 28 (784 inputs) or 14 (196)")
    ap.add_argument("--hidden", type=int, default=128, help="hidden-layer width")
    ap.add_argument("--epochs", type=int, default=8, help="training epochs")
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--count", type=int, default=10000, help="test images to classify on the board")
    ap.add_argument("--show", type=int, default=3, help="digits to draw")
    ap.add_argument("--data-dir", help="cache folder (default: $TINYTPU_DATA or the user cache)")
    ap.add_argument("--retrain", action="store_true", help="train again even if cached weights exist")
    args = ap.parse_args(argv)

    directory = data_dir(args.data_dir)
    ds = load(directory)
    pool = ds.train_images.shape[1] // args.size if ds.train_images.shape[1] == 28 else 1
    net = trained_model(ds, directory, pool, args.hidden, args.epochs, args.seed, args.retrain)
    x_test = prepare(ds.test_images, pool)
    calib = prepare(ds.train_images[np.random.default_rng(args.seed).permutation(len(ds.train_images))[:CALIBRATION]],
                    pool)
    q = quant.quantize(net, calib)
    float_acc = accuracy(net.forward(x_test), ds.test_labels)
    int_acc = accuracy(q.forward(x_test), ds.test_labels)
    print(f"test accuracy: float {100 * float_acc:.2f}%, integer {100 * int_acc:.2f}% ({len(x_test)} images)")

    if not args.port:
        c = compiler.compile_model(q, Target.default(8))
        path = directory / f"mnist-{x_test.shape[1]}-{args.hidden}.npz"
        c.save(path)
        print(f"compiled for N = 8 (default depths): batch {c.batch}, {len(c.tiles)} weight tiles, "
              f"{len(c.ptab_bias)} channels; saved {path}")
        print("add --port COMx to run it on the board")
        return 0

    count = min(args.count, len(x_test))
    with Link.open(args.port, args.baud) as link:
        link.wait_for_silence()
        dev = Device(link)
        c: Compiled = compiler.compile_model(q, Target.from_info(dev.info()))
        print(f"loading onto the board (N = {c.target.n}): {len(c.tiles)} weight tiles, {len(c.ptab_bias)} "
              f"channels, batch {c.batch}")
        t0 = time.perf_counter()
        runner = c.load(dev)
        if not runner.verify():
            print("FAIL: the image read back from the board differs from what was written")
            return 1
        print(f"  loaded and verified in {time.perf_counter() - t0:.1f} s")
        print(f"classifying {count} test images")
        res = run_on_board(runner, q, x_test[:count], progress=1000)

    preds = np.argmax(q.dequantize(res.acc), axis=1)
    labels = ds.test_labels[:count]
    board_acc = accuracy(q.dequantize(res.acc), labels)
    ms = 1000 * res.seconds / max(count, 1)
    print()
    print(show_digits(ds.test_images, labels, preds, args.show))
    print()
    print(f"board accuracy:  {100 * board_acc:.2f}% on {count} images "
          f"(float {100 * accuracy(net.forward(x_test[:count]), labels):.2f}%, "
          f"integer {100 * accuracy(q.forward(x_test[:count]), labels):.2f}%)")
    print(f"exact match:     {res.exact} / {count} outputs equal the integer reference; "
          f"first batch {'equals' if res.machine_ok else 'DIFFERS FROM'} model.Machine")
    print(f"speed:           {ms:.2f} ms per image, {count / res.seconds:.0f} images/s "
          f"({res.seconds:.1f} s, {-(-count // c.batch)} INFERs of {c.batch})")
    ok = res.exact == count and res.machine_ok
    print("PASS" if ok else "FAIL: the board's results differ from the reference")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
