"""Tests for the MNIST demo tinytpu.mnist (task A7.5), with no network access.

A synthetic stand-in for MNIST (small images whose class is the bright quadrant), written in the
same gzipped IDX format, is "downloaded" from a local file:// folder, so the hash check, the cache
and the parser run exactly as they do with the real files. The board half of the demo runs on the
simulated board in tb/test_tinytpu_top.py (``mnist_demo``).
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tinytpu import compiler, mnist, quant
from tinytpu.compiler import Target


def idx_bytes(a: np.ndarray) -> bytes:
    a = np.asarray(a, dtype=np.uint8)
    head = b"\x00\x00\x08" + bytes([a.ndim]) + b"".join(int(d).to_bytes(4, "big") for d in a.shape)
    return gzip.compress(head + a.tobytes())


def quadrant_images(rng: np.random.Generator, n: int, side: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Images with one bright quadrant (the label, 0..3) on a noisy background."""
    labels = rng.integers(0, 4, n)
    img = rng.integers(0, 60, (n, side, side))
    h = side // 2
    for i, lab in enumerate(labels):
        r, c = divmod(int(lab), 2)
        img[i, r * h:(r + 1) * h, c * h:(c + 1) * h] += rng.integers(140, 196)
    return img.astype(np.uint8), labels.astype(np.uint8)


def write_synthetic(folder: Path, seed: int = 0, side: int = 8) -> dict[str, str]:
    """Write the four files; return their SHA-256 table (like mnist.FILES)."""
    rng = np.random.default_rng(seed)
    tr, trl = quadrant_images(rng, 600, side)
    te, tel = quadrant_images(rng, 200, side)
    folder.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, a in zip(mnist.FILES, (tr, trl, te, tel)):
        data = idx_bytes(a)
        (folder / name).write_bytes(data)
        table[name] = hashlib.sha256(data).hexdigest()
    return table


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """A local 'mirror' with synthetic files, and mnist.FILES pointing at their hashes."""
    src = tmp_path / "mirror"
    table = write_synthetic(src)
    monkeypatch.setattr(mnist, "FILES", table)
    return src.as_uri() + "/", table


def silent(_: str) -> None:
    pass


# ------------------------------------------------------------------------------------------------
# Download, cache and parsing
# ------------------------------------------------------------------------------------------------

def test_fetch_downloads_and_checks(mirror, tmp_path):
    url, table = mirror
    cache = tmp_path / "cache"
    lines = []
    mnist.fetch(cache, url, out=lines.append)
    assert len(lines) == 4
    assert {p.name for p in cache.iterdir()} == set(table)
    lines.clear()
    mnist.fetch(cache, url, out=lines.append)              # cached and intact: nothing downloaded
    assert lines == []
    victim = cache / next(iter(table))
    victim.write_bytes(b"corrupt")                          # a damaged cache file is replaced
    mnist.fetch(cache, url, out=lines.append)
    assert len(lines) == 1 and hashlib.sha256(victim.read_bytes()).hexdigest() == table[victim.name]


def test_fetch_rejects_a_bad_download(mirror, tmp_path, monkeypatch):
    url, table = mirror
    bad = dict(table)
    name = next(iter(bad))
    bad[name] = "0" * 64
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="wrong SHA-256"):
        mnist.fetch(cache, url, files=bad, out=silent)
    assert not (cache / name).exists() and not list(cache.glob("*.part"))


def test_pinned_hashes_are_the_real_files():
    """The pinned table covers exactly the four MNIST files, as hex SHA-256."""
    assert list(mnist.FILES) == ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in mnist.FILES.values())
    assert mnist.MIRROR.startswith("https://") and mnist.MIRROR.endswith("/")


def test_read_idx(tmp_path):
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    (tmp_path / "a.gz").write_bytes(idx_bytes(a))
    assert np.array_equal(mnist.read_idx(tmp_path / "a.gz"), a)
    (tmp_path / "l.gz").write_bytes(idx_bytes(np.array([7, 1, 9])))
    assert mnist.read_idx(tmp_path / "l.gz").tolist() == [7, 1, 9]
    (tmp_path / "short.gz").write_bytes(gzip.compress(gzip.decompress(idx_bytes(a))[:-1]))
    with pytest.raises(ValueError, match="expected 24"):
        mnist.read_idx(tmp_path / "short.gz")
    (tmp_path / "long.gz").write_bytes(gzip.compress(gzip.decompress(idx_bytes(a)) + b"\x00"))
    with pytest.raises(ValueError, match="expected 24"):
        mnist.read_idx(tmp_path / "long.gz")
    (tmp_path / "int.gz").write_bytes(gzip.compress(b"\x00\x00\x0c\x01" + (3).to_bytes(4, "big") + bytes(12)))
    with pytest.raises(ValueError, match="unsigned-byte"):
        mnist.read_idx(tmp_path / "int.gz")


def test_data_dir(monkeypatch, tmp_path):
    assert mnist.data_dir(tmp_path) == tmp_path
    monkeypatch.setenv("TINYTPU_DATA", str(tmp_path / "env"))
    assert mnist.data_dir() == tmp_path / "env"
    monkeypatch.delenv("TINYTPU_DATA")
    assert mnist.data_dir().parts[-2:] == ("tinytpu", "mnist")


def test_prepare_scales_and_pools():
    img = np.array([[[0, 255, 51, 51], [255, 255, 51, 51], [0, 0, 102, 0], [0, 0, 0, 0]]], dtype=np.uint8)
    assert np.allclose(mnist.prepare(img), img.reshape(1, 16) / 255)
    assert np.allclose(mnist.prepare(img, pool=2), [[0.75, 0.2, 0.0, 0.1]])
    with pytest.raises(ValueError, match="blocks"):
        mnist.prepare(np.zeros((1, 3, 3)), pool=2)


# ------------------------------------------------------------------------------------------------
# Training, quantization and the integer path
# ------------------------------------------------------------------------------------------------

def test_training_converges_and_quantizes(mirror, tmp_path):
    url, _ = mirror
    cache = tmp_path / "cache"
    mnist.fetch(cache, url, out=silent)
    ds = mnist.load(cache, download=False)
    net = mnist.trained_model(ds, cache, pool=1, hidden=16, epochs=15, seed=1, out=silent)
    assert (net.k_in, net.n_out) == (64, 10)
    x_test = mnist.prepare(ds.test_images)
    assert mnist.accuracy(net.forward(x_test), ds.test_labels) > 0.95
    q = quant.quantize(net, mnist.prepare(ds.train_images[:100]))
    assert mnist.accuracy(q.forward(x_test), ds.test_labels) > 0.95
    assert q.input_scale == pytest.approx(ds.train_images[:100].max() / 255 / 255)
    # Cached weights: the second call loads instead of training.
    lines = []
    again = mnist.trained_model(ds, cache, pool=1, hidden=16, epochs=15, seed=1, out=lines.append)
    assert "using the trained weights" in lines[0]
    assert np.array_equal(again.layers[0].w, net.layers[0].w)


def test_run_on_board_with_machine(mirror, tmp_path):
    """run_on_board's checks, on the Machine backend (the board itself: tb/test_tinytpu_top.py)."""
    url, _ = mirror
    cache = tmp_path / "cache"
    mnist.fetch(cache, url, out=silent)
    ds = mnist.load(cache, download=False)
    net = mnist.train(mnist.prepare(ds.train_images), ds.train_labels, hidden=12, epochs=5, out=silent)
    q = quant.quantize(net, mnist.prepare(ds.train_images[:100]))
    c = compiler.compile_model(q, Target.default(4), batch=7)
    x = mnist.prepare(ds.test_images[:30])
    res = mnist.run_on_board(c.simulator(), q, x, progress=10, out=silent)
    assert res.exact == 30 and res.machine_ok and res.acc.shape == (30, 10)
    assert np.array_equal(np.argmax(q.dequantize(res.acc), 1), np.argmax(q.forward(x), 1))


def test_show_digits():
    img = np.zeros((2, 4, 4), dtype=np.uint8)
    img[0, :2, :2] = 255
    text = mnist.show_digits(img, np.array([1, 2]), np.array([1, 3]), count=2)
    lines = text.splitlines()
    assert len(lines) == 3                                   # 2 text rows per 4 pixel rows + caption
    assert lines[0].startswith("@@  ") and "ok" in lines[2] and "WRONG" in lines[2]
    assert mnist.show_digits(img, np.array([]), np.array([]), count=0) == ""


def test_main_without_a_board(mirror, tmp_path, capsys):
    url, _ = mirror
    cache = tmp_path / "cache"
    mnist.fetch(cache, url, out=silent)                      # main's own fetch then finds them cached
    assert mnist.main(["--data-dir", str(cache), "--hidden", "16", "--epochs", "3", "--show", "0"]) == 0
    out = capsys.readouterr().out
    assert "test accuracy: float" in out and "compiled for N = 8" in out
    c = compiler.Compiled.from_file(cache / "mnist-64-16.npz")
    assert (c.k_in, c.n_out) == (64, 10)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    params = [rng.normal(size=(5, 4)), rng.normal(size=4), rng.normal(size=(4, 3)), rng.normal(size=3)]
    xb = rng.normal(size=(6, 5))
    tb = np.eye(3)[rng.integers(0, 3, 6)]
    _, grads = mnist.loss_and_grads(params, xb, tb)
    eps = 1e-6
    for p, g in zip(params, grads):
        num = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up, _ = mnist.loss_and_grads(params, xb, tb)
            p[idx] = old - eps
            down, _ = mnist.loss_and_grads(params, xb, tb)
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)
        assert np.allclose(g, num, atol=1e-6), (g, num)


class _CorruptingRunner:
    """Wraps a real Runner: records each chunk's size and corrupts chosen output rows."""

    def __init__(self, runner, bad_rows=()):
        self.runner, self.compiled, self.bad, self.chunks, self.seen = runner, runner.compiled, set(bad_rows), [], 0

    def infer_int(self, qx):
        self.chunks.append(len(qx))
        out = self.runner.infer_int(qx).copy()
        for i in range(len(qx)):
            if self.seen + i in self.bad:
                out[i, 0] += 1
        self.seen += len(qx)
        return out


@pytest.mark.parametrize("bad, exact, machine_ok", [((), 30, True), ((20,), 29, True), ((3,), 29, False)])
def test_run_on_board_catches_wrong_results(mirror, tmp_path, bad, exact, machine_ok):
    url, _ = mirror
    cache = tmp_path / "cache"
    mnist.fetch(cache, url, out=silent)
    ds = mnist.load(cache, download=False)
    net = mnist.train(mnist.prepare(ds.train_images), ds.train_labels, hidden=8, epochs=2, out=silent)
    q = quant.quantize(net, mnist.prepare(ds.train_images[:100]))
    c = compiler.compile_model(q, Target.default(4), batch=7)
    r = _CorruptingRunner(c.simulator(), bad)
    res = mnist.run_on_board(r, q, mnist.prepare(ds.test_images[:30]), progress=10, out=silent)
    assert (res.exact, res.machine_ok) == (exact, machine_ok)
    assert all(n % 7 == 0 for n in r.chunks[:-1]) and sum(r.chunks) == 30   # whole batches per INFER run
    assert r.chunks == [7, 7, 7, 7, 2]                                       # progress 10 rounds down to 7
