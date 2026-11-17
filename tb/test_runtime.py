"""Tests for the compiler's runtime and file format (task A7.4), on the model.Machine backend.

The same Runner drives the board through tinytpu.device.Device; that path is tested end to end
on the simulated board in tb/test_tinytpu_top.py (``compiled_model``).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tinytpu import compiler, model, nn, quant
from tinytpu.compiler import Compiled, RunError, Target
from tinytpu.isa import Mem, SeqState
from tinytpu.quant import QLayer, QModel

NS = [2, 4, 8]


def float_model(rng: np.random.Generator, sizes: list[int], relus: list[bool]) -> nn.Model:
    return nn.Model([nn.Dense(rng.normal(0, 0.4, (k, m)), rng.normal(0, 0.1, m), relu=r)
                     for k, m, r in zip(sizes, sizes[1:], relus)])


def compiled_float(n: int, seed: int = 0, batch: int | None = 4, signed: bool = False,
                   relus=(True, False, False)) -> tuple[QModel, Compiled, np.random.Generator]:
    rng = np.random.default_rng(seed)
    sizes = [2 * n + 3, n + 5, n + 1, 7][:len(relus) + 1]
    net = float_model(rng, sizes, list(relus))
    calib = rng.normal(size=(200, sizes[0])) if signed else rng.uniform(0, 1, (200, sizes[0]))
    q = quant.quantize(net, calib)
    return q, compiler.compile_model(q, Target.default(n), batch=batch), rng


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("rows", [0, 1, 4, 9])
def test_infer_matches_reference_for_any_row_count(n, rows):
    """Several batches and a partial one: integer results bit-exact, floats identical."""
    q, c, rng = compiled_float(n)
    r = c.simulator()
    x = rng.uniform(0, 1, (rows, q.k_in))
    qx = q.quantize_input(x)
    assert np.array_equal(r.infer_int(qx), q.forward_int(qx))
    got = r.infer(x)
    assert got.shape == (rows, q.n_out)
    assert np.array_equal(got, q.forward(x))                     # same integers, same scales


@pytest.mark.parametrize("n", NS)
def test_signed_input_and_last_relu(n):
    q, c, rng = compiled_float(n, seed=5, signed=True, relus=(False, True))
    x = rng.normal(size=(6, q.k_in)) * 3                         # some inputs saturate
    got = c.simulator().infer(x)
    assert np.array_equal(got, q.forward(x)) and got.min() >= 0


def test_default_batch_and_padding():
    q, c, rng = compiled_float(4, batch=None)
    assert c.batch == compiler.max_batch(q, c.target)
    x = rng.uniform(0, 1, (c.batch + 1, q.k_in))                 # one full batch and one row
    assert np.array_equal(c.simulator().infer(x), q.forward(x))


def test_input_checks():
    q, c, _ = compiled_float(2)
    r = c.simulator()
    with pytest.raises(ValueError, match="shape"):
        r.infer(np.zeros((1, q.k_in + 1)))
    with pytest.raises(ValueError, match="0..255"):
        r.infer_int(np.full((1, q.k_in), 256))
    with pytest.raises(ValueError, match="finite"):
        r.infer(np.full((1, q.k_in), np.nan))
    big = r.infer(np.full((1, q.k_in), 1e300))                   # saturates instead of overflowing
    assert np.array_equal(big, q.forward(np.full((1, q.k_in), 255 * q.input_scale)))


def test_integer_only_model():
    q = QModel([QLayer(w=[[3], [-2]], bias=[7], mult=[0], shift=[0], relu=False, act_signed=True)])
    c = compiler.compile_model(q, Target.default(2), batch=2)
    r = c.simulator()
    assert r.infer_int(np.array([[-128, 127], [1, 1], [0, 0]])).tolist() == [[-384 - 254 + 7], [8], [7]]
    with pytest.raises(ValueError, match="no scales"):
        r.infer(np.zeros((1, 2)))


def test_wrong_program_is_detected():
    """Another model's program in IMEM: the HALT tag differs, so the run is refused."""
    q, c, rng = compiled_float(4, seed=1)
    _, other, _ = compiled_float(4, seed=2)
    m = model.Machine(n=4)
    r = c.load(m)
    m.load_program(other.program)
    with pytest.raises(RunError, match="tag") as e:
        r.infer(rng.uniform(0, 1, (1, q.k_in)))
    assert e.value.status.halt_code == other.tag
    m.load_program([0])                                          # HALT 0: the tag is almost surely not 0
    if c.tag != 0:
        with pytest.raises(RunError):
            r.infer(rng.uniform(0, 1, (1, q.k_in)))


def test_error_halt_is_reported():
    q, c, rng = compiled_float(2)
    m = model.Machine(n=2)
    r = c.load(m)
    m.imem[0] = 0xFF                                             # illegal opcode
    with pytest.raises(RunError) as e:
        r.infer_int(np.zeros((1, q.k_in), dtype=np.int64))
    assert e.value.status.state == SeqState.HALTED_ERROR


def test_verify_reads_back():
    q, c, _ = compiled_float(4)
    m = model.Machine(n=4)
    r = c.load(m)
    assert r.verify()
    m.wbuf[len(c.tiles) - 1, 3, 3] ^= 1
    assert not r.verify()
    r.write_image()
    assert r.verify()
    m.ptab_shift[1] ^= 1
    assert not r.verify()


def test_target_mismatch_is_refused():
    _, c, _ = compiled_float(4)
    with pytest.raises(ValueError, match="compiled for N = 4"):
        c.load(model.Machine(n=8))
    d = dict(c.target.depths)
    d[Mem.ABUF] *= 2
    with pytest.raises(ValueError, match="ABUF 2048"):
        c.load(model.Machine(n=4, depths=d))


@pytest.mark.parametrize("n", NS)
def test_save_and_load_round_trip(n, tmp_path):
    q, c, rng = compiled_float(n, seed=3)
    c.save(tmp_path / "model")
    c2 = Compiled.from_file(tmp_path / "model.npz")              # numpy adds the suffix
    assert c2.program == c.program and c2.tag == c.tag and c2.layers == c.layers and c2.target == c.target
    assert np.array_equal(c2.tiles, c.tiles) and c2.tiles.dtype == np.int8
    for a in ("ptab_bias", "ptab_mult", "ptab_shift", "out_scale"):
        assert np.array_equal(getattr(c2, a), getattr(c, a))
    assert c2.input_scale == c.input_scale                       # floats survive exactly
    x = rng.uniform(0, 1, (5, q.k_in))
    assert np.array_equal(c2.simulator().infer(x), q.forward(x))


def test_corrupted_file_is_refused(tmp_path):
    _, c, _ = compiled_float(2)
    c.save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz") as f:
        parts = {k: f[k].copy() for k in f.files}
    parts["tiles"][0, 0, 0] ^= 1
    np.savez(tmp_path / "bad.npz", **parts)
    with pytest.raises(ValueError, match="does not match its tag"):
        Compiled.from_file(tmp_path / "bad.npz")
    parts["tiles"][0, 0, 0] ^= 1
    parts["meta"] = np.array(str(parts["meta"]).replace('"format": 1', '"format": 99'))
    np.savez(tmp_path / "future.npz", **parts)
    with pytest.raises(ValueError, match="format 99"):
        Compiled.from_file(tmp_path / "future.npz")


def test_random_integer_models_many_batches():
    rng = random.Random(9)
    for n in NS:
        sizes = [rng.randint(1, 2 * n + 3) for _ in range(3)]
        layers = [QLayer(w=[[rng.randint(-127, 127) for _ in range(b)] for _ in range(a)],
                         bias=[rng.randint(-5000, 5000) for _ in range(b)],
                         mult=[rng.randint(0, 65535) for _ in range(b)], shift=[rng.randint(10, 20)] * b,
                         relu=i == 0, act_signed=(n == 4) if i == 0 else False)
                  for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]
        q = QModel(layers)
        c = compiler.compile_model(q, Target.default(n), batch=3)
        lo, hi = (-128, 127) if q.input_signed else (0, 255)
        qx = np.array([[rng.randint(lo, hi) for _ in range(sizes[0])] for _ in range(11)])
        assert np.array_equal(c.simulator().infer_int(qx), q.forward_int(qx))


class _StubBackend:
    """Reports a given status for every run (states a real Machine never combines with the tag)."""

    def __init__(self, c: Compiled, status: model.Status):
        self.c, self.status = c, status

    def target(self) -> Target:
        return self.c.target

    def infer(self, c: Compiled, vecs: np.ndarray):
        return self.status, np.zeros((c.out_rows, c.target.n), dtype=np.int32)


@pytest.mark.parametrize("state", [SeqState.HALTED_ERROR, SeqState.IDLE, SeqState.RUNNING])
def test_status_other_than_halted_ok_is_refused(state):
    """Even with the right tag in the halt-code field, only HALTED_OK counts as a result."""
    q, c, _ = compiled_float(2)
    r = compiler.Runner(c, _StubBackend(c, model.Status(state, halt_code=c.tag)))
    with pytest.raises(RunError):
        r.infer_int(np.zeros((1, q.k_in), dtype=np.int64))
    ok = compiler.Runner(c, _StubBackend(c, model.Status(SeqState.HALTED_OK, halt_code=c.tag)))
    assert ok.infer_int(np.zeros((1, q.k_in), dtype=np.int64)).tolist() == [[0] * q.n_out]
