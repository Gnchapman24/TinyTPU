"""Tests for code generation, tinytpu.compiler (task A7.3).

A compiled image is loaded into the instruction-level reference model ``model.Machine`` and run;
the accumulator rows it leaves must equal ``QModel.forward_int`` bit-exactly. Random models have
sizes that aren't multiples of N, both input types, and hidden layers with and without ReLU, at
N = 2, 4 and 8. Each fit limit is tested just met and just exceeded, with targets small enough
to make it bind.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tinytpu import compiler, isa, model, nn, quant
from tinytpu.compiler import Compiled, FitError, Target
from tinytpu.isa import Fields, Mem, Opcode, SeqState
from tinytpu.quant import QLayer, QModel

NS = [2, 4, 8]


def random_qmodel(rng: random.Random, sizes: list[int], relus: list[bool], signed: bool) -> QModel:
    """Integer layers with extreme values mixed in (no float path: forward_int is the reference)."""
    layers = []
    for i, (k, n) in enumerate(zip(sizes, sizes[1:])):
        layers.append(QLayer(
            w=[[rng.choice([-127, 127, 0, rng.randint(-127, 127)]) for _ in range(n)] for _ in range(k)],
            bias=[rng.randint(-200000, 200000) for _ in range(n)],
            mult=[rng.choice([1, 65535, rng.randint(0, 65535)]) for _ in range(n)],
            shift=[rng.choice([0, 63, rng.randint(8, 24)]) for _ in range(n)],
            relu=relus[i], act_signed=signed if i == 0 else not relus[i - 1]))
    return QModel(layers)


def dense_qmodel(sizes: list[int], relu: bool = True) -> QModel:
    """A small all-ones integer model, for limit tests (only the sizes matter)."""
    return QModel([QLayer(w=np.ones((k, n), dtype=np.int64), bias=np.zeros(n), mult=np.ones(n), shift=np.zeros(n),
                          relu=relu, act_signed=(i > 0 and not relu))
                   for i, (k, n) in enumerate(zip(sizes, sizes[1:]))])


def run_on_machine(c: Compiled, qx: np.ndarray) -> tuple[model.Status, np.ndarray, model.Machine]:
    """Load the image into a Machine, write one batch of inputs, run, read the output rows."""
    t = c.target
    m = model.Machine(n=t.n, depths=dict(t.depths))
    m.load_program(c.program)
    m.load_weights(c.tiles)
    m.load_params(c.ptab_bias, c.ptab_mult, c.ptab_shift)
    vecs = model.pack_activations(qx, t.n)
    assert len(vecs) == c.in_vectors
    m.load_vectors(vecs, at=c.in_addr)
    status = m.run()
    rows = m.acc[c.out_addr:c.out_addr + c.out_rows]
    return status, model.unpack_acc(rows, c.batch, c.n_out), m


def random_inputs(rng: random.Random, batch: int, k: int, signed: bool) -> np.ndarray:
    lo, hi = (-128, 127) if signed else (0, 255)
    return np.array([[rng.choice([lo, hi, 0, rng.randint(lo, hi)]) for _ in range(k)] for _ in range(batch)])


# ------------------------------------------------------------------------------------------------
# Bit-exact against the integer reference
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("seed", range(8))
def test_random_models_match_forward_int(n, seed):
    rng = random.Random(1000 * n + seed)
    layers = rng.randint(1, 4)
    sizes = [rng.randint(1, 3 * n + 1) for _ in range(layers + 1)]
    relus = [rng.random() < 0.6 for _ in range(layers)]
    signed = rng.random() < 0.5
    q = random_qmodel(rng, sizes, relus, signed)
    target = Target.default(n)
    batch = rng.choice([None, 1, rng.randint(1, 7)])
    c = compiler.compile_model(q, target, batch=batch)
    if batch is None:
        assert c.batch == compiler.max_batch(q, target)
    qx = random_inputs(rng, c.batch, sizes[0], signed)
    status, got, _ = run_on_machine(c, qx)
    assert status.state == SeqState.HALTED_OK and status.halt_code == c.tag, status
    assert np.array_equal(got, q.forward_int(qx))


@pytest.mark.parametrize("n", NS)
def test_quantized_float_model_matches(n):
    """The whole path from floats: quantize, compile, run on Machine, dequantize."""
    rng = np.random.default_rng(n)
    sizes, relus = [3 * n + 3, 2 * n + 1, n + 3, 5], [True, False, False]
    net = nn.Model([nn.Dense(rng.normal(0, 0.3, (k, m)), rng.normal(0, 0.1, m), relu=r)
                    for k, m, r in zip(sizes, sizes[1:], relus)])
    q = quant.quantize(net, rng.uniform(0, 1, (300, sizes[0])))
    c = compiler.compile_model(q, Target.default(n), batch=6)
    x = rng.uniform(0, 1, (6, sizes[0]))
    status, acc, _ = run_on_machine(c, q.quantize_input(x))
    assert status.state == SeqState.HALTED_OK
    assert np.array_equal(acc, q.forward_int(q.quantize_input(x)))
    assert np.allclose(c.out_scale, q.out_scale) and c.input_scale == q.input_scale
    want = net.forward(x)
    assert np.abs(q.dequantize(acc) - want).max() < 0.05 * np.abs(want).max()


@pytest.mark.parametrize("n", NS)
def test_hidden_activations_land_in_alternate_regions(n):
    """The last REQUANT result in each ABUF region equals the integer reference's hidden
    activations (earlier ones are overwritten by design), and a layer never reads the region it
    writes."""
    rng = random.Random(n)
    sizes = [2 * n + 1, n + 2, 3 * n - 1, n + 1, 3]
    relus = [True, False, True, False]
    q = random_qmodel(rng, sizes, relus, signed=False)
    c = compiler.compile_model(q, Target.default(n), batch=3)
    qx = random_inputs(rng, 3, sizes[0], signed=False)
    status, _, m = run_on_machine(c, qx)
    assert status.state == SeqState.HALTED_OK
    x, last_write = qx, {}
    for plan, layer in zip(c.layers[:-1], q.layers[:-1]):
        x = model.requant(model.matmul(x, layer.w, layer.act_signed, bias=layer.bias), layer.mult, layer.shift,
                          layer.relu)
        last_write[plan.a_dst] = (plan, x)
    assert sorted(last_write) == sorted({0, c.layers[0].a_dst})          # both regions were used
    for dst, (plan, want) in last_write.items():
        vecs = m.abuf[dst:dst + c.batch * plan.nt]
        assert np.array_equal(model.unpack_vectors(vecs, c.batch, plan.n_out, signed=not plan.relu), want)
    for plan in c.layers:
        src = set(range(plan.a_src, plan.a_src + c.batch * plan.kt))
        if plan.a_dst is not None:
            assert not src & set(range(plan.a_dst, plan.a_dst + c.batch * plan.nt))


# ------------------------------------------------------------------------------------------------
# Program and image layout
# ------------------------------------------------------------------------------------------------

def test_program_shape_and_fields():
    rng = random.Random(3)
    q = random_qmodel(rng, [20, 13, 9, 5], [True, False, True], signed=True)
    c = compiler.compile_model(q, Target.default(4), batch=2)
    f = [Fields.decode(w) for w in c.program]
    assert [x.opcode for x in f] == [Opcode.MATMUL, Opcode.REQUANT, Opcode.MATMUL, Opcode.REQUANT,
                                     Opcode.MATMUL, Opcode.HALT]
    kts, nts = [5, 4, 3], [4, 3, 2]
    assert [x.kt for x in f[0::2][:3]] == kts and [x.nt for x in f[0::2][:3]] == nts
    assert [x.w_base for x in f[0::2][:3]] == [0, 20, 32]                 # 5*4, then + 4*3
    assert [x.p_base for x in f[:5]] == [0, 0, 16, 16, 28]                # + 4*4, then + 3*4
    assert all(x.m == 2 and x.acc_base == 0 for x in f[:5])
    act_signed = [bool((x.flags >> isa.FLAG_ACT_SIGNED) & 1) for x in f[0::2][:3]]
    assert act_signed == [True, False, True]
    assert [bool((x.flags >> isa.FLAG_RELU) & 1) for x in f[1:4:2]] == [True, False]
    assert all(isa.AccMode((x.flags >> isa.FLAG_ACC_MODE_LSB) & 3) == isa.AccMode.BIAS for x in f[0::2][:3])
    # Regions: A = max(2*5 input, 2*3 layer-1 output) = 10, B from 10.
    assert [x.a_base for x in f[:5]] == [0, 10, 10, 0, 0]
    assert f[-1].w_base == c.tag
    assert (c.in_addr, c.in_vectors, c.out_addr, c.out_rows) == (0, 10, 0, 4)
    assert c.tiles.shape == (32 + 6, 4, 4) and len(c.ptab_bias) == 36
    # The last layer's PTAB entries carry only the bias; padding is all zero.
    assert c.ptab_mult[28:].tolist() == [0] * 8 and c.ptab_shift[28:].tolist() == [0] * 8
    assert c.ptab_bias[28 + 5:].tolist() == [0] * 3
    assert c.ptab_bias[16 + 9:28].tolist() == [0] * 3
    assert len(c.ptab_bytes()) == 8 * 36


def test_tag_tracks_the_image():
    rng = random.Random(4)
    q = random_qmodel(rng, [9, 7, 3], [True, False], signed=False)
    t = Target.default(4)
    c1, c2 = compiler.compile_model(q, t, batch=2), compiler.compile_model(q, t, batch=2)
    assert c1.tag == c2.tag and c1.program == c2.program
    assert compiler.compile_model(q, t, batch=3).tag != c1.tag                 # the program changed
    q.layers[0].w[0, 0] = 1 if q.layers[0].w[0, 0] != 1 else 2
    c3 = compiler.compile_model(q, t, batch=2)
    assert c3.tag != c1.tag                                                    # a weight changed
    q.layers[1].bias[0] += 1
    c4 = compiler.compile_model(q, t, batch=2)
    assert c4.tag != c3.tag and np.array_equal(c4.tiles, c3.tiles)            # only PTAB changed
    q.layers[0].mult[0] ^= 1
    assert compiler.compile_model(q, t, batch=2).tag != c4.tag                 # an M changed


def test_single_layer_and_integer_only():
    q = QModel([QLayer(w=[[1, -1], [2, 3]], bias=[5, -5], mult=[0, 0], shift=[0, 0], relu=False, act_signed=True)])
    c = compiler.compile_model(q, Target.default(2), batch=1)
    assert len(c.program) == 2 and c.input_scale is None and c.out_scale is None
    status, got, _ = run_on_machine(c, np.array([[-128, 127]]))
    assert status.state == SeqState.HALTED_OK
    assert got.tolist() == [[-128 + 254 + 5, 128 + 381 - 5]]


@pytest.mark.parametrize("sizes, batch, tiles, chans", [
    ([784, 128, 10], 5, 1600, 144),     # docs/compiler.md's demo model
    ([196, 64, 10], 20, 216, 80),       # and the 14x14 option
])
def test_design_page_figures(sizes, batch, tiles, chans):
    c = compiler.compile_model(dense_qmodel(sizes), Target.default(8))
    assert (c.batch, len(c.tiles), len(c.ptab_bias)) == (batch, tiles, chans)


# ------------------------------------------------------------------------------------------------
# Limits: each just met, then just exceeded
# ------------------------------------------------------------------------------------------------

def small_target(n=2, max_payload=1 << 20, **depths) -> Target:
    d = {m: 1 << 12 for m in Mem}
    d.update({Mem[k.upper()]: v for k, v in depths.items()})
    return Target(n, d, max_payload)


@pytest.mark.parametrize("sizes, batch, target, match", [
    # IMEM 4: two layers need 4 instructions; three need 6
    ([2, 2, 2], 1, small_target(imem=4), None),
    ([2, 2, 2, 2], 1, small_target(imem=4), "program: 6 instructions"),
    # WBUF 8 at N = 2: 4 -> 4 -> 4 is 4 + 4 tiles; 4 -> 4 -> 5 is 4 + 6
    ([4, 4, 4], 1, small_target(wbuf=8), None),
    ([4, 4, 5], 1, small_target(wbuf=8), "weights: 10 tiles"),
    # PTAB 8 at N = 2: 4 + 4 channels; 4 + 6
    ([2, 4, 4], 1, small_target(ptab=8), None),
    ([2, 4, 5], 1, small_target(ptab=8), "channels: 10"),
    # ACC 8 at N = 2: 4 column tiles, batch 2 is 8 rows; batch 3 is 12
    ([2, 8], 2, small_target(acc=8), None),
    ([2, 6], 3, small_target(acc=8), "layer 0: 3 x 3 = 9 accumulator rows"),
    # ABUF 8 at N = 2: input 2 x 2 + hidden 2 x 2; then hidden 2 x 3
    ([4, 4, 2], 2, small_target(abuf=8), None),
    ([4, 6, 2], 2, small_target(abuf=8), "activations: 4 \\+ 6 = 10"),
    # INFER request: 10 + 5 x 784 = 3930 bytes
    ([784, 16], 5, small_target(8, max_payload=3930), None),
    ([784, 16], 5, small_target(8, max_payload=3929), "INFER request: 3930"),
    # INFER reply: 8 + 3 x 2 x 32 = 200 bytes
    ([8, 16], 3, small_target(8, max_payload=200), None),
    ([8, 16], 3, small_target(8, max_payload=199), "INFER reply: 200"),
    # Counts: 12 bits
    ([8190, 2], 1, small_target(wbuf=1 << 16), None),
    ([8192, 2], 1, small_target(wbuf=1 << 16), "kt = 4096"),
    ([2, 2], 4095, small_target(acc=1 << 16, abuf=1 << 16), None),
    ([2, 2], 4096, small_target(acc=1 << 16, abuf=1 << 16), "batch 4096 exceeds"),
    ([2, 2], 0, small_target(), "at least 1"),
])
def test_limits(sizes, batch, target, match):
    q = dense_qmodel(sizes)
    if match is None:
        c = compiler.compile_model(q, target, batch=batch)
        assert c.batch == batch
    else:
        with pytest.raises(FitError, match=match):
            compiler.compile_model(q, target, batch=batch)


def test_fit_error_lists_every_problem():
    with pytest.raises(FitError) as e:
        compiler.compile_model(dense_qmodel([4, 4, 5]), small_target(wbuf=8, ptab=8, imem=2), batch=1)
    assert len(e.value.problems) == 3
    assert all(s in str(e.value) for s in ("program", "weights", "channels"))


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("seed", range(6))
def test_max_batch_is_the_largest(n, seed):
    rng = random.Random(seed + 50 * n)
    sizes = [rng.randint(1, 40 * n) for _ in range(rng.randint(2, 4))]
    q = dense_qmodel(sizes)
    t = Target.default(n)
    b = compiler.max_batch(q, t)
    compiler.check_fit(q, t, b)
    if b < compiler.MAX_COUNT:
        with pytest.raises(FitError):
            compiler.check_fit(q, t, b + 1)


@pytest.mark.parametrize("acc", [8, 16, 64, 512])
def test_max_batch_exact_value(acc):
    """With only ACC binding: B = ACC depth / NT exactly."""
    t = small_target(acc=acc)
    assert compiler.max_batch(dense_qmodel([2, 8]), t) == acc // 4           # NT = 4 at N = 2


def test_max_batch_when_nothing_fits():
    with pytest.raises(FitError, match="weights"):
        compiler.max_batch(dense_qmodel([800, 300, 10]), Target.default(8))


def test_target_validation():
    Target.default(2)
    for bad in (dict(n=3), dict(n=1)):
        with pytest.raises(ValueError, match="N must"):
            Target(**bad)
    d = dict(isa.DEFAULT_DEPTHS)
    d[Mem.ABUF] = 1000
    with pytest.raises(ValueError, match="ABUF depth"):
        Target(8, d)
    del d[Mem.ABUF]
    with pytest.raises(ValueError, match="cover"):
        Target(8, d)
