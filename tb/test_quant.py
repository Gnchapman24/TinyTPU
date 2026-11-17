"""Tests for the float layer API tinytpu.nn and the quantizer tinytpu.quant (task A7.2).

The quantizer's arithmetic is checked against naive references that share no code with it: exact
rational arithmetic (``fractions.Fraction``) for rounding and the ``M``/``s`` choice, and plain
integer loops for the integer forward pass. Accuracy against the float model is checked on random
networks with sizes that aren't multiples of any array size. Nothing here depends on N.
"""

from __future__ import annotations

import math
import random
import warnings
from fractions import Fraction

import numpy as np
import pytest

from tinytpu import model, nn, quant
from tinytpu.quant import QLayer, QModel, QuantWarning

K_LIMIT = (1 << 31) // model.MAX_PRODUCT      # largest K that can't overflow with a zero bias (66311)


# ------------------------------------------------------------------------------------------------
# Naive references
# ------------------------------------------------------------------------------------------------

def ref_round(v: float) -> int:
    """Round half away from zero, exactly."""
    f = Fraction(v)
    n = math.floor(abs(f) + Fraction(1, 2))
    return n if f >= 0 else -n


def ref_mult_shift(r: float) -> tuple[int, int]:
    """Try every s from 63 down with exact arithmetic; the first M that fits wins."""
    for s in range(63, -1, -1):
        m = ref_round(Fraction(r) * 2 ** s)
        if m <= 0xFFFF:
            return m, s
    raise ValueError("too large")


def ref_wrap(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def ref_forward_int(q: QModel, qx) -> list[list[int]]:
    """Integer loops: matmul + bias with wrapping, then (acc*M + R) >> s and the clamp."""
    rows = [[int(v) for v in r] for r in qx]
    for li, layer in enumerate(q.layers):
        out = []
        for r in rows:
            acc = []
            for c in range(layer.n_out):
                t = int(layer.bias[c]) + sum(r[k] * int(layer.w[k][c]) for k in range(layer.k))
                acc.append(ref_wrap(t))
            out.append(acc)
        if li == len(q.layers) - 1:
            return out
        rows = []
        for acc in out:
            vals = []
            for c, a in enumerate(acc):
                m, s = int(layer.mult[c]), int(layer.shift[c])
                y = (a * m + (0 if s == 0 else 1 << (s - 1))) >> s
                vals.append(min(max(y, 0 if layer.relu else -128), 255 if layer.relu else 127))
            rows.append(vals)
    raise AssertionError("unreachable")


def random_net(rng: np.random.Generator, sizes: list[int], relus: list[bool]) -> nn.Model:
    return nn.Model([nn.Dense(rng.normal(0, 1 / math.sqrt(k), size=(k, n)), rng.normal(0, 0.1, size=n), relu=r)
                     for k, n, r in zip(sizes, sizes[1:], relus)])


# ------------------------------------------------------------------------------------------------
# nn
# ------------------------------------------------------------------------------------------------

def test_nn_forward_and_shapes():
    net = nn.Model([nn.Dense([[1.0, -1.0], [2.0, 0.5]], [0.0, 1.0], relu=True), nn.Dense([[1.0], [3.0]])])
    x = np.array([[1.0, 1.0], [-1.0, 0.0]])
    # layer 0: [3, 0.5] and [-1, 2] -> relu -> [3, 0.5], [0, 2]; layer 1: 3 + 1.5 = 4.5, 0 + 6 = 6
    assert np.array_equal(net.forward(x), [[4.5], [6.0]])
    assert [a.shape for a in net.activations(x)] == [(2, 2), (2, 1)]
    assert (net.k_in, net.n_out) == (2, 1)
    with pytest.raises(ValueError, match="layer 1 takes 3"):
        nn.Model([nn.Dense(np.ones((2, 2))), nn.Dense(np.ones((3, 1)))])
    with pytest.raises(ValueError, match="bias"):
        nn.Dense(np.ones((2, 2)), [1.0])
    with pytest.raises(ValueError, match="finite"):
        nn.Dense([[np.nan]])
    with pytest.raises(ValueError):
        nn.Model([])
    with pytest.raises(ValueError, match="shape"):
        net.forward(np.ones((1, 3)))


# ------------------------------------------------------------------------------------------------
# Rounding and M / s
# ------------------------------------------------------------------------------------------------

def test_round_half_away():
    vals = [-2.5, -1.5, -0.5, -0.49999999, 0.49999999999999994, -0.49999999999999994, 0.0, 0.5, 1.5, 2.5, 2.4999999, 1e9 + 0.5]
    assert quant.round_half_away(vals).tolist() == [ref_round(v) for v in vals]
    assert quant.round_half_away([0.5, 1.5, 2.5]).tolist() == [1, 2, 3]       # not ties-to-even


@pytest.mark.parametrize("r, m, s", [
    (1.0, 32768, 15),              # 2**15 fits; 2**16 doesn't
    (0.5, 32768, 16),
    (3.0, 49152, 14),
    (65535.0, 65535, 0),
    (65535.49, 65535, 0),          # the largest r that s = 0 can take
])
def test_mult_shift_cases(r, m, s):
    got_m, got_s = quant.mult_shift([r])
    assert (int(got_m[0]), int(got_s[0])) == ref_mult_shift(r) == (m, s)


def test_mult_shift_tie_at_boundary():
    """r * 2**s exactly 65535.5 rounds up to 65536, so that s is not allowed."""
    m, s = quant.mult_shift([32767.75])
    assert (int(m[0]), int(s[0])) == (32768, 0)
    assert ref_mult_shift(32767.75) == (32768, 0)


def test_mult_shift_limits():
    with pytest.raises(ValueError, match="too large"):
        quant.mult_shift([65535.5])
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive"):
            quant.mult_shift([bad])
    m, s = quant.mult_shift([2.0 ** -70, 2.0 ** -48, 0.49999999999999994 * 2.0 ** -63])
    assert m.tolist() == [0, 32768, 0] and s.tolist() == [63, 63, 63]
    with pytest.raises(ValueError, match="too large"):
        quant.round_half_away([2.0 ** 62])


def test_mult_shift_random_against_exact():
    rng = random.Random(7)
    rs = [2.0 ** rng.uniform(-60, 15.9) for _ in range(2000)]
    m, s = quant.mult_shift(rs)
    for r, mi, si in zip(rs, m.tolist(), s.tolist()):
        assert (mi, si) == ref_mult_shift(r)
        assert si == 63 or mi >= 32768                 # full precision unless s is at its limit
        assert abs(mi / 2 ** si - r) <= r * 2 ** -16 + 2 ** -(si + 1)


# ------------------------------------------------------------------------------------------------
# Quantizing weights, biases and activations
# ------------------------------------------------------------------------------------------------

def test_weights_per_channel_and_zero_channel():
    w = np.array([[0.5, 0.0, -3.0], [-1.0, 0.0, 1.5], [0.25, 0.0, 0.0]])
    b = np.array([0.0, 0.2, 0.0])
    q = quant.quantize(nn.Model([nn.Dense(w, b)]), np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 0.0]]))
    layer = q.layers[0]
    assert layer.w[:, 0].tolist() == [64, -127, 32]          # 0.5 / (1/127) = 63.5 -> 64
    assert layer.w[:, 1].tolist() == [0, 0, 0]
    assert layer.w[:, 2].tolist() == [-127, 64, 0]           # 1.5 / (3/127) = 63.5 -> 64
    s_x = 1 / 255
    s_w = np.array([1.0, 1.0, 3.0]) / 127
    s_w[1] = 1 / 127                                          # an all-zero channel gets 1/127
    assert layer.w.tolist() == [[ref_round(v) for v in row] for row in (w / s_w).tolist()]
    assert layer.bias.tolist() == [ref_round(v) for v in (b / (s_x * s_w)).tolist()]
    assert np.allclose(q.out_scale, s_x * s_w)
    # The zero channel's output is exactly its bias.
    assert np.allclose(q.forward(np.zeros((1, 3)))[0, 1], 0.2, atol=s_x * s_w[1])


def test_bias_rounding_ties():
    """Biases that land exactly on .5 steps round away from zero."""
    w = np.array([[1.0, 1.0]])                    # s_w = 1/127
    s_x = 1 / 255
    b = np.array([2.5, -2.5]) * (s_x / 127)       # exactly +-2.5 accumulator steps
    q = quant.quantize(nn.Model([nn.Dense(w, b)]), np.array([[1.0]]))
    assert q.layers[0].bias.tolist() == [3, -3]


def test_signedness_and_overrides():
    rng = np.random.default_rng(1)
    net = random_net(rng, [6, 5, 4, 3], [True, False, False])
    pos, neg = rng.uniform(0, 1, (20, 6)), rng.normal(size=(20, 6))
    q = quant.quantize(net, pos)
    assert [l.act_signed for l in q.layers] == [False, False, True]     # input UINT8; after ReLU UINT8; else INT8
    assert math.isclose(q.input_scale, pos.max() / 255)
    q = quant.quantize(net, neg)
    assert q.layers[0].act_signed and math.isclose(q.input_scale, np.abs(neg).max() / 127)
    q = quant.quantize(net, pos, input_signed=True, input_scale=0.01)
    assert q.layers[0].act_signed and q.input_scale == 0.01
    assert q.quantize_input(np.array([[-2.0, -0.005, 0.005, 0.0149, 2.0, 0.0]])).tolist() == [[-128, -1, 1, 1, 127, 0]]
    with pytest.raises(ValueError, match="input_scale"):
        quant.quantize(net, pos, input_scale=0.0)


def test_mnist_style_input_is_exact():
    """Pixels k/255 with a calibrated maximum of 1.0 quantize back to exactly k."""
    net = nn.Model([nn.Dense(np.ones((256, 2)))])
    x = (np.arange(256) / 255.0)[None, :]
    q = quant.quantize(net, x)
    assert q.quantize_input(x).tolist() == [list(range(256))]


def test_hidden_scales_and_requant_params():
    rng = np.random.default_rng(2)
    net = random_net(rng, [7, 9, 4], [True, False])
    x = rng.uniform(0, 1, (50, 7))
    q = quant.quantize(net, x)
    hid = net.activations(x)[0]
    s_x, s_y = x.max() / 255, hid.max() / 255
    s_w = np.abs(net.layers[0].w).max(axis=0) / 127
    for c in range(9):
        assert (int(q.layers[0].mult[c]), int(q.layers[0].shift[c])) == ref_mult_shift(s_x * s_w[c] / s_y)
    s_w2 = np.abs(net.layers[1].w).max(axis=0) / 127
    assert np.allclose(q.out_scale, s_y * s_w2)
    assert q.layers[1].mult.tolist() == [0] * 4 and q.layers[1].shift.tolist() == [0] * 4


def test_percentile_calibration_clips_outliers():
    rng = np.random.default_rng(3)
    net = random_net(rng, [5, 6, 2], [True, False])
    x = rng.uniform(0, 1, (1000, 5))
    x[0] = 50.0                                    # one outlier
    qmax = quant.quantize(net, x)
    qpct = quant.quantize(net, x, calibration="percentile", percentile=99.0)
    assert qpct.input_scale < qmax.input_scale / 10
    assert qpct.quantize_input(x[:1]).max() == 255  # the outlier saturates
    with pytest.raises(ValueError, match="calibration"):
        quant.quantize(net, x, calibration="mean")
    with pytest.raises(ValueError, match="percentile"):
        quant.quantize(net, x, calibration="percentile", percentile=0)


def test_zero_activations_warn():
    net = nn.Model([nn.Dense(-np.ones((3, 2)), relu=True), nn.Dense(np.ones((2, 1)))])
    with pytest.warns(QuantWarning, match="layer 0's output is always zero"):
        q = quant.quantize(net, np.ones((4, 3)))     # ReLU of a negative sum: always 0
    assert q.forward(np.ones((1, 3))).tolist() == [[0.0]]
    with pytest.warns(QuantWarning, match="network input is always zero"):
        quant.quantize(nn.Model([nn.Dense(np.ones((3, 1)))]), np.zeros((2, 3)))


def test_multiplier_limits_through_quantize():
    x = np.array([[1.0, 1.0]])
    # Channel 1's weights are 1e-30 of channel 0's: its multiplier is far below 2**-64.
    net = nn.Model([nn.Dense(np.array([[1.0, 1e-30], [0.0, 0.0]]), relu=True), nn.Dense(np.ones((2, 1)))])
    with pytest.warns(QuantWarning, match=r"channels \[1\].*always output 0"):
        q = quant.quantize(net, x)
    assert q.layers[0].mult.tolist()[1] == 0 and q.layers[0].mult.tolist()[0] > 0
    # Big weights that cancel to a tiny output: the output scale is too fine for M at s = 0.
    net = nn.Model([nn.Dense(np.array([[1000.0], [-1000.0 + 1e-5]]), relu=True), nn.Dense(np.ones((1, 1)))])
    with pytest.raises(ValueError, match="too large"):
        quant.quantize(net, x)


# ------------------------------------------------------------------------------------------------
# Overflow rule
# ------------------------------------------------------------------------------------------------

def test_k_overflow_limit():
    x_ok = np.ones((1, K_LIMIT))
    quant.quantize(nn.Model([nn.Dense(np.ones((K_LIMIT, 1)))]), x_ok)
    with pytest.raises(ValueError, match="overflow"):
        quant.quantize(nn.Model([nn.Dense(np.ones((K_LIMIT + 1, 1)))]), np.ones((1, K_LIMIT + 1)))
    # With K at the limit, the bias has 2**31 - K*32385 - 1 = 1912 steps of room.
    room = (1 << 31) - K_LIMIT * model.MAX_PRODUCT - 1
    step = (1 / 255) * (1 / 127)                                        # s_x * s_w
    quant.quantize(nn.Model([nn.Dense(np.ones((K_LIMIT, 1)), [room * step])]), x_ok)
    with pytest.raises(ValueError, match="layer 0.*overflow"):
        quant.quantize(nn.Model([nn.Dense(np.ones((K_LIMIT, 1)), [(room + 1) * step])]), x_ok)


# ------------------------------------------------------------------------------------------------
# QModel: validation and the integer reference
# ------------------------------------------------------------------------------------------------

def test_qmodel_validation():
    good = dict(w=[[1]], bias=[0], mult=[1], shift=[0], relu=True, act_signed=False)
    QLayer(**good)
    for change, msg in [(dict(w=[[-128]]), "weights"), (dict(bias=[1 << 31]), "INT32"),
                        (dict(mult=[65536]), "M must"), (dict(shift=[64]), "M must"),
                        (dict(bias=[0, 0]), "bias must have shape")]:
        with pytest.raises(ValueError, match=msg):
            QLayer(**{**good, **change})
    a = QLayer(**good)
    with pytest.raises(ValueError, match="act_signed = False"):
        QModel([a, QLayer(**{**good, "act_signed": True})])
    with pytest.raises(ValueError, match="takes 2 inputs"):
        QModel([a, QLayer(**{**good, "w": [[1], [1]]})])
    q = QModel([a])
    with pytest.raises(ValueError, match="no input scale"):
        q.quantize_input(np.ones((1, 1)))
    with pytest.raises(ValueError, match="no output scales"):
        q.dequantize(np.ones((1, 1)))
    with pytest.raises(ValueError, match="0..255"):
        q.forward_int(np.array([[-1]]))


@pytest.mark.parametrize("seed", range(6))
def test_forward_int_matches_naive_loops(seed):
    """Hand-built integer layers with extreme values, both input types, ReLU and no ReLU."""
    rng = random.Random(seed)
    sizes = [rng.randint(1, 13) for _ in range(rng.randint(2, 4))]
    relus = [rng.random() < 0.5 for _ in sizes[1:]]
    signed = rng.random() < 0.5
    layers = []
    for i, (k, n) in enumerate(zip(sizes, sizes[1:])):
        layers.append(QLayer(
            w=[[rng.choice([-127, 127, 0, rng.randint(-127, 127)]) for _ in range(n)] for _ in range(k)],
            bias=[rng.randint(-100000, 100000) for _ in range(n)],
            mult=[rng.choice([0, 1, 65535, rng.randint(0, 65535)]) for _ in range(n)],
            shift=[rng.choice([0, 63, rng.randint(0, 30)]) for _ in range(n)],
            relu=relus[i], act_signed=signed if i == 0 else not relus[i - 1]))
    q = QModel(layers)
    lo, hi = (-128, 127) if signed else (0, 255)
    qx = [[rng.choice([lo, hi, 0, rng.randint(lo, hi)]) for _ in range(sizes[0])] for _ in range(5)]
    got = q.forward_int(np.array(qx))
    assert got.dtype == np.int32
    assert got.tolist() == ref_forward_int(q, qx)


# ------------------------------------------------------------------------------------------------
# Accuracy against the float model
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sizes, relus, signed_input", [
    ([27, 19, 11], [True, False], False),
    ([33, 25, 17, 5], [True, True, False], False),
    ([20, 13, 9, 7], [False, True, False], True),       # an INT8 hidden layer
    ([10, 3], [False], True),                           # one layer: MATMUL only
    ([16, 12, 4], [True, True], False),                 # ReLU on the last layer (host side)
])
def test_accuracy_against_float(sizes, relus, signed_input):
    rng = np.random.default_rng(sum(sizes))
    net = random_net(rng, sizes, relus)

    def draw(n):
        return rng.normal(size=(n, sizes[0])) if signed_input else rng.uniform(0, 1, (n, sizes[0]))

    q = quant.quantize(net, draw(500))
    x = draw(200)
    want, got = net.forward(x), q.forward(x)
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < 0.05, f"relative error {err:.4f}"
    if relus[-1]:
        assert got.min() >= 0
    # The integer path is what forward() uses.
    assert np.array_equal(got, q.dequantize(q.forward_int(q.quantize_input(x))))
