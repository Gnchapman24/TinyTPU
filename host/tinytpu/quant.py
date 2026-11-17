"""Quantizer for the host compiler (task A7.2, ``docs/compiler.md``).

:func:`quantize` turns a float :class:`tinytpu.nn.Model` plus calibration inputs into a
:class:`QModel`: INT8 weights, INT32 biases, and the requantizer's ``M`` and ``s`` per output
channel. All quantization is symmetric, with no zero-points (AGENTS.md §2):

* weights: one scale per output channel, ``max|w| / 127``;
* activations: one scale per layer input, from the calibration data; UINT8 after a ReLU (and for
  a never-negative network input), INT8 otherwise;
* biases: ``round(b / (s_x * s_w))``;
* ``M`` and ``s``: the largest ``s`` in 0..63 with ``round(r * 2**s) <= 65535``, where
  ``r = s_x * s_w / s_y`` (:func:`mult_shift`).

Hidden layers are ``MATMUL`` (bias mode) plus ``REQUANT``. The last layer is ``MATMUL`` only: its
INT32 accumulators are the result, and :meth:`QModel.dequantize` turns them into float logits.

:meth:`QModel.forward_int` is the bit-exact integer reference (level 2 in ``docs/compiler.md``):
it chains :func:`tinytpu.model.matmul` and :func:`tinytpu.model.requant`, so the compiled program
on ``Machine`` and on the board must match it exactly.

Example:
    >>> import numpy as np
    >>> from tinytpu import nn, quant
    >>> rng = np.random.default_rng(0)
    >>> net = nn.Model([nn.Dense(rng.normal(size=(20, 12)), relu=True), nn.Dense(rng.normal(size=(12, 3)))])
    >>> x = rng.uniform(size=(100, 20))
    >>> q = quant.quantize(net, x)
    >>> err = np.abs(q.forward(x) - net.forward(x)).max() / np.abs(net.forward(x)).max()
    >>> bool(err < 0.03)
    True
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from tinytpu import model
from tinytpu.nn import Model

W_MAX = 127
"""Largest weight magnitude: weights are symmetric INT8, -127..127 (AGENTS.md §2)."""
M_MAX = 0xFFFF
"""Largest requantizer multiplier ``M`` (u16)."""
S_MAX = 63
"""Largest requantizer shift ``s`` (u6)."""
INT32_MIN, INT32_MAX = model.INT32_MIN, model.INT32_MAX


class QuantWarning(UserWarning):
    """A quantization that works but loses information (an always-zero channel or layer)."""


def round_half_away(x: ArrayLike) -> np.ndarray:
    """Round to the nearest integer, ties away from zero, as int64.

    ``np.round`` rounds ties to even; the quantizer always uses this instead (``docs/compiler.md``).

    Raises:
        ValueError: If a value is not finite or its magnitude is ``2**62`` or more (it would not
            survive the conversion to int64).
    """
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all() or (np.abs(a) >= 2.0 ** 62).any():
        raise ValueError("value too large to quantize (not finite, or 2**62 or more)")
    mag = np.abs(a)
    whole = np.floor(mag)
    up = (mag - whole) >= 0.5                   # exact; floor(mag + 0.5) is not (0.49999999999999994)
    return (np.sign(a) * (whole + up)).astype(np.int64)


def mult_shift(r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Choose the requantizer's ``M`` and ``s`` for real multipliers ``r`` (one per channel).

    ``s`` is the largest shift in 0..63 with ``round(r * 2**s) <= 65535``, and
    ``M = round(r * 2**s)``. Unless ``s`` is 63, ``M >= 32768``: the full 16 bits of precision.

    Args:
        r: Positive real multipliers.

    Returns:
        ``(M, s)`` as int64 arrays of ``r``'s shape. ``M`` is 0 where ``r`` is below ``2**-64``.

    Raises:
        ValueError: If any ``r`` is not positive and finite, or too large for ``s = 0``
            (``r >= 65535.5``).
    """
    rs = np.asarray(r, dtype=np.float64)
    mult = np.zeros(rs.shape, dtype=np.int64)
    shift = np.zeros(rs.shape, dtype=np.int64)
    for idx, v in np.ndenumerate(rs):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"requantization multiplier must be positive and finite, got {v}")
        if v >= M_MAX + 0.5:
            raise ValueError(f"requantization multiplier {v:.6g} is too large: at s = 0, M would exceed {M_MAX}")
        for s in range(S_MAX, -1, -1):          # largest s first
            scaled = math.ldexp(v, s)           # exact: a power-of-two scaling
            if scaled < M_MAX + 0.5:            # i.e. round-half-up(scaled) <= M_MAX; no int64 cast
                whole = math.floor(scaled)
                mult[idx], shift[idx] = whole + (scaled - whole >= 0.5), s
                break
    return mult, shift


def quantize_input(x: ArrayLike, scale: float, signed: bool, k: int) -> np.ndarray:
    """Quantize float inputs ``(B, k)``: round to steps of ``scale``, then clamp to INT8
    (``signed``) or UINT8. Returns int64. Shared by :class:`QModel` and the compiler's runtime."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != k:
        raise ValueError(f"inputs must have shape (B, {k}), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("inputs must be finite")
    lo, hi = (-128, 127) if signed else (0, 255)
    return np.clip(round_half_away(np.clip(x / scale, lo - 1, hi + 1)), lo, hi)


def dequantize(acc: ArrayLike, out_scale: np.ndarray, relu: bool) -> np.ndarray:
    """Float outputs from the last layer's INT32 accumulators: ``acc * out_scale`` per channel,
    then the last layer's ReLU (done on the host). Shared by :class:`QModel` and the runtime."""
    y = np.asarray(acc, dtype=np.float64) * out_scale
    return np.maximum(y, 0.0) if relu else y


@dataclass
class QLayer:
    """One quantized layer.

    Attributes:
        w: Weights ``(K, Nout)``, int64 in -127..127.
        bias: INT32 bias per output channel ``(Nout,)``, added in ``MATMUL`` bias mode.
        mult: Requantizer ``M`` per channel ``(Nout,)``, 0..65535. Unused (0) on the last layer.
        shift: Requantizer ``s`` per channel ``(Nout,)``, 0..63. Unused (0) on the last layer.
        relu: ReLU after this layer. On a hidden layer it is the ``REQUANT`` relu bit (UINT8
            output); on the last layer it is applied on the host, after dequantization.
        act_signed: Whether this layer's input is INT8 (else UINT8): the ``MATMUL`` flag.
    """

    w: np.ndarray
    bias: np.ndarray
    mult: np.ndarray
    shift: np.ndarray
    relu: bool
    act_signed: bool

    def __post_init__(self) -> None:
        self.w = np.array(self.w, dtype=np.int64)
        if self.w.ndim != 2 or 0 in self.w.shape:
            raise ValueError(f"weights must be a non-empty (K, Nout) matrix, got shape {self.w.shape}")
        n_out = self.w.shape[1]
        self.bias, self.mult, self.shift = (np.array(a, dtype=np.int64) for a in (self.bias, self.mult, self.shift))
        for name, a in (("bias", self.bias), ("mult", self.mult), ("shift", self.shift)):
            if a.shape != (n_out,):
                raise ValueError(f"{name} must have shape ({n_out},), got {a.shape}")
        if np.abs(self.w).max() > W_MAX:
            raise ValueError(f"weights must be in -{W_MAX}..{W_MAX}")
        if self.bias.min() < INT32_MIN or self.bias.max() > INT32_MAX:
            raise ValueError("bias must be INT32")
        if self.mult.min() < 0 or self.mult.max() > M_MAX or self.shift.min() < 0 or self.shift.max() > S_MAX:
            raise ValueError(f"M must be 0..{M_MAX} and s must be 0..{S_MAX}")
        if not model.k_fits(self.k, int(np.abs(self.bias).max())):
            raise ValueError(f"K = {self.k} with max |bias| = {int(np.abs(self.bias).max())} could overflow "
                             f"the INT32 accumulators (K * {model.MAX_PRODUCT} + max|bias| must be < 2**31)")
        self.relu, self.act_signed = bool(self.relu), bool(self.act_signed)

    @property
    def k(self) -> int:
        """Input width K."""
        return self.w.shape[0]

    @property
    def n_out(self) -> int:
        """Output width."""
        return self.w.shape[1]


@dataclass
class QModel:
    """A quantized network: the integer layers plus the scales that connect them to floats.

    Can be built by :func:`quantize`, or directly from integer layers (the scales are then
    optional, and only :meth:`forward_int` works without them).

    Attributes:
        layers: The layers, input first.
        input_scale: Float value of one input step (``x ≈ input_scale * q_x``), or ``None``.
        out_scale: Float value of one accumulator step per output channel of the last layer
            (``logit ≈ out_scale * acc``), or ``None``.
    """

    layers: list[QLayer]
    input_scale: float | None = None
    out_scale: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.layers = list(self.layers)
        if not self.layers:
            raise ValueError("a model needs at least one layer")
        for i, (a, b) in enumerate(zip(self.layers, self.layers[1:])):
            if b.k != a.n_out:
                raise ValueError(f"layer {i + 1} takes {b.k} inputs but layer {i} gives {a.n_out}")
            if b.act_signed == a.relu:
                raise ValueError(f"layer {i + 1} must have act_signed = {not a.relu}: layer {i} "
                                 f"{'has' if a.relu else 'has no'} ReLU, so its output is "
                                 f"{'UINT8' if a.relu else 'INT8'}")
        if self.out_scale is not None:
            self.out_scale = np.array(self.out_scale, dtype=np.float64)
            if self.out_scale.shape != (self.n_out,):
                raise ValueError(f"out_scale must have shape ({self.n_out},)")

    @property
    def k_in(self) -> int:
        """Input width."""
        return self.layers[0].k

    @property
    def n_out(self) -> int:
        """Output width."""
        return self.layers[-1].n_out

    @property
    def input_signed(self) -> bool:
        """Whether the network input is INT8 (else UINT8)."""
        return self.layers[0].act_signed

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        """Quantize float inputs ``(B, K)``: round to steps of :attr:`input_scale`, then clamp to
        INT8 or UINT8. Returns int64."""
        if self.input_scale is None:
            raise ValueError("this model has no input scale; pass integer inputs to forward_int")
        return quantize_input(x, self.input_scale, self.input_signed, self.k_in)

    def forward_int(self, qx: np.ndarray) -> np.ndarray:
        """Bit-exact integer reference: the last layer's INT32 accumulators for integer inputs.

        Args:
            qx: Inputs ``(B, K)``: INT8 or UINT8 values, as the input signedness says.

        Returns:
            ``(B, Nout)`` int32, exactly what the board leaves in ``ACC``.
        """
        x = np.asarray(qx, dtype=np.int64)
        lo, hi = (-128, 127) if self.input_signed else (0, 255)
        if x.ndim != 2 or x.shape[1] != self.k_in:
            raise ValueError(f"inputs must have shape (B, {self.k_in}), got {x.shape}")
        if x.size and (x.min() < lo or x.max() > hi):
            raise ValueError(f"inputs must be in {lo}..{hi}")
        for layer in self.layers[:-1]:
            acc = model.matmul(x, layer.w, layer.act_signed, bias=layer.bias)
            x = model.requant(acc, layer.mult, layer.shift, relu=layer.relu)
        last = self.layers[-1]
        return model.matmul(x, last.w, last.act_signed, bias=last.bias)

    def dequantize(self, acc: np.ndarray) -> np.ndarray:
        """Float logits from the last layer's accumulators (and its ReLU, done here on the host)."""
        if self.out_scale is None:
            raise ValueError("this model has no output scales")
        return dequantize(acc, self.out_scale, self.layers[-1].relu)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Float in, float out, through the integer model: what the board computes."""
        return self.dequantize(self.forward_int(self.quantize_input(x)))


def _stat(a: np.ndarray, calibration: str, percentile: float) -> float:
    a = np.abs(np.asarray(a, dtype=np.float64))
    if calibration == "max":
        return float(a.max(initial=0.0))
    return float(np.percentile(a, percentile)) if a.size else 0.0


def _scale(stat: float, levels: int, what: str) -> float:
    if stat > 0:
        return stat / levels
    warnings.warn(f"{what} is always zero in the calibration data; using scale 1", QuantWarning, stacklevel=3)
    return 1.0


def quantize(net: Model, calib_x: np.ndarray, *, calibration: str = "max", percentile: float = 99.99,
             input_scale: float | None = None, input_signed: bool | None = None) -> QModel:
    """Quantize a float model for the board.

    Args:
        net: The float model.
        calib_x: Representative inputs ``(C, K)``. They set the activation scales.
        calibration: ``"max"`` (default): an activation scale covers the largest magnitude seen.
            ``"percentile"``: it covers the ``percentile``-th percentile of magnitudes, and larger
            values saturate.
        percentile: The percentile for ``calibration="percentile"``.
        input_scale: Overrides the input scale (float value of one input step).
        input_signed: Overrides the input type: INT8 (``True``) or UINT8 (``False``). By default
            the input is UINT8 if no calibration input is negative.

    Returns:
        The quantized model.

    Raises:
        ValueError: If an argument is invalid, a layer could overflow the INT32 accumulators, or a
            requantization multiplier is too large for the hardware.
    """
    if calibration not in ("max", "percentile"):
        raise ValueError(f"calibration must be 'max' or 'percentile', got {calibration!r}")
    if calibration == "percentile" and not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    acts = net.activations(calib_x)                 # also checks calib_x's shape
    x = np.asarray(calib_x, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("calibration needs at least one input")
    signed = bool(x.min() < 0) if input_signed is None else bool(input_signed)
    if input_scale is None:
        s_x = _scale(_stat(x, calibration, percentile), 127 if signed else 255, "the network input")
    elif not (math.isfinite(input_scale) and input_scale > 0):
        raise ValueError(f"input_scale must be positive and finite, got {input_scale}")
    else:
        s_x = float(input_scale)
    first_scale = s_x

    layers: list[QLayer] = []
    out_scale = None
    last = len(net.layers) - 1
    for i, layer in enumerate(net.layers):
        peak = np.abs(layer.w).max(axis=0)
        s_w = np.where(peak > 0, peak / W_MAX, 1.0 / W_MAX)
        q_w = np.clip(round_half_away(layer.w / s_w), -W_MAX, W_MAX)
        q_b = round_half_away(layer.b / (s_x * s_w))
        max_b = int(np.abs(q_b).max())
        if not model.k_fits(layer.k, max_b):
            raise ValueError(f"layer {i}: K = {layer.k} with a quantized bias up to {max_b} could overflow the "
                             f"INT32 accumulators (K * {model.MAX_PRODUCT} + max|bias| must be < 2**31)")
        if i < last:
            s_y = _scale(_stat(acts[i], calibration, percentile), 255 if layer.relu else 127,
                         f"layer {i}'s output")
            mult, shift = mult_shift(s_x * s_w / s_y)
            if (mult == 0).any():
                warnings.warn(f"layer {i}: channels {np.flatnonzero(mult == 0).tolist()} have a multiplier "
                              "below 2**-64 and always output 0", QuantWarning, stacklevel=2)
        else:
            mult = shift = np.zeros(layer.n_out, dtype=np.int64)
            out_scale = s_x * s_w
        layers.append(QLayer(q_w, q_b, mult, shift, relu=layer.relu, act_signed=signed))
        if i < last:
            signed, s_x = not layer.relu, s_y
    return QModel(layers, input_scale=first_scale, out_scale=out_scale)
