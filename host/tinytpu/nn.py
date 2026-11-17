"""Float layer API for the host compiler (task A7.2, ``docs/compiler.md``).

A network is a :class:`Model`: a list of fully connected :class:`Dense` layers, each with an
optional ReLU. Weights are ``(K, Nout)``, so a layer computes ``x @ w + b``, the same convention
as :func:`tinytpu.model.matmul`. (A PyTorch ``nn.Linear`` weight is ``(Nout, K)``: transpose it.)

This is the float reference (level 1 in ``docs/compiler.md``). :func:`tinytpu.quant.quantize`
turns it into integers for the board.

Example:
    >>> import numpy as np
    >>> from tinytpu import nn
    >>> net = nn.Model([nn.Dense(np.eye(2), relu=True), nn.Dense([[1.0], [-1.0]], [0.5])])
    >>> net.forward(np.array([[3.0, -2.0]]))
    array([[3.5]])
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


class Dense:
    """A fully connected layer: ``y = x @ w + b``, then an optional ReLU.

    Args:
        w: Weights, shape ``(K, Nout)``.
        b: Bias, shape ``(Nout,)``; zeros if omitted.
        relu: Apply a ReLU to the output.

    Raises:
        ValueError: If a shape is wrong or a value is not finite.
    """

    def __init__(self, w: ArrayLike, b: ArrayLike | None = None, relu: bool = False):
        self.w = np.array(w, dtype=np.float64)
        if self.w.ndim != 2 or 0 in self.w.shape:
            raise ValueError(f"weights must be a non-empty (K, Nout) matrix, got shape {self.w.shape}")
        self.b = np.zeros(self.w.shape[1]) if b is None else np.array(b, dtype=np.float64)
        if self.b.shape != (self.w.shape[1],):
            raise ValueError(f"bias must have shape ({self.w.shape[1]},), got {self.b.shape}")
        if not (np.isfinite(self.w).all() and np.isfinite(self.b).all()):
            raise ValueError("weights and bias must be finite")
        self.relu = bool(relu)

    @property
    def k(self) -> int:
        """Input width K."""
        return self.w.shape[0]

    @property
    def n_out(self) -> int:
        """Output width."""
        return self.w.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the layer's float output for inputs ``x`` of shape ``(B, K)``."""
        y = np.asarray(x, dtype=np.float64) @ self.w + self.b
        return np.maximum(y, 0.0) if self.relu else y

    def __repr__(self) -> str:
        return f"Dense({self.k} -> {self.n_out}{', relu' if self.relu else ''})"


class Model:
    """A stack of :class:`Dense` layers.

    Args:
        layers: The layers, input first. Each layer's K must equal the previous one's output width.

    Raises:
        ValueError: If there are no layers or the widths don't chain.
    """

    def __init__(self, layers: Sequence[Dense]):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("a model needs at least one layer")
        for i, (a, b) in enumerate(zip(self.layers, self.layers[1:])):
            if b.k != a.n_out:
                raise ValueError(f"layer {i + 1} takes {b.k} inputs but layer {i} gives {a.n_out}")

    @property
    def k_in(self) -> int:
        """Input width."""
        return self.layers[0].k

    @property
    def n_out(self) -> int:
        """Output width."""
        return self.layers[-1].n_out

    def activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Return every layer's float output (after its ReLU) for inputs ``x`` of shape ``(B, K)``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.k_in:
            raise ValueError(f"inputs must have shape (B, {self.k_in}), got {x.shape}")
        out = []
        for layer in self.layers:
            x = layer.forward(x)
            out.append(x)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the model's float output for inputs ``x`` of shape ``(B, K)``."""
        return self.activations(x)[-1]

    def __repr__(self) -> str:
        return f"Model({', '.join(map(repr, self.layers))})"
