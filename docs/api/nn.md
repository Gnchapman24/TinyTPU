# `tinytpu.nn`

The float layer API of the [host compiler](../compiler.md): a `Model` is a list of `Dense` layers, each with an optional ReLU. Weights are `(K, Nout)`, so a layer computes `x @ w + b`.

::: tinytpu.nn
