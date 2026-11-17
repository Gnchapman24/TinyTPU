# `tinytpu.quant`

The quantizer of the [host compiler](../compiler.md#quantization-tinytpuquant): it turns a float `nn.Model` and calibration inputs into a `QModel` of INT8 weights, INT32 biases and requantizer parameters. `QModel.forward_int` is the bit-exact integer reference that compiled programs must match.

`tb/test_quant.py` checks the rounding and the `M`/`s` choice against exact rational arithmetic, the integer forward pass against plain integer loops, and the accuracy against the float model.

::: tinytpu.quant
