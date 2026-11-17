# `tinytpu.compiler`

Code generation and runtime for the [host compiler](../compiler.md#code-generation-tinytpucompiler): `compile_model` turns a quantized `QModel` into a `Compiled` image for one `Target` (N and the memory depths), after checking every limit.

`tb/test_compiler.py` runs compiled images on `model.Machine` at N = 2, 4 and 8 and compares them bit-exactly with `QModel.forward_int`, and tests each limit just met and just exceeded. `tb/test_runtime.py` covers the runtime (`Runner`), the file format and their checks on `Machine`, and the `compiled_model` test in `tb/test_tinytpu_top.py` runs a compiled model on the simulated board through `Device`.

::: tinytpu.compiler
