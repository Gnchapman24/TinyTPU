# `tinytpu.mnist`

The MNIST demo of the [host compiler](../compiler.md#mnist-demo-python-m-tinytpumnist): download with hash checks, NumPy training, quantization, compilation, and classification on the board. User guide: [Using the board](../host.md#6-mnist-digits-on-the-board).

`tb/test_mnist.py` tests it on a synthetic dataset in the same file format, with no network access; `mnist_demo` in `tb/test_tinytpu_top.py` runs its board half on the simulated board.

::: tinytpu.mnist
