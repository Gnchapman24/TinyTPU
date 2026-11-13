# TinyTPU

A minimal INT8 Tensor Processing Unit for the Digilent Basys3 (Xilinx Artix-7 `xc7a35t`).

- 8×8 weight-stationary systolic array, one DSP48E1 per MAC
- INT8 weights and activations (activations selectable signed or unsigned), INT32 accumulation
- Per-channel requantization with optional ReLU
- Host link over the on-board USB-UART at 1 Mbaud, with CRC-16 packets

**Status:** early stage. See [RTL modules](rtl/index.md) for the generated module reference.

## Building these docs

In WSL, from the repository root:

```bash
make docs          # regenerate docs/rtl/ and run mkdocs build --strict
make docs-serve    # live preview at http://127.0.0.1:8000
```

Module pages are generated from the RTL source. To change a module's page, edit its
`/** ... */` header comment or the `//` comments on its parameter and port lines.
