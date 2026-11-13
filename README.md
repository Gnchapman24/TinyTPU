# TinyTPU

A miniature TPU implementation on the Artix-7 FPGA platform: a minimal INT8 Tensor Processing Unit for the Digilent Basys3 (Xilinx Artix-7 `xc7a35t`).

TinyTPU implements an 8x8 2D weight-stationary systolic array designed for extremely fast matrix multiplication operations. It includes a custom minimal instruction set architecture (ISA) with operations for loading data, executing computations, and storing results.

- 8×8 weight-stationary systolic array, one DSP48E1 per MAC
- INT8 weights and activations (activations selectable signed or unsigned), INT32 accumulation
- Per-channel requantization with optional ReLU
- Host link over the on-board USB-UART at 1 Mbaud, with CRC-16 packets

**Status:** early stage. The design decisions are recorded in [AGENTS.md](AGENTS.md), and the work plan is in [TASKS.md](TASKS.md).

## Toolchain

| Side | Tools |
|---|---|
| WSL2 (Ubuntu) | OSS CAD Suite (Verilator, Yosys + yosys-slang, SymbiYosys), cocotb, MkDocs |
| Windows | Vivado 2025.2, host driver (pyserial) |

Exact versions are pinned in `tools/versions.env` (planned).
