# `tinytpu.model`

The bit-exact reference model:
- **Math level:** `matmul`, `requant` and `wrap32`. The RTL datapath is checked against these.
- **Machine level:** `Machine` runs ISA programs on the five memories exactly as the [ISA](../isa.md) defines, including the error halts.
- **Layout helpers:** they produce the canonical memory layouts from the [memory map](../memory_map.md).

`tb/test_model.py` checks it against naive pure-Python references, and runs the worked example from the ISA page.

::: tinytpu.model
