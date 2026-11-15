# `tinytpu.baseline`

The tier-3 regression check: `syn/build.tcl` writes `<top>_metrics.json` after routing, and this compares it with the committed `syn/baseline.json` using the rules of AGENTS.md §5. `syn\vivado.ps1 -Top tinytpu_top` runs it automatically once a baseline exists.

::: tinytpu.baseline
