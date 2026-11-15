"""Compare a build's metrics with the committed baseline (task A5.2; rules in AGENTS.md §5, tier 3).

``syn/build.tcl`` writes ``<top>_metrics.json`` after routing: Slice LUTs (``lut``), LUT as memory,
LUT cells before packing, FFs, DSP, BRAM (tiles), WNS / WHS (ns) and Vivado's vectorless power
estimate (W). This compares it with
``syn/baseline.json``:

* **fail** if LUT or FF grew by more than 10 %, if the DSP or BRAM count changed at all, if
  setup or hold slack is negative, or if the top level or part differ;
* **report** the other figures (LUT as memory, LUT cells, slack, power) and every change, for review.

The baseline is created once from the first full build and then only changed deliberately, in
its own reviewed commit with the reason in the message (``--create``, which refuses to replace
an existing baseline unless ``--force`` is given).

Usage (Windows, after ``.\\syn\\vivado.ps1 -Top tinytpu_top``)::

    uv run --no-default-groups python -m tinytpu.baseline $env:LOCALAPPDATA\\tinytpu\\vivado\\tinytpu_top\\tinytpu_top_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parents[2] / "syn" / "baseline.json"
"""``syn/baseline.json`` in the repo this package is installed from."""

GROWTH_LIMIT = 0.10
"""LUT and FF may grow by at most this fraction of the baseline."""
GROWTH = ("lut", "ff")
EXACT = ("dsp", "bram")
NON_NEGATIVE = ("wns_ns", "whs_ns")
INFO = ("lut_memory", "lut_cells", "power_total_w", "power_dynamic_w")
IDENTITY = ("top", "part")


@dataclass(frozen=True)
class Row:
    """One compared metric.

    Attributes:
        name: Metric name.
        baseline: Value in the baseline (``None`` if missing).
        now: Value in the new build (``None`` if missing).
        ok: False if this metric breaks a rule.
        note: The rule or the change, for the report.
    """

    name: str
    baseline: object
    now: object
    ok: bool
    note: str


def compare(now: dict, base: dict) -> list[Row]:
    """Apply the baseline rules; return one row per metric."""
    rows: list[Row] = []
    for k in IDENTITY:
        rows.append(Row(k, base.get(k), now.get(k), base.get(k) == now.get(k),
                        "" if base.get(k) == now.get(k) else "different design: not comparable"))
    for k in GROWTH:
        b, n = base.get(k), now.get(k)
        if b is None or n is None:
            rows.append(Row(k, b, n, False, "missing"))
            continue
        growth = (n - b) / b if b else 0.0
        ok = growth <= GROWTH_LIMIT
        rows.append(Row(k, b, n, ok, f"{growth:+.1%}" + ("" if ok else f" (limit +{GROWTH_LIMIT:.0%})")))
    for k in EXACT:
        b, n = base.get(k), now.get(k)
        ok = b is not None and b == n
        rows.append(Row(k, b, n, ok, "" if ok else "must not change"))
    for k in NON_NEGATIVE:
        b, n = base.get(k), now.get(k)
        ok = n is not None and n >= 0
        delta = f"{n - b:+.3f}" if isinstance(b, (int, float)) and isinstance(n, (int, float)) else ""
        rows.append(Row(k, b, n, ok, delta + ("" if ok else " (timing not met)")))
    for k in INFO:
        b, n = base.get(k), now.get(k)
        delta = f"{n - b:+.3g}" if isinstance(b, (int, float)) and isinstance(n, (int, float)) else ""
        rows.append(Row(k, b, n, True, delta))
    return rows


def report(rows: list[Row], out=print) -> bool:
    """Print the comparison as a table; return True if every rule holds."""
    out(f"{'metric':<16} {'baseline':>16} {'now':>16}  note")
    for r in rows:
        mark = "" if r.ok else "  <-- FAIL"
        out(f"{r.name:<16} {r.baseline!s:>16} {r.now!s:>16}  {r.note}{mark}")
    bad = [r.name for r in rows if not r.ok]
    out("\nbaseline check: " + ("PASS" if not bad else "FAIL (" + ", ".join(bad) + ")"))
    return not bad


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(prog="python -m tinytpu.baseline", description=__doc__.splitlines()[0])
    ap.add_argument("metrics", type=Path, help="<top>_metrics.json written by syn/build.tcl")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="baseline file (default: syn/baseline.json)")
    ap.add_argument("--create", metavar="REASON",
                    help="write the metrics as the new baseline instead of comparing; REASON is recorded in it")
    ap.add_argument("--force", action="store_true", help="with --create: replace an existing baseline")
    args = ap.parse_args(argv)

    now = json.loads(args.metrics.read_text())
    if args.create is not None:
        if args.baseline.exists() and not args.force:
            print(f"{args.baseline} exists; a baseline is only replaced deliberately (--force).", file=sys.stderr)
            return 2
        if not args.create.strip():
            print("--create needs a reason.", file=sys.stderr)
            return 2
        data = {"reason": args.create.strip(), **now}
        args.baseline.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {args.baseline}; commit it on its own, with the reason in the message")
        return 0
    if not args.baseline.exists():
        print(f"no baseline at {args.baseline} (create it once with --create)", file=sys.stderr)
        return 2
    base = json.loads(args.baseline.read_text())
    return 0 if report(compare(now, base)) else 1


if __name__ == "__main__":
    sys.exit(main())
