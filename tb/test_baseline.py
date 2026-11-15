"""Tests for tinytpu.baseline (task A5.2): the AGENTS.md regression rules for implementation metrics."""

from __future__ import annotations

import json

import pytest

from tinytpu import baseline

BASE = {"top": "tinytpu_top", "part": "xc7a35tcpg236-1", "vivado": "2025.2", "lut": 3700, "lut_memory": 494, "lut_cells": 4061,
        "ff": 4185, "dsp": 66, "bram": 39, "wns_ns": 1.075, "whs_ns": 0.035, "power_total_w": 0.3,
        "power_dynamic_w": 0.22}


def check(**changes) -> tuple[bool, dict[str, baseline.Row]]:
    now = {**BASE, **changes}
    rows = baseline.compare(now, BASE)
    return all(r.ok for r in rows), {r.name: r for r in rows}


def test_identical_build_passes():
    ok, rows = check()
    assert ok
    assert rows["lut"].note == "+0.0%"


@pytest.mark.parametrize("name", ["lut", "ff"])
def test_growth_limit_is_ten_percent(name):
    limit = int(BASE[name] * 1.10)
    assert check(**{name: limit})[0]                       # exactly +10 % (rounded down) is fine
    ok, rows = check(**{name: limit + 2})
    assert not ok and not rows[name].ok and "limit" in rows[name].note
    assert check(**{name: BASE[name] // 2})[0]             # shrinking is always fine


@pytest.mark.parametrize("name, value", [("dsp", 65), ("dsp", 67), ("bram", 38.5), ("bram", 40)])
def test_dsp_and_bram_must_not_change(name, value):
    ok, rows = check(**{name: value})
    assert not ok and rows[name].note == "must not change"


@pytest.mark.parametrize("name", ["wns_ns", "whs_ns"])
def test_negative_slack_fails(name):
    assert check(**{name: 0.0})[0]
    ok, rows = check(**{name: -0.01})
    assert not ok and "timing not met" in rows[name].note


def test_power_and_lut_details_are_reported_not_checked():
    ok, rows = check(power_total_w=0.9, lut_memory=600, lut_cells=9000)
    assert ok and rows["power_total_w"].note == "+0.6" and rows["lut_memory"].note == "+106"


def test_another_design_is_not_comparable():
    assert not check(top="loopback_top")[0]
    assert not check(part="xc7a100tcsg324-1")[0]


def test_missing_metric_fails():
    now = dict(BASE)
    del now["lut"]
    assert not all(r.ok for r in baseline.compare(now, BASE))


def test_cli_compare_create_and_refuse_to_overwrite(tmp_path, capsys):
    metrics = tmp_path / "m.json"
    base = tmp_path / "baseline.json"
    metrics.write_text(json.dumps(BASE))
    assert baseline.main([str(metrics), "--baseline", str(base)]) == 2            # no baseline yet
    assert baseline.main([str(metrics), "--baseline", str(base), "--create", " "]) == 2   # needs a reason
    assert baseline.main([str(metrics), "--baseline", str(base), "--create", "first full build"]) == 0
    assert json.loads(base.read_text())["reason"] == "first full build"
    assert baseline.main([str(metrics), "--baseline", str(base), "--create", "again"]) == 2   # not replaced
    assert baseline.main([str(metrics), "--baseline", str(base)]) == 0
    assert "baseline check: PASS" in capsys.readouterr().out
    metrics.write_text(json.dumps({**BASE, "dsp": 65}))
    assert baseline.main([str(metrics), "--baseline", str(base)]) == 1
    assert "FAIL (dsp)" in capsys.readouterr().out
    assert baseline.main([str(metrics), "--baseline", str(base), "--create", "redo", "--force"]) == 0
