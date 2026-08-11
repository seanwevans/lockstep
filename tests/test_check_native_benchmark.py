"""Unit tests for the native benchmark invariant gate.

``scripts/check_native_benchmark.py`` is a CI gate, so its pass/fail logic is
worth pinning directly rather than only exercising it through the workflow. These
tests drive the module in-process with synthetic reports (no clang, no compiled
benchmark run), covering: a clean pass, ABI drift, checksum drift beyond
tolerance, checksum noise within tolerance, a dropped workload, and the
round-trip that the shipped baseline matches its own projection.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_native_benchmark", REPO_ROOT / "scripts" / "check_native_benchmark.py"
)
assert _SPEC and _SPEC.loader
check_native_benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_native_benchmark)

BASELINE_PATH = REPO_ROOT / "benchmarks" / "baselines" / "native.json"


def _report(**overrides: dict) -> dict:
    """A minimal run_native.py-shaped report for two workloads."""
    rows = {
        "alpha": {"workload": "alpha", "rows_per_tick": 1024, "arena_bytes": 4096, "checksum": 10.0, "mrows_per_sec": 500.0},
        "beta": {"workload": "beta", "rows_per_tick": 2048, "arena_bytes": 8192, "checksum": 20.0, "mrows_per_sec": 600.0},
    }
    for name, patch in overrides.items():
        rows[name] = {**rows[name], **patch}
    return {"kind": "native_execution", "results": list(rows.values())}


def _invoke(baseline_path: Path, current_path: Path, update: bool = False) -> int:
    import sys

    argv = [
        "check_native_benchmark.py",
        "--baseline",
        str(baseline_path),
        "--current",
        str(current_path),
    ]
    if update:
        argv.append("--update")
    old = sys.argv
    sys.argv = argv
    try:
        return check_native_benchmark.main()
    finally:
        sys.argv = old


def _baseline_of(report: dict) -> dict:
    return check_native_benchmark._baseline_from_results(report)


def test_matching_report_passes(tmp_path: Path) -> None:
    report = _report()
    baseline = _baseline_of(report)
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(report), encoding="utf-8")
    assert _invoke(b, c) == 0


def test_checksum_within_tolerance_passes(tmp_path: Path) -> None:
    baseline = _baseline_of(_report())
    # A sub-1e-6 relative wobble must not trip the gate.
    current = _report(alpha={"checksum": 10.0 * (1 + 1e-7)})
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(current), encoding="utf-8")
    assert _invoke(b, c) == 0


def test_checksum_drift_fails(tmp_path: Path) -> None:
    baseline = _baseline_of(_report())
    current = _report(alpha={"checksum": 11.0})
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(current), encoding="utf-8")
    assert _invoke(b, c) == 1


def test_abi_drift_fails(tmp_path: Path) -> None:
    baseline = _baseline_of(_report())
    current = _report(beta={"arena_bytes": 8192 + 64})
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(current), encoding="utf-8")
    assert _invoke(b, c) == 1


def test_missing_workload_fails(tmp_path: Path) -> None:
    baseline = _baseline_of(_report())
    current = {"results": [_report()["results"][0]]}  # drop 'beta'
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(current), encoding="utf-8")
    assert _invoke(b, c) == 1


def test_update_writes_projection(tmp_path: Path) -> None:
    report = _report()
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    c.write_text(json.dumps(report), encoding="utf-8")
    assert _invoke(b, c, update=True) == 0
    written = json.loads(b.read_text(encoding="utf-8"))
    assert set(written["workloads"]) == {"alpha", "beta"}
    assert written["workloads"]["alpha"]["arena_bytes"] == 4096
    # Throughput is intentionally not part of the gated baseline.
    assert "mrows_per_sec" not in written["workloads"]["alpha"]


def test_shipped_baseline_is_self_consistent() -> None:
    """The committed baseline must be a valid, non-empty invariants file."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["workloads"], "shipped native baseline has no workloads"
    for name, fields in baseline["workloads"].items():
        assert set(fields) == {"rows_per_tick", "arena_bytes", "checksum"}, name
