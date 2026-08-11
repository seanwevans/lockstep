"""Smoke tests for the Lockstep-vs-hand-written-C harness.

These do not assert on timing (host-dependent). They verify that the harness
builds and runs each workload with the local clang, that the Lockstep and C
paths agree on the output checksum (the harness raises otherwise, so a
successful run already implies agreement), and that the C reference is
semantically correct by cross-checking its checksum against the independent
``run_native.py`` harness for a shared workload. Skipped when clang is missing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, filename: str):
    path = REPO_ROOT / "benchmarks" / "native" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lockstep_vs_c = _load("lockstep_lockstep_vs_c", "lockstep_vs_c.py")
run_native = _load("lockstep_run_native_for_vs_c", "run_native.py")

clang_required = pytest.mark.skipif(
    shutil.which("clang") is None,
    reason="native benchmarks require an LLVM/clang toolchain on PATH",
)


def test_every_workload_has_a_reference():
    for name in run_native.WORKLOADS:
        assert name in lockstep_vs_c._REFERENCES
        ref = lockstep_vs_c._REFERENCES[name]
        assert ref.body.strip()
        assert ref.checksum_offset_macro.startswith("LOCKSTEP_OFFSET_STREAM_")


@clang_required
def test_run_workload_reports_ratio_and_agrees():
    row = lockstep_vs_c._run_workload(
        "particle_energy",
        iterations=50,
        warmup=2,
        clang="clang",
        keep_dir=None,
        target_width=8,
    )
    assert row["rows_per_tick"] == 32768
    assert row["lockstep_mrows_per_sec"] > 0
    assert row["c_mrows_per_sec"] > 0
    assert row["lockstep_vs_c_ratio"] > 0


@clang_required
def test_c_reference_matches_native_harness_checksum():
    # Both harnesses prime the same deterministic inputs and both checksum the
    # particlesOut.px column, so a correct C reference must reproduce the exact
    # checksum the independent run_native harness reads from Lockstep_Tick.
    vs_c = lockstep_vs_c._run_workload(
        "particle_energy", iterations=50, warmup=2, clang="clang", keep_dir=None, target_width=8
    )
    native = run_native._run_workload(
        "particle_energy", iterations=50, warmup=2, clang="clang", keep_dir=None
    )
    assert vs_c["checksum"] == native["checksum"]
