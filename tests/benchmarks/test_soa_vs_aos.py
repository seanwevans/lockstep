"""Smoke tests for the SoA-vs-AoS layout micro-benchmark.

These do not pin absolute timings (host-dependent). They verify the micro-
benchmark builds and runs with the local clang, that the two layouts compute
identical results (the comparison is only valid if they agree), and that SoA
is not slower than AoS for the fully-vectorizable kernel. Skipped when clang
is unavailable.
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

_MODULE_PATH = REPO_ROOT / "benchmarks" / "native" / "soa_vs_aos.py"
_spec = importlib.util.spec_from_file_location("lockstep_soa_vs_aos", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
soa_vs_aos = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = soa_vs_aos
_spec.loader.exec_module(soa_vs_aos)

clang_required = pytest.mark.skipif(
    shutil.which("clang") is None,
    reason="the SoA-vs-AoS micro-benchmark requires clang on PATH",
)


def test_iterations_scale_down_with_size():
    assert soa_vs_aos._iterations_for(1_000) > soa_vs_aos._iterations_for(1_000_000)
    assert soa_vs_aos._iterations_for(10**9) >= 20


def test_results_agree_accepts_matching_and_rejects_divergent():
    base = {
        "integrate_check_aos": 1.2345,
        "integrate_check_soa": 1.2345,
        "energy_aos": 100.0,
        "energy_soa": 100.02,  # within 1e-3 relative
    }
    assert soa_vs_aos._results_agree(base)

    divergent = dict(base, energy_soa=110.0)
    assert not soa_vs_aos._results_agree(divergent)


@clang_required
def test_microbench_runs_and_soa_is_competitive():
    results = soa_vs_aos.run([4_000, 64_000], clang="clang")
    assert len(results) == 2
    for row in results:
        # SoA vectorizes the full-field kernel; it must not be slower than AoS.
        assert row["integrate_soa_speedup"] >= 1.0
        assert row["integrate_soa_mrows_per_sec"] > 0
        assert row["energy_soa_speedup"] > 0
