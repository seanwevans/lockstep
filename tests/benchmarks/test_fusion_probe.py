"""Smoke tests for the multi-stage fusion probe.

These do not pin absolute timings (host-dependent). They verify the probe builds
and runs with the local clang, that the fused and unfused variants compute
identical results (the comparison is only valid if they agree), and that fusion
is not slower. Skipped when clang is unavailable.
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

_MODULE_PATH = REPO_ROOT / "benchmarks" / "native" / "fusion_probe.py"
_spec = importlib.util.spec_from_file_location("lockstep_fusion_probe", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
fusion_probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fusion_probe
_spec.loader.exec_module(fusion_probe)

clang_required = pytest.mark.skipif(
    shutil.which("clang") is None,
    reason="the fusion probe requires clang on PATH",
)


def test_results_agree_accepts_matching_and_rejects_divergent():
    base = {
        "unfused_score": 500.0,
        "fused_score": 500.02,  # within 1e-3 relative
        "unfused_count": 1000,
        "fused_count": 1000,
    }
    assert fusion_probe._results_agree(base)

    assert not fusion_probe._results_agree(dict(base, fused_score=600.0))
    assert not fusion_probe._results_agree(dict(base, fused_count=999))


@clang_required
def test_fusion_probe_runs_and_fusion_is_not_slower():
    results = fusion_probe.run([4_000, 64_000], clang="clang")
    assert len(results) == 2
    for row in results:
        assert row["fusion_speedup"] >= 1.0
        assert row["fused_mrows_per_sec"] > 0
        assert row["unfused_mrows_per_sec"] > 0
