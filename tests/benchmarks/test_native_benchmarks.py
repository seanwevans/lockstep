"""Smoke tests for the native execution benchmark harness.

These do not assert on timing (which is host-dependent). They verify that the
harness can compile a real workload's generated code with the local clang
toolchain, run it, and produce a well-formed, deterministic report. Skipped
when clang is unavailable so non-toolchain environments stay green.
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

_MODULE_PATH = REPO_ROOT / "benchmarks" / "native" / "run_native.py"
_spec = importlib.util.spec_from_file_location("lockstep_run_native", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
run_native = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass annotation resolution can find the module.
sys.modules[_spec.name] = run_native
_spec.loader.exec_module(run_native)

clang_required = pytest.mark.skipif(
    shutil.which("clang") is None,
    reason="native benchmarks require an LLVM/clang toolchain on PATH",
)


def test_arena_bytes_from_ir_sums_leaf_sizes():
    ir = (
        '%"struct.Lockstep_Arena" = type '
        "{[8 x i32], [8 x float], [4 x double], float}"
    )
    # 8*4 + 8*4 + 4*8 + 4 = 32 + 32 + 32 + 4 = 100
    assert run_native._arena_bytes_from_ir(ir) == 100


def test_build_plan_sizes_accumulator_arena_beyond_header():
    source = (REPO_ROOT / "benchmarks" / "workloads" / "particle_energy.lock").read_text()
    plan = run_native._build_plan("particle_energy", source)

    # The workload declares two Particle streams of 32768; inputs must be primed.
    assert plan.capacity == 32768
    assert plan.rows == 32768
    assert plan.input_fields, "expected primeable input leaves"
    assert plan.checksum_field is not None and not plan.checksum_field.is_integer

    # The fold accumulator is laid out as one float per row in the IR ABI, so the
    # true arena is larger than a scalar-accumulator layout (2 * 6 * 32768 * 4).
    assert plan.arena_bytes > 2 * 6 * 32768 * 4


@clang_required
def test_run_workload_end_to_end_is_deterministic():
    first = run_native._run_workload(
        "particle_energy", iterations=50, warmup=2, clang="clang", keep_dir=None
    )
    second = run_native._run_workload(
        "particle_energy", iterations=50, warmup=2, clang="clang", keep_dir=None
    )

    assert first["rows_per_tick"] == 32768
    assert first["mrows_per_sec"] > 0
    assert first["arena_gib_per_sec"] > 0
    # Same build + same deterministic priming => identical checksum.
    assert first["checksum"] == second["checksum"]
