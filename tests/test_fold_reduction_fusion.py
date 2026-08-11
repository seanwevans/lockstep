"""Fold-into-kernel fusion: correctness of the register-carried reduction.

When an accumulator is written by a single standalone kernel route and consumed
by exactly one ``fold``, the backend carries the reduction in a loop-carried
register across the route loop instead of materializing the per-row accumulator
buffer and strip-mining it afterwards (see ``_lower_reduction_route`` in
``codegen.py``).  That removes the O(rows) buffer traffic that made accumulator
pipelines trail hand-written C.

The structural half of this contract (the fused IR shape, and that a multi-fold
accumulator is *not* fused) lives in ``tests/test_compiler_api.py``.  This module
pins the property that actually matters: the folded scalar the fused path
produces is **numerically identical** to the un-fused strip-mine path and to an
independent host computation.  It compiles the same reduction two ways -- once so
it fuses (one fold) and once so it cannot (two folds over the same accumulator,
which must keep the buffer) -- runs both, and reads the folded uniform back out
of the arena.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from lockstep_compiler.compiler import compile_lockstep

REPO_ROOT = Path(__file__).resolve().parents[1]
INTRINSICS_C = REPO_ROOT / "benchmarks" / "native" / "lockstep_intrinsics.c"

CAPACITY = 4096

# ``e`` is written by one kernel route (``Acc``) as a running sum, then reduced.
# With a single fold it is reduction-fusible; with a second fold over the same
# accumulator it is not (the buffer must persist to feed both folds).
_FUSED = """
shader Acc(in float src, accum float e) {{ e = e + (src * src); }}

pipeline P {{
    stream<float, {cap}> in0;
    accumulator<float> e;
    uniform float total;

    bind {{
        e = Acc(in0, e);
        uniform float total = fold sum(e);
    }}
}}
"""

_UNFUSED = """
shader Acc(in float src, accum float e) {{ e = e + (src * src); }}

pipeline P {{
    stream<float, {cap}> in0;
    accumulator<float> e;
    uniform float total;
    uniform float peak;

    bind {{
        e = Acc(in0, e);
        uniform float total = fold sum(e);
        uniform float peak = fold max(e);
    }}
}}
"""


def _arena_bytes(header: str) -> int:
    match = re.search(r"#define\s+LOCKSTEP_ARENA_BYTES\s+(\d+)", header)
    assert match is not None, "header did not define LOCKSTEP_ARENA_BYTES"
    return int(match.group(1))


def _build_and_run_total(source: str, work: Path) -> float:
    work.mkdir(parents=True, exist_ok=True)
    result = compile_lockstep(
        source, semantic_validator=lambda _tree, **_kwargs: [], target_width=8
    )
    ir = result.llvm_ir or ""
    arena_bytes = _arena_bytes(result.c_header or "")
    # ``in0`` (cap floats) then ``e`` (cap floats) then the uniform(s).  ``total``
    # is the first uniform, so it sits right after the two float columns.
    total_offset = 2 * CAPACITY * 4

    driver = f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CAP ((size_t){CAPACITY})
void Lockstep_Tick(void* arena);
int main(void) {{
    uint8_t* base = (uint8_t*)calloc(1, (size_t){arena_bytes});
    if (!base) return 2;
    float* in0 = (float*)(base + 0);
    for (size_t i = 0; i < CAP; ++i) in0[i] = (float)((i % 97) + 1) * 0.125f;
    Lockstep_Tick(base);
    printf("%.7g\\n", (double)*(float*)(base + {total_offset}));
    free(base);
    return 0;
}}
"""
    (work / "mod.ll").write_text(ir, encoding="utf-8")
    (work / "driver.c").write_text(driver, encoding="utf-8")
    exe = work / "run"
    clang = shutil.which("clang")
    assert clang is not None
    compile_proc = subprocess.run(
        [
            clang,
            "-O2",
            "-Wno-override-module",
            str(work / "driver.c"),
            str(work / "mod.ll"),
            str(INTRINSICS_C),
            "-o",
            str(exe),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return float(run.stdout.strip())


def test_fused_reduction_selection_is_as_expected() -> None:
    """The one-fold pipeline fuses; the two-fold pipeline keeps the buffer."""
    fused_ir = compile_lockstep(
        _FUSED.format(cap=CAPACITY),
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=8,
    ).llvm_ir
    unfused_ir = compile_lockstep(
        _UNFUSED.format(cap=CAPACITY),
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=8,
    ).llvm_ir

    assert "reduce_e_acc" in fused_ir and "fold_e_strip_cond" not in fused_ir
    assert "reduce_e_acc" not in unfused_ir and "fold_e_strip_cond" in unfused_ir


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_fused_fold_value_matches_unfused_and_reference() -> None:
    """The fused register reduction folds to the same scalar as the strip-mine
    path and as an independent host sum."""
    expected = sum(
        (((i % 97) + 1) * 0.125) ** 2 for i in range(CAPACITY)
    )
    with tempfile.TemporaryDirectory(prefix="lsfold_") as tmp:
        tmp_path = Path(tmp)
        fused = _build_and_run_total(_FUSED.format(cap=CAPACITY), tmp_path / "f")
        unfused = _build_and_run_total(_UNFUSED.format(cap=CAPACITY), tmp_path / "u")

    assert fused == pytest.approx(unfused, rel=1e-6, abs=1e-3)
    assert fused == pytest.approx(expected, rel=1e-4, abs=1e-1)
