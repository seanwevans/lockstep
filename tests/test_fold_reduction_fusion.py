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

Both programs are ordinary Lockstep that passes the semantic validator, and the
folded uniform is read at the ``LOCKSTEP_OFFSET_UNIFORM_*`` offset the generated
header publishes.  Earlier revisions of this module did neither: they disabled
the validator (``semantic_validator=lambda _tree, **_kwargs: []``) to compile a
program with three semantic errors -- a kernel bound to an output it had no
``out`` parameter for, and uniforms declared twice -- and then read the fold
result at a hand-computed byte offset that only existed *because* of the
duplicate declaration.  So the fusion was verified in a shape no valid program
could express, and the module could not have caught a fold whose result never
reached the arena at all.  ``test_programs_pass_the_semantic_validator`` keeps it
that way.
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
shader Acc(in float src, out float dst, accum float e) {{
    e = e + (src * src);
    dst = src;
}}

pipeline P {{
    stream<float, {cap}> in0;
    stream<float, {cap}> out0;
    accumulator<float> e;

    bind {{
        out0 = Acc(in0, out0, e);
        uniform float total = fold sum(e);
    }}
}}
"""

_UNFUSED = """
shader Acc(in float src, out float dst, accum float e) {{
    e = e + (src * src);
    dst = src;
}}

pipeline P {{
    stream<float, {cap}> in0;
    stream<float, {cap}> out0;
    accumulator<float> e;

    bind {{
        out0 = Acc(in0, out0, e);
        uniform float total = fold sum(e);
        uniform float peak = fold max(e);
    }}
}}
"""


def _define(header: str, name: str) -> int:
    match = re.search(rf"#define\s+{name}\s+(\d+)", header)
    assert match is not None, f"{name} missing from generated header"
    return int(match.group(1))


def _build_and_run_total(source: str, work: Path) -> float:
    work.mkdir(parents=True, exist_ok=True)
    result = compile_lockstep(source, target_width=8)
    ir = result.llvm_ir or ""
    header = result.c_header or ""
    arena_bytes = _define(header, "LOCKSTEP_ARENA_BYTES")
    # Read the folded uniform where the header says it lives, rather than
    # reconstructing the arena layout here -- a hand-computed offset silently
    # reads the wrong slot the moment the layout changes.
    total_offset = _define(header, "LOCKSTEP_OFFSET_UNIFORM_TOTAL")
    in_offset = _define(header, "LOCKSTEP_OFFSET_STREAM_IN0")

    driver = f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CAP ((size_t){CAPACITY})
void Lockstep_Tick(void* arena);
int main(void) {{
    uint8_t* base = (uint8_t*)calloc(1, (size_t){arena_bytes});
    if (!base) return 2;
    float* in0 = (float*)(base + {in_offset});
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


def test_programs_pass_the_semantic_validator() -> None:
    """Both fixtures must be programs a user could actually write.

    Verifying fusion on a program the compiler rejects proves nothing about the
    compiler, so this is a guard against quietly reintroducing a
    ``semantic_validator`` override to make a malformed fixture compile.
    """
    for source in (_FUSED, _UNFUSED):
        result = compile_lockstep(source.format(cap=CAPACITY), target_width=8)
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert not errors, [d.message for d in errors]


def test_fused_reduction_selection_is_as_expected() -> None:
    """The one-fold pipeline fuses; the two-fold pipeline keeps the buffer."""
    fused_ir = compile_lockstep(_FUSED.format(cap=CAPACITY), target_width=8).llvm_ir
    unfused_ir = compile_lockstep(_UNFUSED.format(cap=CAPACITY), target_width=8).llvm_ir

    assert "reduce_e_acc" in fused_ir and "fold_e_strip_cond" not in fused_ir
    assert "reduce_e_acc" not in unfused_ir and "fold_e_strip_cond" in unfused_ir


@pytest.mark.parametrize(
    "label,source", [("fused", _FUSED), ("unfused", _UNFUSED)], ids=["fused", "unfused"]
)
def test_fold_result_is_stored_at_the_published_offset(label: str, source: str) -> None:
    """Both lowerings must write the reduced scalar to the arena.

    Asserted against the byte offset from the header rather than the SSA value's
    name: the fused path names it ``*_final`` and the strip-mine path names it
    ``fold_reduce*``, so a name-based check silently covers only one of them.
    """
    result = compile_lockstep(source.format(cap=CAPACITY), target_width=8)
    ir = result.llvm_ir or ""
    offset = _define(result.c_header or "", "LOCKSTEP_OFFSET_UNIFORM_TOTAL")

    # Follow the pointer the backend builds for that offset all the way to the
    # store, so this cannot pass on an unrelated store elsewhere in the tick.
    byte_ptr = re.search(
        rf'%"([\w.]+)" = getelementptr i8, i8\* %"[\w.]+", i32 {offset}\b', ir
    )
    assert byte_ptr is not None, f"{label}: nothing addresses the uniform at {offset}"
    typed = re.search(
        rf'%"([\w.]+)" = bitcast i8\* %"{re.escape(byte_ptr.group(1))}" to float\*', ir
    )
    assert typed is not None, f"{label}: uniform pointer at {offset} is never typed"
    assert re.search(
        rf'store float %"[\w.]+", float\* %"{re.escape(typed.group(1))}"', ir
    ), f"{label}: reduction is computed but never stored to the uniform slot"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_fused_fold_value_matches_unfused_and_reference() -> None:
    """The fused register reduction folds to the same scalar as the strip-mine
    path and as an independent host sum."""
    expected = sum((((i % 97) + 1) * 0.125) ** 2 for i in range(CAPACITY))
    with tempfile.TemporaryDirectory(prefix="lsfold_") as tmp:
        tmp_path = Path(tmp)
        fused = _build_and_run_total(_FUSED.format(cap=CAPACITY), tmp_path / "f")
        unfused = _build_and_run_total(_UNFUSED.format(cap=CAPACITY), tmp_path / "u")

    assert fused == pytest.approx(unfused, rel=1e-6, abs=1e-3)
    assert fused == pytest.approx(expected, rel=1e-4, abs=1e-1)
