"""A folded uniform must reach the host.

``uniform T x = fold op(y);`` declares ``x``.  Before this was wired up, that
declaration produced no arena slot: ``_lower_fold_route`` silently returned
without storing (its ``uniform_name not in uniform_slots`` guard), the generated
header exposed no offset for ``x``, and the host had no way to read the folded
scalar back.  LLVM then saw the whole accumulator dataflow as dead and deleted
it -- for a copy-shaped pipeline the entire ``Lockstep_Tick`` collapsed to a
column copy with no arithmetic at all, so benchmarks that thought they were
timing a reduction were timing a memcpy.

These tests pin the contract end to end: the header exposes an offset for every
folded uniform, the IR stores the reduction there, and a host that allocates
``LOCKSTEP_ARENA_BYTES`` and calls ``Lockstep_Tick`` reads back the value an
independent computation predicts.
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

CAPACITY = 2048

# A plain, valid pipeline: one kernel route accumulates, one fold reduces.
# Nothing here bypasses the semantic validator.
PROGRAM = f"""
shader Square(in float src, out float dst, accum float energy) {{
    energy = energy + (src * src);
    dst = src;
}}

pipeline Sim {{
    stream<float, {CAPACITY}> samplesIn;
    stream<float, {CAPACITY}> samplesOut;
    accumulator<float> energy;

    bind {{
        samplesOut = Square(samplesIn, samplesOut, energy);
        uniform float totalEnergy = fold sum(energy);
    }}
}}
"""


def _define(header: str, name: str) -> int:
    match = re.search(rf"#define\s+{name}\s+(\d+)", header)
    assert match is not None, f"{name} missing from generated header"
    return int(match.group(1))


def test_folded_uniform_gets_a_header_slot():
    result = compile_lockstep(PROGRAM, verbose=False)
    header = result.c_header or ""
    assert "float uniform_totalEnergy_value;" in header
    # The slot must be addressable by the host, not merely reserved.
    assert _define(header, "LOCKSTEP_OFFSET_UNIFORM_TOTALENERGY") > 0


def test_fold_result_is_stored_to_the_arena():
    result = compile_lockstep(PROGRAM, verbose=False)
    ir = result.llvm_ir or ""
    # Whichever lowering runs (register-carried or strip-mined), the reduced
    # scalar has to be written somewhere, not just computed and dropped.
    finals = re.findall(r'%"[A-Za-z0-9_.]*_final(?:\.\d+)?"', ir)
    assert finals, "no fold reduction found in generated IR"
    for final in set(finals):
        assert (
            f"store float {final}" in ir
        ), f"fold result {final} is computed but never stored"


def test_accumulation_survives_as_dead_code_would_not():
    """The reduction must still be in the IR as real arithmetic."""
    ir = compile_lockstep(PROGRAM, verbose=False).llvm_ir or ""
    assert "fadd" in ir, "accumulator addition missing from generated IR"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_host_reads_folded_uniform_from_arena():
    result = compile_lockstep(PROGRAM, verbose=False)
    header = result.c_header or ""
    arena_bytes = _define(header, "LOCKSTEP_ARENA_BYTES")
    total_offset = _define(header, "LOCKSTEP_OFFSET_UNIFORM_TOTALENERGY")
    in_offset = _define(header, "LOCKSTEP_OFFSET_STREAM_SAMPLESIN")

    driver = f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CAP ((size_t){CAPACITY})
void Lockstep_Tick(void* arena);
int main(void) {{
    uint8_t* base = (uint8_t*)calloc(1, (size_t){arena_bytes});
    if (!base) return 2;
    float* in = (float*)(base + {in_offset});
    for (size_t i = 0; i < CAP; ++i) in[i] = (float)((i % 64) + 1) * 0.25f;
    Lockstep_Tick(base);
    printf("%.7g\\n", (double)*(float*)(base + {total_offset}));
    free(base);
    return 0;
}}
"""
    expected = sum((((i % 64) + 1) * 0.25) ** 2 for i in range(CAPACITY))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "mod.ll").write_text(result.llvm_ir or "", encoding="utf-8")
        (work / "driver.c").write_text(driver, encoding="utf-8")
        exe = work / "run"
        clang = shutil.which("clang")
        assert clang is not None
        build = subprocess.run(
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
        assert build.returncode == 0, build.stderr
        run = subprocess.run(
            [str(exe)], capture_output=True, text=True, timeout=120, check=False
        )
        assert run.returncode == 0, run.stderr
        assert float(run.stdout.strip()) == pytest.approx(expected, rel=1e-4)
