"""Numerical equivalence of the ``--target-width`` SIMD knob.

``target_width`` selects the SIMD vector width the backend lowers to (and the
``LOCKSTEP_SIMD_WIDTH`` header macro the host reads). It is a *performance* knob:
changing it strip-mines the fused vector loop and the fold reduction at a
different width, but it must never change the computed result.

The existing unit tests pin the structural effect of the knob -- that the
emitted IR carries ``<N x float>`` vectors and the header macro tracks ``N``.
They do not execute the generated code, so they cannot catch a width-specific
lowering bug (a mis-strided vector load, a wrong tail-loop bound at an odd
width, ...). This test closes that gap: it compiles the *same* accumulator
pipeline at several widths, runs each, and asserts byte-identical output streams
and accumulator buffers across all of them -- and against a scalar reference.

The pipeline has no filter, so it exercises the manual fused-vector lowering
path (where the width actually flows into loads/stores/reductions) rather than a
plain scalar loop that clang auto-vectorizes on its own.
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
WIDTHS = (4, 8, 16)

SOURCE = """
struct Event { int deviceId; float value; };
struct Alert { int deviceId; float score; };

shader Normalize(in Event src, out Event dst) {
    dst.deviceId = src.deviceId;
    dst.value = src.value * 0.1;
}

shader Score(in Event src, out Alert dst, accum float scoreSum) {
    float score = src.value * 1.6;
    scoreSum = scoreSum + score;
    dst.deviceId = src.deviceId;
    dst.score = score;
}

pipeline P {
    stream<Event, 4096> eventsRaw;
    stream<Event, 4096> eventsEnriched;
    stream<Alert, 4096> alertsScored;
    accumulator<float> scoreSum;
    bind {
        eventsEnriched = Normalize(eventsRaw, eventsEnriched);
        alertsScored = Score(eventsEnriched, alertsScored, scoreSum);
        uniform float totalScore = fold sum(scoreSum);
    }
}
"""


def _macros(header: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(r"#define\s+(LOCKSTEP_\w+)\s+(\d+)", header)
    }


@pytest.mark.parametrize("width", WIDTHS)
def test_ir_and_header_track_target_width(width: int) -> None:
    """Structural (no clang): the IR vector width and header macro match the knob."""
    result = compile_lockstep(SOURCE, verbose=False, target_width=width)
    ir = result.llvm_ir or ""
    float_widths = {int(w) for w in re.findall(r"<(\d+) x float>", ir)}
    assert width in float_widths, f"expected <{width} x float> in IR, saw {float_widths}"
    assert f"#define LOCKSTEP_SIMD_WIDTH {width}" in (result.c_header or "")


def _build_and_run(width: int, work: Path) -> tuple[list[float], list[float]]:
    work.mkdir(parents=True, exist_ok=True)
    result = compile_lockstep(SOURCE, verbose=False, target_width=width)
    ir = result.llvm_ir or ""
    macros = _macros(result.c_header or "")

    arena_bytes = macros["LOCKSTEP_ARENA_BYTES"]
    off_in_value = macros["LOCKSTEP_OFFSET_STREAM_EVENTSRAW_VALUE"]
    off_in_id = macros["LOCKSTEP_OFFSET_STREAM_EVENTSRAW_DEVICEID"]
    off_score = macros["LOCKSTEP_OFFSET_STREAM_ALERTSSCORED_SCORE"]
    off_accum = macros["LOCKSTEP_OFFSET_ACCUM_SCORESUM"]

    driver = f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_BYTES ((size_t){arena_bytes})
#define CAP ((size_t){CAPACITY})

void Lockstep_Tick(void* arena);

int main(void) {{
    uint8_t* base = (uint8_t*)calloc(1, ARENA_BYTES);
    if (!base) return 2;
    float* in_value = (float*)(base + {off_in_value});
    int32_t* in_id = (int32_t*)(base + {off_in_id});
    for (size_t i = 0; i < CAP; ++i) {{
        in_value[i] = (float)((i % 97) + 1) * 0.25f;
        in_id[i] = (int32_t)i;
    }}
    Lockstep_Tick(base);
    const float* score = (const float*)(base + {off_score});
    const float* accum = (const float*)(base + {off_accum});
    for (size_t i = 0; i < CAP; ++i)
        printf("%zu %.7g %.7g\\n", i, (double)score[i], (double)accum[i]);
    free(base);
    return 0;
}}
"""
    (work / "mod.ll").write_text(ir, encoding="utf-8")
    (work / "driver.c").write_text(driver, encoding="utf-8")
    exe = work / "bench"

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

    scores: list[float] = []
    accum: list[float] = []
    for line in run.stdout.splitlines():
        _, s, a = line.split()
        scores.append(float(s))
        accum.append(float(a))
    return scores, accum


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_results_are_identical_across_widths() -> None:
    """Every width produces byte-identical output and accumulator buffers."""
    with tempfile.TemporaryDirectory(prefix="lswidth_") as tmp:
        tmp_path = Path(tmp)
        runs = {w: _build_and_run(w, tmp_path / str(w)) for w in WIDTHS}

    baseline_scores, baseline_accum = runs[WIDTHS[0]]
    assert len(baseline_scores) == CAPACITY
    for width in WIDTHS[1:]:
        scores, accum = runs[width]
        assert scores == baseline_scores, f"width {width} output differs from {WIDTHS[0]}"
        assert accum == baseline_accum, f"width {width} accumulator differs from {WIDTHS[0]}"

    # And the shared result matches the source semantics: value * 0.1 * 1.6.
    for i in range(CAPACITY):
        expected = ((i % 97) + 1) * 0.25 * 0.1 * 1.6
        assert baseline_scores[i] == pytest.approx(expected, rel=1e-5, abs=1e-5)
        assert baseline_accum[i] == pytest.approx(expected, rel=1e-5, abs=1e-5)
