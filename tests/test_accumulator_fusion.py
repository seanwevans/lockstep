"""Fusion of accumulator pipeline stages.

The code generator plans stage fusion in the optimizer but historically refused
to emit the single fused vector loop whenever a stage took an ``accum``
parameter -- ``_can_vectorize_fused_group`` bailed on ``param.modifier ==
"accum"``.  That forced accumulator pipelines into one strip-mined loop *per
stage*, each streaming the whole arena and materializing the intermediate
streams the optimizer said it would eliminate.

These tests pin the two guarantees of lifting that restriction:

* **Structural** (always runs): a pure multi-shader accumulator pipeline with no
  filter now emits exactly one fused loop, the accumulator is written inside it,
  and the eliminated intermediate stream gets no per-stage loop of its own.
* **Differential** (requires ``clang``): the fused pipeline computes bit-for-bit
  the same output stream and the same per-row accumulator buffer as the scalar
  path.  The scalar reference is produced from the *same* source by appending a
  keep-all filter, which makes the group fall back to per-stage scalar loops
  (codegen still refuses to fuse groups containing a filter), so the only
  variable between the two runs is whether the accumulating stages were fused.
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

# A two-stage pipeline: Normalize feeds Score, Score accumulates.  No filter, so
# with the accum restriction lifted the whole group fuses into one loop.
FUSED_SOURCE = """
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

# Same computation, but a trailing keep-all filter forces the group to bail to
# per-stage scalar loops (codegen does not fuse groups that contain a filter).
# The accumulator buffer and the alertsScored output are written identically.
SCALAR_SOURCE = """
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

filter KeepAll(in Alert src, out Alert dst) {
    dst.deviceId = src.deviceId;
    dst.score = src.score;
}

pipeline P {
    stream<Event, 4096> eventsRaw;
    stream<Event, 4096> eventsEnriched;
    stream<Alert, 4096> alertsScored;
    stream<Alert, 4096> alertsKept;
    accumulator<float> scoreSum;
    bind {
        eventsEnriched = Normalize(eventsRaw, eventsEnriched);
        alertsScored = Score(eventsEnriched, alertsScored, scoreSum);
        alertsKept = KeepAll(alertsScored, alertsKept);
        uniform float totalScore = fold sum(scoreSum);
    }
}
"""


def _macros(header: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(r"#define\s+(LOCKSTEP_\w+)\s+(\d+)", header)
    }


def test_accumulator_group_emits_single_fused_loop() -> None:
    """The accumulating group fuses: one fused loop, accumulator stored inside."""
    result = compile_lockstep(FUSED_SOURCE, verbose=False)
    ir = result.llvm_ir or ""

    fused_loops = set(re.findall(r"fused_(\d+)_cond", ir))
    assert fused_loops == {"0"}, f"expected exactly one fused loop, got {fused_loops}"

    # The accumulator is written from within the fused loop rather than a
    # separate per-stage pass.
    assert "fused_store_scoreSum" in ir or "accum_scoreSum" in ir

    # The eliminated intermediate (eventsEnriched) must not get its own
    # per-stage kernel loop -- that is exactly the traffic fusion removes.
    assert "route_Normalize_cond" not in ir
    assert "route_Score_cond" not in ir


def test_accumulator_restriction_was_the_only_blocker() -> None:
    """Sanity: the same pipeline with a filter still falls back to scalar loops."""
    ir = compile_lockstep(SCALAR_SOURCE, verbose=False).llvm_ir or ""
    # Filter in the group -> no fused loop; stages emitted as scalar routes.
    assert "fused_0_cond" not in ir
    assert "route_Score_cond" in ir


def _build_and_run(source: str, work: Path) -> tuple[list[float], list[float]]:
    """Compile ``source`` to a native executable that primes inputs, ticks once,
    and dumps the alertsScored.score column and the scoreSum accumulator buffer.
    Returns (scores, accum)."""
    work.mkdir(parents=True, exist_ok=True)
    result = compile_lockstep(source, verbose=False)
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
    ir_path = work / "mod.ll"
    driver_path = work / "driver.c"
    exe_path = work / "bench"
    ir_path.write_text(ir, encoding="utf-8")
    driver_path.write_text(driver, encoding="utf-8")

    clang = shutil.which("clang")
    assert clang is not None
    compile_proc = subprocess.run(
        [
            clang,
            "-O2",
            "-Wno-override-module",
            str(driver_path),
            str(ir_path),
            str(INTRINSICS_C),
            "-o",
            str(exe_path),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr

    scores: list[float] = []
    accum: list[float] = []
    for line in run.stdout.splitlines():
        _, s, a = line.split()
        scores.append(float(s))
        accum.append(float(a))
    return scores, accum


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_fused_accumulator_matches_scalar_path() -> None:
    """Fused and scalar codegen produce identical output and accumulator buffers."""
    with tempfile.TemporaryDirectory(prefix="lsfuse_") as tmp:
        work = Path(tmp)
        fused_scores, fused_accum = _build_and_run(FUSED_SOURCE, work / "f")
        scalar_scores, scalar_accum = _build_and_run(SCALAR_SOURCE, work / "s")

    assert len(fused_scores) == CAPACITY
    assert fused_scores == scalar_scores
    assert fused_accum == scalar_accum

    # And both match the source semantics: score = value * 0.1 * 1.6.
    for i in range(CAPACITY):
        expected = ((i % 97) + 1) * 0.25 * 0.1 * 1.6
        assert fused_scores[i] == pytest.approx(expected, rel=1e-5, abs=1e-5)
        assert fused_accum[i] == pytest.approx(expected, rel=1e-5, abs=1e-5)
