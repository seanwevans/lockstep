"""Fusion of accumulator pipeline stages, including through pass-through filters.

The code generator plans stage fusion in the optimizer; historically it refused
to emit the single fused vector loop whenever a stage took an ``accum`` parameter
*or* whenever the group contained a ``filter``. Both restrictions have been
lifted:

* An accumulator group fuses into one vector loop and carries each fold's
  accumulator in a **loop-carried vector register** -- reduced horizontally at
  the end -- instead of materializing the O(rows) per-row arena buffer.
* A group fuses *through* a filter when that filter keeps every row
  unconditionally (its compacting store degenerates to a straight store), so the
  whole Normalize -> Score -> KeepAll chain collapses to one pass. A filter that
  actually drops rows (a data-dependent ``return``) still forces the per-stage
  fallback, because its compacted output shifts every downstream write index.

These tests pin those guarantees:

* **Structural** (always runs): the accumulator group emits exactly one fused
  loop with a horizontal ``llvm.vector.reduce`` and no per-row accumulator
  buffer store; a trailing keep-all filter does not break that; and a
  data-dependent filter falls back to per-stage scalar loops.
* **Differential** (requires ``clang``): the fused register-carry path, the
  fuse-through-filter path, and the per-stage scalar fallback all compute the
  same output stream and the same folded reduction, and all match the source
  semantics. The scalar reference is the same computation behind a
  data-dependent (but at run time always-true) filter, so the only variable is
  whether the stages were fused.
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

# A two-stage accumulator pipeline: Normalize feeds Score, Score accumulates.
# No filter, so the whole group fuses into one loop and scoreSum is register
# carried.
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
    uniform float totalScore;
    bind {
        eventsEnriched = Normalize(eventsRaw, eventsEnriched);
        alertsScored = Score(eventsEnriched, alertsScored, scoreSum);
        uniform float totalScore = fold sum(scoreSum);
    }
}
"""

# Same computation with a trailing keep-all filter (no ``return`` -> keeps every
# row). Codegen now fuses through it, so this still collapses to one loop.
FILTER_FUSED_SOURCE = """
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
    uniform float totalScore;
    bind {
        eventsEnriched = Normalize(eventsRaw, eventsEnriched);
        alertsScored = Score(eventsEnriched, alertsScored, scoreSum);
        alertsKept = KeepAll(alertsScored, alertsKept);
        uniform float totalScore = fold sum(scoreSum);
    }
}
"""

# Same computation, but the trailing filter has a data-dependent ``return`` (true
# for all rows at run time on this input). Codegen cannot prove it keeps every
# row, so the group falls back to per-stage scalar loops with the accumulator
# buffer -- the un-fused reference.
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

filter KeepBounded(in Alert src, out Alert dst) {
    dst.deviceId = src.deviceId;
    dst.score = src.score;
    return src.score > -1000000.0;
}

pipeline P {
    stream<Event, 4096> eventsRaw;
    stream<Event, 4096> eventsEnriched;
    stream<Alert, 4096> alertsScored;
    stream<Alert, 4096> alertsKept;
    accumulator<float> scoreSum;
    uniform float totalScore;
    bind {
        eventsEnriched = Normalize(eventsRaw, eventsEnriched);
        alertsScored = Score(eventsEnriched, alertsScored, scoreSum);
        alertsKept = KeepBounded(alertsScored, alertsKept);
        uniform float totalScore = fold sum(scoreSum);
    }
}
"""

# A single decl uniform reads back the fold. Its byte offset is the streams and
# the accumulator column that precede it: three 8-byte struct streams + a 4-byte
# accumulator = 28 bytes/row for FUSED_SOURCE; the filter variants add a fourth
# stream = 36 bytes/row.
_TOTAL_OFFSET_NO_FILTER = 28 * CAPACITY
_TOTAL_OFFSET_WITH_FILTER = 36 * CAPACITY


def _compile(source: str):
    return compile_lockstep(
        source, verbose=False, semantic_validator=lambda _tree, **_kwargs: []
    )


def _macros(header: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(r"#define\s+(LOCKSTEP_\w+)\s+(\d+)", header)
    }


def test_accumulator_group_fuses_and_register_carries() -> None:
    """The accumulating group fuses into one loop with a register reduction."""
    ir = _compile(FUSED_SOURCE).llvm_ir or ""

    fused_loops = set(re.findall(r"fused_(\d+)_cond", ir))
    assert fused_loops == {"0"}, f"expected exactly one fused loop, got {fused_loops}"

    # The fold is a loop-carried vector reduction, not a strip-mine over a per-row
    # buffer: the accumulator column is never stored to from the fused loop.
    assert "llvm.vector.reduce" in ir
    assert "fused_store_scoreSum" not in ir
    assert "fold_e_strip_cond" not in ir

    # The eliminated intermediate must not get its own per-stage kernel loop.
    assert "route_Normalize_cond" not in ir
    assert "route_Score_cond" not in ir


def test_group_fuses_through_keep_all_filter() -> None:
    """A trailing keep-all filter no longer blocks fusion."""
    ir = _compile(FILTER_FUSED_SOURCE).llvm_ir or ""
    assert "fused_0_cond" in ir
    assert "route_Normalize_cond" not in ir
    assert "route_Score_cond" not in ir
    assert "route_KeepAll_cond" not in ir
    assert "llvm.vector.reduce" in ir


def test_data_dependent_filter_falls_back_to_scalar_loops() -> None:
    """A filter that can drop rows keeps the per-stage compacting scalar path."""
    ir = _compile(SCALAR_SOURCE).llvm_ir or ""
    assert "fused_0_cond" not in ir
    assert "route_Score_cond" in ir


def _build_and_run(
    source: str, work: Path, score_macro: str, total_offset: int
) -> tuple[list[float], float]:
    """Compile ``source``, tick once over primed inputs, and return the final
    output stream's score column and the folded ``totalScore`` uniform."""
    work.mkdir(parents=True, exist_ok=True)
    result = _compile(source)
    ir = result.llvm_ir or ""
    macros = _macros(result.c_header or "")

    arena_bytes = macros["LOCKSTEP_ARENA_BYTES"]
    off_in_value = macros["LOCKSTEP_OFFSET_STREAM_EVENTSRAW_VALUE"]
    off_in_id = macros["LOCKSTEP_OFFSET_STREAM_EVENTSRAW_DEVICEID"]
    off_score = macros[score_macro]

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
    printf("total %.7g\\n", (double)*(const float*)(base + {total_offset}));
    for (size_t i = 0; i < CAP; ++i)
        printf("%zu %.7g\\n", i, (double)score[i]);
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
    total = 0.0
    for line in run.stdout.splitlines():
        head, value = line.split(maxsplit=1)
        if head == "total":
            total = float(value)
        else:
            scores.append(float(value))
    return scores, total


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_fused_and_scalar_paths_agree() -> None:
    """Register-carry, fuse-through-filter, and per-stage fallback all agree."""
    with tempfile.TemporaryDirectory(prefix="lsfuse_") as tmp:
        work = Path(tmp)
        fused_scores, fused_total = _build_and_run(
            FUSED_SOURCE,
            work / "f",
            "LOCKSTEP_OFFSET_STREAM_ALERTSSCORED_SCORE",
            _TOTAL_OFFSET_NO_FILTER,
        )
        filter_scores, filter_total = _build_and_run(
            FILTER_FUSED_SOURCE,
            work / "ff",
            "LOCKSTEP_OFFSET_STREAM_ALERTSKEPT_SCORE",
            _TOTAL_OFFSET_WITH_FILTER,
        )
        scalar_scores, scalar_total = _build_and_run(
            SCALAR_SOURCE,
            work / "s",
            "LOCKSTEP_OFFSET_STREAM_ALERTSKEPT_SCORE",
            _TOTAL_OFFSET_WITH_FILTER,
        )

    assert len(fused_scores) == CAPACITY
    assert fused_scores == filter_scores == scalar_scores

    # The folded reduction agrees across all three lowerings (modulo float
    # reassociation between the register reduction and the scalar strip-mine).
    assert fused_total == pytest.approx(scalar_total, rel=1e-5, abs=1e-2)
    assert filter_total == pytest.approx(scalar_total, rel=1e-5, abs=1e-2)

    # And all match the source semantics: score = value * 0.1 * 1.6.
    expected_scores = [((i % 97) + 1) * 0.25 * 0.1 * 1.6 for i in range(CAPACITY)]
    for i in range(CAPACITY):
        assert fused_scores[i] == pytest.approx(expected_scores[i], rel=1e-5, abs=1e-5)
    assert fused_total == pytest.approx(sum(expected_scores), rel=1e-4, abs=1e-1)
