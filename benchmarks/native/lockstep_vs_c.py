#!/usr/bin/env python3
"""Compare Lockstep's shipped machine code against hand-written idiomatic C.

The other native harnesses in this directory pit Lockstep against *itself* --
SoA vs AoS layout, fused vs per-stage loops. They all show Lockstep's model
winning. This one is the honest external question: for the same computation,
does the code Lockstep actually emits keep up with the obvious single-pass C
loop a human would write by hand?

For each workload it:

1. compiles the real ``.lock`` source to LLVM IR via ``compile_lockstep`` (the
   same entry point ``run_native.py`` uses) -- this is the ``Lockstep_Tick``
   Lockstep ships;
2. links that IR against a hand-written C reference kernel that computes the
   *same* end-to-end transform in a single fused pass over the *same* SoA arena
   layout (identical byte offsets, taken from the generated header);
3. primes two identical arenas, times ``Lockstep_Tick`` and the C reference in
   the same process, and verifies both produce the same output checksum before
   reporting throughput and the ``lockstep / c`` ratio.

A ratio >= 1.0 means Lockstep matches or beats hand-written C; < 1.0 means the C
baseline wins. The interesting, honest result is that they diverge by workload:

* ``particle_energy`` is a single fused kernel, so Lockstep's codegen already
  emits essentially the loop a human would -- expect near parity.
* ``telemetry_filter_aggregation`` and ``multi_stage_pipeline`` are multi-stage
  pipelines ending in a filter/accumulator. Codegen lowers them to one
  strip-mined loop *per stage*, materializing the intermediate streams the
  optimizer's own plan says it eliminates. The single-pass C reference makes one
  trip through memory instead of several, so it wins -- and this harness measures
  by how much. That gap is the throughput a fusion-through-filter codegen change
  would recover (see ``fusion_probe.py`` for the same delta measured internally).

Built with ``clang -O3 -march=native -ffast-math`` (matching the reduction
codegen and the sibling probes); absolute numbers are host-dependent, so read
the ratio.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
for _p in (str(BASE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lockstep_compiler.compiler import compile_lockstep  # noqa: E402

# Reuse the arena/offset parsing already proven out by the native harness.
from run_native import (  # noqa: E402
    INTRINSICS_C,
    WORKLOAD_DIR,
    BenchmarkError,
    StreamField,
    _build_plan,
    _offset_macros,
)


@dataclass
class Reference:
    """A hand-written C competitor for one workload.

    ``body`` is the fused kernel (operates on ``uint8_t* base`` using the
    generated ``LOCKSTEP_OFFSET_STREAM_*`` macros). ``checksum_offset_macro`` is
    the output float column both paths are summed over to prove they agree.
    """

    body: str
    checksum_offset_macro: str


# --- Hand-written single-pass competitors, one per workload -----------------
#
# Each reference computes the exact end-to-end transform of the corresponding
# .lock pipeline (see benchmarks/workloads/*.lock), in one loop, using the same
# per-stage float arithmetic so the checksum matches Lockstep bit-for-bit modulo
# reassociation. The arena byte offsets come from the generated header, so both
# the reference and Lockstep_Tick read/write the identical SoA columns.

_REFERENCES: dict[str, Reference] = {
    # Single fused kernel: integrate positions/velocities + accumulate kinetic
    # energy. Lockstep already fuses this, so the reference is the same loop.
    "particle_energy": Reference(
        checksum_offset_macro="LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_PX",
        body=r"""
    const int32_t* in_id   = (const int32_t*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_ID);
    const float*   in_px    = (const float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_PX);
    const float*   in_py    = (const float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_PY);
    const float*   in_vx    = (const float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_VX);
    const float*   in_vy    = (const float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_VY);
    const float*   in_mass  = (const float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESIN_MASS);
    int32_t* o_id   = (int32_t*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_ID);
    float*   o_px   = (float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_PX);
    float*   o_py   = (float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_PY);
    float*   o_vx   = (float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_VX);
    float*   o_vy   = (float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_VY);
    float*   o_mass = (float*)(base + LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_MASS);
    float ke = 0.0f;
    for (size_t i = 0; i < CAP; ++i) {
        float vx = in_vx[i], vy = in_vy[i];
        ke += 0.5f * in_mass[i] * (vx * vx + vy * vy);
        o_id[i]   = in_id[i];
        o_px[i]   = in_px[i] + vx;
        o_py[i]   = in_py[i] + vy;
        o_vx[i]   = vx * 0.995f;
        o_vy[i]   = vy * 0.995f;
        o_mass[i] = in_mass[i];
    }
    bench_keep_f = ke;
""",
    ),
    # filter (KeepHealthy) -> shader+accumulate (AggregateReadings). Lockstep
    # emits two strip-mined loops and materializes telemetryFiltered; the
    # reference does it in one pass.
    "telemetry_filter_aggregation": Reference(
        checksum_offset_macro="LOCKSTEP_OFFSET_STREAM_TELEMETRYAGGREGATED_READING",
        body=r"""
    const int32_t* in_dev = (const int32_t*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYRAW_DEVICEID);
    const float*   in_read = (const float*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYRAW_READING);
    const uint8_t* in_ok   = (const uint8_t*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYRAW_HEALTHY);
    int32_t* o_dev  = (int32_t*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYAGGREGATED_DEVICEID);
    float*   o_read = (float*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYAGGREGATED_READING);
    uint8_t* o_ok   = (uint8_t*)(base + LOCKSTEP_OFFSET_STREAM_TELEMETRYAGGREGATED_HEALTHY);
    float total = 0.0f; int32_t count = 0;
    for (size_t i = 0; i < CAP; ++i) {
        float r = in_read[i];
        o_dev[i]  = in_dev[i];
        o_read[i] = r;
        o_ok[i]   = in_ok[i];
        total += r; count += 1;
    }
    bench_keep_f = total + (float)count;
""",
    ),
    # Normalize -> Score(+accumulate) -> KeepActive(filter). Lockstep emits three
    # strip-mined loops and materializes eventsEnriched + alertsScored; the
    # reference collapses the whole chain to one pass.
    "multi_stage_pipeline": Reference(
        checksum_offset_macro="LOCKSTEP_OFFSET_STREAM_ALERTSACTIVE_SCORE",
        body=r"""
    const int32_t* in_dev = (const int32_t*)(base + LOCKSTEP_OFFSET_STREAM_EVENTSRAW_DEVICEID);
    const float*   in_val = (const float*)(base + LOCKSTEP_OFFSET_STREAM_EVENTSRAW_VALUE);
    const uint8_t* in_act = (const uint8_t*)(base + LOCKSTEP_OFFSET_STREAM_EVENTSRAW_ACTIVE);
    int32_t* o_dev   = (int32_t*)(base + LOCKSTEP_OFFSET_STREAM_ALERTSACTIVE_DEVICEID);
    float*   o_score = (float*)(base + LOCKSTEP_OFFSET_STREAM_ALERTSACTIVE_SCORE);
    uint8_t* o_act   = (uint8_t*)(base + LOCKSTEP_OFFSET_STREAM_ALERTSACTIVE_ACTIVE);
    float ssum = 0.0f; int32_t scount = 0;
    for (size_t i = 0; i < CAP; ++i) {
        float norm = in_val[i] * 0.1f;   /* Normalize */
        float s = norm * 1.6f;           /* Score */
        o_dev[i]   = in_dev[i];          /* KeepActive copy */
        o_score[i] = s;
        o_act[i]   = in_act[i];
        ssum += s; scount += 1;
    }
    bench_keep_f = ssum + (float)scount;
""",
    ),
}

WORKLOADS = tuple(_REFERENCES)


def _prime_lines(input_fields: list[StreamField]) -> str:
    """Deterministic input priming, byte-identical to run_native.py's driver."""

    lines: list[str] = []
    for idx, field in enumerate(input_fields):
        cast = f"({field.c_type}*)(base + {field.offset})"
        if field.is_integer:
            value = f"({field.c_type})((i * 2654435761u) + {idx}u)"
        else:
            value = (
                f"({field.c_type})(0.5f + ((float)((i + {idx}) % 4096) * 0.000244140625f))"
            )
        lines.append(
            f"    {{ {field.c_type}* col = {cast}; "
            f"for (size_t i = 0; i < CAP; ++i) col[i] = {value}; }}"
        )
    return "\n".join(lines) if lines else "    /* no primeable inputs */"


def _offset_defines(offsets: dict[str, int]) -> str:
    return "\n".join(f"#define {name} {value}" for name, value in sorted(offsets.items()))


def _render_driver(
    arena_bytes: int,
    capacity: int,
    offsets: dict[str, int],
    input_fields: list[StreamField],
    reference: Reference,
    iterations: int,
    warmup: int,
) -> str:
    checksum_offset = offsets[reference.checksum_offset_macro]
    return f"""/* Auto-generated by benchmarks/native/lockstep_vs_c.py -- do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Portability shim: the MSVC/UCRT toolchain used by clang on Windows lacks
   POSIX clock_gettime and C11 aligned_alloc, so provide equivalents. */
#if defined(_WIN32)
#include <windows.h>
#include <malloc.h>
static void* bench_aligned_alloc(size_t align, size_t size) {{
    return _aligned_malloc(size, align);
}}
static void bench_aligned_free(void* p) {{ _aligned_free(p); }}
static double bench_now_ns(void) {{
    LARGE_INTEGER freq, ctr;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    return (double)ctr.QuadPart * 1e9 / (double)freq.QuadPart;
}}
#else
#include <time.h>
static void* bench_aligned_alloc(size_t align, size_t size) {{
    size_t rounded = (size + align - 1) & ~(align - 1);  /* C11 requires a multiple */
    return aligned_alloc(align, rounded);
}}
static void bench_aligned_free(void* p) {{ free(p); }}
static double bench_now_ns(void) {{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}}
#endif

#define ARENA_BYTES ((size_t){arena_bytes})
#define CAP ((size_t){capacity})
#define ITERS {iterations}
#define WARMUP {warmup}
#define CHECKSUM_OFFSET ((size_t){checksum_offset})

{_offset_defines(offsets)}

void Lockstep_Tick(void* arena);

/* Sink for the C reference's accumulator so the optimizer cannot delete the
   fold. Volatile store per timed call is negligible against a full arena pass. */
static volatile float bench_keep_f = 0.0f;

static void prime(uint8_t* base) {{
{_prime_lines(input_fields)}
}}

static void c_reference(uint8_t* base) {{
{reference.body}
}}

static double checksum(const uint8_t* base) {{
    const float* col = (const float*)(base + CHECKSUM_OFFSET);
    double acc = 0.0;
    for (size_t i = 0; i < CAP; ++i) acc += (double)col[i];
    return acc;
}}

int main(void) {{
    uint8_t* a_ls = (uint8_t*)bench_aligned_alloc(64, ARENA_BYTES);
    uint8_t* a_c  = (uint8_t*)bench_aligned_alloc(64, ARENA_BYTES);
    if (a_ls == NULL || a_c == NULL) {{ fprintf(stderr, "alloc failed\\n"); return 2; }}
    memset(a_ls, 0, ARENA_BYTES);
    memset(a_c, 0, ARENA_BYTES);
    prime(a_ls);
    prime(a_c);

    for (int w = 0; w < WARMUP; ++w) {{ Lockstep_Tick(a_ls); c_reference(a_c); }}

    double t0 = bench_now_ns();
    for (int it = 0; it < ITERS; ++it) Lockstep_Tick(a_ls);
    double t1 = bench_now_ns();
    for (int it = 0; it < ITERS; ++it) c_reference(a_c);
    double t2 = bench_now_ns();

    double ls_ns = (t1 - t0) / (double)ITERS;
    double c_ns  = (t2 - t1) / (double)ITERS;

    printf("{{\\"lockstep_ns\\": %.3f, \\"c_ns\\": %.3f, "
           "\\"checksum_lockstep\\": %.6f, \\"checksum_c\\": %.6f}}\\n",
           ls_ns, c_ns, checksum(a_ls), checksum(a_c));
    bench_aligned_free(a_ls);
    bench_aligned_free(a_c);
    return 0;
}}
"""


def _run_workload(
    name: str,
    iterations: int,
    warmup: int,
    clang: str,
    keep_dir: Path | None,
    target_width: int,
) -> dict[str, object]:
    source_path = WORKLOAD_DIR / f"{name}.lock"
    if not source_path.exists():
        raise BenchmarkError(f"workload source not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")

    plan = _build_plan(name, source, target_width)

    # Recompile once for the header so the reference can address every column by
    # its generated offset macro (the plan only carries input/checksum leaves).
    result = compile_lockstep(source, verbose=False, target_width=target_width)
    offsets = _offset_macros(getattr(result, "c_header", "") or "")

    reference = _REFERENCES[name]
    if reference.checksum_offset_macro not in offsets:
        raise BenchmarkError(
            f"{name}: expected offset macro {reference.checksum_offset_macro} "
            "not found in generated header"
        )

    work_dir = keep_dir / name if keep_dir else Path(tempfile.mkdtemp(prefix=f"lsvc_{name}_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    ir_path = work_dir / f"{name}.ll"
    driver_path = work_dir / f"{name}_driver.c"
    exe_path = work_dir / f"{name}_bench"
    ir_path.write_text(plan.ir_text, encoding="utf-8")
    driver_path.write_text(
        _render_driver(
            plan.arena_bytes,
            plan.capacity,
            offsets,
            plan.input_fields,
            reference,
            iterations,
            warmup,
        ),
        encoding="utf-8",
    )

    compile_cmd = [
        clang,
        "-O3",
        "-march=native",
        "-ffast-math",
        "-Wno-override-module",
        str(driver_path),
        str(ir_path),
        str(INTRINSICS_C),
        "-o",
        str(exe_path),
        "-lm",
    ]
    proc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BenchmarkError(f"{name}: clang failed ({proc.returncode})\n{proc.stderr.strip()}")

    run = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        raise BenchmarkError(
            f"{name}: benchmark executable failed ({run.returncode})\n{run.stderr.strip()}"
        )

    try:
        measured = json.loads(run.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BenchmarkError(f"{name}: could not parse driver output: {run.stdout!r}") from exc

    cs_ls = float(measured["checksum_lockstep"])
    cs_c = float(measured["checksum_c"])
    scale = max(1.0, abs(cs_ls), abs(cs_c))
    if abs(cs_ls - cs_c) > 1e-3 * scale:
        raise BenchmarkError(
            f"{name}: Lockstep and C reference disagree "
            f"(checksum {cs_ls} vs {cs_c}); comparison is invalid"
        )

    ls_ns = float(measured["lockstep_ns"])
    c_ns = float(measured["c_ns"])
    rows = plan.rows

    def mrows(ns: float) -> float:
        return (rows / (ns * 1e-9)) / 1e6 if ns > 0 else 0.0

    if keep_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "workload": name,
        "target_width": target_width,
        "rows_per_tick": rows,
        "lockstep_per_tick_us": round(ls_ns / 1000.0, 4),
        "c_per_tick_us": round(c_ns / 1000.0, 4),
        "lockstep_mrows_per_sec": round(mrows(ls_ns), 1),
        "c_mrows_per_sec": round(mrows(c_ns), 1),
        # >= 1.0: Lockstep matches/beats hand-written C.  < 1.0: C wins.
        "lockstep_vs_c_ratio": round(c_ns / ls_ns, 3) if ls_ns > 0 else 0.0,
        "checksum": round(cs_ls, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        action="append",
        help="Run only the selected workload(s); repeatable.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Timed calls per kernel per workload (default: 2000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed warmup iterations before measurement (default: 20).",
    )
    parser.add_argument("--clang", default="clang", help="C/LLVM compiler to use.")
    parser.add_argument(
        "--target-width",
        type=int,
        default=8,
        help="SIMD vector width the compiler lowers Lockstep to (default: 8).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path.")
    parser.add_argument(
        "--keep-artifacts",
        type=Path,
        help="Keep generated IR/driver/exe under this directory for inspection.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.target_width <= 0:
        raise SystemExit("--target-width must be positive")

    clang = shutil.which(args.clang)
    if clang is None:
        raise SystemExit(
            f"'{args.clang}' not found on PATH; native benchmarks require an LLVM/clang toolchain."
        )

    selected = args.workload or list(WORKLOADS)
    results: list[dict[str, object]] = []
    for name in selected:
        results.append(
            _run_workload(
                name, args.iterations, args.warmup, clang, args.keep_artifacts, args.target_width
            )
        )

    payload = {
        "schema_version": 1,
        "kind": "lockstep_vs_c",
        "iterations": args.iterations,
        "target_width": args.target_width,
        "results": results,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"Lockstep vs hand-written C ({args.iterations} iters each, "
        f"target-width {args.target_width}, {clang})"
    )
    print("ratio = C time / Lockstep time; >= 1.0 Lockstep matches/beats C, < 1.0 C wins.\n")
    print("| workload | rows/tick | Lockstep Mrows/s | C Mrows/s | ratio |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            f"| {row['workload']} | {row['rows_per_tick']} "
            f"| {row['lockstep_mrows_per_sec']} | {row['c_mrows_per_sec']} "
            f"| {row['lockstep_vs_c_ratio']}x |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
